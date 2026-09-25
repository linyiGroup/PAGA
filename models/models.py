import torch
import torch.nn as nn


class Adaptation(nn.Module):
    def __init__(self, input_dim, feat_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Gating(nn.Module):
    def __init__(self, feat_dim=64, sequence_length=34):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feat_dim, sequence_length),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class Alignment(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.proj_1 = nn.Linear(feat_dim, feat_dim)
        self.proj_2 = nn.Linear(feat_dim, feat_dim)
        self.proj_3 = nn.Linear(feat_dim, feat_dim)
        self.relu = nn.ReLU()

    def forward(self, f1, f2, f_f):
        f_1 = self.proj_1(self.relu(f1))
        f_2 = self.proj_2(self.relu(f2))
        f_f = self.proj_3(self.relu(f_f))
        return torch.cat([f_1, f_2, f_f], dim=-1)

class AGAttention(nn.Module):

    def __init__(
            self,
            query_input_dim,
            context_input_dim,
            feat_dim=64,
            num_heads=1,
            sequence_length=34,
    ):
        super().__init__()

        if feat_dim % num_heads != 0:
            raise ValueError(
                "feat_dim must be divisible by num_heads"
            )

        self.output_dim = feat_dim * 3

        self.A_query = Adaptation(
            input_dim=query_input_dim,
            feat_dim=feat_dim,
        )

        self.A_context = Adaptation(
            input_dim=context_input_dim,
            feat_dim=feat_dim,
        )

        self.G_query = Gating(feat_dim, sequence_length)
        self.G_context = Gating(feat_dim, sequence_length)

        self.query_norm = nn.LayerNorm(feat_dim)
        self.context_norm = nn.LayerNorm(feat_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.alignment = Alignment(feat_dim)

    def forward(
        self,
        query,
        context,
        query_mask=None,
        context_mask=None,
    ):
        if query_mask is not None:
            query_mask = query_mask.bool()

        if context_mask is not None:
            context_mask = context_mask.bool()

            if torch.any(context_mask.sum(dim=1) == 0):
                raise ValueError(
                    "Each sample must contain at least "
                    "one valid context residue"
                )

        # Adaptation
        query_A = self.A_query(query)
        context_A = self.A_context(context)

        if query_mask is not None:
            query_A = query_A * query_mask.unsqueeze(-1).to(
                query_A.dtype
            )

        if context_mask is not None:
            context_A = context_A * context_mask.unsqueeze(-1).to(
                context_A.dtype
            )

        # Gating
        joint_features = torch.cat([query_A, context_A], dim=1)

        if query_mask is not None and context_mask is not None:
            joint_mask = torch.cat([query_mask, context_mask], dim=1)
            joint_mask_float = joint_mask.unsqueeze(-1).to(joint_features.dtype)

            global_features = (joint_features * joint_mask_float).sum(dim=1)
            valid_count = joint_mask_float.sum(dim=1).clamp(min=1.0)
            global_features = global_features / valid_count
        else:
            global_features = joint_features.mean(dim=1)

        query_G = self.G_query(global_features).unsqueeze(-1)
        context_G = self.G_context(global_features).unsqueeze(-1)

        if query_mask is not None:
            query_G = query_G * query_mask.unsqueeze(-1).to(query_G.dtype)

        if context_mask is not None:
            context_G = context_G * context_mask.unsqueeze(-1).to(context_G.dtype)

        query_AG = query_A * query_G
        context_AG = context_A * context_G

        # Layer normalization
        query_features = self.query_norm(query_AG)
        context_features = self.context_norm(context_AG)

        if query_mask is not None:
            query_features = (
                query_features
                * query_mask.unsqueeze(-1).to(query_features.dtype)
            )

        if context_mask is not None:
            context_features = (
                context_features
                * context_mask.unsqueeze(-1).to(
                    context_features.dtype
                )
            )

        # Cross-attention
        key_padding_mask = None

        if context_mask is not None:
            key_padding_mask = ~context_mask

        interaction, attention_weights = self.cross_attention(
            query=query_features,
            key=context_features,
            value=context_features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        if query_mask is not None:
            interaction = interaction * query_mask.unsqueeze(-1).to(interaction.dtype)

        if query_A.size(1) != context_A.size(1):
            raise ValueError("Query and context must use the same padded length")
        output = self.alignment(
            f1=query_A,
            f2=context_A,
            f_f=interaction,
        )

        if query_mask is not None:
            output = (
                output
                * query_mask.unsqueeze(-1).to(output.dtype)
            )

        return (
            output,
            attention_weights,
            query_G,
            context_G,
            query_A,
            context_A,
            interaction,
        )

class BindingClassifier(nn.Module):
    def __init__(self, input_dim, sequence_length=34):
        super().__init__()
        self.relu = nn.ReLU()
        self.cls = nn.Linear(input_dim * sequence_length, 2)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.cls(self.relu(x))

class PAGAModel(nn.Module):

    def __init__(
        self,
        input_dim=960,
        feat_dim=64,
        num_heads=1,
        sequence_length=34,
    ):
        super().__init__()

        alignment_dim = feat_dim * 3

        # Peptide-HLA interaction
        self.phla_AG = AGAttention(
            query_input_dim=input_dim,
            context_input_dim=input_dim,
            feat_dim=feat_dim,
            num_heads=num_heads,
        )

        # pHLA-TCR interaction
        #
        # The first AGAttention returns 3 * feat_dim.
        self.tcr_phla_AG = AGAttention(
            query_input_dim=alignment_dim,
            context_input_dim=input_dim,
            feat_dim=feat_dim,
            num_heads=num_heads,
        )

        # The second AGAttention also returns 3 * feat_dim.
        self.classifier = BindingClassifier(
            input_dim=alignment_dim,
            sequence_length=34,
        )

    def forward(
        self,
        hla_embedding,
        peptide_embedding,
        tcr_embedding,
        hla_mask=None,
        peptide_mask=None,
        tcr_mask=None,
        return_details=False,
    ):
        (
            phla_features,
            phla_attention,
            peptide_G,
            hla_G,
            peptide_A,
            hla_A,
            phla_interaction,
        ) = self.phla_AG(
            query=peptide_embedding,
            context=hla_embedding,
            query_mask=peptide_mask,
            context_mask=hla_mask,
        )

        (
            interaction_features,
            tcr_phla_attention,
            phla_G,
            tcr_G,
            phla_A,
            tcr_A,
            tcr_phla_interaction,
        ) = self.tcr_phla_AG(
            query=phla_features,
            context=tcr_embedding,
            query_mask=peptide_mask,
            context_mask=tcr_mask,
        )

        logits = self.classifier(interaction_features)

        if not return_details:
            return logits

        details = {
            "peptide_A": peptide_A,
            "hla_A": hla_A,
            "peptide_G": peptide_G,
            "hla_G": hla_G,
            "phla_interaction": phla_interaction,
            "phla_features": phla_features,
            "phla_attention": phla_attention,
            "phla_A": phla_A,
            "tcr_A": tcr_A,
            "phla_G": phla_G,
            "tcr_G": tcr_G,
            "tcr_phla_interaction": tcr_phla_interaction,
            "interaction_features": interaction_features,
            "tcr_phla_attention": tcr_phla_attention,
        }

        return logits, details

