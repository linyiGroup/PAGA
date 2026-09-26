# PAGA: Progressive Adaptive Gated Attention enhances multi-source heterogeneous feature fusion for TCR-pHLA binding prediction

This repository contains the release implementation of **PAGA**, a two-stage neural network for predicting TCR–peptide–HLA binding. 

PAGA uses an Attention-Gated Attention (AGAttention) module at each stage. Each module performs:

1. feature adaptation to a common latent dimension;
2. residue-level gating;
3. cross-attention to model interactions between the two sequences; and
4. alignment of the adapted and interaction features.

The first AGAttention module models peptide–HLA (`pHLA`) binding. The second module uses the learned pHLA representation as the query and models pHLA–TCR (`pTCR`) binding. The resulting features are passed to a two-class binding classifier.

## Repository structure

```text
PAGA_release/
├── 01_generate_embeddings.py       # Generate residue-level ESMc embeddings
├── 02_train_val_PAGA.py            # Train and validate one experiment setting
├── 03_test_PAGA.py                 # Independent and external test evaluation
├── 04_evaluate_test_PAGA.py        # Validation, independent, and external evaluation
├── configs/
│   ├── config_embedding.yaml        # Embedding-generation configuration
│   ├── config_model.yaml            # PAGA architecture configuration
│   ├── config_random.yaml           # random experiment configuration
│   ├── config_reftcr.yaml           # reftcr experiment configuration
│   ├── config_unipep.yaml           # unipep experiment configuration
│   └── requirements.txt             # Python dependencies
├── datasets/
│   ├── data_TCRpHLA/
│   │   ├── random/                  # train/validation/test/dbpepneo data
│   │   ├── reftcr/                  # train/validation/test/dbpepneo data
│   │   └── unipep/                  # train/validation/test/dbpepneo data
│   ├── data_interpretability/
│   │   ├── pHLA/                    # data for pHLA interpretability analysis
│   │   └── pTCR/                    # data for pTCR interpretability analysis
│   ├── dataset.py                   # PyTorch dataset and embedding loader
│   └── hla_classI_sequence_dict.csv # HLA allele-to-sequence mapping
├── final_model_states/
│   ├── PAGA_random.pt               # Released random checkpoint
│   ├── PAGA_reftcr.pt               # Released reftcr checkpoint
│   └── PAGA_unipep.pt               # Released unipep checkpoint
├── models/
│   ├── ESMc300m.pt                  # ESMc-300M model used for embeddings
│   └── models.py                    # PAGA model definition
├── outputs/                         # Evaluation logs
└── utils/                           # Embedding, configuration, training, and metric utilities
```

## Environment setup

The release was prepared for Python 3.12. Create an environment and install the pinned dependencies:

```bash
conda create -n paga python=3.12
conda activate paga
pip install -r configs/requirements.txt
```

The dependency list contains PyTorch and the ESM SDK. For GPU execution, install a PyTorch build compatible with the CUDA version on the target machine if the pinned wheel is not suitable.

## Dataset format

The training and evaluation CSV files are organized as follows:

```text
datasets/data_TCRpHLA/<experiment>/
├── train_data.csv
├── val_data.csv
├── test_data.csv
└── dbpepneo_data.csv
```

`<experiment>` is one of `random`, `reftcr`, or `unipep`.

Every input CSV must contain at least these columns:

| Column | Description |
| --- | --- |
| `CDR3` | TCR CDR3β amino-acid sequence |
| `MT_pep` | peptide/antigen amino-acid sequence |
| `HLA_sequence` | HLA class-I pseudo-sequence |
| `Label` | Binding label, normally `0` or `1` |

`HLA_type` and `peplen` are retained as metadata but are not required by the PyTorch dataset. Sequences are converted to uppercase, whitespace is removed, and the current configuration uses a maximum sequence length of 34 residues. The released HLA pseudo-sequences and model configuration use a residue embedding dimension of 960.

## Embedding generation

The model consumes residue-level ESMc embeddings. The release includes `models/ESMc300m.pt`, but generated sequence embeddings are not stored in the repository. Generate them from the project root with:

```bash
python 01_generate_embeddings.py
```

The default embedding configuration reads CSV files below `datasets/data_TCRpHLA/` and writes:

```text
models/embeddingDict.pkl
models/ESMcEmbedding/
```

The HLA mapping file uses the columns `HLA` and `sequence` and is configured in `configs/config_embedding.yaml`. If a new dataset uses HLA alleles instead of `HLA_sequence`, update the mapping settings and ensure that the allele names are present in `datasets/hla_classI_sequence_dict.csv`.

### Embedding path configuration

The three experiment configuration files currently point to an embedding store in a sibling checkout:

```yaml
embedding_dict_path: "../PAGA/models/ESMc/ESMcEmbedding/embeddingDict.pkl"
embedding_root: "../PAGA/models/ESMc/ESMcEmbedding"
```

For a standalone `PAGA_release` checkout, first run the embedding-generation script and then change these two entries in `configs/config_random.yaml`, `configs/config_reftcr.yaml`, and `configs/config_unipep.yaml` to:

```yaml
embedding_dict_path: "./models/embeddingDict.pkl"
embedding_root: "./models/ESMcEmbedding"
```

Alternatively, keep the default paths when the original `PAGA` project is available at `../PAGA` and already contains the required embedding dictionary and `.npy` files.

## Using the released checkpoints

The release provides three checkpoints:

```text
final_model_states/PAGA_random.pt
final_model_states/PAGA_reftcr.pt
final_model_states/PAGA_unipep.pt
```

The evaluation scripts do not take command-line arguments. They select an experiment through the module-level `EXP` variable and load the checkpoint specified by `testing.model_path` in the corresponding configuration file.

### Evaluation with the default settings

`03_test_PAGA.py` defaults to `EXP = "unipep"` and evaluates the independent and external test sets:

```bash
python 03_test_PAGA.py
```

`04_evaluate_test_PAGA.py` defaults to `EXP = "reftcr"` and evaluates the validation, independent test, and external test sets:

```bash
python 04_evaluate_test_PAGA.py
```

To evaluate another released setting, edit the `EXP` value at the top of the script to `random`, `reftcr`, or `unipep`, and verify that the selected configuration points to the matching checkpoint. For example:

```python
EXP = "random"
```

The scripts report:

- ROC-AUC;
- PR-AUC / average precision;
- PPVn;
- accuracy;
- F1 score; and
- Matthews correlation coefficient (MCC).

Logs are written under `outputs/<experiment>/test_log.txt`.

## Training and validation

`02_train_val_PAGA.py` trains one experiment setting and validates it after every epoch. The script currently defaults to:

```python
EXP = "unipep"
```

Set `EXP` to `random`, `reftcr`, or `unipep` before starting a different run:

```bash
python 02_train_val_PAGA.py
```

Training uses the corresponding `configs/config_<experiment>.yaml` file and `configs/config_model.yaml`. Checkpoints and the training log are written below `outputs/<experiment>/`; the training log filename used by the script is `traing_log.txt`.

Important training settings are configured in the experiment YAML files, including batch size, number of epochs, learning rate, weight decay, device, and number of data-loader workers. The default model configuration is:

```yaml
input_dim: 960
feat_dim: 64
num_heads: 1
sequence_length: 34
```

## Reproducibility notes

- Run all commands from the `PAGA_release` project root so that the relative paths in the YAML files resolve correctly.
- Keep the same sequence preprocessing, maximum length, embedding model, and embedding dictionary when comparing results.
- The provided scripts do not set explicit random seeds; repeated training runs can therefore differ slightly.
- If a GPU is unavailable, the training and evaluation utilities fall back to CPU when constructing the device, although embedding generation and the YAML device setting may need to be adjusted for the local machine.
- The released model states are intended for direct evaluation; training from scratch is optional.

## Included evaluation logs

The `outputs/` directory contains the evaluation logs generated for the three released experiment settings. These logs are provided as reference outputs; rerunning evaluation may produce small numerical differences because of hardware and software versions.

## Citation

The corresponding manuscript is not yet published. A citation entry will be added when the paper is available.

## License and contact

No separate license file is included in this release. Please contact the project authors before redistributing the code, model checkpoints, or datasets.
