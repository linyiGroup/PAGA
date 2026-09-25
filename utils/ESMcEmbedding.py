import os
import pickle
from pathlib import Path
import blosum

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import EsmSequenceTokenizer


VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def cleanSequence(sequence):
    if pd.isna(sequence):
        return None

    sequence = "".join(str(sequence).split()).upper()

    if not sequence:
        return None

    if not set(sequence).issubset(VALID_AMINO_ACIDS):
        return None

    return sequence


def loadEmbeddingDict(path):
    if not os.path.exists(path):
        return {}

    with open(path, "rb") as file:
        return pickle.load(file)


def saveEmbeddingDict(embeddingDict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(embeddingDict, file)


def fetchUnembeddedDataPathList(dirsPath):
    dataPathList = []

    for directory in dirsPath:
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Data directory not found: {directory}"
            )

        dataPathList.extend(
            str(path)
            for path in directory.rglob("*.csv")
        )

    return sorted(dataPathList)


def loadHLAMapping(hlaMapping):
    dictionaryData = pd.read_csv(
        hlaMapping["dictionaryPath"]
    )

    alleleColumn = hlaMapping["dictionaryAlleleColumn"]
    sequenceColumn = hlaMapping["dictionarySequenceColumn"]

    if alleleColumn not in dictionaryData.columns:
        raise KeyError(
            f"HLA dictionary column not found: {alleleColumn}"
        )

    if sequenceColumn not in dictionaryData.columns:
        raise KeyError(
            f"HLA sequence column not found: {sequenceColumn}"
        )

    hlaSequenceDict = {}

    for _, row in dictionaryData.iterrows():
        allele = str(row[alleleColumn]).strip().upper()
        sequence = cleanSequence(row[sequenceColumn])

        if allele and sequence:
            hlaSequenceDict[allele] = sequence

    return hlaSequenceDict


def fetchUnembeddedSequenceSet(
    dataPathList,
    squenceColumnName,
    hlaMapping=None,
):
    seqSet = set()
    missingHLA = set()

    useHLAMapping = (
        hlaMapping is not None
        and hlaMapping.get("enabled", False)
    )

    if useHLAMapping:
        hlaSequenceDict = loadHLAMapping(hlaMapping)
        hlaAlleleColumn = hlaMapping["dataAlleleColumn"]

    for path in dataPathList:
        data = pd.read_csv(path)

        # Collect CDR3 beta, peptide and existing HLA sequences.
        for column in squenceColumnName:
            if column not in data.columns:
                continue

            for value in data[column].dropna().unique():
                sequence = cleanSequence(value)

                if sequence:
                    seqSet.add(sequence)

        # Map HLA class I alleles to amino acid sequences.
        if useHLAMapping and hlaAlleleColumn in data.columns:
            for value in data[hlaAlleleColumn].dropna().unique():
                allele = str(value).strip().upper()

                if allele in hlaSequenceDict:
                    seqSet.add(hlaSequenceDict[allele])
                else:
                    missingHLA.add(allele)

    if missingHLA:
        examples = ", ".join(sorted(missingHLA)[:20])

        raise ValueError(
            f"{len(missingHLA)} HLA alleles could not be mapped. "
            f"Examples: {examples}"
        )

    return seqSet


def loadESMcModel(path, device):
    model = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model = model.to(torch.device(device))
    model.eval()

    return model


def encodeOneSeq(
    sequence,
    model,
    maxSeqLength,
    device,
):
    if len(sequence) > maxSeqLength:
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds "
            f"maxSeqLength {maxSeqLength}: {sequence}"
        )

    tokenizer = EsmSequenceTokenizer()

    protein = ESMProtein(sequence=sequence)
    proteinTensor = model.encode(protein).to(device)

    paddingLength = maxSeqLength - len(sequence)

    padding = torch.full(
        (paddingLength,),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )

    eos = torch.tensor(
        [tokenizer.eos_token_id],
        dtype=torch.long,
        device=device,
    )

    proteinTensor.sequence = torch.cat(
        [
            proteinTensor.sequence[:-1],
            padding,
            eos,
        ]
    )

    return proteinTensor


def embedOneSeq(model, proteinTensor):
    output = model.logits(
        proteinTensor,
        LogitsConfig(return_embeddings=True),
    )

    # Remove CLS and EOS embeddings.
    return output.embeddings[0, 1:-1]


def saveEmbedding(
    model,
    seqSet,
    embeddingDict,
    maxSeqLength,
    embeddingDirPath,
    maxFilesPerDir,
    device,
    errorLogPath="",
):
    embeddingRoot = Path(embeddingDirPath)
    embeddingRoot.mkdir(parents=True, exist_ok=True)

    if errorLogPath and os.path.exists(errorLogPath):
        os.remove(errorLogPath)

    currentDirNum = 1

    with torch.no_grad():
        for sequence in tqdm(sorted(seqSet)):
            if sequence in embeddingDict:
                continue

            try:
                while True:
                    saveDirectory = (
                        embeddingRoot / str(currentDirNum)
                    )
                    saveDirectory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    fileCount = len(
                        list(saveDirectory.glob("*.npy"))
                    )

                    if fileCount < maxFilesPerDir:
                        break

                    currentDirNum += 1

                proteinTensor = encodeOneSeq(
                    sequence,
                    model,
                    maxSeqLength,
                    device,
                )

                embedding = embedOneSeq(
                    model,
                    proteinTensor,
                )

                embedding = (
                    embedding.float().cpu().numpy()
                )

                savePath = (
                    saveDirectory / f"{sequence}.npy"
                )

                np.save(savePath, embedding)

                relativePath = Path(
                    str(currentDirNum),
                    f"{sequence}.npy",
                )

                embeddingDict[sequence] = (
                    relativePath.as_posix()
                )

            except Exception as error:
                message = (
                    f"Error: {error}\n"
                    f"Sequence: {sequence}\n\n"
                )

                print(message)

                if errorLogPath:
                    with open(
                        errorLogPath,
                        "a",
                        encoding="utf-8",
                    ) as file:
                        file.write(message)

    return embeddingDict


def saveAtchleyDict(seqSet, maxSeqLength, atchleyFactorPath, savePath):
    """
    基于 seqSet 离线生成 Atchley 特征，并持久化保存为 pkl 字典
    """
    atchleyDict = loadEmbeddingDict(savePath)
    atchleyFactor = pd.read_csv(atchleyFactorPath, index_col=0)

    # 过滤出还没算过的序列
    unprocessed_seqs = [seq for seq in seqSet if seq not in atchleyDict]
    if not unprocessed_seqs:
        return atchleyDict

    for seq in tqdm(sorted(unprocessed_seqs), desc="Generating Atchley"):
        try:
            # 查表
            embedding = [atchleyFactor.loc[a].tolist() for a in seq]

            # Padding (老项目的逻辑：不够 maxSeqLength 就用0补齐)
            pad_len = maxSeqLength - len(embedding)
            if pad_len > 0:
                embedding += [[0.0] * len(embedding[0])] * pad_len

            atchleyDict[seq] = np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error generating Atchley for seq: {seq}. Error: {e}")

    saveEmbeddingDict(atchleyDict, savePath)
    return atchleyDict


def saveBlosumDict(seqSet, maxSeqLength, blosumNum, savePath):
    """
    基于 seqSet 离线生成指定版本(如 50, 62)的 BLOSUM 特征
    """
    blosumDict = loadEmbeddingDict(savePath)
    bm = blosum.BLOSUM(blosumNum)
    star_pad = list(bm["*"].values())  # 老项目的逻辑：用 '*' 对应的向量进行 Padding

    unprocessed_seqs = [seq for seq in seqSet if seq not in blosumDict]
    if not unprocessed_seqs:
        return blosumDict

    for seq in tqdm(sorted(unprocessed_seqs), desc=f"Generating BLOSUM {blosumNum}"):
        try:
            # 查表
            embedding = [list(bm[a].values()) for a in seq]

            # Padding
            pad_len = maxSeqLength - len(embedding)
            if pad_len > 0:
                embedding += [star_pad] * pad_len

            blosumDict[seq] = np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error generating BLOSUM {blosumNum} for seq: {seq}. Error: {e}")

    saveEmbeddingDict(blosumDict, savePath)
    return blosumDict