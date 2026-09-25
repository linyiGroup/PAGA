import sys
# 必须先将当前目录加入环境变量，否则找不到本地的 utils 包
sys.path.append("./")

from utils.utils import load_config
from utils.ESMcEmbedding import (
    loadEmbeddingDict,
    fetchUnembeddedDataPathList,
    fetchUnembeddedSequenceSet,
    loadESMcModel,
    saveEmbedding,
    saveEmbeddingDict,
    saveAtchleyDict,
    saveBlosumDict
)

CONFIG_PATH = "./configs/config_embedding.yaml"

config = load_config(CONFIG_PATH)

# Load the existing sequence-to-embedding dictionary.
embeddingDict = loadEmbeddingDict(
    config["embeddingDictPath"]
)

# Find all CSV files containing TCR, peptide, and HLA data.
dataPathList = fetchUnembeddedDataPathList(
    config["originDataDirs"]
)

# Collect unique CDR3 beta, peptide, and mapped HLA sequences.
seqSet = fetchUnembeddedSequenceSet(
    dataPathList,
    config["squenceColumnName"],
    config["hlaMapping"],
)

print(
    "Total number of sequences requiring embedding:",
    len(seqSet),
)

# Load the local ESMc model.
ESMcModel = loadESMcModel(
    config["ESMcPath"],
    config["device"],
)

ESMcModel.eval()

# Generate and save residue-level embeddings.
embeddingDict = saveEmbedding(
    ESMcModel,
    seqSet,
    embeddingDict,
    config["maxSeqLength"],
    config["embeddingDirPath"],
    config["maxFilesPerDir"],
    config["device"],
    config["errorLogPath"],
)

# Save the updated sequence-to-file dictionary.
saveEmbeddingDict(
    embeddingDict,
    config["embeddingDictPath"],
)

print("Embedding dictionary has been updated.")
print(
    "Total number of embedded sequences:",
    len(embeddingDict),
)
