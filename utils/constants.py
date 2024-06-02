# model names
TRANSFORMER_XL = "transformer_xl"
GPT2 = "gpt2"
GPT2_MEDIUM = "gpt2-medium"
GPT2_LARGE = "gpt2-large"
GPT2_XL = "gpt2-xl"
T5 = "t5-base"
T5_LARGE = "t5-large"
T5_3B = "t5-3b"
OPT = "opt"
OPT_125M = "opt-125m"
OPT_350M = "opt-350m"
OPT_1_3B = "opt-1.3b"
OPT_2_7B = "opt-2.7b"
OPT_13B = "opt-13b"
OPT_30B = "opt-30b"
MISTRAL = "mistral"
MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
LLAMA = "llama"
LLAMA_7B = "meta-llama/Llama-2-7b"
LLAMA_13B = "meta-llama/Llama-2-13b"

BLOOM = "bloom"
BLOOM_560M = "bloom-560m"
BLOOM_1b = "bloom-1b1"
BLOOM_3b = "bloom-3b"
BLOOM_7b1 = "bloom-7b1"

# accelerators
SVD = "SVD"
PCA = "PCA"
QUANTIZATION = "QUANTIZATION"
QUANTIZATION_GPU = "QUANTIZATION_GPU"
QUANTIZATION_NVIDIA = "QUANTIZATION_NVIDIA"
PRUNING = "PRUNING"

# Quantization types
DynamicQ = "Dynamic Quantization"
StaticQ = "Static Quantization"
QAT = "Quantization Aware Training"

# Quantization example input
EXAMPLE_INPUT = "Hello, my name is"

# datasets
BILLSUM = "billsum"
WIKITEXT2 = "wikitext-2-raw-v1"
SQUAD = "squad"
OPENWEBTEXT = "openwebtext"
SAMSUM = "samsum"
REFINED_WEB = "refined_web"

# GLUE
QNLI = "qnli"
SST2 = "sst2"
COLA = "cola"
STSB = "stsb"
RTE = "rte"
MRPC = "mrpc"
GLUE = [QNLI, SST2, COLA, STSB, RTE, MRPC]

# Super GLUE
WIC = "wic"
BOOLQ = "boolq"
MULTIRC = "multirc"
SUPERGLUE = [WIC, BOOLQ, MULTIRC]

# Number of blocks
NUM_BLOCKS_GPT2 = 12
NUM_BLOCKS_GPT2_MEDIUM = 24


# Mistral Token Ids
MISTRAL_YES_ID = 5081
MISTRAL_NO_ID = 708

# Context Length
MISTRAL_CONTEXT_LENGTH = 8192
