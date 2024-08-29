# DATA_PATH=/home/ai/data/llava/dataset/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
DATA_PATH="/home/user27/AudioTinyLLaVA/dev.json"
FINETUNE_DATA_PATH=/home/user27/AudioTinyLLaVA/dev.json #finetune annotation file path
IMAGE_PATH=None #pretrain image dir
FINETUNE_IMAGE_PATH=None #finetune image dir

LLM_VERSION=microsoft/phi-2 # llm path in huggingface
# apple/OpenELM-270M-Instruc
# LLM_VERSION=tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B
VT_VERSION=languagebind #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=llama #chat template, other options are: phi, llama, gemmma, etc
VERSION=clotho_audio_caption #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=3072 #max model length for llm
CUDA_VISIBLE_DEVICES=0,1

bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
# bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
