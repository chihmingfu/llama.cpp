#!/bin/bash

MODEL_DIR="./models"
MODEL_FILE="Llama-3.2-1B-Instruct-IQ3_M.gguf"
PROMPT="Hello, how are you?"
TOKEN_NUM=128
THREADS=1
OUTPUT_FILE="llama_cli_output.txt"

# 參數處理
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --prompt)
      PROMPT="$2"
      shift
      ;;
    --tokens)
      TOKEN_NUM="$2"
      shift
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift
      ;;
    --model)
      MODEL_FILE="$2"
      shift
      ;;
    *)
      ;;
  esac
  shift
done

echo "[INFO] Running llama-cli with model: $MODEL_FILE"
echo "[INFO] Prompt: $PROMPT"
time ./build/bin/llama-cli \
  -m ${MODEL_DIR}/${MODEL_FILE} \
  -p "${PROMPT}" -n ${TOKEN_NUM} --no-mmap --threads ${THREADS} | tee ${OUTPUT_FILE}
