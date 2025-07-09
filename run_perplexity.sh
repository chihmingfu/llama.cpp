#!/bin/bash

MODEL_DIR="./models"
DATA_FILE="./wikitext-2-raw/wiki.test.raw"

echo "[INFO] Running llama-perplexity with Q4_K_M..."
time ./build/bin/llama-perplexity -m ${MODEL_DIR}/Llama-3.2-1B-Instruct-Q4_K_M.gguf -f ${DATA_FILE}

echo "[INFO] Running llama-perplexity with IQ3_M..."
time ./build/bin/llama-perplexity -m ${MODEL_DIR}/Llama-3.2-1B-Instruct-IQ3_M.gguf -f ${DATA_FILE}
