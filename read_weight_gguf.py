from gguf import GGUFReader
import numpy as np

# 載入 GGUF 模型
model_path= "./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# 讀入你的 GGUF 模型檔案
reader = GGUFReader(model_path)

# 取出所有 tensor name
#tensor_names = [tensor.name for tensor in reader.tensors]

# 印出或進一步過濾
#print(tensor_names)
if 1:
    norm_weights = [
        t.name 
        for t in reader.tensors 
        if "norm" in t.name and "weight" in t.name
    ]
    print(norm_weights)
    
tensor = next(t for t in reader.tensors if t.name == "blk.0.attn_k.weight")
print("shape:", tensor.shape)
print("dtype:", tensor.dtype)