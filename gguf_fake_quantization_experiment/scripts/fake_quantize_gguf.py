#!/usr/bin/env python3
"""
GGUF Fake Quantization工具 - 專門用於Layer 0 FFN norm權重的BF16 fake quantization

使用方法：
python fake_quantize_gguf.py input.gguf output.gguf --layers "0"
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np

# 加載本地gguf包
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

logger = logging.getLogger(__name__)

def fake_quantize_to_bf16(tensor_f32):
    """
    將F32 tensor進行BF16 fake quantization
    
    BF16格式：1位符號 + 8位指數 + 7位尾數
    方法：保留F32的高16位，低16位清零
    
    Args:
        tensor_f32: 輸入的F32 tensor
    
    Returns:
        fake quantized F32 tensor (BF16精度但F32格式)
    """
    # 將F32轉為32位整數表示
    f32_bits = tensor_f32.astype(np.float32).view(np.uint32)
    
    # 保留高16位（符號+指數+部分尾數），低16位清零
    bf16_bits = (f32_bits >> 16) << 16
    
    # 轉回F32格式
    return bf16_bits.view(np.float32)

def analyze_quantization_effect(original, fake_quantized, tensor_name):
    """分析fake quantization的效果"""
    abs_diff = np.abs(original - fake_quantized)
    rel_diff = abs_diff / (np.abs(original) + 1e-8)
    
    stats = {
        'name': tensor_name,
        'max_abs_diff': np.max(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'max_rel_diff': np.max(rel_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'orig_range': (np.min(original), np.max(original)),
        'fake_range': (np.min(fake_quantized), np.max(fake_quantized)),
        'shape': original.shape,
        'dtype_orig': str(original.dtype),
        'dtype_fake': str(fake_quantized.dtype)
    }
    
    return stats

def fake_quantize_gguf(input_path, output_path, target_layers):
    """
    對GGUF文件中的FFN norm權重進行fake quantization
    
    Args:
        input_path: 輸入GGUF文件路徑
        output_path: 輸出GGUF文件路徑  
        target_layers: 要處理的層列表
    """
    print(f"讀取GGUF文件: {input_path}")
    reader = GGUFReader(input_path)
    
    # 獲取模型架構信息
    arch = reader.get_field("general.architecture")
    if arch:
        arch_name = bytes(arch.parts[arch.data[0]]).decode('utf-8')
    else:
        arch_name = "llama"  # 默認架構
    
    print(f"模型架構: {arch_name}")
    print(f"總tensor數量: {len(reader.tensors)}")
    
    # 創建新的writer
    writer = GGUFWriter(output_path, arch_name)
    
    # 複製所有metadata
    print("複製metadata...")
    copied_fields = 0
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue  # 跳過GGUF內部字段
        try:
            value = field.contents()
            if field.types and len(field.types) > 0:
                value_type = field.types[0]
                # 使用通用的add方法
                if hasattr(writer, 'add_string') and value_type.name == "STRING":
                    writer.add_string(key, value)
                elif hasattr(writer, 'add_uint32') and value_type.name == "UINT32":
                    writer.add_uint32(key, value)
                elif hasattr(writer, 'add_uint64') and value_type.name == "UINT64":
                    writer.add_uint64(key, value)
                elif hasattr(writer, 'add_float32') and value_type.name == "FLOAT32":
                    writer.add_float32(key, value)
                elif hasattr(writer, 'add_array') and value_type.name == "ARRAY":
                    writer.add_array(key, value)
                else:
                    # 嘗試通用添加方法
                    writer.add_string(key, str(value))
                copied_fields += 1
        except Exception as e:
            logger.warning(f"跳過field {key}: {e}")
    
    print(f"複製了 {copied_fields} 個metadata字段")
    
    # 處理tensors
    print("處理tensors...")
    modified_count = 0
    total_ffn_norm = 0
    quantization_stats = []
    
    for tensor in reader.tensors:
        tensor_name = tensor.name
        tensor_data = tensor.data.copy()
        tensor_type = tensor.tensor_type
        
        # 檢查是否為目標權重（FFN norm或rope_freqs）
        if ".ffn_norm.weight" in tensor_name or tensor_name == "rope_freqs.weight":
            total_ffn_norm += 1
            
            # 提取層號或處理特殊權重
            if tensor_name == "rope_freqs.weight":
                # rope_freqs.weight特殊處理，假設為layer 0
                layer_idx = 0
                should_process = 0 in target_layers
            else:
                parts = tensor_name.split(".")
                if len(parts) >= 2 and parts[0] == "blk":
                    try:
                        layer_idx = int(parts[1])
                        should_process = layer_idx in target_layers
                    except ValueError:
                        should_process = False
                else:
                    should_process = False
            
            if should_process:
                print(f"量化 {tensor_name} (層 {layer_idx}): {tensor_type.name} -> BF16 fake")
                
                # 執行fake quantization
                original_data = tensor_data.astype(np.float32)
                fake_quantized_data = fake_quantize_to_bf16(original_data)
                
                # 分析量化效果
                stats = analyze_quantization_effect(original_data, fake_quantized_data, tensor_name)
                quantization_stats.append(stats)
                
                # 顯示量化效果
                print(f"  最大絕對差異: {stats['max_abs_diff']:.2e}")
                print(f"  平均絕對差異: {stats['mean_abs_diff']:.2e}")
                print(f"  最大相對差異: {stats['max_rel_diff']:.2%}")
                print(f"  平均相對差異: {stats['mean_rel_diff']:.2%}")
                
                # 添加fake quantized tensor（保持F32格式）
                writer.add_tensor(tensor_name, fake_quantized_data, raw_dtype=GGMLQuantizationType.F32)
                modified_count += 1
            else:
                # 不在目標層，保持原樣
                writer.add_tensor(tensor_name, tensor_data, raw_dtype=tensor_type)
        else:
            # 非FFN norm權重，保持原樣
            writer.add_tensor(tensor_name, tensor_data, raw_dtype=tensor_type)
    
    # 寫入文件
    print("寫入新的GGUF文件...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"完成！修改了 {modified_count}/{total_ffn_norm} 個FFN norm權重")
    print(f"輸出文件: {output_path}")
    
    return quantization_stats

def save_quantization_stats(stats, output_file):
    """保存量化統計結果"""
    import json
    
    # 轉換numpy類型為可序列化的類型
    serializable_stats = []
    for stat in stats:
        serializable_stat = {}
        for key, value in stat.items():
            if isinstance(value, np.ndarray):
                serializable_stat[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                serializable_stat[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                serializable_stat[key] = int(value)
            else:
                serializable_stat[key] = value
        serializable_stats.append(serializable_stat)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    
    print(f"量化統計結果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="GGUF FFN norm fake quantization tool")
    parser.add_argument("input_file", nargs="?", default="/wsfs/home/jimmy.fu/Works/llama.cpp/models/Llama-3.2-1B-Instruct-f16.gguf", help="輸入GGUF文件路徑")
    parser.add_argument("output_file", nargs="?", default="/home/jimmy.fu/Works/llama.cpp/gguf_fake_quantization_experiment/data/llama32_fake_quant_test.gguf", help="輸出GGUF文件路徑")
    parser.add_argument("--layers", default="0", help="要處理的層，格式: '0' 或 '0,1,2'")
    parser.add_argument("--stats-output", help="量化統計結果輸出文件路徑")
    parser.add_argument("--verbose", action="store_true", help="詳細輸出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # 解析層列表
    try:
        if args.layers.lower() == "all":
            target_layers = list(range(1000))  # 處理所有層
        else:
            target_layers = [int(x.strip()) for x in args.layers.split(",")]
        print(f"目標層: {target_layers}")
    except ValueError:
        print("錯誤：層格式應為 '0' 或 '0,1,2'")
        sys.exit(1)
    
    # 檢查輸入文件
    if not Path(args.input_file).exists():
        print(f"錯誤：輸入文件不存在: {args.input_file}")
        sys.exit(1)
    
    try:
        start_time = time.time()
        stats = fake_quantize_gguf(args.input_file, args.output_file, target_layers)
        end_time = time.time()
        
        print(f"處理時間: {end_time - start_time:.2f} 秒")
        
        # 保存統計結果
        if args.stats_output:
            save_quantization_stats(stats, args.stats_output)
        elif stats:
            # 默認保存到results目錄
            stats_file = Path("results/layer0/numerical_analysis.json")
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            save_quantization_stats(stats, str(stats_file))
            
    except Exception as e:
        print(f"錯誤：{e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()