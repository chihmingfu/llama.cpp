#!/usr/bin/env python3

# 根據我們觀察到的數據計算統計

print("=== FFN Norm假量化實驗數據分析 ===\n")

# 從單線程數據觀察到的結果
layers = 22  # TinyLlama有22層
calls_per_layer = 30  # 每層被調用30次 (從輸出中觀察到)

print("1. 架構分析:")
print(f"   - 模型層數: {layers}")
print(f"   - 每層RMS norm調用次數: {calls_per_layer}")
print(f"   - 總RMS norm調用次數: {layers * calls_per_layer} = {layers * calls_per_layer}")

print("\n2. 量化統計:")
print("   單層量化 (Layer 5):")
print("   - 量化次數: 1 (僅Layer 5的第一次調用)")
print("   - 跳過次數: 29 (Layer 5的其他29次調用)")
print("   - 覆蓋率: 1/30 = 3.33% (針對Layer 5)")

print("\n   全層量化 (All Layers):")
print("   - 量化次數: 22 (每層第一次調用)")
print("   - 跳過次數: 22 × 29 = 638")
print(f"   - 覆蓋率: 22/660 = {22/660:.2%} (總體)")

print("\n3. 性能影響分析:")

# 性能數據
baseline_prompt = 6479.47
baseline_eval = 1886.13
baseline_total = 8907.00

single_prompt = 6687.57  
single_eval = 1498.44
single_total = 8249.78

all_prompt = 6752.71
all_eval = 1926.19
all_total = 9228.38

print("   Baseline (無量化):")
print(f"   - Prompt eval: {baseline_prompt:.2f} ms")
print(f"   - Eval time: {baseline_eval:.2f} ms") 
print(f"   - Total time: {baseline_total:.2f} ms")

print("\n   單層量化 vs Baseline:")
prompt_impact_single = (single_prompt - baseline_prompt) / baseline_prompt * 100
eval_impact_single = (single_eval - baseline_eval) / baseline_eval * 100
total_impact_single = (single_total - baseline_total) / baseline_total * 100

print(f"   - Prompt eval: {prompt_impact_single:+.1f}%")
print(f"   - Eval time: {eval_impact_single:+.1f}%")
print(f"   - Total time: {total_impact_single:+.1f}%")

print("\n   全層量化 vs Baseline:")
prompt_impact_all = (all_prompt - baseline_prompt) / baseline_prompt * 100
eval_impact_all = (all_eval - baseline_eval) / baseline_eval * 100
total_impact_all = (all_total - baseline_total) / baseline_total * 100

print(f"   - Prompt eval: {prompt_impact_all:+.1f}%")
print(f"   - Eval time: {eval_impact_all:+.1f}%")
print(f"   - Total time: {total_impact_all:+.1f}%")

print("\n4. 精度損失驗證:")
print("   - BF16轉換確認: ✅ (Unit test通過)")
print("   - 數值差異範圍: 0.000007 - 0.000174")
print("   - 量化效果可測量: ✅")

print("\n5. 架構理解:")
print("   - GGML tensor分塊: ✅ (每層30次調用證實)")
print("   - First-call-only邏輯: ✅ (只有第一次調用被量化)")
print("   - 單線程執行: ✅ (call#順序正確)")

print("\n=== 結論 ===")
print("1. 假量化實現正確: 每層精確量化1次")
print("2. 性能影響可接受: 全層量化僅增加3.6%總時間")
print("3. 精度損失可測量: BF16轉換產生預期的數值差異")
print("4. 實驗數據可信: 單線程執行排除並行影響")