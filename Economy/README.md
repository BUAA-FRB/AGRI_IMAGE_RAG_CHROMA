# Economy 模块（经济损失估计 + 本地 Qwen 分析 + 多模态可视化报告）

## 输入
- output/latest.json
- datasets/data2018_miniscale/field_stats.json

## 输出（默认）
- output/economy_out/economy_output.json
- output/economy_out/economy_prompt.txt
- output/economy_out/llm_analysis.md
- output/economy_out/economy_report.md
- output/assets/charts/<runid>_<timestamp>/*.png + charts_data.json

## 核心逻辑
1. 从 latest.json 的 predicted_labels 得到灾害标签与置信度
2. 若 query 不是数据集 tile：使用 evidence_context 中距离最小的 tile_id 作为 proxy
3. 从 field_stats.json 读取该 tile 的 image_area 与 label_areas
4. 受灾比例 r = sum(label_areas[label]) / image_area
5. 减产映射（启发式）：f = r * within_loss + r^2 * spillover
6. 不确定性：按 base_uncertainty 与 (1-confidence) 扩展，输出 P10/P50/P90
7. 总损失避免双重计数：TotalLoss = BaseFieldValue * CombinedYieldLoss
8. 图表：趋势推演（基线 vs 处置后情景）+ 归因柱状图 + 归因饼图
9. 可选：调用本地 Qwen（Transformers）把结构化数据写成“专家式经济研判文本”，并拼进最终报告

## 运行
- 全链路（含 LLM）：
  python -m Economy.cli --latest output/latest.json --field-stats datasets/data2018_miniscale/field_stats.json

- 只生成规则报告与图表：
  python -m Economy.cli --no-llm

## 本地 Qwen 依赖
建议安装：
- torch
- transformers
- accelerate
- sentencepiece
- matplotlib

模型路径默认：
- ./models/Qwen/Qwe2.5-3B-Instruct
并会自动尝试：
- ./models/Qwen/Qwen2.5-3B-Instruct
