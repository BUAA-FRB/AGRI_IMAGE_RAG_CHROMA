# Report Agent（智能农业报告生成）

该模块用于读取 `advice_out/advice_output.json`（上一个阶段的预测 + 专家建议打包输出），并调用本地
`Qwen2.5-3B-Instruct` 模型生成一份**高质量 Markdown 报告**。

## 目录放置位置

请将本目录放在你的项目根目录下，与 `Advice/`、`agri_rag/` **同级**：

```
project_root/
  Advice/
  agri_rag/
  Report/      # 本目录
  advice_out/
  models/
```

## 依赖

建议在你的 Conda 环境中安装（按需）：

- torch
- transformers
- accelerate

本目录提供 `requirements.txt` 作为参考。

## 快速开始

在项目根目录执行：

```bash
python -m Report.cli --input advice_out/advice_output.json --out_dir report_out
```

可选参数示例：

```bash
python -m Report.cli   --input advice_out/advice_output.json   --model_path models/Qwen/Qwen2.5-3B-Instruct   --out_dir report_out   --max_new_tokens 1600   --temperature 0.4   --top_p 0.85
```

输出将默认保存为：

- `report_out/report_<id>_<YYYYMMDD-HHMMSS>.md`

## 说明

- 该 Agent **不依赖** `Advice/` 或 `agri_rag/` 的代码，避免耦合；只读取 JSON。
- 报告内容会包含：
  - 研判结论（标签、置信度、原因、结论、不确定性）
  - 灾前/灾中/灾后行动清单（来自上阶段 adviser_plan）
  - 损失评估与取证清单
  - 恢复重建与监测计划
  - 知识依据附录（将检索 Top-K 映射为 [K1]..[K6]）

祝你项目推进顺利 🙂
