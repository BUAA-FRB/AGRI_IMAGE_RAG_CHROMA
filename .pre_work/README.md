# .pre_work — 气象-水文-土壤风险预警 Agent（灾前）

这个 Agent 用于把 **未来 7–30 天** 的气象预报 + 过去 7–30 天的实况/再分析（可选）+ 土壤属性（可选）
转成“风险时间表 + 触发阈值 + 行动窗口 + 图文报告”。

## 你会得到什么
- `risk_report.md`：图文并茂的 Markdown 报告（含图表、表格、阈值触发解释）
- `bundle.json`：可复现的结构化输出（原始数据、指标、风险得分、阈值命中）
- `assets/charts/*.png`：图表（降雨/ET0、水分收支、土壤湿度、风险曲线、触发热力图）

## 数据源（默认）
- Open‑Meteo（无需 key）：历史/预报气象 + ET0 + 土壤湿度/温度等变量  
- SoilGrids（无需 key，可选）：土壤砂/粉/黏比例、有机碳等，用于推断入渗与持水能力（提升“可预见性”）

---

## 快速开始

### 1) 安装依赖
```bash
pip install -r .pre_work/requirements.txt
```

### 2) 单地块运行（推荐：直接运行 runner，无需改 PYTHONPATH）
```bash
python .pre_work/run_agent.py --name "Field-A" --lat 39.90 --lon 116.40 --days_hist 14 --days_fore 16 --extend_to_30
```

### 3) 多地块批处理
```bash
python .pre_work/run_agent.py --sites .pre_work/examples/sites.json --extend_to_30
```

### 4) 使用本地 Qwen 输出“专家口吻解释”
```bash
python .pre_work/run_agent.py --name "Field-A" --lat 39.90 --lon 116.40 \
  --model_path ./models/Qwen/Qwen2.5-3B-Instruct
```

如果你暂时不想加载大模型：
```bash
python .pre_work/run_agent.py --name "Field-A" --lat 39.90 --lon 116.40 --no_llm
```

---

## 输出目录
默认输出到仓库根目录：`.pre_work_out/<地块名>/<run_id>/`。  
你可以用 `--out_root` 改成别的目录。

每次运行都会产出：
- `risk_report.md`
- `bundle.json`
- `risk_table_next30.csv`
- `assets/charts/*.png`
- （可选）`llm_analysis.md` 与 `risk_prompt.txt`
