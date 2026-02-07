# AGRI_IMAGE_RAG_CHROMA

农业图像 RAG：基于 CLIP 嵌入 + Chroma 向量库 + Qwen3-VL 的农田灾害/异常检索与预测。

## 功能概览

- **建库**：扫描 Agriculture-Vision 风格数据集，用 CLIP 对 RGB（可选 NIR）图做嵌入，写入 Chroma。
- **检索**：支持「文本→图像」「图像→图像」检索（`query-text` / `query-image`）。
- **预测**：检索 top-k 证据后，用 Qwen3-VL 做多模态推理（`predict-text` / `predict-image`）。
- **带 Rerank 版本**（`agri_rag_with_rerank`）：可选 Qwen 文本重排、评估脚本（`eval`）。

## 数据集约定

目录需包含：

- `field_images/rgb/`：RGB 图像（`.jpg`）
- （可选）`field_images/nir/`：近红外
- `*_splits.json`：train/val/test 划分
- （可选）`field_labels/`、`field_stats.json` 等

从 `dataset_root` 起可自动查找 `field_images/rgb` 与 splits 文件。

## 使用方式

推荐使用带 rerank 的 CLI（子命令更全）：

```bash
# 安装依赖后，从项目根目录执行
python -m agri_rag_with_rerank.cli <子命令> [参数]
```

### 常用子命令

| 子命令 | 说明 |
|--------|------|
| `index` | 将数据集建入 Chroma（`--dataset_root`、`--db_dir`、`--collection`） |
| `query-text` | 文本检索图像（`--query`、`--top_k`） |
| `query-image` | 以图搜图（`--image_path`） |
| `predict-text` | 文本 → 检索 → Qwen3-VL 预测 |
| `predict-image` | 图像 → 检索 → Qwen3-VL 预测 |
| `prompt-text` / `prompt-image` | 仅生成 RAG 提示，不调用模型 |
| `eval` | 在指定 split 上跑评估，输出报告与 jsonl |

### 示例

```bash
# 建库（仅 RGB）
python -m agri_rag_with_rerank.cli index --dataset_root ./data/agri_vision --db_dir ./data/chroma_db

# 文本检索
python -m agri_rag_with_rerank.cli query-text --db_dir ./data/chroma_db --query "农田积水" --top_k 5

# 图像预测（带 NIR 证据展示）
python -m agri_rag_with_rerank.cli predict-image --db_dir ./data/chroma_db --image_path /path/to/img.jpg --include_nir_evidence --show_hits
```

## 项目结构

```
agri_rag/                 # 基础版：index / query / predict
  ├── dataset.py          # Agriculture-Vision 数据扫描与 TileRecord
  ├── embedder.py         # CLIP 图像/文本嵌入（支持 RGB+NIR）
  ├── chroma_store.py     # Chroma 建库与检索
  ├── rag_predictor.py    # RAG + Qwen3-VL 预测
  ├── qwen3_vl_client.py  # Qwen3-VL 调用
  └── cli.py

agri_rag_with_rerank/     # 扩展版：在上述基础上增加
  ├── reranker.py         # Qwen 文本重排
  ├── evaluator.py        # 评估流程
  ├── logging_utils.py    # 日志配置
  └── cli.py              # 完整 CLI（含 eval、rerank 等）
```

## 依赖概要

- Python 3.10+
- `torch`、`transformers`（CLIP）
- `chromadb`
- `Pillow`、`tqdm`
- Qwen3-VL 相关（如 `qwen-vl-utils` 等，见代码 import）

未提供 `requirements.txt` 时，可根据上述及 `agri_rag` / `agri_rag_with_rerank` 内 import 自行整理环境。

## 配置要点

- **Chroma**：默认 `--db_dir ./data/chroma_db`，`--collection agri_global`；支持 `--patches` 建 patch 级集合。
- **Qwen3-VL**：`--qwen_model` 可为 HuggingFace 模型 id 或本地路径；`--qwen_local_root` 用于解析本地缓存。
- **Rerank**：在 `agri_rag_with_rerank` 中通过 `--rerank_mode qwen` 开启，可配合 `--rerank_top_n` 等使用。
