# Nutrient Deficiency Downstream Agent (农业灾害预测：作物营养缺乏下游 Agent)

本工程提供一个**可直接落地**的“nutrient_deficiency（作物营养缺乏）”下游 Agent：
- 读取你上游的 `output/latest.json`（`agri_rag.final_output.v2`）
- 对 query 图像（默认 `Demo/query.jpg` 或上游 `query_image_path`）做**指数/活力**分析，生成：
  - 严重度热力图 PNG（透明背景，适合叠加）
  - 严重度斑块 GeoJSON（可挤出做 3D）
  - 采样点 GeoJSON
  - 图文并茂的 Markdown 报告（包含证据图/查询图/统计图）
- 输出到当前目录下的 `nutrient_output/`
- 附带一个 **3D 前端（React + deck.gl + MapLibre）**，可把热力图/斑块/采样点叠加在 3D 地形上展示（默认使用 MapLibre 的公开 DEM 示例源，无需 token；也支持你替换为自部署瓦片）。

> 说明：你的示例 `latest.json` 中主标签是 `waterway`。本 Agent 会输出 `is_applicable=false`，但仍会把 evidence 里出现的 `nutrient_deficiency`（如 E5）做展示与对照，并支持 `--force` 强制对 query 图进行缺乏分析（用于 demo/回测）。

---

## 目录结构

```
agri_nutrient_agent_project/
  run_agent.py                 # 运行 Agent（生成 nutrient_output）
  serve.py                     # FastAPI 静态服务（给前端用）
  nutrient_agent/              # Agent 代码
  web/                         # 3D 前端（Vite + React + deck.gl + MapLibre）
  nutrient_output/             # 运行后生成（默认）
```

---

## 1) 安装依赖（后端）

建议新建虚拟环境：

```bash
pip install -r requirements.txt
```

如果你希望启用本地 Qwen 模型生成“更自然的双语解释”（可选）：

```bash
pip install -r requirements-llm.txt
```

---

## 2) 运行 Agent（生成输出）

在项目根目录执行：

```bash
python agri_nutrient_agent_project\run_agent.py --input output/latest.json --query Demo/query.jpg
python agri_nutrient_agent_project\run_agent.py --input output/latest.json --query Demo/query.jpg --force --llm qwen2.5 --qwen25_path models/Qwen/Qwen2.5-3B-Instruct
```

强制运行营养缺乏分析（即使上游没预测到）：

```bash
python run_agent.py --input output/latest.json --query Demo/query.jpg --force
```

输出默认写到 `./nutrient_output/`。

---

## 3) 启动本地服务 + 运行 3D 前端

### 3.1 启动后端静态服务（提供 nutrient_output 与 assets）
```bash
python agri_nutrient_agent_project\serve.py --host 0.0.0.0 --port 8000
```

- 输出 JSON: `http://localhost:8000/api/nutrient/latest`
- 静态资源：`http://localhost:8000/nutrient_output/...`

### 3.2 启动前端
```bash
cd web
npm install
npm run dev
```

打开 `http://localhost:5173`，你将看到：
- 3D 地形底图（MapLibre terrain DEM 示例）
- 严重度热力图叠加（BitmapLayer）
- 严重度斑块挤出（GeoJsonLayer extruded）
- 采样点（ScatterplotLayer）
- 右侧证据墙（显示 query & evidence 预览图）

---

## 4) 与你的工程对齐（你最可能会改的点）

- 地理参考（真实经纬度/田块边界）：
  - 如果你有 `field_meta.json`（GeoJSON 边界/bbox），可以在 `run_agent.py` 里加 `--field_meta ...`
  - 或在 `nutrient_agent/config.py` 里把默认中心点改成你真实地块中心
- DEM/瓦片自部署：
  - 前端 `web/src/mapStyle.ts` 里 terrain source 的 `url` 可替换为你自建的 `tiles.json`（Terrain-RGB/Mapzen Terrarium 都可）

---

## 5) 输出说明（nutrient_output/）

- `nutrient_deficiency_output.json`：主输出（含 `frontend_scene`，前端按这个自动渲染图层）
- `report.md`：多模态 Markdown 报告（含图片）
- `assets/`：拷贝的 query/evidence 预览图 + 生成的指数/热力图
- `geojson/`：严重度斑块与采样点 GeoJSON

祝你 demo 顺利。
