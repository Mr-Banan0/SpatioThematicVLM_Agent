# SpatioThematicVLM Agent

基于 **Qwen2.5-VL-7B-Instruct** 视觉语言大模型与 **LangGraph** 多智能体工作流的专题地图自动化分析系统。

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [安装部署](#安装部署)
- [快速开始](#快速开始)
- [评测框架](#评测框架)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 项目概述

SpatioThematicVLM Agent 接收一张专题地图图片（PNG/JPG），通过多智能体协作流水线完成：

1. **视觉特征提取** — 识别图例、标题、颜色标尺等地图视觉元素
2. **语义主题分析** — 推断地图的地理含义与空间分布模式
3. **空间异常检测** — 发现数据空缺、分布突变等异常
4. **报告自动生成** — 输出结构化的 Markdown 地理分析报告

可选支持上传 Shapefile（ZIP 压缩包），通过 ArcPy 服务提取真实 GIS 元数据辅助分析。

## 系统架构

```
┌─────────────┐     POST /analyze     ┌──────────────────────────────────────┐
│  Streamlit   │ ──────────────────▶  │        FastAPI (port 8000)           │
│  前端 UI     │ ◀──────────────────  │                                      │
│ (port 8501)  │     JSON response    │   LangGraph StateGraph 工作流:       │
└─────────────┘                       │                                      │
                                      │   Supervisor                         │
                                      │     ├── GIS Preprocessor (可选)      │
                                      │     ├── Visual Agent    ─┐           │
                                      │     ├── Semantic Agent   │ Qwen2.5   │
                                      │     ├── Anomaly Agent    │ -VL-7B    │
                                      │     └── Report Agent    ─┘           │
                                      └───────────┬──────────────────────────┘
                                                  │ (可选)
                                      ┌───────────▼──────────────┐
                                      │  ArcPy GIS Preprocessor  │
                                      │     (port 8002)          │
                                      │  TIF/SHP → PNG + 元数据   │
                                      └──────────────────────────┘
```

## 环境要求

| 项目 | 最低要求 |
|------|----------|
| Python | 3.10+ |
| GPU | NVIDIA GPU，显存 ≥ 10GB（4-bit 量化） |
| CUDA | 11.8+ |
| 磁盘 | ~15GB（模型权重自动下载） |
| ArcPy | 仅使用 GIS 预处理服务时需要（需 ArcGIS Pro 环境） |

> 如果没有 GPU，模型会回退到 CPU，但推理速度会非常慢（单次调用 > 10 分钟），不建议用于实际测试。

## 安装部署

### 1. 克隆项目

```bash
git clone <repo-url>
cd SpatioThematicVLM_Agent-main
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 首次运行时，Qwen2.5-VL-7B-Instruct 模型权重会从 Hugging Face 自动下载（约 4-5GB，4-bit 量化版本）。如需提前下载：
>
> ```bash
> python -c "from transformers import Qwen2_5_VLForConditionalGeneration; Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')"
> ```

### 4. 验证 GPU 可用

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 快速开始

### 方式一：通过 API 直接调用

**启动后端服务：**

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

**发送分析请求：**

```bash
# 仅上传地图图片
curl -X POST http://localhost:8000/analyze \
  -F "image_file=@test.png"

# 上传图片 + Shapefile
curl -X POST http://localhost:8000/analyze \
  -F "image_file=@test.png" \
  -F "shp_zip=@test_data.zip"
```

返回的 JSON 中 `report_markdown` 字段即为生成的分析报告。

### 方式二：通过 Streamlit UI

**先启动后端，再启动前端：**

```bash
# 终端 1：后端
uvicorn app.app:app --host 0.0.0.0 --port 8000

# 终端 2：前端
streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501`，在左侧上传地图图片，点击「开始 GIS 分析」。

### 方式三：使用 ArcPy GIS 预处理（可选）

如果需要从原始 GeoTIFF 或 Shapefile 开始分析（而非已做好的地图图片），需要启动 ArcPy 服务：

```bash
# 终端 3：需要在 ArcGIS Pro 的 Python 环境中运行
python gis_preprocessor_service.py
```

该服务监听 `localhost:8002`，将 TIF/SHP 转换为预览 PNG 并提取 CRS、要素数等元数据。

---

## 评测框架

项目内置了一套完整的自动化评测体系（`evaluation/`），**无需 Ground Truth 或专家标注**，覆盖五个研究问题：

| RQ | 研究问题 | 核心指标 |
|----|----------|----------|
| RQ1 | 结构化输出有效性 | JSON 合规率、Pydantic 通过率、字段完整度 |
| RQ2 | 跨运行稳定性 | Jaccard 相似度、输出丰富度标准差 |
| RQ3 | 生成参数敏感性 | temperature/top_p/max_tokens 与合规率、延迟的关系 |
| RQ4 | 输入鲁棒性 | 10 种图片扰动下的合规率与一致性 |
| RQ5 | 多智能体流水线一致性 | 跨层引用率（ROUGE-L）、主题重叠度 |

### 运行评测

```bash
# 快速模式 — 验证流程是否跑通（约 10-20 分钟）
python -m evaluation.run_all --image test.png --mode quick

# 标准模式 — 出论文数据（约 2-3 小时）
python -m evaluation.run_all --image test.png --mode standard

# 完整模式 — 细粒度参数网格（约 5-8 小时）
python -m evaluation.run_all --image test.png --mode full
```

### 三种模式对比

| 配置项 | quick | standard | full |
|--------|-------|----------|------|
| 一致性测试次数 | 3 | 5 | 10 |
| Temperature 值 | 3 个 | 5 个 | 11 个 |
| top_p 值 | 2 个 | 4 个 | 8 个 |
| max_tokens 值 | 2 个 | 5 个 | 6 个 |
| 每参数重复次数 | 1 | 2 | 3 |
| 图片扰动 | 否 | 是（10种） | 是（10种） |
| Prompt 变体 | 否 | 是（4种） | 是（4种） |

### 可选参数

```bash
# 跳过参数敏感性实验（只跑一致性测试）
python -m evaluation.run_all --image test.png --mode standard --skip-sensitivity

# 跳过图片鲁棒性实验
python -m evaluation.run_all --image test.png --mode standard --skip-robustness

# 从已有 JSON 结果重新生成图表（不重跑实验）
python -m evaluation.run_all --figures-only
```

### 后台运行（推荐用于长时间实验）

```bash
nohup python -m evaluation.run_all --image test.png --mode standard > eval_log.txt 2>&1 &
tail -f eval_log.txt
```

### 产出文件

运行完成后生成：

```
evaluation/
├── results/                            # 原始数据（JSON）
│   ├── experiment_config.json          # 实验配置与耗时
│   ├── consistency_results.json        # RQ1 + RQ2 + RQ5
│   ├── temperature_sweep.json          # RQ3
│   ├── top_p_sweep.json               # RQ3
│   ├── max_tokens_sweep.json          # RQ3
│   ├── robustness_results.json        # RQ4
│   └── prompt_variant_results.json    # RQ3 补充
│
├── figures/                            # 论文配图（300dpi PNG）
│   ├── fig1_schema_compliance.png      # 各节点 schema 合规率
│   ├── fig2_field_completeness.png     # 字段完整度
│   ├── fig3_consistency.png            # 跨运行输出稳定性
│   ├── fig4_sensitivity_temperature.png # 温度敏感性三合一图
│   ├── fig4_sensitivity_top_p.png      # top_p 敏感性
│   ├── fig4_sensitivity_max_tokens.png # token 预算敏感性
│   ├── fig5_robustness.png             # 图片扰动鲁棒性
│   ├── fig6_latency.png               # 各节点延迟分布
│   ├── fig7_coherence.png              # 流水线一致性热力图
│   └── fig8_prompt_variants.png        # Prompt 策略对比
│
└── perturbed_images/                   # RQ4 生成的扰动图片
```

### 论文章节建议

```
4. Evaluation
   4.1 Experimental Setup       ← experiment_config.json
   4.2 RQ1: Structural Validity ← Fig 1, Fig 2
   4.3 RQ2: Output Stability    ← Fig 3
   4.4 RQ3: Parameter Sensitivity ← Fig 4, Fig 8
   4.5 RQ4: Input Robustness    ← Fig 5
   4.6 RQ5: Pipeline Coherence  ← Fig 7
   4.7 Performance Analysis     ← Fig 6
5. Discussion
   - 最佳参数配置建议
   - Prompt 工程策略对比
   - 局限性与改进方向
```

---

## 项目结构

```
SpatioThematicVLM_Agent-main/
│
├── app/                          # 核心应用
│   ├── __init__.py
│   ├── app.py                    # FastAPI 主应用 + LangGraph 工作流
│   └── nodes.py                  # 全部智能体节点 + VLM 调用
│
├── evaluation/                   # 评测框架
│   ├── __init__.py
│   ├── metrics.py                # 自动化评测指标
│   ├── runner.py                 # Pipeline 执行引擎
│   ├── sensitivity.py            # 敏感性分析实验
│   ├── visualize.py              # 论文级可视化
│   └── run_all.py                # CLI 总入口
│
├── streamlit_app.py              # Streamlit 前端 UI
├── gis_preprocessor_service.py   # ArcPy GIS 预处理服务
├── requirements.txt              # Python 依赖
├── test.png                      # 示例测试地图
├── test_data.zip                 # 示例 Shapefile
├── uploads/                      # 上传文件存储
├── temp_gis/                     # GIS 临时文件
└── README.md                     # 本文档
```

---

## 常见问题

### Q: 模型下载失败 / 速度很慢

设置 Hugging Face 镜像源：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或手动下载模型后指定本地路径，修改 `app/nodes.py` 中的 `MODEL_NAME`：

```python
MODEL_NAME = "/path/to/local/Qwen2.5-VL-7B-Instruct"
```

### Q: CUDA out of memory

模型默认使用 4-bit 量化，显存需求约 8-10GB。如果仍然 OOM：

- 确认没有其他进程占用 GPU：`nvidia-smi`
- 减小图片分辨率：修改 `nodes.py` 中 `max_pixels` 参数
- 减小 `max_new_tokens`（默认 512）

### Q: 评测中途中断，能恢复吗

每个实验完成后结果会立即保存到 `evaluation/results/` 目录。如果中断：

1. 已完成的实验结果不会丢失
2. 重新运行时会覆盖同名结果文件
3. 可以用 `--skip-sensitivity` 或 `--skip-robustness` 跳过已跑完的部分
4. 如果只需要重新生成图表：`python -m evaluation.run_all --figures-only`

### Q: Streamlit 的「与 GIS Agent 对话」功能报错

当前版本的 `app/app.py` 尚未实现 `/chat` 接口，该功能暂不可用。如需启用，需要在后端添加对话路由。

### Q: 没有 ArcGIS 环境，GIS 预处理能用吗

不能。`gis_preprocessor_service.py` 依赖 ArcPy，必须在 ArcGIS Pro 的 Python 环境中运行。如果不使用 GIS 预处理功能，直接上传已做好的地图图片即可，不影响核心分析流程。
