# LitVerify AI — 学术文献引用智能验证智能体

针对赛题《AI 推荐的文献是真是假？》实现的文献引用验证智能体：接收一条（或一批）
文献信息，结合**本地规则引擎**与 **CrossRef / OpenAlex / arXiv / PubMed / Semantic Scholar / DBLP / DataCite / DOIDB** 等学术 API，
自动判定真实性、给出 0–100 可信度分数与问题明细。

- **单条验证**：粘贴任意格式引用（或仅 DOI / 标题），输出 可信 / 可疑 / 虚假 判定 + 规则评分明细。
- **批量验证**：上传 CSV / Excel，自动识别字段列逐条核验，回填中文结果列。
- **虚假特征分析**：按 生成模型 / 学术领域 / 主题 分组统计虚假率与主要失效规则（对应赛题任务一）。
- **数据画图**：用自然语言提需求，DeepSeek 生成 Python 绘图代码，应用在受限沙箱中执行后用内置 Plotly 渲染器展示（柱状/饼/热力/旭日/雷达等不限类型；执行失败自动重试一轮，无 Key 时回退默认图表，界面可查看生成代码）。
- **智能问答 / AI 解读**（可选）：依赖 DeepSeek，需要配置 API Key。截图请先转成文本再粘贴核验。
- **报告导出**：一键导出 HTML / PDF 验证报告。

> 验证核心（单条 / 批量 / 虚假特征 / 导出）**无需任何 API Key**，仅用本地规则 + 公开学术 API 即可运行。
> DeepSeek 仅用于"AI 解读 / 智能问答"等增强功能。

## 1. 环境要求

- Python **3.10+**（推荐 3.12）
- 可访问公网（调用 CrossRef / OpenAlex / arXiv / PubMed / Semantic Scholar / DBLP / DataCite / DOIDB 等）

## 2. 安装

### Windows（一键）
```bat
setup.bat   :: 创建 .venv 并安装依赖（仅需一次）
```

### macOS / Linux 或手动
```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## 3. 配置 API Key（可选）

仅在需要"AI 解读 / 智能问答"时需要。复制 `.env.example` 为 `.env` 并填写：

```bash
cp .env.example .env
# 编辑 .env：填入 DEEPSEEK_API_KEY，以及 CROSSREF_EMAIL / OPENALEX_EMAIL（礼貌访问，建议填写）
# 可选：SEMANTIC_SCHOLAR_API_KEY / NCBI_API_KEY / WANFANG_APP_KEY / WANFANG_APP_SECRET
```

也可在应用内右上角「⚙ 设置」里临时填写 API Key。

## 4. 启动

### Windows
```bat
run.bat
```

### macOS / Linux 或手动
```bash
.venv/bin/python -m streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

## 5. 使用测试数据

仓库同级目录 `测试数据/文献数据(1).xlsx` 为赛题样例（200 条，含 生成模型 / 学术领域 /
有关主题 / 完整标题 / 作者姓名 / 期刊会议 / 发表年份 / DOI 等列）。

1. 启动应用，点击输入框左侧「＋」→ 选择 **📊 批量验证**。
2. 上传该 Excel（或拖拽到页面）。
3. 系统自动识别中文字段列（标题 / 作者 / DOI / 年份 / 期刊），逐条核验。

**预期输出**：在原表基础上回填以下列——
`验证结果`（可信 / 可疑 / 虚假）、`可信度分数`（0–100）、`虚假特征`（命中的失效规则）、
`命中DOI`、`命中标题`；页面给出真实 / 可疑 / 虚假计数与得分分布图，可下载结果 CSV。

再切换 **🔬 虚假特征** 即可看到按模型 / 领域 / 主题分组的虚假率统计。

> 该样例多为 AI 生成的伪造引用（DOI 无法解析），预期大多判为「虚假」。

## 6. 单条验证示例

在 **📋 单条验证** 模式下输入：

| 输入 | 预期 |
|---|---|
| `doi:10.1038/nature14539` | 可信（CrossRef 命中 LeCun《Deep learning》） |
| `Vaswani A. Attention is all you need. NeurIPS, 2017.` | 可信 / 可疑（标题命中，无 DOI 稍弱） |
| `Smith J. Quantum hyperdrive learning. Nature, 2024. doi:10.1038/fake.99999` | 虚假（DOI 无法解析） |

## 7. 运行测试

```bash
.venv/bin/python -m pytest          # 全部单元测试
.venv/bin/python -m ruff check .    # 静态检查（可选）
```

## 8. 目录结构

```
app.py                # Streamlit 入口
config/               # 设置、提示词、规则 YAML
services/             # 解析、规则引擎、API 客户端、批量处理、导出
llm/                  # DeepSeek 客户端与解读
utils/                # DOI / 相似度 / 缓存 / 会话 / DataFrame 助手
ui/                   # 聊天外壳、主题、侧边栏、卡片（theme.css 为样式表）
db/                   # SQLite 历史记录
tests/                # 单元 + 集成测试
```

## 9. 规则体系（任务二）

加权规则分三类，综合得分 ≥80 判可信、40–79 可疑、<40 虚假（阈值可在设置中调整）：

- **DOI 维度**：DOI 格式正则、DOI 在 CrossRef / OpenAlex / PubMed / Semantic Scholar / DBLP / DataCite / DOIDB 等来源可解析
- **外部库比对**：标题 / 作者 / 期刊 / 卷期页一致性、arXiv 解析、多源外部库一致性
- **本地元数据（无需联网）**：作者姓名格式、期刊/会议名称合理性（学术关键词）、标题长度与异常字符、年份范围（1900–2026）

规则权重与阈值见 `config/rules_default.yaml`，也可在应用「⚙ 设置」内调整并导入 / 导出。
