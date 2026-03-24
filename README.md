# NovelCraft OS (赛博主编与小说生成引擎)

> 一个基于多智能体协作 (Multi-Agent) 和动态知识库挂载的专业级 AI 网文创作工作站。

NovelCraft OS 是一套 **AI 原生创作操作系统**：拒绝大模型常见的“塑料味”和“流水账”，通过严苛的法则提炼、可挂载的战术规则、以及“骨肉分离”的本地存储架构，实现对生成质量和工程可控性的降维管控。

---

## ✨ 核心设计哲学

- **AI 原生**：不是“聊天框套壳”，而是围绕写作生产链设计完整操作流。
- **工作流引擎**：将创作拆分为规划、起草、审校等可追踪节点，便于迭代与诊断。
- **知识库解耦**：写作法则以独立 JSON 资产沉淀，可提炼、可分组、可挂载、可复用。
- **本地优先与隐私友好**：项目数据、正文、法则均落地本地，不依赖云端数据库。

---

## 🚀 核心特性

### 🛠️ 骨肉分离架构
- 告别单文件 JSON 灾难。
- **元数据**（总纲、卷纲、章纲、挂载关系）与 **正文 Markdown** 物理隔离。
- 支撑百万字长篇时仍保持稳定与可维护。

### 🧠 五维赛博档案 (Cyber Archive)
- 覆盖五大实体：**人物 / 世界 / 势力 / 资产 / 时间线**。
- 采用 Master-Detail 主从编辑体验：左侧清单，右侧动态表单。
- 支持 AI 定向生成并即时回填表单，减少手工搬运。

### ⚖️ 法则炼金炉 (Rule Forge)
- 把名家片段、教程干货、避毒规范反向蒸馏为 `WritingRule`。
- 产物是结构化 JSON 指令：**Do’s (`positive_instructions`) / Don’ts (`negative_constraints`)**。
- 支持法则分组（Group）管理，便于大型法则库治理。

### 🎒 战术背包挂载 (Tactical Backpack)
- 创作前动态勾选法则并挂载到全局状态。
- 挂载舱按分组折叠展示，创作时可快速切换策略组合。
- 规则注入链路统一，避免“提示词散落各处”的维护灾难。

### 🤖 多智能体流 (Agentic Workflow)
- **Planner**：章节节拍与推进规划。
- **Drafter**：正文落地与风格执行。
- **Checker**：一致性与约束校验。
- 在工作流中形成可复盘、可重试、可观测的流水线创作机制。

---

## 🧭 项目结构 (Project Structure)

```tree
NovelCraft-OS/
├─ agents/                    # 多智能体角色（supervisor / planner / drafter / checker）
├─ core/                      # 核心模型、LLM 网关、数据库封装、Schema 定义
├─ ui/
│  └─ flet_app.py             # Flet 桌面应用主入口
├─ workflow/
│  └─ graph.py                # Agentic 工作流编排
├─ .env.example               # 环境变量模板（仅示例，不含真实密钥）
├─ .gitignore
└─ data/                      # 🔒 私有资产金库（被 .gitignore 忽略）
   ├─ projects/
   │  └─ <项目名>/
   │     ├─ meta.json         # 大纲/档案/挂载关系等元数据
   │     └─ chapters/         # 正文章节 Markdown（vol_x_ch_y.md）
   └─ knowledge_base/
      └─ rules/               # 写作法则库 JSON（可分组、可挂载）
```

> `data/` 已在 `.gitignore` 中忽略，默认不会提交，适合作为你的私有创作资产与知识库仓储。

---

## 🧩 环境配置与安装 (Installation & Setup)

### 1) Python 版本
- 推荐：**Python 3.10+**

### 2) 创建虚拟环境

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3) 安装依赖

如果你维护了 `requirements.txt`：

```bash
pip install -r requirements.txt
```

核心依赖至少应包含：
- `flet`
- `instructor`
- `pydantic`
- `python-dotenv`
- `litellm`
- `langgraph`
- `tenacity`

可直接执行：

```bash
pip install flet instructor pydantic python-dotenv litellm langgraph tenacity
```

### 4) 安全配置（非常重要）

系统已弃用明文 API Key 持久化。请使用 `.env`：

```bash
copy .env.example .env
```

然后编辑 `.env`，填入真实密钥：

```env
DEEPSEEK_API_KEY="your_real_key_here"
```

> `.env` 已被 `.gitignore` 忽略，**绝对不要提交真实密钥**。

---

## ⚡ 快速上手 (Quick Start Workflow)

### 启动 UI

```bash
flet run -r ui/flet_app.py
```

### 标准创作流

1. 在 `.env` 中配置模型密钥。  
2. 打开 **法则炼金炉**，提炼并保存写作规则。  
3. 创建项目，完善 **赛博档案**（人物/世界/势力/资产/时间线）。  
4. 打开 **战术背包**挂载法则，按流程生成总纲、卷纲与正文。  

---

## 🛡️ 工程实践建议

- 法则保持“小步迭代”：每次新增一条高价值法则，避免一次性堆叠过多噪声。
- 挂载遵循“场景最小集”：每次创作只挂载当前任务强相关规则。
- 正文坚持文件化：章节内容始终写入 `data/projects/<项目>/chapters/`，不要回灌巨量正文进元数据。
- 如遇结构化输出抖动，优先检查法则提炼输入文本质量与分组治理是否清晰。

---

## 📌 项目定位

NovelCraft OS 不是“单次问答式写文助手”，而是面向中长篇、可持续生产的 **AI 写作基础设施**。  
它的目标是把“灵感、规则、流程、产出”统一进一个可复盘、可迁移、可工程化维护的创作系统。

