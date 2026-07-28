# 项目中文注释与 README 中文化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目自有 Python 代码、测试、冒烟脚本中的英文注释统一调整为中文，补充关键职责注释，并将 README 完整中文化。

**Architecture:** 本次为纯文档化变更，按业务源码、测试与脚本、README 三个边界分批处理。每批只修改注释或面向读者的文档文字，最后通过静态扫描、Ruff、完整测试和差异复核证明运行逻辑与接口契约未变。

**Tech Stack:** Python 3.12、FastAPI、LangGraph、pytest、Ruff、Bash、Markdown

## Global Constraints

- 仅修改 `src/agent_app/**/*.py`、`tests/**/*.py`、`scripts/smoke.sh` 和 `README.md`。
- 将已有英文注释与 docstring 翻译为自然、准确的中文，并补充核心职责、接口边界和复杂流程说明。
- 不修改运行逻辑、类型声明、函数签名、导入、错误消息、日志内容、提示词、API 数据、测试断言或运行时依赖。
- 保留 `# type: ignore`、`# pragma: no cover`、`# noqa`、`# shellcheck` 等工具标记以及必要的技术名词和代码符号。
- 不添加解释简单赋值、直接返回或明显条件判断的低价值逐行注释。

---

### Task 1: 业务源码中文注释

**Files:**
- Modify: `src/agent_app/**/*.py`
- Test: `tests/unit/**/*.py`
- Test: `tests/integration/**/*.py`
- Test: `tests/contract/**/*.py`

**Interfaces:**
- Consumes: 当前业务源码中的模块、类、函数、协议、数据模型和 LangGraph 节点定义。
- Produces: 行为完全不变、关键职责和执行边界具有中文说明的业务源码。

- [ ] **Step 1: 建立业务源码注释清单**

运行：

```bash
rg -n --glob '*.py' '^\s*(#|("""|\x27\x27\x27))' src/agent_app
```

逐文件识别英文模块 docstring、类/函数 docstring、普通行注释，以及缺少说明的核心公共符号；工具标记和代码符号记为保留项。

- [ ] **Step 2: 中文化 API、基础设施与通用模型注释**

修改以下目录和文件中的注释/docstring：

```text
src/agent_app/__init__.py
src/agent_app/api/
src/agent_app/config.py
src/agent_app/errors.py
src/agent_app/infrastructure/
src/agent_app/logging.py
src/agent_app/main.py
src/agent_app/schemas/
```

重点说明应用装配、依赖注入、稳定错误边界、SSE 编码、请求响应契约和日志脱敏原因。仅在核心公共符号缺少说明时新增简洁中文 docstring。

- [ ] **Step 3: 中文化 Deep Agents、编排、服务与工作流注释**

修改以下目录中的注释/docstring：

```text
src/agent_app/deep_agents/
src/agent_app/orchestration/
src/agent_app/services/
src/agent_app/workflows/
```

重点说明第三方边界隔离、受限能力、路由优先级、图状态流转、流式事件降级、检查点语义和摘要子图职责。保留 Deep Agents、LangGraph、SSE 等技术名词以及类型检查标记。

- [ ] **Step 4: 检查业务源码差异只涉及注释**

运行：

```bash
git diff --word-diff=porcelain -- 'src/agent_app/**/*.py'
git diff --check -- 'src/agent_app/**/*.py'
```

预期：差异仅出现在注释/docstring；`git diff --check` 退出码为 0。

- [ ] **Step 5: 运行源码相关验证**

运行：

```bash
uv run ruff check src/agent_app
uv run ruff format --check src/agent_app
uv run pytest tests/unit tests/integration tests/contract
```

预期：Ruff 两项检查通过，所有测试通过。

- [ ] **Step 6: 提交业务源码注释**

```bash
git add src/agent_app
git commit -m "docs: translate source comments to Chinese"
```

### Task 2: 测试代码与冒烟脚本中文注释

**Files:**
- Modify: `tests/**/*.py`
- Modify: `scripts/smoke.sh`
- Test: `tests/**/*.py`

**Interfaces:**
- Consumes: 当前测试场景、测试替身、fixture 和冒烟测试步骤。
- Produces: 测试行为不变、测试目的和脚本步骤均以中文说明的验证代码。

- [ ] **Step 1: 建立测试与脚本注释清单**

运行：

```bash
rg -n --glob '*.py' '^\s*(#|("""|\x27\x27\x27))' tests
rg -n '^\s*#' scripts/smoke.sh
```

识别模块 docstring、测试替身说明、辅助函数说明、分组标题和脚本步骤注释；保留 shebang 与 shellcheck 指令。

- [ ] **Step 2: 中文化测试注释并补充测试意图**

将 `tests/**/*.py` 中已有英文注释/docstring 调整为中文。补充说明仅用于解释测试替身职责、fixture 隔离边界、端到端场景和不直观的失败路径，不改测试名称、测试数据、断言或 fixture 行为。

- [ ] **Step 3: 中文化冒烟脚本注释**

将 `scripts/smoke.sh` 的前置条件、阶段标题和步骤说明调整为中文。保留 `#!/usr/bin/env bash`、`# shellcheck disable=SC1091`、命令、变量、URL 和请求载荷原样。

- [ ] **Step 4: 验证测试与脚本**

运行：

```bash
uv run ruff check tests
uv run ruff format --check tests
bash -n scripts/smoke.sh
uv run pytest
git diff --check -- tests scripts/smoke.sh
```

预期：Ruff、Bash 语法检查、完整测试和差异检查均通过。

- [ ] **Step 5: 提交测试与脚本注释**

```bash
git add tests scripts/smoke.sh
git commit -m "docs: translate test and script comments to Chinese"
```

### Task 3: README 中文化与全局验收

**Files:**
- Modify: `README.md`
- Verify: `src/agent_app/**/*.py`
- Verify: `tests/**/*.py`
- Verify: `scripts/smoke.sh`

**Interfaces:**
- Consumes: 当前架构、安装运行方式、API 示例、测试命令、目录结构和限制说明。
- Produces: 保持技术契约和操作步骤不变的中文项目说明，以及全范围验收结果。

- [ ] **Step 1: 中文化 README 说明文字**

翻译标题、简介、架构图说明、路由优先级、前置条件、安装运行、API、测试、目录结构和限制。保留以下内容原样：

```text
FastAPI / LangGraph / Deep Agents / SSE / API / JSON
命令、路径、URL、环境变量
请求与响应字段、枚举值、示例载荷
组件类名和技能名
```

- [ ] **Step 2: 复核 README 示例和语义**

逐段对照修改前内容，确认功能、优先级、限制和操作步骤没有遗漏；检查 Markdown 代码围栏、列表、链接、架构图对齐和 JSON 示例完整性。

- [ ] **Step 3: 扫描残留英文注释**

运行：

```bash
rg -n --glob '*.py' '^\s*#.*[A-Za-z]{2}|^\s*("""|\x27\x27\x27).*?[A-Za-z]{2}' src/agent_app tests
rg -n '^\s*#.*[A-Za-z]{2}' scripts/smoke.sh
```

逐条复核结果：仅允许 shebang、工具标记、技术名词、代码符号或无法合理翻译的协议名称；任何解释性英文都调整为中文。

- [ ] **Step 4: 运行最终验证**

运行：

```bash
uv run ruff check .
uv run ruff format --check .
bash -n scripts/smoke.sh
uv run pytest
git diff --check
```

预期：所有命令退出码均为 0，完整测试无失败。

- [ ] **Step 5: 最终差异复核**

运行：

```bash
git diff --stat HEAD~2
git diff HEAD~2 -- README.md src/agent_app tests scripts/smoke.sh
```

确认 Python/Bash 差异仅涉及注释，README 仅调整面向读者的说明文字，未改动运行字符串、接口契约或代码逻辑。

- [ ] **Step 6: 提交 README**

```bash
git add README.md
git commit -m "docs: translate README to Chinese"
```

