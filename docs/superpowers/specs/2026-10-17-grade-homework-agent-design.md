# 作业批改 Agent 化改造设计（grade_homework）

- 日期：2026-10-17
- 状态：已实现（`deep_agents/grade_homework/`）
- 决策：把 `grade_homework` 从固定 LangGraph 工作流改造为 Deep Agent，**用 agent 替换工作流**，路由名与 `result` 契约保持不变。

## 1. 背景与调研结论

改造前 `grade_homework` 是 `fetch_image → grade_llm（结构化视觉输出）→ calc_verify`
的三节点固定图：学科路由、批改规则、错因口径全部内嵌在一条 system prompt 里，
无法按学科加载不同规范，也无法在判分之外补充错因诊断与学情反馈。

联网调研（三路并行：国内中小学批改规范、批改系统实现架构、Agent Skills 生态）
的核心结论：

1. **政策红线**（教育部教基厅函〔2021〕13 号 + 《教师生成式人工智能应用指引》）：
   全批全改、及时反馈、鼓励信息技术辅助诊断；但 AI 不得将作文/开放性作业的
   自动批改结果作为学生最终评价。→ 人工复核必须落成机器可判定的显式字段，
   而不是事后抽查。
2. **可靠性杠杆**（GradePilot / AutoGradAI / 腾讯云教育智能批改 / GradingBench /
   arXiv 2502.13789 的共识）：答案库与结构化 rubric 是第一杠杆；"生成→校验"
   比单次输出抗幻觉；错因诊断比判分难（模型常把一切归为"计算错误"，
   通用模型错因分类准确率可低于 30%）；人工复核边界要写成 confidence/flag。
3. **分工结构**：主流是"确定性感知/校验流水线 + agent 判断 + 职能子代理"，
   而非单 agent 端到端（GradingBench 证明定位/识别错误是端到端的主瓶颈）。
4. **技能工程**（Agent Skills 规范 + superpowers/writing-skills）：description 是
   唯一路由键，只写"何时用"不复述流程；长规范拆成主技能 + 学科/职能子技能；
   确定性计算（加权、统计、校验）必须交给代码而非技能文本。

主要参考：[教育部作业管理通知原文](https://www.gov.cn/zhengce/zhengceku/2021-04/25/content_5602131.htm)、
[教师生成式 AI 应用指引报道](https://app.xinhuanet.com/news/article.html?articleId=c54096a11819252e79e01cef2ce56fa2)、
[GradePilot](https://github.com/aimilet/GradePilot)、[AutoGradAI](https://github.com/rice-awa/AutoGradAI)、
[educators-toolkit](https://github.com/SkillMedev/educators-toolkit)、
[腾讯云教育智能批改](https://cloud.tencent.cn/document/product/1341/138926)、
[GradingBench](https://github.com/ERRORSEMI/GradingBench)、
[MathCCS 错因诊断](https://arxiv.org/abs/2502.13789)、
[Agent Skills 规范](https://agentskills.io/specification)、
[Anthropic Agent Skills 工程博客](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)、
[writing-skills](https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md)。

## 2. 架构：三层执行链

```
GradeHomeworkExecutor（统一执行器，注册为 deep agent "grade_homework"）
│
├─ 感知层（确定性代码，images.py）
│   extract_urls（仅含 :// 的片段）→ SSRF 校验 → 下载 → 压缩 JPEG
│   → 以 image_url 内容块注入当前用户消息（工具消息不支持图片）
│   → 多轮追问无新图时复用会话历史中的图片
│
├─ 判断层（agent，deepagents 运行时）
│   主代理（视觉模型）按技能逐题批改
│   ├─ 工具 verify_expressions：算式精确复算（复用 calc.py）
│   ├─ 工具 check_grading_result：契约校验 + 服务端后处理预演
│   ├─ 子代理 error_analyst：错因诊断与学情（response_format=DiagnosisResult）
│   └─ 子代理 feedback_writer：学生评语（response_format=FeedbackReport）
│   主图 task 工具限流 3 次（ToolCallLimitMiddleware）
│
└─ 校验层（确定性代码，postprocess.py，与工具共用同一实现）
    finalize_answer：提取最后一个可校验 JSON → fix_stats（统计重算）
    → verify_math（复算纠错，翻判数值确不符的题）→ apply_review_policy
    （人审裁定）→ 返回 GradeResult 契约
```

关键不变量：**agent 输出的数字不直接决定对外结果**。统计、复算、人审标记全部
由服务端在返回前强制重跑；agent 即使数错题数、算错准确率也会被纠正。

### 2.1 契约兼容

- 路由名不变：旧客户端 `task_type=grade_homework` 经路由别名按同名 Deep Agent
  路由（仅接受完全匹配的注册名，拼写错误仍 422）；新客户端可用
  `execution_mode=deep_agent&agent_type=grade_homework`。
- `result` 字段全保留（subject/questions/expressions/calc_verified/accuracy/…），
  新增 `diagnosis`、`feedback`、`needs_human_review`、`review_reasons`、
  每题 `answer_type`/`knowledge_point`/`needs_review`/`review_reason`（增量兼容）。
- 模型配置沿用 `openai.grade_homework_model`（含 base_url/api_key 覆盖）；
  图片限制沿用 `grade_homework` 配置段，新增 `max_pages`。

## 3. 采纳的 SKILL（6 个，`src/agent_app/skills/grade-homework/`）

布局遵循 Agent Skills 规范：容器 `/skills/grade-homework/` 下的一级子目录即技能，
frontmatter `name` 与目录同名、description 唯一且只写触发条件。主代理与两个
子代理共用同一容器（启动只注入 name+description，命中后自行读取正文）。

| 技能 | 使用者 | 来源/依据 | 承载内容 |
| --- | --- | --- | --- |
| `grading-orchestration` | 主代理（必读） | GradePilot 的阶段化流程 + GradeBench"先定位识别再判分" | 工作顺序、分题与跨图合并、判定口径、题干逐字为准、输出契约 JSON 示例、自查清单、人审合规 |
| `math-grading` | 主代理（判数学时） | 沿用原工作流 prompt + 教研惯例（步骤分/单位答句/格式） | 逐步骤判定、expressions 逐字转写与复算三态、格式规范、打分口径、错因归类、复核升级 |
| `chinese-grading` | 主代理（判语文时） | 教务处批改惯例（泗水实验小学等）+ 千帆作文 rubric | 逐字比对/形近字、阅读按要点给分、病句修改、作文四维评等第（内容/语言/结构/书写） |
| `english-grading` | 主代理（判英语时） | 学校英语批改惯例（四线格/书写规范） | 逐字母拼写比对、语法点核对表、书面表达四维评档、订正惯例 |
| `error-diagnosis` | error_analyst + 主代理 | arXiv 2502.13789 两级错因 taxonomy + 万方错因分类文献 | 9 大类 category + 子类 error_type、判定顺序（禁止全归计算错误）、知识点标签、advice 写法、学情汇总 |
| `feedback-writing` | feedback_writer + 主代理 | educators-toolkit 的 Glow-Grow-Go + 批语合规要求 | 亮点/改进/下一步结构、具体不套模板的语气规范、AI 评语只是预批的内容边界 |

每个技能正文统一含 **Verification**（给出最终回答前自查）与 **Escalate**/
人工复核段，对齐"技能必须带验证段"的工程实践。

### 3.1 技能/工具/代码的职责边界

- **技能（SKILL.md）**：判断标准、流程顺序、语气规范——模型需要"理解"的部分。
- **工具（@tool）**：模型循环内需要的确定性反馈（复算结果、契约校验错误）。
- **执行器代码**：每次都必须发生的约束（下载安全、统计、复算、人审裁定、
  契约提取）——不是对模型的建议，而是保证。

## 4. 关键实现决策与取舍

1. **图片进消息而非工具**：OpenAI 兼容端点的工具消息不支持 `image_url` 内容块，
   视觉判断放在主代理的用户消息里最稳；代价是多轮会话历史携带图片 token。
2. **子代理只做纯文本推理**：子代理经 `task` 工具调度、拿不到图片，故错因与
   评语以主代理的逐题判定为输入——避免"再读一遍图"的感知误差与双倍视觉调用；
   `response_format` 结构化契约让子代理结果可被主代理原样并入最终 JSON。
3. **最终回答带 JSON 而非主图 response_format**：DeepAgentAdapter 以文本 delta
   聚合 answer，主图结构化输出会绕过该路径；编排技能约定最终回答必须含
   ```json 代码块，`finalize_answer` 用 raw_decode 扫描取最后一个可校验对象，
   与旧工作流的兜底解析同构。
4. **兼容别名只放宽 AUTO**：显式 `execution_mode=workflow` + agent 名仍然 422，
   错误语义不模糊。
5. **删除旧工作流**：`workflows/grade_homework/` 移除，`calc.py` 迁入
   `deep_agents/grade_homework/`（工具与后处理共用），旧测试
   `test_grade_workflow.py` 删除、`test_grade_calc.py` 迁移并改导入。

## 5. 测试与验证

- 单元（`tests/unit/deep_agents/test_grade_agent.py`，20 例）：URL 提取/校验、
  图片下载与 SSRF 拒绝、多模态消息注入、统计重算与复算纠错（沿用旧回归用例）、
  人审裁定、两个工具、执行器（含追问复用历史图片、页数上限、不可解析回答）、
  装配透传、子代理契约、技能布局校验（name==目录名、description 唯一）。
- 集成（`tests/integration/test_grade_agent.py`）：ScriptedModel 驱动真实
  deepagents 运行时——"校验工具→最终回答"完整回环 + 服务端复算翻判；
  `task` 派发 error_analyst 且主/子代理 system prompt 均注入技能清单
  （验证 `SubAgent.skills` 经父级 CompositeBackend 解析）。
- 迁移与更新：`test_grade_calc.py`（导入路径）、`test_catalog.py`（3 个 agent）、
  `workflows/test_catalog.py`（2 个 workflow）、`test_router.py`/`test_task_service.py`
  （兼容别名）、`test_main.py`（装配模型选择）。

## 6. 后续演进（本轮未做）

- 答案库/题库检索工具（answer-based 优先于 answer-free，可靠性第一杠杆）。
- 批注坐标回显（腾讯云式逐题定位，降低人工复核成本）。
- 错题本/学情持久化（当前仅进程内检查点）。
- 批改质量离线评测集（分学科/学段/题型分层，对齐人工批改基线）。
