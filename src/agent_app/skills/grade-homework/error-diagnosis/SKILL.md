---
name: error-diagnosis
description: Use this skill when the graded result contains wrong or partial questions and the mistake must be explained as a cause plus a knowledge point (错因诊断、错因分类、知识点标签、薄弱点分析、高频错误、教师讲评建议). Used by the error_analyst subagent and by the main agent when it writes per-question reasons.
---

# 错因诊断与学情分析规范

## 两级错因分类

一级 `category` 必须从这个集合里选（不要自造）：

| category | 含义 | 典型表现 |
| --- | --- | --- |
| `knowledge` | 知识性错误 | 公式、定义、规则根本没掌握，整类题都不会 |
| `concept` | 概念混淆 | 面积/周长、进率、形近字、相似语法点混用 |
| `method` | 策略/方法性错误 | 思路方向错、方法选择错、解题步骤结构错 |
| `calculation` | 计算错误 | 方法正确，单纯运算或口诀出错 |
| `careless` | 粗心 | 抄错数字、漏写单位、跳步、漏答 |
| `reading` | 审题不清 | 看错要求、漏条件、答非所问 |
| `habit` | 习惯与规范 | 不写过程、格式、书写、不写答句 |
| `language` | 语言表达 | 拼写、语法、用词 |
| `other` | 其他/不确定 | 证据不足时使用，并在 reason 里说明 |

二级 `error_type` 是用中文写的具体子类，例如"进位遗漏""口诀记错""单位换算进率错"
"过去式形式错""形近字己/已混淆"。子类必须与一级自洽。

## 判定顺序（不要跳过过程）

1. 先看学生写的**过程**：方法对不对？
   - 方法就错 → `method`（或 `knowledge`，如果整类题都不会）；
2. 方法对但结果错 → `calculation` 或 `careless`：
   - 数字算错、口诀错、进位借位错 → `calculation`；
   - 抄错数字、看错符号、漏写 → `careless`；
3. 答案对但格式/单位/答句缺 → `habit`；
4. 完全答偏 → `reading`；
5. 拼写、语法、用词 → `language`。

**不要把一切都归为"计算错误"**：这是 AI 批改最常见的失败模式。若你只能用
`calculation` 解释所有错题，说明你没有认真看过程，请重新逐题核对；
确实无法归类时用 `other` 并说明证据不足。

## 知识点标签

- 每题给一个 `knowledge_point`，用教材语言（"两位数乘两位数""分数加减法"
  "一般过去时""形近字辨析""找中心句"）；
- 同一份作业里相同知识点用**完全相同的写法**，不要一处写"乘法"、一处写"两位数乘法"；
- 学情汇总时按知识点聚合，不要按题号罗列。

## 面向学生的建议（advice）

- 一句话、可执行、学生看得懂：写"下次算 12×12 先把 10 和 2 分开算"，
  而不是"加强计算能力"；
- 只针对该题暴露的具体问题，不写通用学习建议；
- 语气客观、不指责（避免"太粗心""不认真"这类评价性语言）。

## 学情汇总

- `top_errors`：出现次数或影响最大的错因，最多 3 条，写成"错因 + 出现次数"；
- `weak_points`：薄弱知识点，最多 5 个，按出现频率排序；
- `teaching_advice`：给教师的 1–3 句讲评建议（先讲共性错误，再点个别辅导对象类型），
  不给具体学生姓名。

## Verification

- 每个诊断项的 `category` 都在允许集合内，且与 `error_type` 自洽；
- 诊断覆盖了所有 `wrong`/`partial` 的题目，没有遗漏；
- 没有出现"所有错误都是计算错误"的退化结果；
- `advice` 都是可直接执行的动作，而不是评价；
- 知识点标签在同一份作业内写法一致；
- 只诊断已有判定，没有改动对错结论或分数。

## Escalate（不要硬下结论）

- 证据不足（学生过程没写、字迹不清）：`category=other`，reason 写"过程缺失，
  无法定位错因"，并建议教师面批；
- 学生的错误可能源于题干歧义或题目本身有误：在 reason 里说明，建议人工复核，
  不要把它归为学生知识性问题。
