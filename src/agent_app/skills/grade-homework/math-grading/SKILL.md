---
name: math-grading
description: Use this skill when the homework subject is mathematics (算术、口算、竖式、脱式、解方程、应用题、几何计算、单位换算) and the work must be judged step by step. Covers 过程分与结果分、算式逐字转写、单位与答句规范、错因归类与打分口径.
---

# 数学作业批改规范

## 逐步骤判定

1. 识别学生手写的**每一步**演算（竖式、脱式、分步算式、解方程步骤）；
2. 逐步检查 `process_correct`：方法是否成立、步骤是否符合教材规范；
3. 再检查 `result_correct`：最终结果是否正确；
4. 过程对、结果错 → `verdict=partial`；方法就错 → `verdict=wrong`。

## 算式必须交给计算工具

数值不要心算。把学生写的每个算式**逐字转写**进 `expressions`：

- `raw`：原样转写（保留单位、等号、中文、括号），如 `"4×5+4=24（名）"`；
- `claimed`：学生写的结果，**只写纯数值**，如 `"24"`（单位留在 raw 里）。

然后调用 `verify_expressions` 复算，按结果处理：

| 复算结果 | 处理 |
| --- | --- |
| `calc_verified: true` | 数值一致，仍按你的过程判断定性 |
| `calc_verified: false` | 数值确定不符 → 该题至少 `partial`，结果判错，`error_detail.expected` 填正确值 |
| `calc_verified: null` | 无法复算（算式不可求值或 claimed 非数值）→ 不作为翻判依据 |

转写只抄写、不重构：禁止按自己的理解改写算式或代入"你认为的"题干数字。

## 格式与规范（按学校惯例判定，扣分依据要写进 comment）

- 应用题：单位名称加括号、答语写完整；缺单位/缺答句按 `habit` 记错因；
- 竖式写在横式下方居中，脱式等号上下对齐、不跳步；
- 解方程先写「解」，等号对齐；分数结果须最简、假分数与带分数按题目要求；
- 进位、借位、约分、余数、四舍五入都纳入判定；
- 错误率很高的作业（如整页计算普遍错误）在 `summary` 里提示"建议面批/重做"。

## 打分口径

- 有明确分值依据（题目给分、教师要求）时按步骤给过程分：写 `score`/`full_score`；
- 没有依据时一律留 `null`，不要凭空造分；
- `partial` 必须说明"哪一步对、哪一步错"。

## 错因归类（一级用 category，子类写 error_type）

- `calculation`：方法对、单纯算错（含口诀记错、进位遗漏）；
- `method`：思路或方法选择错误（如该用除法用了乘法、方程列错）；
- `reading`：看错数字、漏条件、抄错题；
- `concept`：概念/公式记混（如面积与周长、进率）；
- `habit`：单位、答句、书写格式、不写过程；
- `knowledge`：该知识点根本没掌握，整类题都不会。

## Verification

- 每个算式都进过 `verify_expressions`；
- `calc_verified: false` 的表达式对应的题目没有被判成 `correct`；
- 应用题检查过单位与答句；
- 用 `partial` 的题目写清了过程分依据；
- 题干数字与你复算所用的数字一致（逐字核对图片）。

## Escalate（转人工复核）

- 题干或条件识别不确定 → `verdict=unreadable`、`confidence=low`、`needs_review=true`；
- 学生用了超出本年级范围但正确的解法（如用方程解算术题）→ 不判错，
  `needs_review=true`，`review_reason` 写"解法非常规，需教师确认是否给分"；
- 图形题依赖图中标注（角度、长度）而标注模糊 → 不给结论，转人工。
