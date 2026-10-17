---
name: grading-orchestration
description: Use this skill whenever a homework photo (or several photos of the same homework) must be graded into the structured grade contract — including 批改作业、改作业、检查作业、作业照片识别分题、判断对错、给出成绩与评语. Always read this skill first, before any subject skill, because it defines the working order and the exact output JSON the caller expects.
---

# 作业批改编排规范（主 agent 专用）

## 输入

- 同一份作业的 1–8 张图片：可能是正反面、上下两半，或不同题目的多页；
- 可选：用户补充说明（年级、只批某几题、要求只判对错不给分等）。

## 工作顺序

1. 判断学科（`subject`：math / chinese / english / other）并给出 `subject_confidence`；
2. 读取对应学科技能（`math-grading` / `chinese-grading` / `english-grading`），
   `other` 或混合作业按题目要求逐题判定；
3. **逐题批改（一题一批）**：识别题干、学生作答、学生演算过程，逐题给出判定；
4. 数学题的每一道算式先用 `verify_expressions` 复算，再下结论；
5. 有错题（`wrong` / `partial`）时派发 `error_analyst` 子代理做错因与学情诊断；
6. 需要面向学生的详细评语时派发 `feedback_writer` 子代理；
7. 调 `check_grading_result` 校验并取回 normalized 结果；
8. 最终回答 = 一句简短说明 + 结果 JSON（原样使用 normalized）。

## 分题与跨图规则

- 题号 `no` 从 1 开始连续编号；跨图视为一份作业整体，不重复、不跳号；
- 同一道题被拍到两张图（上半题在前一张、下半题在后一张）必须合并为一题；
- 只有题号、没有可判内容的涂写（姓名、日期、页码）忽略，不计入 `questions`；
- 题目本身识别不到或字迹完全无法辨认时，不要猜：该题 `verdict=unreadable`、
  `confidence=low`、`needs_review=true`。

## 判定口径

- `verdict`：`correct` 完全正确 / `wrong` 错误 / `partial` 部分正确（按步骤给分）/
  `unreadable` 无法判定；
- 分别给出 `process_correct`（过程）与 `result_correct`（结果）：小学数学与应用题
  要区分"过程对、结果错"与"过程就错"；
- `answer_type`：客观题 `objective`；作文、说理、观点表达等开放性题目标 `open`；
- `confidence`：识别与学生答案清楚时为 `high`；题干模糊、字迹潦草时为 `low`；
- `knowledge_point`：一个知识点标签（如"两位数乘法""分数加减""一般过去时"）；
- `comment`：只写给该题的一句话提示，正确题留空。

## 题干逐字为准（最重要的一条）

图片里的数字和文字就是唯一事实来源。常见题往往有多个流传版本，**严禁**凭记忆、
常识或"这道题应该是……"改写、补全图片中的数字、单位与条件。若你的复算结果与
图片不一致而你要判学生错，先回头核对：题干转写是否与图片完全一致？无法确认时
按 `unreadable` + `needs_review` 处理，绝不硬判。

## 输出契约（最终回答的 JSON）

```json
{
  "subject": "math",
  "subject_confidence": 0.95,
  "summary": "计算基本准确，注意应用题单位与答句。",
  "score": null,
  "full_score": null,
  "questions": [
    {
      "no": 1,
      "question_text": "12×12=?",
      "student_answer": "154",
      "knowledge_point": "两位数乘法",
      "process_correct": false,
      "result_correct": false,
      "verdict": "wrong",
      "answer_type": "objective",
      "expressions": [{"raw": "12×12", "claimed": "154"}],
      "error_detail": {
        "category": "calculation",
        "error_type": "口诀记错",
        "expected": 144,
        "correct_form": "12×12=144",
        "reason": "12×12 应为 144"
      },
      "comment": "再算一遍 12×12。",
      "confidence": "high",
      "needs_review": false,
      "review_reason": null
    }
  ],
  "diagnosis": {"items": [], "top_errors": [], "weak_points": [], "teaching_advice": ""},
  "feedback": {"glow": [], "grow": [], "go": "", "parent_note": null},
  "degraded": false
}
```

`total_questions` / `correct_count` / `wrong_count` / `accuracy` / `needs_human_review`
由服务端重算与裁定，你可以不填；填了也会被覆盖。

## Verification（给出最终回答前自查）

- 每道题的 `no` 是否连续无重复，且覆盖了图片里所有可判题目（全批全改）；
- 判 `wrong`/`partial` 的题是否都有 `error_detail`（至少 category + reason）；
- 数学题的每个算式是否都进过 `verify_expressions`，且 `calc_verified === false`
  的表达式在判定中被当作错误处理；
- 是否已调用 `check_grading_result` 且 `ok=true`；
- 最终回答里的 JSON 是否与 normalized 完全一致（数字不得手改）；
- 低置信/不可读/开放性题目是否已置 `needs_review=true` 并写明 `review_reason`。

## 人工复核与合规

AI 只出**预批**：教师确认后才作为学生最终评价（教育部《关于加强义务教育学校
作业管理的通知》《教师生成式人工智能应用指引》）。因此以下情况必须显式标出来，
而不是硬给结论：题干或答案有歧义、字迹潦草、学生用了非常规但正确的解法、
作文等开放性题目、识别置信度低。服务端会据此置 `needs_human_review`。

## 常见错误

- 只打整页"一个大勾"：必须逐题给出判定，不能笼统说"整体不错"；
- 用记忆中的题目版本覆盖图片里的数字；
- 把作文当作客观题直接给分数结论；
- 自行编造分数：没有明确分值依据时 `score`/`full_score` 留 `null`；
- 最终回答只写文字不给 JSON。
