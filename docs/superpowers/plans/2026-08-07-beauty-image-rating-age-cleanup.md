# Beauty Image Rating Age Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all adult/minor detection, messaging, fields, penalties, and labels from `beauty-image-rating` because upstream guarantees that every input depicts a legal adult.

**Architecture:** Keep the existing single-file Skill and scoring model. Remove the age-gating branch completely, rename the two age-qualified style labels, and update every dependent table, workflow step, example, and consistency rule so the document has one coherent output contract.

**Tech Stack:** Markdown, YAML frontmatter, JSON examples, shell-based static validation.

## Global Constraints

- Do not add new runtime dependencies or files under the Skill directory.
- Preserve all composition, quality, preference, scoring, and non-age-related manual-review behavior.
- Output must remain JSON-only when the Skill is used.
- Do not commit changes unless the user explicitly requests a commit.

---

### Task 1: Remove Age-Gating Behavior

**Files:**
- Modify: `src/agent_app/skills/beauty_image_rating/SKILL.md:1`
- Reference: `docs/superpowers/specs/2026-08-07-beauty-image-rating-age-cleanup-design.md:1`

**Interfaces:**
- Consumes: Upstream images already guaranteed to depict legal adults.
- Produces: A JSON-only image-rating contract whose first content block after `image_id` is `primary_subject`, with no age-related fields or values.

- [ ] **Step 1: Run a baseline forward test**

Ask a fresh agent to use the current Skill for a representative image-rating request where the caller explicitly says all inputs are legal adults. Record that the current instructions still require `age_gate`, age review states, or age-qualified labels; this is the expected failing behavior.

- [ ] **Step 2: Normalize YAML frontmatter**

Replace the malformed opening block with exactly two fields and a valid closing delimiter:

```yaml
---
name: beauty-image-rating
description: 对女性人物图片进行多标签分类、构图分析和个人偏好评分。适用于本地图片整理、图片检索、筛选和排序。输出严格的结构化 JSON，不进行 OCR，不识别人物身份。
---
```

- [ ] **Step 3: Remove age-specific rules and outputs**

Apply these document-wide changes:

```text
删除：成年状态检查整节
删除：任务目标中的成年状态安全检查
删除：真实年龄推测禁令（年龄不再属于本 Skill 的分析维度）
删除：年龄相关扣分、REVIEW 等级、置信度规则和执行步骤
删除：示例 JSON 中的 age_gate
删除：动漫/绘画/AI 图片中的现实年龄或未成年判断规则
重命名：成年可爱风 -> 可爱风
重命名：成人JK风 -> JK风
```

Retain `manual_review` only for non-age ambiguity such as `primary_subject_uncertain`.

- [ ] **Step 4: Verify forbidden terms are absent**

Run:

```bash
if rg -n '成年|未成年|年龄|adult|minor|youthful|萝莉' src/agent_app/skills/beauty_image_rating/SKILL.md; then
  exit 1
fi
```

Expected: exit code `0` with no matches.

- [ ] **Step 5: Verify renamed labels are consistent**

Run:

```bash
rg -n '可爱风|JK风' src/agent_app/skills/beauty_image_rating/SKILL.md
```

Expected: every former adult-qualified style occurrence now uses `可爱风` or `JK风` in label lists, scoring tables, and preference configuration.

- [ ] **Step 6: Parse YAML and JSON examples**

Run:

```bash
uv run python - <<'PY'
import json
import re
from pathlib import Path

import yaml

path = Path("src/agent_app/skills/beauty_image_rating/SKILL.md")
text = path.read_text()
frontmatter = text.split("---", 2)[1]
metadata = yaml.safe_load(frontmatter)
assert set(metadata) == {"name", "description"}
for block in re.findall(r"```json\n(.*?)\n```", text, re.DOTALL):
    json.loads(block)
PY
```

Expected: exit code `0` and no output.

- [ ] **Step 7: Run final diff checks**

Run:

```bash
git diff --check -- src/agent_app/skills/beauty_image_rating/SKILL.md
git diff -- src/agent_app/skills/beauty_image_rating/SKILL.md
```

Expected: no whitespace errors; diff contains only frontmatter normalization and age-related cleanup.

- [ ] **Step 8: Run a post-change forward test**

Ask a fresh agent to use the revised Skill for the same representative request. Confirm its proposed JSON contains no age gate, age warning, age review reason, or age-qualified tag, while preserving scoring and non-age manual-review rules.
