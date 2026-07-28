---
name: solution-planning
description: Guides the agent to decompose open-ended tasks into structured, verifiable plans.
---

# Solution Planning Skill

When asked to solve an open-ended or complex task, follow this structured process:

1. **Restate the goal.** Summarize the objective and the success criteria in your own words so the user can confirm understanding.
2. **Separate facts from assumptions.** List what is known from the provided information. Explicitly mark anything you are inferring or assuming.
3. **Decompose the task.** Break the problem into ordered, verifiable actions. Each step should have a clear deliverable and a way to check it succeeded.
4. **Identify dependencies, risks, and mitigations.** Note where steps depend on each other, what could go wrong, and how you would handle it.
5. **Ask for missing information.** When a gap would materially change the plan, ask the user. Otherwise make an explicit bounded assumption and move forward.
6. **Stay within your tools.** You have task planning, a virtual file system, and read-only references. Do not claim to have performed external research, shell execution, or host-file changes.
7. **Produce a plan.** End with a concise proposed plan (ordered steps, owners/roles if relevant, acceptance checks) and a short list of unresolved decisions for the user.
