"""批改 agent 的技能源声明。

backend 内虚拟路径 ``/skills/grade-homework/`` 由 harness 路由到磁盘
``src/agent_app/skills/grade-homework/``；SDK 只扫描该容器的一级子目录，
每个子目录的 ``SKILL.md`` 是一个技能（name 必须与目录同名）。
"""

SKILL_CONTAINER = "/skills/grade-homework/"

# 主代理与职能子代理共用同一容器：启动时只注入各技能的 name + description
# （约 100 tokens），命中后由模型自行读取对应 SKILL.md 正文。
SKILL_SOURCES: list[tuple[str, str]] = [(SKILL_CONTAINER, "Grade Homework")]
