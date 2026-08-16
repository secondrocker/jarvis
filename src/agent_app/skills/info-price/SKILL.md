---
name: info-price
description: Query China regional construction material info-prices (信息价/指导价)
  by region, material and period; supports fuzzy material matching, cross-region
  comparison and price trend analysis with charts.
---

# 信息价查询技能

## 何时使用

用户询问某地某期建材信息价、指导价、预算价，或要求跨地区比价、价格
走势分析时使用本技能。

## 数据获取流程

1. 读取 `references/websites.md`，按 地区 → 站点类型 匹配数据源：
   官方造价站 → 期刊汇编站 → 比价站交叉验证。
2. 用 web_search 定位目标页面（关键词：`{地区} {年份}{月份} 信息价 {材料}`），
   再用 web_fetch 读取正文表格。
3. 期数表达统一归一化后输出：`YYYY-MM`（如 2026-07）或 `YYYY年第N期`。

## 材料名模糊匹配

常见别名映射（探测时逐个尝试）：

| 通用名 | 规格别名 |
|---|---|
| 螺纹钢 | HRB400、HRB500、三级钢、四级钢 |
| 圆钢 | HPB300、一级钢 |
| 水泥 | P.O 42.5、P.O 52.5、普通硅酸盐水泥 |
| 商品混凝土 | C20/C25/C30/C35/C40、预拌混凝土、砼 |
| 砂 | 中砂、粗砂、机制砂、天然砂 |
| 石子 | 5-16、5-25、5-31.5、碎石、卵石 |

同一材料命中多规格时全部返回并逐条标注规格。

## 数据规范

- 价格单位必须统一（元/吨、元/m³），单位不同的数据不并入同一序列。
- 含税价与除税价必须标注，不得混用。
- 每条数据必须携带来源域名；无法溯源的价格不输出。
- 来源之间冲突时并列呈现，不擅自取舍。
