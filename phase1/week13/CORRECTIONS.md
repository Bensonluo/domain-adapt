# Week 13 纠错记录

## C13-01：算法描述与实际配置边界

- 原问题：理论笔记容易让读者理解为 DPO/GRPO 都必须加载独立 reference model。
- 修复：注明 DPO 可通过 adapter-disable 等方式形成 reference；本项目 Week 17 的 GRPO `beta=0` 不加载 reference。理论公式与具体实现配置分开记录。

## 当前状态

理论推导主体可保留；实现性结论以对应框架版本和实际 run config 为准。
