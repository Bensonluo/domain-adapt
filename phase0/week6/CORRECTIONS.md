# Week 6 纠错记录

> 审查日期：2026-08-28｜状态：`EXTERNAL_SUBSTITUTE / PARTIAL`

## C1：外部案例被写成原计划全部完成

- 原问题：README 一边保留未通过验收，一边声明所有交付物已在外部项目完成。
- 为什么错：外部 `medical_entity` 使用不同模型、数据规模和结构化任务，且原计划的人工开放问答评估没有完成。
- 已修复：定义为“替代性交付”，只证明完成了结构化实体匹配 SFT 案例。
- 已修复：外部 repo 链接固定到 commit `9267c7c569eeb9f2b14d0a1cf0faa67c831d7126`。
- 关闭条件：继续固定数据版本、运行配置、checkpoint 和结果 manifest；若坚持原退出标准，还需补原任务。

## C2：Week 6 与 Week 8 不是同一实验链

- 原问题：Week 6 引用 `medical_entity/Qwen`，Week 8 改为 `master_data/Gemma 26B`，总结把数据与结果放在同一成果表。
- 为什么错：模型、任务和数据全部改变，无法形成训练到评估的闭环。
- 已修复：两个案例在 manifest 和总结中拆开。

## C3：“零泄漏”表述过宽

- 原问题：按药品编码/查询隔离被概括为零泄漏。
- 为什么错：这只能说明某一标识层的分离，仍需检查模板、归一化 query、候选集合和同生成器分布重合。
- 修复方法：改为准确描述已执行的分组规则，并要求 code/query/template/candidate 四层 overlap 报告。

## C4：masking 失败时不能继续训练

- 原问题：assistant marker 找不到时退化为全序列 loss，并称“比报错好”。
- 为什么错：训练目标被静默改变，比立即失败更难发现且会污染实验。
- 已修复：训练脚本改为 fail-closed。

## C5：实现注释虚报 MinHash 去重

- 原问题：`dataset_prep.py` 顶部写成“exact match + MinHash”，实际只对规范化后的完整 JSON 做 MD5 精确去重。
- 为什么错：精确去重不能发现改写、模板变体和字段轻微差异形成的近重复；错误标注会让读者误判泄漏控制强度。
- 已修复：注释与 README 统一为“MD5 精确去重”，MinHash/语义近重复审计仍作为待补能力，不计入现有证据。

## C6：领域训练链没有 dev 评估

- 原问题：`domain_sft.py` 只接收训练集，训练过程中没有独立 dev 指标，却在验收要求中讨论过拟合和模型选择。
- 为什么错：只看 train loss 不能判断泛化，最后一个 epoch 也不一定是最佳 checkpoint。
- 已修复：强制传入预先分组切分的 `--eval_data`，每个 epoch 评估 `eval_loss`，保存并恢复最佳 checkpoint，同时显式记录 seed。

## C7：把调参集合命名成 test

- 原问题：`dataset_prep.py` 默认输出 `test_*`，但该集合会传给 Trainer 反复监控并选择 checkpoint。
- 为什么错：参与模型选择的集合本质上是 dev；继续称为 test 容易在迭代后误当作未见测试集，造成乐观偏差。
- 已修复：CLI 改为 `--eval_output/--eval_ratio` 并默认输出 `dev_*`；旧参数仅兼容。最终 blind test 必须独立构建、冻结配置后只运行一次。

## C8：chat template 后可能重复添加 special tokens

- 原问题：与 Week 5 相同，格式化后的 ChatML 文本再次 tokenize 时沿用 tokenizer 默认 special-token 行为。
- 为什么错：不同基座的 BOS/EOS 默认策略不同，会产生隐藏的模板差异。
- 已修复：显式 `add_special_tokens=False`，并继续由 assistant marker 检查 fail-closed。

## C9：把长度阈值称为“质量过滤”

- 原问题：`filter_quality` 只检查总字符数和 assistant 字符数，却被文档作为低质量过滤与质量分布证据。
- 为什么错：长度不能判断事实正确性、来源可靠性、覆盖、难度、指令一致性或污染；短答案也可能完全正确。
- 已修复：函数改名为 `passes_length_checks`，输出和 README 限定为最低长度启发式；完整质量审计仍需单独执行。

## C10：截断可静默删除后续 assistant turn

- 原问题：至少一轮 assistant 完整时，后续 turn 在截断点之外不会触发原 fail-closed 检查。
- 为什么错：训练目标与原始多轮样本不一致，长样本系统性少监督。
- 已修复：原始 assistant turn 数必须与 tokenized 后识别数完全一致，否则拒绝样本。

## C11：不均衡组使 dev 比例高度依赖一次随机顺序

- 原问题：组打乱后顺序累加到目标比例；组大小为 99/1 时，可能得到 99% 或 1% dev。
- 为什么错：虽然没有组泄漏，但评估规模和方差被 seed 偶然性支配，计划比例失去意义。
- 已修复：校验 `0 < eval_ratio < 1`，在多个确定性随机排列的候选前缀中选择最接近目标样本数且保留非空 train 的组集合，并报告实际比例。若大组本身超过目标，实际比例仍应如实记录，不能伪称精确分层。
