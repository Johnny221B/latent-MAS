# Docs Index

`docs/` 现在按“长期主文档、参考资料、记录、计划”四层组织。

## 1. 主文档

这些文件描述当前版本的稳定事实，应在大版本更新后优先同步：

- [training_pipeline.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/training_pipeline.md)：当前 train/eval pipeline 的主说明
- [method.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/method.md)：方法定义与当前实现边界
- [training_pipeline_risks.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/training_pipeline_risks.md)：当前仍有效的风险与限制
- [agent_workflow.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/agent_workflow.md)：训练与评测时的 agent 执行流程
- [prompt_flow.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/prompt_flow.md)：prompt 的加载与使用链路
- [ours_json_log_format.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/ours_json_log_format.md)：评测 JSON 输出格式

## 2. 参考资料

这些文件用于辅助阅读、论文对照或历史分析，不定义当前版本事实：

- [reference/paper.pdf](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/reference/paper.pdf)
- [reference/paper_codebase_analysis.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/reference/paper_codebase_analysis.md)

## 3. Records

这些文件是带日期的版本记录，不应替代主文档：

- `records/changes/`：变更记录
- `records/experiments/`：实验日志
- `records/issues/`：某一日期版本的问题跟踪
- `records/sessions/`：会话汇总

当前已有：

- [change_log_2026-03-19.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/records/changes/change_log_2026-03-19.md)
- [probe64_experiment_log_2026-03-19.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/records/experiments/probe64_experiment_log_2026-03-19.md)
- [current_version_issues_2026-03-19.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/records/issues/current_version_issues_2026-03-19.md)
- [session_report_2026-03-19.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/records/sessions/session_report_2026-03-19.md)

## 4. Plans

[docs/plans/](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/plans) 保存执行计划和调试计划。它们记录“当时准备怎么做”，不是“当前版本实际已经是什么”。

## 5. 维护规则

当出现影响 train/eval/config/output/docs 语义的大版本提交时，至少同步检查：

- `training_pipeline.md`
- `method.md`
- `agent_workflow.md`
- `ours_json_log_format.md`
- `README.md`
- 本索引文件

如果改动只是一次实验或某次调试，不要把 dated record 误当成主文档更新；应优先判断信息应该进入主文档还是进入 `records/`。
