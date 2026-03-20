# AGENTS

## Documentation Maintenance

- Treat `docs/training_pipeline.md` as the canonical description of the current train/eval pipeline.
- Treat `docs/method.md`, `docs/agent_workflow.md`, `docs/prompt_flow.md`, and `docs/ours_json_log_format.md` as companion documents that must stay aligned with runtime behavior.
- After any major version change that affects train flow, eval flow, experiment config semantics, output artifacts, logging shape, or prompt behavior, update the affected main docs in the same work session.
- Update `docs/README.md` whenever files are added, moved, reclassified, or promoted between main docs, reference docs, records, and plans.
- Keep dated logs, experiment notes, issue trackers, and session summaries under `docs/records/`; do not let them become the source of truth for the current version.
- If a change only affects a dated experiment or one-off debugging run, record it under `docs/records/` instead of rewriting method or pipeline docs unless the runtime behavior actually changed.
- When docs and code diverge, fix the docs immediately rather than preserving stale explanations.
