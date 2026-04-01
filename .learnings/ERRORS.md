## [ERR-20260401-001] shell-quoting-rg-pattern

**Logged**: 2026-04-01T00:08:39-04:00
**Priority**: low
**Status**: pending
**Area**: docs

### Summary
A shell `rg` command failed because a backtick-containing pattern was wrapped with mismatched quotes.

### Error
```text
/usr/bin/bash: -c: line 1: unexpected EOF while looking for matching ``'
/usr/bin/bash: -c: line 2: syntax error: unexpected end of file
```

### Context
- Command/operation attempted: `rg -n ... docs/...`
- Cause: mixed shell quotes around a regex pattern containing a backtick
- Environment: Codex bash shell in repository root

### Suggested Fix
Prefer single-quoted regex patterns without embedded backticks, or split the search into multiple simpler commands when auditing docs.

### Metadata
- Reproducible: yes
- Related Files: docs/training_pipeline.md

---
