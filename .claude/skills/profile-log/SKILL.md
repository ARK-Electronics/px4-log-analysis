---
name: profile-log
description: Analyze PX4 ULog file size by topic breakdown. Use when the user wants to understand what's taking up space in a .ulg log file.
argument-hint: <path-to-ulg-file>
allowed-tools: Bash(python3 *)
---

# Profile ULog File Size

Run the size profiler on the given ULog file and interpret the results.

## Before running

1. Resolve the log file path:
   - First check if it already exists under `logs/` in this repo (by filename match).
   - If not found, check `~/Downloads/`.
   - If found outside the repo, catalogue it: create `logs/<log-name>/`, copy the `.ulg` there.

## Command

```bash
python3 scripts/profile_log_size.py <resolved-log-path>
```

## After running

1. Summarize the key findings: file size, duration, SDLOG_PROFILE, and the top space consumers.
2. Call out anything unusual — topics with unexpectedly high rates, oversized messages, or categories that dominate disproportionately.
3. If asked, suggest optimization opportunities (lower rates, unnecessary topics, SDLOG_PROFILE changes).
4. Create or update the `logs/<log-name>/README.md` catalogue entry per the convention in CLAUDE.md.
