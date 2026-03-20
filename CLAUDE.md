# PX4 ULog Analysis Toolkit

Tools for analyzing PX4 ULog flight log files — size profiling, topic breakdowns, and optimization insights.

## Dependencies

- Python 3.10+
- `pyulog` — install with `pip install pyulog`

## Skills

| Skill | Description |
|-------|-------------|
| `/profile-log <path>` | Run the size profiler on a `.ulg` file and interpret results |
| `/accel-vibration <path>` | Analyze accel vibration, compute PSDs/spectrograms, identify notch filter targets |

## Directory Layout

- `scripts/` — Analysis scripts used by skills
- `logs/` — Catalogued log sessions (one subdirectory per log)
- `ideas/` — One file per unimplemented feature idea (remove when implemented)
- `.claude/skills/` — Claude Code skill definitions

## Log Catalogue Convention

Each analysed log gets a directory under `logs/` containing the `.ulg` file and a `README.md`.

The README uses YAML frontmatter for indexing:

```yaml
---
log_file: example.ulg
date_catalogued: 2026-03-19
skills_used:
  - profile-log
  - gps-interference
tags:
  - high-rate
  - optical-flow
---
```

- **`skills_used`** — list of skill names run on this log (enables queries like "which logs have had GPS interference analysis")
- **`tags`** — freeform labels for flight characteristics, hardware, issues found, etc.
- **Body** — summary of the log, followed by an analysis history with dated entries per session

### Querying the catalogue

To find logs analysed with a specific skill:
```bash
grep -rl "gps-interference" logs/*/README.md
```

### When starting a new session with a log

1. Copy the `.ulg` into `logs/<short-name>/`
2. Create or update the `README.md` with frontmatter and analysis notes
3. Append a dated entry to the Analysis History section after each skill run
