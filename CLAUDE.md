# PX4 ULog Analysis Toolkit

Tools for analyzing PX4 ULog flight log files — size profiling, topic breakdowns, and optimization insights.

## Tools

| Tool | Location | Description |
|------|----------|-------------|
| Size profiler | `size_profiling/profile_log_size.py` | Break down a `.ulg` file by topic size, rate, and category |

## Quick Start

```bash
python3 size_profiling/profile_log_size.py path/to/log.ulg
```

## Dependencies

- Python 3.10+
- `pyulog` — install with `pip install pyulog`

## Directory Layout

- `size_profiling/` — Log size analysis tools
- `tmp/` — Scratch space for decompiled/exported data (gitignored)
