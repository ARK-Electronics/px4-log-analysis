# PX4 ULog Analysis Toolkit

A collection of tools for analyzing PX4 ULog flight log files, exposed as [Claude Code](https://claude.com/claude-code) skills.

## Setup

```bash
pip install pyulog
```

Requires Python 3.10+.

## Usage

Analysis tools are available as Claude Code slash commands:

```
/profile-log logs/my-flight/my-flight.ulg
```

Scripts can also be run directly:

```bash
python3 scripts/profile_log_size.py path/to/log.ulg
```

## Available Skills

| Skill | Description |
|-------|-------------|
| `/profile-log` | Break down a `.ulg` file by topic size, rate, and category |

## Structure

```
scripts/        Analysis scripts
logs/           Catalogued logs — one dir per log with README and .ulg file
ideas/          Planned features — one file per idea
.claude/skills/ Claude Code skill definitions
```

Each log directory contains a `README.md` with YAML frontmatter tracking which skills have been run and freeform tags, making the catalogue searchable (e.g. `grep -rl "gps-interference" logs/*/README.md`).
