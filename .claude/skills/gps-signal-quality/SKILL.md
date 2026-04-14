---
name: gps-signal-quality
description: Profile GPS signal quality from a PX4 ULog — satellites, fix type, receiver accuracy, EKF innovations, stationary drift, and raw-position jitter. Supports a two-log compare mode for before/after flights (e.g. driver A/B, firmware bisects, receiver swaps). Use when the user wants to validate GPS behavior across a change.
argument-hint: <log-a.ulg> [log-b.ulg] [--label-a NAME] [--label-b NAME]
allowed-tools: Bash(python3 *)
---

# GPS Signal Quality Analysis

Run the GPS signal quality analyzer on one or two ULog files and interpret the results. Two logs triggers compare mode — every time-series plot gets overlaid curves and every summary value gets a Δ column.

## Before running

1. Resolve each log file path:
   - First check if it already exists under `logs/` in this repo (by filename match).
   - If not found, check `~/Downloads/`.
   - If found outside the repo, catalogue it: create `logs/<log-name>/`, copy the `.ulg` there.
2. Set the output directory to log_a's `logs/<log-name>/` directory (or a shared dir when comparing).
3. If the user mentioned what the logs represent (e.g. "before-fix" / "after-fix", "stock driver" / "patched driver", "hardware A" / "hardware B") pass those via `--label-a` / `--label-b` — they show up in every plot legend and the verdict sentence.

## Command

```bash
# Single log
python3 scripts/gps_signal_quality.py <log-a.ulg> --output-dir <logs/log-name/>

# Compare two logs
python3 scripts/gps_signal_quality.py <log-a.ulg> <log-b.ulg> \
    --output-dir <logs/compare-name/> \
    --label-a "before-fix" --label-b "after-fix"
```

## After running

1. Open the PDF for the user: `xdg-open <output-dir>/gps_signal_quality.pdf` (or `gps_signal_quality_compare.pdf` in compare mode).
2. Read the pages you need with the Read tool:
   - page 1: cover + summary table + how-to-read
   - page 2: satellites_used and fix_type timeseries
   - page 3: receiver eph/epv/HDOP/VDOP
   - page 4: s_variance_m_s, c_variance_rad (velocity/course accuracy)
   - page 5: noise_per_ms, jamming_indicator, AGC (environment)
   - page 6: estimator_aid_src_gnss_* innovations + rejections
   - page 7: position_drift_rate_* from estimator_gps_status (the money plot)
   - page 8: post-EKF eph/epv/evh/evv
   - page 9: d(lat,lon,alt)/dt — raw position jitter
   - page 10: pass/fail checks (single) or Δ table + verdict (compare)
3. Interpret the findings. Pull from the summary text file `gps_signal_quality.txt` as well — it has the same stats in plain text and is easy to quote into a PR comment.

### What to look for

- **Satellite count jump** on the same sky = receiver tracking more signals (e.g. L5 / E5a / B2a enabled). The X20 / ZED-F9P-style driver bugs typically show up as a ~30–50% sat-count reduction when L5 is silenced.
- **eph / epv drop** = multi-band dual-frequency correction working — removes ionospheric delay. Expect 2–3× smaller eph on a good fix vs. L1-only.
- **TTFF improvement** = faster time to first 3D fix. Stored as `ttff_s` in the summary.
- **Fix-type upgrade** = 3→5 (3D → RTK-float) or 5→6 (→ RTK-fixed) once RTCM corrections are wired up.
- **Stationary drift rate** (page 7) is the cleanest pass/fail signal: with the aircraft sitting still, how fast does the EKF's position estimate walk? Compare medians, not peaks.
- **Noise/AGC parity** (page 5) is a sanity check — if these differ a lot between logs, the RF environments weren't comparable and other deltas should be interpreted with that caveat.

### Compare-mode verdict wording

The script auto-generates a one-sentence verdict on page 10 / in the text file. When relaying it to the user, add context:
- Mention the satellite-count delta *and* the drift-rate delta together — one without the other is weaker evidence.
- If `noise_per_ms` differs > 20% between logs, say so: you can't cleanly attribute deltas to the driver change.
- If either log has `jamming_indicator` p95 > 100, flag possible RF interference.

### Missing topics

Some topics may not be in the log:
- `estimator_aid_src_gnss_*` (pos/vel/hgt) — absent on older logs. Page 6 will show a "topic missing" banner.
- `estimator_gps_status` — absent on older logs. Page 7 (the money plot) will be empty.
- `satellite_info` — per-satellite CN0 / band breakdown. **Not consumed by this script** but worth mentioning to the user: if they're trying to prove L5-band reception directly, enabling this topic (`SDLOG_PROFILE` includes it, or custom profile) gives the strongest evidence. ARK_FPV excludes it by default.

### Dual-GPS boards

If the board has two GPS receivers, both `sensor_gps` multi_ids will be populated. The script currently auto-picks the one with more samples and logs which it chose (as a `NOTE:` line). If the user cares about the other receiver, say so and we can add an `--gps-instance` flag.

### PlotJuggler naming note

When a log has multiple `sensor_gps` subscriptions (common on boards that subscribe multi_id 0 *and* 1 regardless of which receivers are wired up), PlotJuggler shows the topic as `sensor_gps.00` / `sensor_gps.01`. The script handles this transparently — you don't need to rename anything.

## After interpreting

- Quote the verdict line and the 2–3 most impactful deltas into your response.
- If this is validating a driver PR, surface the satellite-count and drift-rate deltas together as evidence.
- Create or update `logs/<log-name>/README.md` per the catalogue convention in CLAUDE.md; tag the README with `skills_used: - gps-signal-quality` and a dated analysis-history entry.
