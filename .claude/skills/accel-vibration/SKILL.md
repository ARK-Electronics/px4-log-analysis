---
name: accel-vibration
description: Analyze accelerometer vibration from a PX4 ULog, compute PSDs, spectrograms, identify notch filter targets, and compare Z velocity estimation. Use when the user wants to evaluate accel vibration or tune notch filter parameters.
argument-hint: <path-to-ulg-file>
allowed-tools: Bash(python3 *)
---

# Accelerometer Vibration Analysis

Run the accel vibration analyzer on the given ULog file and interpret the results.

## Before running

1. Resolve the log file path:
   - First check if it already exists under `logs/` in this repo (by filename match).
   - If not found, check `~/Downloads/`.
   - If found outside the repo, catalogue it: create `logs/<log-name>/`, copy the `.ulg` there.
2. Set the output directory to the log's `logs/<log-name>/` directory.

## Command

```bash
python3 scripts/accel_vibration.py <resolved-log-path> --output-dir <logs/log-name/>
```

## After running

1. Open the combined PDF for the user (`xdg-open <output-dir>/analysis.pdf`). It contains all plots plus an interpretation guide as the last page. Then read the individual PNGs yourself with the Read tool so you can interpret them.
2. Interpret the findings:
   - Vibration severity (accel_vibration_metric: <1.0 good, 1-3 moderate, >3 high)
   - Whether vibration peaks track motor RPM (candidates for ESC RPM DNF)
   - Whether peaks are fixed-frequency (candidates for static notch filters)
   - Whether the gyro FFT frequency range covers the dominant accel peaks
   - Z velocity estimation quality if range sensor data is available
3. Recommend parameter settings:
   - `IMU_ACC_DNF_EN` (0=off, 1=ESC RPM, 2=FFT, 3=both)
   - `IMU_ACC_DNF_BW` (bandwidth — wider catches offset peaks, narrower preserves signal)
   - `IMU_ACC_DNF_HMC` (harmonics — set to cover all visible harmonic peaks)
   - `IMU_ACC_NF0_FRQ/BW`, `IMU_ACC_NF1_FRQ/BW` (static notches for fixed resonances)
   - `IMU_ACCEL_CUTOFF` (LPF — usually leave at 30 Hz unless specific reason to change)
4. If this is a comparison flight (DNF enabled vs baseline), quantify the improvement.
5. Create or update the `logs/<log-name>/README.md` catalogue entry per the convention in CLAUDE.md.

## Notes

- The script analyzes `sensor_accel_fifo` (raw pre-filter data) for the vibration spectrum. This shows what the filters need to deal with, not what passes through them.
- To measure filter effectiveness, compare `accel_vibration_metric` between flights or look at EKF Z velocity error.
- Motor frequency = ESC RPM / 60. The script overlays this on the spectrogram.
- All outputs (plots, summary.txt) go into the log's catalogue directory under `logs/`.
