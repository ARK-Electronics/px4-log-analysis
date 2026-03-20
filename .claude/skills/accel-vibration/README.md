# Accelerometer Vibration Analysis

## What this skill does

Analyzes accelerometer vibration from a PX4 ULog to identify vibration sources, evaluate filter effectiveness, and recommend notch filter parameters. Generates plots and a text summary.

## Background

### Why accel vibration matters

Multirotor motors produce vibrations at their rotation frequency and harmonics. These vibrations corrupt the accelerometer data that the EKF integrates for velocity and position estimates. The result is altitude bobbing, position drift, and degraded flight performance.

PX4's gyro processing has a full filtering stack (dynamic notch filters + static notch filters + low-pass filter) operating on high-rate FIFO data. As of the `feat/accel-dynamic-notch-filters` branch, the accelerometer processing has the same capability.

### The filter chain

Accel data passes through these filters in order (all per-axis):

1. **ESC RPM dynamic notch** — tracks each motor's rotation frequency from `esc_status`. Places a notch at the fundamental and configured harmonics (2x, 3x, etc.). Requires DShot telemetry or CAN ESC feedback.

2. **FFT dynamic notch** — uses peak frequencies detected by the gyro FFT module (`sensor_gyro_fft`). Targets whatever vibration peaks the FFT identifies, even if they're not motor-related.

3. **Static notch 0** (`IMU_ACC_NF0_FRQ`) — fixed-frequency notch for a known structural resonance.

4. **Static notch 1** (`IMU_ACC_NF1_FRQ`) — second fixed-frequency notch.

5. **Low-pass filter** (`IMU_ACCEL_CUTOFF`, default 30 Hz) — 2nd-order Butterworth that rolls off everything above the cutoff.

Dynamic notches go first because they're narrowband and preserve signal. The LPF goes last as a broadband safety net.

### When to use each filter type

| Filter | Use when... |
|--------|-------------|
| ESC RPM DNF | Motor vibration peaks track RPM (most common case). Requires ESC telemetry. |
| FFT DNF | Vibration peaks that shift with flight conditions but aren't directly at motor RPM. Requires `IMU_GYRO_FFT_EN=1` and peaks within the FFT frequency range. |
| Static notch | Fixed-frequency structural resonance that doesn't change with throttle. Rare — most resonances shift at least somewhat. |
| LPF only | Broadband noise with no clear peaks. Lowering the cutoff adds phase lag, so only do this as a last resort. |

## What the script analyzes

### 1. Vibration metrics (`vehicle_imu_status`)

- `accel_vibration_metric` — high-frequency vibration energy in the accel data (m/s^2)
- `gyro_vibration_metric` — same for gyro (rad/s)
- Thresholds: <1.0 good, 1-3 moderate, >3 high
- **Note**: This metric is computed from raw data in `VehicleIMU`, *before* the notch filters in `VehicleAcceleration`. It does NOT reflect DNF effectiveness. To measure filter improvement, compare EKF performance (Z velocity error) between flights.

### 2. Accel FIFO power spectral density (`sensor_accel_fifo`)

The core analysis. Extracts raw int16 FIFO samples, scales them to m/s^2, and computes the power spectral density (PSD) using Welch's method.

- **What you're looking at**: The PSD shows signal power vs frequency. Narrowband peaks are vibration sources — these are what notch filters target. The broadband noise floor is what the LPF handles.
- **Why raw/pre-filter data**: `sensor_accel_fifo` is the raw sensor output before any filtering. This shows what the filters need to deal with. The filtered output (`vehicle_acceleration`) is published at ~200 Hz, too low to see the high-frequency peaks.
- **Sample rate**: Typically 4-8 kHz depending on the IMU. The script computes this from the `dt` field.

### 3. Spectrogram (time-frequency plot)

Shows how the accel vibration spectrum changes over time, with motor RPM overlaid.

- **What you're looking at**: Bright horizontal bands are vibration peaks. If they track the green motor harmonic lines (H1, H2, H3), they're motor-frequency vibrations — perfect targets for ESC RPM DNF.
- **If bands don't align with harmonics**: Either the motor pole count (`DSHOT_MOT_POL`) is wrong (common!), or the vibration source isn't motor-related.
- **If bands are perfectly horizontal** (don't change with RPM): structural resonance — use a static notch filter.

### 4. Peak-to-motor alignment

Computes the ratio of each detected PSD peak to the motor fundamental frequency. If the ratio is close to an integer (1.0, 2.0, 3.0), the peak is a motor harmonic.

Watch for systematic offsets. A consistent ~20% offset across all harmonics usually means `DSHOT_MOT_POL` is wrong. Common motor magnet counts: 12, 14, 24, 28 — verify by counting magnets or checking the motor spec sheet.

### 5. Z velocity comparison (if range sensor available)

Compares the EKF's Z velocity estimate (from accel integration) against the derivative of a range sensor (lidar/sonar). When `EKF2_RNG_CTRL=0`, the EKF doesn't fuse range data, so the Z velocity is purely accel-derived — any error is directly attributable to accel quality.

- **RMS error**: The primary metric. Lower is better. Typical values: 0.1-0.3 m/s good, 0.3-0.6 moderate, >0.6 poor.
- **Bias drift**: Slow wander in the error is accel bias drift (expected without Z aiding), not vibration-related.
- **High-frequency noise**: Rapid oscillations in Z velocity that correlate with `accel_vibration_metric` are vibration leaking through the filters.

### 6. Gyro FFT coverage

Checks whether the gyro FFT frequency range (`IMU_GYRO_FFT_MIN` to `IMU_GYRO_FFT_MAX`) covers the dominant accel vibration peaks. If the motor fundamental is above `IMU_GYRO_FFT_MAX`, the FFT-based DNF cannot help — use ESC RPM DNF instead.

## Parameters reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMU_ACC_DNF_EN` | 0 | Enable dynamic notch: 0=off, 1=ESC RPM, 2=FFT, 3=both |
| `IMU_ACC_DNF_BW` | 15 Hz | Bandwidth per notch. Wider catches offset peaks but removes more signal. |
| `IMU_ACC_DNF_HMC` | 3 | Number of harmonics (1-7). Set to cover all visible harmonic peaks. |
| `IMU_ACC_DNF_MIN` | 25 Hz | Minimum notch frequency. Notches park here when RPM drops. |
| `IMU_ACC_NF0_FRQ` | 0 | Static notch 0 center frequency (0=disabled) |
| `IMU_ACC_NF0_BW` | 20 Hz | Static notch 0 bandwidth |
| `IMU_ACC_NF1_FRQ` | 0 | Static notch 1 center frequency (0=disabled) |
| `IMU_ACC_NF1_BW` | 20 Hz | Static notch 1 bandwidth |
| `IMU_ACCEL_CUTOFF` | 30 Hz | Low-pass filter cutoff. Lower = less noise but more phase lag. |
| `DSHOT_MOT_POL` | — | Motor pole count. Wrong value = ESC RPM DNF targets wrong frequencies. |

## Outputs

| File | Description |
|------|-------------|
| `accel_psd.png` | PSD for X/Y/Z axes. Left: full range. Right: 0-500 Hz zoomed with motor harmonics marked. |
| `accel_spectrogram.png` | Z-axis spectrogram with motor RPM overlay and vibration metric time series. |
| `z_velocity.png` | EKF Z velocity vs range sensor derivative (only if range sensor data exists). |
| `vibration_summary.png` | Vibration metrics and ESC RPM overview with current filter parameters. |
| `summary.txt` | Text summary with findings, peak analysis, and parameter listing. |
