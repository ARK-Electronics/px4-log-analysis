#!/usr/bin/env python3
"""
PX4 accelerometer vibration analyzer.

Extracts accel FIFO data from a ULog, computes power spectral density,
generates spectrograms, identifies notch filter targets, and evaluates
Z velocity estimation quality.

Usage:
    python3 accel_vibration.py <log.ulg> [--output-dir <dir>]

Outputs:
    - accel_psd.png          PSD for all 3 axes during hover (0-500 Hz + full range)
    - accel_spectrogram.png  Z-axis spectrogram with motor RPM overlay
    - z_velocity.png         EKF Z vel vs range sensor derivative (if range data available)
    - vibration_summary.png  Vibration metrics, ESC RPM, and filter status overview
    - summary.txt            Text summary with findings and recommendations
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from pyulog import ULog
except ImportError:
    print("Error: pyulog not installed. Run: pip install pyulog", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

try:
    from scipy import signal as scipy_signal
except ImportError:
    print("Error: scipy not installed. Run: pip install scipy", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# ULog helpers
# ---------------------------------------------------------------------------

def get_topic(ulog, topic_name, multi_id=0):
    """Return the first matching dataset for a topic name and multi_id."""
    for d in ulog.data_list:
        if d.name == topic_name and d.multi_id == multi_id:
            return d
    return None


def get_param(ulog, name, default=None):
    """Get an initial parameter value from the log."""
    return ulog.initial_parameters.get(name, default)


def timestamps_to_seconds(ts_us, start_us):
    """Convert microsecond timestamps to seconds relative to start."""
    return (ts_us.astype(np.int64) - np.int64(start_us)) / 1e6


# ---------------------------------------------------------------------------
# Flight phase detection
# ---------------------------------------------------------------------------

def detect_flight_phases(ulog):
    """Detect armed/flight segments from vehicle_status arming_state."""
    start_us = ulog.start_timestamp
    info = {"start_us": start_us, "armed_start_s": None, "armed_end_s": None}

    # Use arming_state (2 = armed) from vehicle_status
    vstatus = get_topic(ulog, "vehicle_status")
    if vstatus is not None and "arming_state" in vstatus.data:
        ts = timestamps_to_seconds(vstatus.data["timestamp"], start_us)
        armed_mask = vstatus.data["arming_state"] == 2
        armed_idx = np.where(armed_mask)[0]
        if len(armed_idx) > 0:
            info["armed_start_s"] = ts[armed_idx[0]]
            info["armed_end_s"] = ts[armed_idx[-1]]
            return info

    # Fallback: use actuator_motors control > 0
    motors = get_topic(ulog, "actuator_motors")
    if motors is not None:
        ts = timestamps_to_seconds(motors.data["timestamp"], start_us)
        # pyulog expands arrays: control[0], control[1], ...
        active = np.zeros(len(ts), dtype=bool)
        for i in range(12):
            key = f"control[{i}]"
            if key in motors.data:
                active |= (motors.data[key] > 0.05)
        active_idx = np.where(active)[0]
        if len(active_idx) > 0:
            info["armed_start_s"] = ts[active_idx[0]]
            info["armed_end_s"] = ts[active_idx[-1]]
            return info

    # Last resort: use full log duration
    end_us = ulog.last_timestamp
    info["armed_start_s"] = 0
    info["armed_end_s"] = (end_us - start_us) / 1e6
    return info


def detect_hover_segment(ulog, phases):
    """Find a stable hover segment (middle 60% of armed time)."""
    if phases["armed_start_s"] is None:
        return None, None
    t0 = phases["armed_start_s"]
    t1 = phases["armed_end_s"]
    duration = t1 - t0
    hover_start = t0 + duration * 0.2
    hover_end = t1 - duration * 0.2
    return hover_start, hover_end


# ---------------------------------------------------------------------------
# Accel FIFO extraction and PSD
# ---------------------------------------------------------------------------

def extract_accel_fifo(ulog, t_start_s, t_end_s):
    """
    Extract raw accel FIFO samples for a time window.

    Returns (sample_rate_hz, data_xyz, time_info) where data_xyz is a dict
    with keys 'x', 'y', 'z' each containing a 1D float array of
    concatenated scaled samples, and time_info is a dict with:
      - actual_start_s: timestamp of the first FIFO message
      - time_scale: ratio of actual elapsed time to assumed (samples/rate) time,
        used to correct the spectrogram time axis for inter-batch gaps.
    """
    fifo = get_topic(ulog, "sensor_accel_fifo")
    if fifo is None:
        return None, None, None

    start_us = ulog.start_timestamp
    ts = timestamps_to_seconds(fifo.data["timestamp"], start_us)

    # Filter to time window
    mask = (ts >= t_start_s) & (ts <= t_end_s)
    if not np.any(mask):
        return None, None, None

    ts_masked = ts[mask]
    actual_start_s = float(ts_masked[0])
    actual_duration = float(ts_masked[-1] - ts_masked[0])
    dt_us = fifo.data["dt"][mask]
    scale = fifo.data["scale"][mask]
    samples = fifo.data["samples"][mask]

    # Compute sample rate from median dt
    median_dt = np.median(dt_us)
    sample_rate_hz = 1e6 / median_dt

    data_xyz = {"x": [], "y": [], "z": []}

    # pyulog expands FIFO arrays into individual fields: x[0], x[1], ..., x[31]
    max_fifo = 32
    n_msgs = int(np.sum(mask))

    for axis in ["x", "y", "z"]:
        # Reassemble the array from individual fields
        raw_cols = []
        for j in range(max_fifo):
            key = f"{axis}[{j}]"
            if key in fifo.data:
                raw_cols.append(fifo.data[key][mask])
            else:
                break
        if not raw_cols:
            return None, None, None
        raw = np.column_stack(raw_cols)  # shape: (n_msgs, num_cols)

        for i in range(n_msgs):
            n = int(samples[i])
            s = float(scale[i])
            n = min(n, raw.shape[1])
            data_xyz[axis].append(raw[i, :n].astype(np.float64) * s)
        data_xyz[axis] = np.concatenate(data_xyz[axis])

    total_samples = sum(len(data_xyz[a]) for a in ["x"])  # all axes same length
    assumed_duration = total_samples / sample_rate_hz
    time_scale = actual_duration / assumed_duration if assumed_duration > 0 else 1.0

    time_info = {"actual_start_s": actual_start_s, "time_scale": time_scale}
    return sample_rate_hz, data_xyz, time_info


def compute_psd(data, sample_rate_hz, nperseg=None):
    """Compute PSD using Welch's method. Returns (freqs, psd)."""
    if nperseg is None:
        nperseg = min(4096, len(data) // 4)
    freqs, psd = scipy_signal.welch(data, fs=sample_rate_hz, nperseg=nperseg,
                                     noverlap=nperseg // 2, window="hann")
    return freqs, psd


def find_psd_peaks(freqs, psd, min_freq=20, max_freq=2000, prominence_db=5):
    """Find peaks in the PSD above a prominence threshold."""
    psd_db = 10 * np.log10(psd + 1e-30)
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs_m = freqs[mask]
    psd_db_m = psd_db[mask]

    peaks, props = scipy_signal.find_peaks(psd_db_m, prominence=prominence_db,
                                            distance=5)
    result = []
    for p in peaks:
        result.append({
            "freq_hz": freqs_m[p],
            "psd_db": psd_db_m[p],
            "prominence_db": props["prominences"][peaks == p][0] if len(peaks) > 0 else 0,
        })
    result.sort(key=lambda x: x["psd_db"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# ESC RPM extraction
# ---------------------------------------------------------------------------

def extract_esc_rpm(ulog, t_start_s, t_end_s):
    """Extract ESC RPM data for a time window. Returns (times_s, rpm_per_motor)."""
    esc = get_topic(ulog, "esc_status")
    if esc is None:
        return None, None

    start_us = ulog.start_timestamp
    ts = timestamps_to_seconds(esc.data["timestamp"], start_us)
    mask = (ts >= t_start_s) & (ts <= t_end_s)
    if not np.any(mask):
        return None, None

    ts_masked = ts[mask]
    esc_count = int(np.median(esc.data["esc_count"][mask]))

    rpm_per_motor = []
    for i in range(min(esc_count, 12)):
        key = f"esc[{i}].esc_rpm"
        if key in esc.data:
            rpm = esc.data[key][mask].astype(np.float64)
            if np.any(rpm != 0):
                rpm_per_motor.append(rpm)

    return ts_masked, rpm_per_motor


# ---------------------------------------------------------------------------
# Vibration metrics
# ---------------------------------------------------------------------------

def extract_vibration_metrics(ulog):
    """Extract accel and gyro vibration metrics from vehicle_imu_status."""
    imu_status = get_topic(ulog, "vehicle_imu_status")
    if imu_status is None:
        return None

    start_us = ulog.start_timestamp
    ts = timestamps_to_seconds(imu_status.data["timestamp"], start_us)

    result = {"time_s": ts}

    if "accel_vibration_metric" in imu_status.data:
        result["accel_vib"] = imu_status.data["accel_vibration_metric"]
    if "gyro_vibration_metric" in imu_status.data:
        result["gyro_vib"] = imu_status.data["gyro_vibration_metric"]

    return result


# ---------------------------------------------------------------------------
# Z velocity comparison
# ---------------------------------------------------------------------------

def extract_z_velocity(ulog):
    """Extract EKF Z velocity and range sensor data for comparison."""
    start_us = ulog.start_timestamp
    result = {}

    # EKF Z velocity
    lpos = get_topic(ulog, "vehicle_local_position")
    if lpos is not None:
        result["ekf_time_s"] = timestamps_to_seconds(lpos.data["timestamp"], start_us)
        result["ekf_vz"] = lpos.data["vz"]
        result["ekf_z"] = lpos.data["z"]

    # Range sensor
    dist = get_topic(ulog, "distance_sensor")
    if dist is not None:
        ts = timestamps_to_seconds(dist.data["timestamp"], start_us)
        distance = dist.data["current_distance"]
        # Store full-rate altitude data
        result["range_alt_time_s"] = ts
        result["range_alt"] = distance
        # Compute derivative (velocity) — uses midpoints
        dt = np.diff(ts)
        dz = np.diff(distance)
        valid = dt > 0.001  # avoid division by zero
        vz_range = np.zeros_like(dt)
        vz_range[valid] = -dz[valid] / dt[valid]  # negate: range sensor is NED-up, EKF is FRD
        # Smooth with simple moving average
        kernel_size = 5
        if len(vz_range) > kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            vz_range = np.convolve(vz_range, kernel, mode="same")
        result["range_time_s"] = (ts[:-1] + ts[1:]) / 2  # midpoint timestamps
        result["range_vz"] = vz_range

    return result


# ---------------------------------------------------------------------------
# FFT peak extraction
# ---------------------------------------------------------------------------

def extract_fft_peaks(ulog, t_start_s, t_end_s):
    """Extract gyro FFT peak frequencies during a time window."""
    fft_data = get_topic(ulog, "sensor_gyro_fft")
    if fft_data is None:
        return None

    start_us = ulog.start_timestamp
    ts = timestamps_to_seconds(fft_data.data["timestamp"], start_us)
    mask = (ts >= t_start_s) & (ts <= t_end_s)
    if not np.any(mask):
        return None

    result = {}
    for axis in ["x", "y", "z"]:
        peaks = []
        for i in range(3):  # up to 3 peaks per axis
            key = f"peak_frequencies_{axis}[{i}]"
            if key in fft_data.data:
                vals = fft_data.data[key][mask]
                peaks.append(vals[vals > 0])
        result[axis] = peaks

    return result


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(data, sample_rate_hz, nperseg=2048):
    """Compute spectrogram. Returns (times, freqs, Sxx)."""
    freqs, times, Sxx = scipy_signal.spectrogram(
        data, fs=sample_rate_hz, nperseg=nperseg,
        noverlap=nperseg // 2, window="hann"
    )
    return times, freqs, Sxx


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_psd(psd_data, sample_rate_hz, peaks_by_axis, motor_freq_hz, params, output_path):
    """
    Plot PSD for all 3 axes.

    Top row: full range. Bottom row: zoomed 0-500 Hz with motor harmonics and filter bands.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col")
    fig.suptitle("Accelerometer FIFO Power Spectral Density (raw, pre-filter)", fontsize=14)

    dnf_en = int(params.get("IMU_ACC_DNF_EN", 0))
    dnf_bw = float(params.get("IMU_ACC_DNF_BW", 30))
    dnf_hmc = int(params.get("IMU_ACC_DNF_HMC", 1))

    axis_names = ["X", "Y", "Z"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, axis in enumerate(["x", "y", "z"]):
        freqs, psd = psd_data[axis]
        psd_db = 10 * np.log10(psd + 1e-30)

        for col, (fmax, label) in enumerate([(sample_rate_hz / 2, "Full range"),
                                                (500, "0-500 Hz (dashed = motor harmonics)")]):
            ax = axes[i, col]
            fmask = freqs <= fmax
            ax.plot(freqs[fmask], psd_db[fmask], color=colors[i], linewidth=0.7)
            ax.set_ylabel(f"{axis_names[i]} PSD (dB)")
            ax.grid(True, alpha=0.3)

            if col == 1 and motor_freq_hz is not None:
                # Mark motor harmonics and DNF bands
                for h in range(1, 5):
                    f = motor_freq_hz * h
                    if f <= fmax:
                        ax.axvline(f, color="red", linestyle="--", alpha=0.5)
                        # Show DNF notch band if ESC RPM enabled and within harmonic count
                        if (dnf_en & 1) and h <= dnf_hmc:
                            ax.axvspan(f - dnf_bw / 2, f + dnf_bw / 2,
                                       color="blue", alpha=0.08)
                        if i == 2:  # label in bottom row
                            ax.annotate(f"{h}x ({f:.0f} Hz)",
                                        xy=(f, 0), xycoords=("data", "axes fraction"),
                                        fontsize=8, color="red",
                                        ha="center", va="top",
                                        xytext=(0, -2), textcoords="offset points")

                # Show static notch filters
                for nf in ["IMU_ACC_NF0_FRQ", "IMU_ACC_NF1_FRQ"]:
                    nf_freq = float(params.get(nf, 0))
                    nf_bw = float(params.get(nf.replace("FRQ", "BW"), 20))
                    if nf_freq > 0 and nf_freq <= fmax:
                        ax.axvspan(nf_freq - nf_bw / 2, nf_freq + nf_bw / 2,
                                   color="green", alpha=0.12)
                        ax.axvline(nf_freq, color="green", linestyle="-",
                                   alpha=0.6, linewidth=1)

                # Mark detected peaks
                for p in peaks_by_axis.get(axis, [])[:5]:
                    if p["freq_hz"] <= fmax:
                        ax.annotate(f'{p["freq_hz"]:.0f} Hz',
                                    xy=(p["freq_hz"], p["psd_db"]),
                                    fontsize=7, color="darkred",
                                    arrowprops=dict(arrowstyle="->", color="darkred"),
                                    textcoords="offset points", xytext=(10, 10))

            if i == 0:
                ax.set_title(label)
            if i == 2:
                ax.set_xlabel("Frequency (Hz)")

    # Add DNF band legend annotation on the top-right subplot
    if dnf_en & 1 and motor_freq_hz is not None:
        axes[0, 1].annotate(f"Blue shading = DNF band (\u00b1{dnf_bw/2:.0f} Hz)",
                            xy=(1, 1), xycoords="axes fraction",
                            fontsize=8, color="blue", ha="right", va="bottom",
                            xytext=(-5, 2), textcoords="offset points")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    return fig


def plot_spectrogram(spec_data, motor_times_s, motor_freq_hz_arr, fifo_t_start_s,
                     vib_metrics, phases, params, output_path):
    """
    Plot Z-axis spectrogram with motor RPM overlay, filter bands, and vibration metric.
    """
    times, freqs, Sxx = spec_data

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[3, 1, 1],
                              sharex=True)
    fig.suptitle("Z-Axis Accel Spectrogram vs Motor RPM", fontsize=14)

    # Spectrogram
    ax = axes[0]
    Sxx_db = 10 * np.log10(Sxx + 1e-30)
    vmin, vmax = np.percentile(Sxx_db, [5, 99])
    ax.pcolormesh(times + fifo_t_start_s, freqs, Sxx_db, vmin=vmin, vmax=vmax,
                  shading="gouraud", cmap="inferno")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, min(1500, freqs[-1]))

    # Overlay motor harmonics and DNF bands
    dnf_en = int(params.get("IMU_ACC_DNF_EN", 0))
    dnf_bw = float(params.get("IMU_ACC_DNF_BW", 30))
    dnf_hmc = int(params.get("IMU_ACC_DNF_HMC", 1))

    if motor_times_s is not None and motor_freq_hz_arr is not None:
        median_freq = float(np.median(motor_freq_hz_arr))
        for h in range(1, dnf_hmc + 1):
            freq_h = motor_freq_hz_arr * h
            ax.plot(motor_times_s, freq_h, color="deepskyblue",
                    linewidth=1.2, alpha=0.85)
            median_h = median_freq * h
            # Show DNF band if ESC RPM notch is enabled (bit 0)
            if dnf_en & 1:
                ax.fill_between(motor_times_s,
                                freq_h - dnf_bw / 2, freq_h + dnf_bw / 2,
                                color="deepskyblue", alpha=0.15)
                ax.annotate(f"{h}x motor ({median_h:.0f} Hz)  DNF \u00b1{dnf_bw/2:.0f}",
                            xy=(motor_times_s[-1], freq_h[-1]),
                            fontsize=9, color="deepskyblue", fontweight="bold",
                            va="center", ha="left",
                            xytext=(5, 0), textcoords="offset points")
            else:
                ax.annotate(f"{h}x motor ({median_h:.0f} Hz)",
                            xy=(motor_times_s[-1], freq_h[-1]),
                            fontsize=9, color="deepskyblue", fontweight="bold",
                            va="center", ha="left",
                            xytext=(5, 0), textcoords="offset points")

    # Static notch filters
    for nf, color in [("IMU_ACC_NF0_FRQ", "lime"), ("IMU_ACC_NF1_FRQ", "cyan")]:
        nf_freq = float(params.get(nf, 0))
        nf_bw = float(params.get(nf.replace("FRQ", "BW"), 20))
        if nf_freq > 0:
            ax.axhspan(nf_freq - nf_bw / 2, nf_freq + nf_bw / 2,
                        color=color, alpha=0.2)
            ax.axhline(nf_freq, color=color, linestyle="-", linewidth=1, alpha=0.7)
            ax.annotate(f"Static notch {nf_freq:.0f} Hz",
                        xy=(times[0] + fifo_t_start_s, nf_freq),
                        fontsize=9, color=color, fontweight="bold",
                        va="bottom", xytext=(5, 3), textcoords="offset points")

    # LPF cutoff
    lpf_cutoff = float(params.get("IMU_ACCEL_CUTOFF", 30))
    ax.axhline(lpf_cutoff, color="black", linestyle=":", linewidth=1.5, alpha=0.9)
    ax.annotate(f"LPF cutoff {lpf_cutoff:.0f} Hz",
                xy=(times[0] + fifo_t_start_s, lpf_cutoff),
                fontsize=9, color="black", fontweight="bold",
                va="bottom", xytext=(5, 3), textcoords="offset points")

    # FFT max range (if FFT DNF enabled)
    if dnf_en & 2:
        fft_max = float(params.get("IMU_GYRO_FFT_MAX", 150))
        ax.axhline(fft_max, color="black", linestyle="--", linewidth=1.5, alpha=0.9)
        ax.annotate(f"FFT max {fft_max:.0f} Hz (DNF blind above)",
                    xy=(times[-1] + fifo_t_start_s, fft_max),
                    fontsize=9, color="black", fontweight="bold",
                    va="bottom", ha="right", xytext=(-5, 3), textcoords="offset points")

    # Motor RPM
    ax = axes[1]
    if motor_times_s is not None:
        ax.plot(motor_times_s, motor_freq_hz_arr * 60, linewidth=0.8)
        ax.set_ylabel("Motor RPM")
        ax.grid(True, alpha=0.3)

    # Vibration metric
    ax = axes[2]
    if vib_metrics is not None and "accel_vib" in vib_metrics:
        ax.plot(vib_metrics["time_s"], vib_metrics["accel_vib"],
                color="red", linewidth=0.8)
        ax.set_ylabel("Accel Vib Metric")
        ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    return fig


def _highpass(signal, timestamps, cutoff_period_s=5.0):
    """Remove low-frequency content using a moving average subtraction."""
    dt = np.median(np.diff(timestamps))
    window = max(3, int(cutoff_period_s / dt)) | 1  # ensure odd
    kernel = np.ones(window) / window
    low_freq = np.convolve(signal, kernel, mode="same")
    return signal - low_freq, low_freq


def plot_z_velocity(z_vel_data, phases, output_path):
    """Plot EKF Z velocity vs range sensor derivative with detrended comparison."""
    if "ekf_vz" not in z_vel_data or "range_vz" not in z_vel_data:
        print("  Skipping Z velocity plot (missing EKF or range sensor data)")
        return

    t0 = phases["armed_start_s"] or 0
    t1 = phases["armed_end_s"] or z_vel_data["ekf_time_s"][-1]

    # Interpolate range Vz to EKF timestamps for aligned comparison
    ekf_t = z_vel_data["ekf_time_s"]
    ekf_vz = z_vel_data["ekf_vz"]
    rng_t = z_vel_data["range_time_s"]
    rng_vz = z_vel_data["range_vz"]

    common_mask = (ekf_t >= max(t0, rng_t[0])) & (ekf_t <= min(t1, rng_t[-1]))
    if not np.any(common_mask):
        print("  Skipping Z velocity plot (no overlapping data)")
        return

    ekf_t_c = ekf_t[common_mask]
    ekf_vz_c = ekf_vz[common_mask]
    rng_vz_interp = np.interp(ekf_t_c, rng_t, rng_vz)

    # Decompose into high-freq (dynamic) and low-freq (bias) components
    ekf_hf, ekf_lf = _highpass(ekf_vz_c, ekf_t_c)
    rng_hf, rng_lf = _highpass(rng_vz_interp, ekf_t_c)

    dynamic_error = ekf_hf - rng_hf
    bias = ekf_lf - rng_lf
    dynamic_rms = float(np.sqrt(np.mean(dynamic_error**2)))
    bias_rms = float(np.sqrt(np.mean(bias**2)))

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Z Velocity: Accel Quality vs Range Sensor (ground truth)", fontsize=14)

    # Panel 1: Range sensor altitude (flight context)
    ax = axes[0]
    if "range_alt" in z_vel_data:
        rng_alt_t = z_vel_data["range_alt_time_s"]
        rng_alt = z_vel_data["range_alt"]
        rng_alt_mask = (rng_alt_t >= t0) & (rng_alt_t <= t1)
        ax.plot(rng_alt_t[rng_alt_mask], rng_alt[rng_alt_mask],
                color="#1f77b4", linewidth=0.8)
    ax.set_ylabel("Range alt (m)")
    ax.set_title("Range sensor altitude (flight context)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Detrended velocity overlay
    ax = axes[1]
    ax.plot(ekf_t_c, rng_hf, label="Range sensor (detrended)",
            linewidth=0.8, alpha=0.7, color="#ff7f0e")
    ax.plot(ekf_t_c, ekf_hf, label="EKF Vz (detrended)",
            linewidth=0.8, color="#1f77b4")
    ax.set_ylabel("Vz high-freq (m/s)")
    ax.set_title("Detrended velocity — do short-term dynamics match?", fontsize=10)
    ax.set_xlim(t0, t1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Dynamic error + bias
    ax = axes[2]
    ax.plot(ekf_t_c, dynamic_error, color="black", linewidth=0.8,
            label=f"Dynamic error (RMS: {dynamic_rms:.3f} m/s)")
    ax.plot(ekf_t_c, bias, color="red", linewidth=1.0, alpha=0.8,
            label=f"Accel bias drift (RMS: {bias_rms:.3f} m/s)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Error (m/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    return fig, {"dynamic_rms": dynamic_rms, "bias_rms": bias_rms}


def plot_vibration_summary(vib_metrics, esc_data, phases, params, output_path):
    """Overview plot: vibration metrics + ESC RPM + key params."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle("Vibration & ESC Overview", fontsize=14)

    t0 = phases["armed_start_s"] or 0
    t1 = phases["armed_end_s"] or 60

    # Vibration metrics
    ax = axes[0]
    if vib_metrics and "accel_vib" in vib_metrics:
        ax.plot(vib_metrics["time_s"], vib_metrics["accel_vib"],
                label="Accel vibration", color="red", linewidth=0.8)
    if vib_metrics and "gyro_vib" in vib_metrics:
        ax.plot(vib_metrics["time_s"], vib_metrics["gyro_vib"],
                label="Gyro vibration", color="blue", linewidth=0.8)
    ax.set_ylabel("Vibration Metric")
    ax.set_xlim(t0, t1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ESC RPM
    ax = axes[1]
    if esc_data[0] is not None:
        esc_ts, rpm_list = esc_data
        for i, rpm in enumerate(rpm_list):
            ax.plot(esc_ts, rpm, linewidth=0.8, label=f"Motor {i}")
        ax.set_ylabel("RPM")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")

    # Add params annotation
    param_text = "\n".join([f"{k}: {v}" for k, v in sorted(params.items())])
    fig.text(0.99, 0.01, param_text, fontsize=7, family="monospace",
             verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Interpretation guide page
# ---------------------------------------------------------------------------

GUIDE_TEXT = [
    ("Vibration Summary (p2)",
     "Top: post-filter accel & gyro vibration metrics.\n"
     "  Accel: <1 good, 1\u20133 moderate, >3 high\n"
     "Bottom: per-motor RPM (check for spread)."),

    ("Spectrogram (p3)",
     "Z-axis vibration vs time. Blue lines = motor RPM.\n"
     "Shaded bands = DNF notch coverage.\n"
     "Bright bands tracking blue = motor vibration.\n"
     "Bright bands at fixed freq = structural resonance."),

    ("Accel PSD (p4)",
     "Right: 0-500 Hz zoom. Red dashed = motor harmonics.\n"
     "Peaks on dashed lines \u2192 DNF target.\n"
     "Peaks off dashed lines \u2192 static notch target."),

    ("Z Velocity (p5)",
     "EKF Vz vs range sensor (detrended).\n"
     "Black = dynamic error (vibration noise).\n"
     "Red = accel bias drift (rectification error).\n"
     "RMS: <0.1 good, 0.1\u20130.3 moderate, >0.3 high."),
]

# Filter parameters to display on the guide page, grouped
def _get_filter_param_groups(params, esc_count=4):
    """Build param groups, including only pole counts for active motors."""
    active_poles = [f"DSHOT_MOT_POL{i}" for i in range(1, esc_count + 1)
                    if f"DSHOT_MOT_POL{i}" in params]
    # Deduplicate: if all active motors share the same pole count, show once
    pole_values = [params[k] for k in active_poles]
    if pole_values and len(set(pole_values)) == 1:
        motor_params = [active_poles[0]]
    else:
        motor_params = active_poles

    return [
        (f"Motor (1-{esc_count})", motor_params),
        ("Accel LPF", ["IMU_ACCEL_CUTOFF"]),
        ("Accel Dynamic Notch", ["IMU_ACC_DNF_EN", "IMU_ACC_DNF_BW",
                                 "IMU_ACC_DNF_HMC", "IMU_ACC_DNF_MIN"]),
        ("Accel Static Notch", ["IMU_ACC_NF0_FRQ", "IMU_ACC_NF0_BW",
                                "IMU_ACC_NF1_FRQ", "IMU_ACC_NF1_BW"]),
        ("Gyro Dynamic Notch", ["IMU_GYRO_DNF_EN", "IMU_GYRO_DNF_BW"]),
        ("Gyro FFT", ["IMU_GYRO_FFT_EN", "IMU_GYRO_FFT_MAX"]),
    ]

_DNF_EN_LABELS = {0: "off", 1: "ESC RPM", 2: "FFT", 3: "ESC RPM + FFT"}


def render_guide_page(params, esc_count=4):
    """Render the guide + filter params as page 1 of the report."""
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#fafafa")

    # Title
    fig.text(0.5, 0.97, "Accelerometer Vibration Report",
             fontsize=18, fontweight="bold", ha="center", va="top",
             family="sans-serif")

    # --- Left side: Plot guide (compact) ---
    fig.text(0.06, 0.91, "Plot Guide",
             fontsize=13, fontweight="bold", va="top",
             family="sans-serif", color="#333333")

    section_colors = ["#d32f2f", "#1565c0", "#2e7d32", "#6a1b9a"]
    y = 0.86
    for i, (title, body) in enumerate(GUIDE_TEXT):
        color = section_colors[i]
        fig.patches.append(plt.Rectangle(
            (0.05, y - 0.003), 0.004, 0.018,
            transform=fig.transFigure, facecolor=color, clip_on=False))
        fig.text(0.065, y, title,
                 fontsize=10, fontweight="bold", va="top",
                 family="sans-serif", color=color)
        fig.text(0.07, y - 0.025, body,
                 fontsize=8.5, va="top", family="sans-serif",
                 linespacing=1.5, color="#444444")
        y -= 0.025 + 0.02 * (body.count("\n") + 1) + 0.03

    # --- Right side: Filter parameters ---
    fig.text(0.55, 0.91, "Current Filter Parameters",
             fontsize=13, fontweight="bold", va="top",
             family="sans-serif", color="#333333")

    y = 0.86
    for group_name, param_names in _get_filter_param_groups(params, esc_count):
        fig.text(0.56, y, group_name,
                 fontsize=10, fontweight="bold", va="top",
                 family="sans-serif", color="#555555")
        y -= 0.03
        for pname in param_names:
            val = params.get(pname, "N/A")
            # Add human-readable label for DNF_EN params
            extra = ""
            if pname.endswith("DNF_EN") and val != "N/A":
                extra = f"  ({_DNF_EN_LABELS.get(int(val), '?')})"
            fig.text(0.58, y, f"{pname}", fontsize=8.5, va="top",
                     family="monospace", color="#444444")
            fig.text(0.82, y, f"{val}{extra}", fontsize=8.5, va="top",
                     family="monospace", color="#111111", fontweight="bold")
            y -= 0.022
        y -= 0.015

    # Footer
    fig.text(0.5, 0.02, "Generated by accel_vibration.py",
             fontsize=8, ha="center", color="#999999", family="sans-serif")

    return fig


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def generate_summary(vib_metrics, phases, psd_peaks, motor_freq_hz,
                     fft_peaks, params, z_vel_rms, sample_rate_hz):
    """Generate a text summary of findings."""
    lines = []
    lines.append("=" * 70)
    lines.append("ACCELEROMETER VIBRATION ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Flight info
    if phases["armed_start_s"] is not None:
        duration = phases["armed_end_s"] - phases["armed_start_s"]
        lines.append(f"Flight duration:  {duration:.1f} s")
    if sample_rate_hz:
        lines.append(f"Accel FIFO rate:  {sample_rate_hz:.1f} Hz")
    lines.append("")

    # Vibration metrics
    if vib_metrics and "accel_vib" in vib_metrics:
        t0, t1 = phases["armed_start_s"], phases["armed_end_s"]
        mask = (vib_metrics["time_s"] >= t0) & (vib_metrics["time_s"] <= t1)
        accel_v = vib_metrics["accel_vib"][mask]
        gyro_v = vib_metrics.get("gyro_vib", np.array([]))[mask] if "gyro_vib" in vib_metrics else np.array([])

        lines.append("VIBRATION METRICS (in-flight)")
        lines.append(f"  Accel:  mean={np.mean(accel_v):.2f}  p95={np.percentile(accel_v, 95):.2f}  max={np.max(accel_v):.2f}")
        if len(gyro_v) > 0:
            lines.append(f"  Gyro:   mean={np.mean(gyro_v):.2f}  p95={np.percentile(gyro_v, 95):.2f}  max={np.max(gyro_v):.2f}")
        lines.append(f"  Assessment: {'GOOD' if np.mean(accel_v) < 1.0 else 'MODERATE' if np.mean(accel_v) < 3.0 else 'HIGH'}")
        lines.append("")

    # Motor frequency
    if motor_freq_hz is not None:
        lines.append("MOTOR FREQUENCY")
        lines.append(f"  Hover RPM:    {motor_freq_hz * 60:.0f}")
        lines.append(f"  Fundamental:  {motor_freq_hz:.1f} Hz")
        for h in range(2, 5):
            lines.append(f"  Harmonic {h}:   {motor_freq_hz * h:.1f} Hz")
        lines.append("")

    # PSD peaks
    lines.append("DOMINANT ACCEL PSD PEAKS")
    for axis in ["x", "y", "z"]:
        peaks = psd_peaks.get(axis, [])
        if peaks:
            peak_strs = [f"{p['freq_hz']:.0f} Hz ({p['psd_db']:.1f} dB)" for p in peaks[:5]]
            lines.append(f"  {axis.upper()}: {', '.join(peak_strs)}")
    lines.append("")

    # Peak vs motor frequency alignment
    if motor_freq_hz is not None and psd_peaks:
        lines.append("PEAK-TO-MOTOR ALIGNMENT")
        all_peaks = []
        for axis in ["x", "y", "z"]:
            all_peaks.extend(psd_peaks.get(axis, [])[:3])
        for p in sorted(all_peaks, key=lambda x: x["freq_hz"])[:8]:
            freq = p["freq_hz"]
            ratio = freq / motor_freq_hz
            nearest_harmonic = round(ratio)
            offset_hz = freq - motor_freq_hz * nearest_harmonic
            offset_pct = (offset_hz / (motor_freq_hz * nearest_harmonic)) * 100 if nearest_harmonic > 0 else 0
            lines.append(f"  {freq:.0f} Hz = {ratio:.2f}x motor  (nearest H{nearest_harmonic}, offset {offset_hz:+.1f} Hz / {offset_pct:+.1f}%)")
        lines.append("")

    # FFT coverage
    fft_max = params.get("IMU_GYRO_FFT_MAX", None)
    if fft_max is not None:
        lines.append("GYRO FFT COVERAGE")
        lines.append(f"  FFT range: {params.get('IMU_GYRO_FFT_MIN', '?')}-{fft_max} Hz")
        if motor_freq_hz and motor_freq_hz > float(fft_max):
            lines.append(f"  WARNING: Motor fundamental ({motor_freq_hz:.0f} Hz) is ABOVE FFT max ({fft_max} Hz)")
            lines.append(f"  FFT-based DNF cannot target motor vibration peaks")
        lines.append("")

    # Z velocity
    if z_vel_rms is not None:
        lines.append("Z VELOCITY ESTIMATION (vs range sensor)")
        lines.append(f"  Dynamic RMS (vibration noise):   {z_vel_rms['dynamic_rms']:.3f} m/s")
        lines.append(f"  Bias RMS (accel drift):           {z_vel_rms['bias_rms']:.3f} m/s")
        lines.append("")

    # Current filter params
    lines.append("CURRENT FILTER PARAMETERS")
    filter_params = ["IMU_ACCEL_CUTOFF", "IMU_ACC_DNF_EN", "IMU_ACC_DNF_BW",
                     "IMU_ACC_DNF_HMC", "IMU_ACC_DNF_MIN",
                     "IMU_ACC_NF0_FRQ", "IMU_ACC_NF0_BW",
                     "IMU_ACC_NF1_FRQ", "IMU_ACC_NF1_BW",
                     "IMU_GYRO_DNF_EN", "IMU_GYRO_DNF_BW",
                     "IMU_GYRO_FFT_EN", "IMU_GYRO_FFT_MAX"]
    for p in filter_params:
        val = params.get(p, "N/A")
        lines.append(f"  {p:25s} = {val}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PX4 accelerometer vibration analyzer")
    parser.add_argument("ulog_file", help="Path to .ulg file")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for plots (default: same dir as log)")
    args = parser.parse_args()

    if not os.path.isfile(args.ulog_file):
        print(f"Error: file not found: {args.ulog_file}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.ulog_file))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {args.ulog_file}...")
    ulog = ULog(args.ulog_file)
    print(f"  Duration: {(ulog.last_timestamp - ulog.start_timestamp) / 1e6:.1f} s")
    print(f"  Topics: {len(ulog.data_list)}")

    # Collect relevant parameters
    param_names = [
        "IMU_ACCEL_CUTOFF", "IMU_ACC_DNF_EN", "IMU_ACC_DNF_BW", "IMU_ACC_DNF_HMC",
        "IMU_ACC_DNF_MIN", "IMU_ACC_NF0_FRQ", "IMU_ACC_NF0_BW", "IMU_ACC_NF1_FRQ",
        "IMU_ACC_NF1_BW", "IMU_GYRO_DNF_EN", "IMU_GYRO_DNF_BW", "IMU_GYRO_DNF_HMC",
        "IMU_GYRO_FFT_EN", "IMU_GYRO_FFT_MAX", "IMU_GYRO_FFT_MIN", "IMU_GYRO_FFT_SNR",
    ] + [f"DSHOT_MOT_POL{i}" for i in range(1, 13)]
    params = {}
    for p in param_names:
        val = get_param(ulog, p)
        if val is not None:
            params[p] = val

    print(f"\nKey params: IMU_ACC_DNF_EN={params.get('IMU_ACC_DNF_EN', 'N/A')}, "
          f"IMU_ACCEL_CUTOFF={params.get('IMU_ACCEL_CUTOFF', 'N/A')}")

    # Detect flight phases
    phases = detect_flight_phases(ulog)
    print(f"Armed: {phases['armed_start_s']:.1f}s - {phases['armed_end_s']:.1f}s")

    hover_start, hover_end = detect_hover_segment(ulog, phases)
    print(f"Hover segment: {hover_start:.1f}s - {hover_end:.1f}s")

    # Extract vibration metrics
    print("\nExtracting vibration metrics...")
    vib_metrics = extract_vibration_metrics(ulog)

    # Extract ESC RPM
    print("Extracting ESC RPM...")
    esc_ts, rpm_list = extract_esc_rpm(ulog, phases["armed_start_s"], phases["armed_end_s"])

    motor_freq_hz = None
    motor_freq_hz_arr = None
    if rpm_list and len(rpm_list) > 0:
        # Average across motors for frequency calculation
        rpm_avg = np.mean(rpm_list, axis=0)
        motor_freq_hz_arr = np.abs(rpm_avg) / 60.0
        # Median during hover
        hover_mask = (esc_ts >= hover_start) & (esc_ts <= hover_end)
        if np.any(hover_mask):
            motor_freq_hz = float(np.median(motor_freq_hz_arr[hover_mask]))
            print(f"  Hover motor freq: {motor_freq_hz:.1f} Hz ({motor_freq_hz * 60:.0f} RPM)")

    # Extract accel FIFO and compute PSD
    print("Extracting accel FIFO data...")
    sample_rate_hz, fifo_data, _ti = extract_accel_fifo(ulog, hover_start, hover_end)

    psd_data = {}
    psd_peaks = {}
    if fifo_data is not None:
        print(f"  Sample rate: {sample_rate_hz:.1f} Hz")
        print(f"  Samples per axis: {len(fifo_data['x'])}")
        print("Computing PSD...")
        for axis in ["x", "y", "z"]:
            freqs, psd = compute_psd(fifo_data[axis], sample_rate_hz)
            psd_data[axis] = (freqs, psd)
            peaks = find_psd_peaks(freqs, psd, min_freq=20,
                                    max_freq=sample_rate_hz / 2)
            psd_peaks[axis] = peaks
            if peaks:
                top = peaks[0]
                print(f"  {axis.upper()} dominant peak: {top['freq_hz']:.1f} Hz ({top['psd_db']:.1f} dB)")
    else:
        print("  WARNING: No sensor_accel_fifo data found!")

    # Extract FFT peaks
    print("Extracting gyro FFT peaks...")
    fft_peaks = extract_fft_peaks(ulog, hover_start, hover_end)

    # Extract Z velocity data
    print("Extracting Z velocity data...")
    z_vel_data = extract_z_velocity(ulog)

    # Generate plots
    print("\nGenerating plots...")

    fig_vibsummary = plot_vibration_summary(
        vib_metrics, (esc_ts, rpm_list), phases, params,
        os.path.join(output_dir, "vibration_summary.png"))

    fig_spectrogram = None
    if fifo_data is not None:
        sr_full, fifo_full, fifo_ti = extract_accel_fifo(ulog,
                                                 phases["armed_start_s"],
                                                 phases["armed_end_s"])
        if fifo_full is not None:
            spec_data = compute_spectrogram(fifo_full["z"], sr_full)
            # Scale spectrogram times to correct for inter-batch gaps
            times, freqs, Sxx = spec_data
            times = times * fifo_ti["time_scale"]
            spec_data = (times, freqs, Sxx)
            fig_spectrogram = plot_spectrogram(
                spec_data, esc_ts, motor_freq_hz_arr,
                fifo_ti["actual_start_s"], vib_metrics, phases, params,
                os.path.join(output_dir, "accel_spectrogram.png"))

    fig_psd = None
    if psd_data:
        fig_psd = plot_psd(psd_data, sample_rate_hz, psd_peaks, motor_freq_hz,
                           params, os.path.join(output_dir, "accel_psd.png"))

    z_vel_rms = None
    fig_zvel = None
    if "ekf_vz" in z_vel_data and "range_vz" in z_vel_data:
        fig_zvel, z_vel_rms = plot_z_velocity(
            z_vel_data, phases, os.path.join(output_dir, "z_velocity.png"))

    # Save combined PDF: guide first, then plots by importance
    pdf_path = os.path.join(output_dir, "analysis.pdf")
    esc_count = len(rpm_list) if rpm_list else 4
    pdf_figures = [render_guide_page(params, esc_count)]  # page 1: guide + params
    # page 2: vibration overview (first thing to check)
    pdf_figures.append(fig_vibsummary)
    # page 3: spectrogram (identifies what's causing vibration)
    if fig_spectrogram is not None:
        pdf_figures.append(fig_spectrogram)
    # page 4: PSD (detailed frequency analysis)
    if fig_psd is not None:
        pdf_figures.append(fig_psd)
    # page 5: Z velocity (impact on estimation)
    if fig_zvel is not None:
        pdf_figures.append(fig_zvel)

    with PdfPages(pdf_path) as pdf:
        for fig in pdf_figures:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  Saved: {pdf_path}")

    # Generate text summary
    summary = generate_summary(vib_metrics, phases, psd_peaks, motor_freq_hz,
                                fft_peaks, params, z_vel_rms, sample_rate_hz)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")

    # Print summary to stdout
    print()
    print(summary)


if __name__ == "__main__":
    main()
