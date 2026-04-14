#!/usr/bin/env python3
"""
PX4 magnetometer-thrust correlation analyzer.

Investigates the correlation between thrust and magnetometer readings to
assess whether mag power compensation is needed, and if already enabled,
whether the current coefficients are effective.

Usage:
    python3 mag_thrust_compensation.py <log.ulg> [--output-dir <dir>]

Outputs:
    - mag_thrust_analysis.pdf    Combined PDF with all plots
    - mag_thrust_summary.txt     Text summary with findings and recommendations
"""

import argparse
import os
import sys

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
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
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


def get_all_instances(ulog, topic_name):
    """Return all multi-instances of a topic, keyed by multi_id."""
    instances = {}
    for d in ulog.data_list:
        if d.name == topic_name:
            instances[d.multi_id] = d
    return instances


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

    vstatus = get_topic(ulog, "vehicle_status")
    if vstatus is not None and "arming_state" in vstatus.data:
        ts = timestamps_to_seconds(vstatus.data["timestamp"], start_us)
        armed_mask = vstatus.data["arming_state"] == 2
        armed_idx = np.where(armed_mask)[0]
        if len(armed_idx) > 0:
            info["armed_start_s"] = ts[armed_idx[0]]
            info["armed_end_s"] = ts[armed_idx[-1]]
            return info

    # Fallback: actuator_motors
    motors = get_topic(ulog, "actuator_motors")
    if motors is not None:
        ts = timestamps_to_seconds(motors.data["timestamp"], start_us)
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

    end_us = ulog.last_timestamp
    info["armed_start_s"] = 0
    info["armed_end_s"] = (end_us - start_us) / 1e6
    return info


def detect_hover_segment(phases):
    """Find a stable hover segment (middle 60% of armed time)."""
    if phases["armed_start_s"] is None:
        return None, None
    t0 = phases["armed_start_s"]
    t1 = phases["armed_end_s"]
    duration = t1 - t0
    return t0 + duration * 0.2, t1 - duration * 0.2


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_mag_data(ulog):
    """Extract raw sensor_mag and corrected vehicle_magnetometer data."""
    start_us = ulog.start_timestamp
    result = {"raw": {}, "corrected": {}}

    # Raw sensor_mag (multiple instances possible)
    raw_instances = get_all_instances(ulog, "sensor_mag")
    for mid, dataset in raw_instances.items():
        result["raw"][mid] = {
            "time_s": timestamps_to_seconds(dataset.data["timestamp"], start_us),
            "x": dataset.data["x"],
            "y": dataset.data["y"],
            "z": dataset.data["z"],
            "device_id": dataset.data["device_id"],
        }

    # Corrected vehicle_magnetometer
    vmag = get_topic(ulog, "vehicle_magnetometer")
    if vmag is not None:
        result["corrected"] = {
            "time_s": timestamps_to_seconds(vmag.data["timestamp"], start_us),
            "x": vmag.data["magnetometer_ga[0]"],
            "y": vmag.data["magnetometer_ga[1]"],
            "z": vmag.data["magnetometer_ga[2]"],
        }

    return result


def extract_thrust(ulog):
    """Extract vehicle_thrust_setpoint z-component (negated, constrained 0..1)."""
    start_us = ulog.start_timestamp
    vts = get_topic(ulog, "vehicle_thrust_setpoint")
    if vts is None:
        return {}
    thrust_z = vts.data["xyz[2]"]
    return {
        "time_s": timestamps_to_seconds(vts.data["timestamp"], start_us),
        "thrust": np.clip(-thrust_z, 0.0, 1.0),
    }


def extract_battery(ulog):
    """Extract battery current for current-based compensation analysis."""
    start_us = ulog.start_timestamp
    bat = get_topic(ulog, "battery_status")
    if bat is None:
        return {}
    result = {
        "time_s": timestamps_to_seconds(bat.data["timestamp"], start_us),
    }
    if "current_a" in bat.data:
        result["current_a"] = bat.data["current_a"]
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def interpolate_to_mag(mag_time, signal_time, signal_values):
    """Interpolate a signal onto mag timestamps."""
    return np.interp(mag_time, signal_time, signal_values)


def detrend_signal(signal, time_s, cutoff_period=10.0):
    """Remove low-frequency trends (thermal drift, etc.) via high-pass."""
    if len(signal) < 10:
        return signal
    dt = np.median(np.diff(time_s))
    if dt <= 0:
        return signal - np.mean(signal)
    # Simple: subtract a moving average
    window = max(3, int(cutoff_period / dt))
    if window >= len(signal):
        return signal - np.mean(signal)
    kernel = np.ones(window) / window
    trend = np.convolve(signal, kernel, mode="same")
    # Fix edges
    trend[:window // 2] = trend[window // 2]
    trend[-(window // 2):] = trend[-(window // 2)]
    return signal - trend


def compute_correlations(mag_data, thrust_data, battery_data,
                         hover_start, hover_end):
    """Compute per-axis correlations between mag and thrust/current."""
    result = {}

    # Use corrected mag if available, else first raw instance
    if mag_data["corrected"]:
        mag = mag_data["corrected"]
        result["mag_source"] = "vehicle_magnetometer (corrected)"
    elif mag_data["raw"]:
        first_id = min(mag_data["raw"].keys())
        mag = mag_data["raw"][first_id]
        result["mag_source"] = f"sensor_mag (instance {first_id}, raw)"
    else:
        return result

    mag_t = mag["time_s"]
    hover_mask = (mag_t >= hover_start) & (mag_t <= hover_end)

    if np.sum(hover_mask) < 20:
        print("  WARNING: too few mag samples in hover segment")
        return result

    result["axes"] = {}

    for axis in ["x", "y", "z"]:
        ax_result = {}
        mag_hover = mag[axis][hover_mask]
        mag_hover_t = mag_t[hover_mask]

        # Detrend to isolate thrust-correlated variation from thermal drift
        mag_detrended = detrend_signal(mag_hover, mag_hover_t, cutoff_period=10.0)
        ax_result["mean"] = float(np.mean(mag_hover))
        ax_result["std"] = float(np.std(mag_hover))
        ax_result["detrended_std"] = float(np.std(mag_detrended))

        # Correlation with thrust
        if thrust_data:
            thrust_interp = interpolate_to_mag(
                mag_hover_t, thrust_data["time_s"], thrust_data["thrust"])
            thrust_detrended = detrend_signal(
                thrust_interp, mag_hover_t, cutoff_period=10.0)

            if np.std(thrust_detrended) > 1e-6 and np.std(mag_detrended) > 1e-6:
                corr = float(np.corrcoef(thrust_detrended, mag_detrended)[0, 1])
                ax_result["thrust_corr"] = corr

                # Linear fit: mag = slope * thrust + offset
                slope, offset = np.polyfit(thrust_interp, mag_hover, 1)
                ax_result["thrust_slope"] = float(slope)
                ax_result["thrust_offset"] = float(offset)
            else:
                ax_result["thrust_corr"] = 0.0

        # Correlation with battery current
        if battery_data and "current_a" in battery_data:
            current_interp = interpolate_to_mag(
                mag_hover_t, battery_data["time_s"],
                battery_data["current_a"])
            current_detrended = detrend_signal(
                current_interp, mag_hover_t, cutoff_period=10.0)

            if np.std(current_detrended) > 1e-6 and np.std(mag_detrended) > 1e-6:
                corr = float(np.corrcoef(current_detrended, mag_detrended)[0, 1])
                ax_result["current_corr"] = corr

                slope, offset = np.polyfit(current_interp * 0.001, mag_hover, 1)
                ax_result["current_slope"] = float(slope)  # Gauss per kA
            else:
                ax_result["current_corr"] = 0.0

        result["axes"][axis] = ax_result

    return result


# Also analyze raw mag for comparison when corrected is available
def compute_raw_correlations(mag_data, thrust_data, hover_start, hover_end):
    """Compute correlations on raw sensor_mag data (all instances)."""
    results = {}
    for mid, mag in mag_data["raw"].items():
        mag_t = mag["time_s"]
        hover_mask = (mag_t >= hover_start) & (mag_t <= hover_end)
        if np.sum(hover_mask) < 20 or not thrust_data:
            continue

        instance_result = {}
        for axis in ["x", "y", "z"]:
            mag_hover = mag[axis][hover_mask]
            mag_hover_t = mag_t[hover_mask]
            mag_detrended = detrend_signal(mag_hover, mag_hover_t, cutoff_period=10.0)

            thrust_interp = interpolate_to_mag(
                mag_hover_t, thrust_data["time_s"], thrust_data["thrust"])
            thrust_detrended = detrend_signal(
                thrust_interp, mag_hover_t, cutoff_period=10.0)

            if np.std(thrust_detrended) > 1e-6 and np.std(mag_detrended) > 1e-6:
                corr = float(np.corrcoef(thrust_detrended, mag_detrended)[0, 1])
                slope, offset = np.polyfit(thrust_interp, mag_hover, 1)
                instance_result[axis] = {
                    "thrust_corr": corr,
                    "thrust_slope": float(slope),
                }
            else:
                instance_result[axis] = {"thrust_corr": 0.0, "thrust_slope": 0.0}

        results[mid] = instance_result

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mag_timeseries(mag_data, thrust_data, phases, hover_start, hover_end):
    """Plot mag XYZ and thrust over time."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Magnetometer & Thrust Time Series", fontsize=14)

    # Corrected mag
    if mag_data["corrected"]:
        mag = mag_data["corrected"]
        label_prefix = "vehicle_magnetometer"
    elif mag_data["raw"]:
        first_id = min(mag_data["raw"].keys())
        mag = mag_data["raw"][first_id]
        label_prefix = f"sensor_mag[{first_id}]"
    else:
        plt.close(fig)
        return None

    colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728"}
    for i, axis in enumerate(["x", "y", "z"]):
        axes[i].plot(mag["time_s"], mag[axis], color=colors[axis],
                     linewidth=0.5, label=f"{label_prefix} {axis.upper()}")
        axes[i].set_ylabel(f"Mag {axis.upper()} (Ga)")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, alpha=0.3)

        # Shade hover segment
        if hover_start is not None:
            axes[i].axvspan(hover_start, hover_end, alpha=0.1, color="green",
                            label="hover" if i == 0 else None)

    # Thrust
    if thrust_data:
        axes[3].plot(thrust_data["time_s"], thrust_data["thrust"],
                     color="#ff7f0e", linewidth=0.5, label="thrust Z")
        if hover_start is not None:
            axes[3].axvspan(hover_start, hover_end, alpha=0.1, color="green")
    axes[3].set_ylabel("Thrust")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # Mark armed region
    if phases["armed_start_s"] is not None:
        for ax in axes:
            ax.axvline(phases["armed_start_s"], color="gray", ls="--",
                       alpha=0.5, linewidth=0.8)
            ax.axvline(phases["armed_end_s"], color="gray", ls="--",
                       alpha=0.5, linewidth=0.8)

    fig.tight_layout()
    return fig


def plot_detrended_overlay(mag_data, thrust_data, hover_start, hover_end):
    """Plot detrended mag overlaid with detrended thrust (scaled) per axis."""
    if mag_data["corrected"]:
        mag = mag_data["corrected"]
        src = "vehicle_magnetometer"
    elif mag_data["raw"]:
        first_id = min(mag_data["raw"].keys())
        mag = mag_data["raw"][first_id]
        src = f"sensor_mag[{first_id}]"
    else:
        return None

    if not thrust_data:
        return None

    mag_t = mag["time_s"]
    hover_mask = (mag_t >= hover_start) & (mag_t <= hover_end)
    if np.sum(hover_mask) < 20:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Detrended Mag vs Thrust (hover segment)\nSource: {src}",
                 fontsize=13)

    colors_mag = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728"}
    mag_hover_t = mag_t[hover_mask]

    thrust_interp = interpolate_to_mag(
        mag_hover_t, thrust_data["time_s"], thrust_data["thrust"])
    thrust_detrended = detrend_signal(thrust_interp, mag_hover_t, cutoff_period=10.0)

    for i, axis in enumerate(["x", "y", "z"]):
        mag_detrended = detrend_signal(
            mag[axis][hover_mask], mag_hover_t, cutoff_period=10.0)

        ax = axes[i]
        ax.plot(mag_hover_t, mag_detrended, color=colors_mag[axis],
                linewidth=0.5, alpha=0.8, label=f"mag {axis.upper()} (detrended)")

        # Scale thrust to match mag amplitude for visual comparison
        if np.std(thrust_detrended) > 1e-6:
            scale = np.std(mag_detrended) / np.std(thrust_detrended)
            ax.plot(mag_hover_t, thrust_detrended * scale,
                    color="#ff7f0e", linewidth=0.8, alpha=0.7,
                    label=f"thrust (scaled)")

            corr = np.corrcoef(thrust_detrended, mag_detrended)[0, 1]
            ax.set_title(f"{axis.upper()}-axis    r = {corr:+.3f}", fontsize=10,
                         loc="left")

        ax.set_ylabel("Gauss (detrended)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_scatter(mag_data, thrust_data, hover_start, hover_end, cal_instance=0):
    """Scatter plots: mag vs thrust for each axis during hover."""
    if mag_data["corrected"]:
        mag = mag_data["corrected"]
        src = "vehicle_magnetometer"
    elif mag_data["raw"]:
        first_id = min(mag_data["raw"].keys())
        mag = mag_data["raw"][first_id]
        src = f"sensor_mag[{first_id}]"
    else:
        return None

    if not thrust_data:
        return None

    mag_t = mag["time_s"]
    hover_mask = (mag_t >= hover_start) & (mag_t <= hover_end)
    if np.sum(hover_mask) < 20:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Mag vs Thrust Scatter (hover)\nSource: {src}", fontsize=13)

    colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728"}
    param_axis = {"x": "X", "y": "Y", "z": "Z"}
    mag_hover_t = mag_t[hover_mask]
    thrust_interp = interpolate_to_mag(
        mag_hover_t, thrust_data["time_s"], thrust_data["thrust"])

    for i, axis in enumerate(["x", "y", "z"]):
        mag_vals = mag[axis][hover_mask]
        ax = axes[i]
        ax.scatter(thrust_interp, mag_vals, s=1, alpha=0.3, color=colors[axis])

        # Linear fit
        if np.std(thrust_interp) > 1e-6:
            slope, offset = np.polyfit(thrust_interp, mag_vals, 1)
            t_range = np.array([thrust_interp.min(), thrust_interp.max()])
            param_name = f"CAL_MAG{cal_instance}_{param_axis[axis]}COMP"
            ax.plot(t_range, slope * t_range + offset, "k--", linewidth=1.5,
                    label=f"slope={slope:.4f} Ga/unit")
            corr = np.corrcoef(thrust_interp, mag_vals)[0, 1]
            ax.set_title(f"{axis.upper()}-axis  r={corr:+.3f}", fontsize=10)
            ax.text(0.5, 0.02, f"{param_name} = {-slope:.3f}",
                    transform=ax.transAxes, ha="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#D8D8D8",
                              ec="0.5", alpha=0.9))
            ax.legend(fontsize=8)

        ax.set_xlabel("Thrust")
        ax.set_ylabel(f"Mag {axis.upper()} (Ga)")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_raw_vs_corrected(mag_data, thrust_data, hover_start, hover_end,
                          cal_instance=0):
    """If both raw and corrected mag exist, show the effect of compensation."""
    if not mag_data["corrected"] or not mag_data["raw"]:
        return None
    if not thrust_data:
        return None

    first_id = min(mag_data["raw"].keys())
    raw = mag_data["raw"][first_id]
    corrected = mag_data["corrected"]

    fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex="col")
    fig.suptitle("Raw vs Corrected Mag-Thrust Correlation", fontsize=14)

    param_axis = {"x": "X", "y": "Y", "z": "Z"}

    for i, axis in enumerate(["x", "y", "z"]):
        for j, (mag, label) in enumerate([(raw, "Raw"), (corrected, "Corrected")]):
            mag_t = mag["time_s"]
            hover_mask = (mag_t >= hover_start) & (mag_t <= hover_end)
            if np.sum(hover_mask) < 20:
                continue

            mag_hover_t = mag_t[hover_mask]
            mag_vals = mag[axis][hover_mask]
            thrust_interp = interpolate_to_mag(
                mag_hover_t, thrust_data["time_s"], thrust_data["thrust"])

            ax = axes[i][j]
            ax.scatter(thrust_interp, mag_vals, s=1, alpha=0.3)
            if np.std(thrust_interp) > 1e-6:
                slope, offset = np.polyfit(thrust_interp, mag_vals, 1)
                t_range = np.array([thrust_interp.min(), thrust_interp.max()])
                ax.plot(t_range, slope * t_range + offset, "k--", linewidth=1.5,
                        label=f"slope={slope:.4f}")
                corr = np.corrcoef(thrust_interp, mag_vals)[0, 1]
                ax.set_title(f"{label} {axis.upper()}  r={corr:+.3f}", fontsize=10)
                ax.legend(fontsize=8)

                # Label raw side with recommended param value
                if j == 0:
                    param_name = (f"CAL_MAG{cal_instance}_"
                                  f"{param_axis[axis]}COMP")
                    ax.text(0.5, 0.02, f"{param_name} = {-slope:.3f}",
                            transform=ax.transAxes, ha="center", fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="#D8D8D8",
                                      ec="0.5", alpha=0.9))
                else:
                    ax.text(0.5, 0.02, "residual after compensation",
                            transform=ax.transAxes, ha="center", fontsize=8,
                            fontstyle="italic", color="0.4")

            ax.set_ylabel(f"Mag {axis.upper()} (Ga)")
            if i == 2:
                ax.set_xlabel("Thrust")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_multi_instance_comparison(mag_data, thrust_data, hover_start, hover_end,
                                   cal_instance_map=None):
    """Compare thrust correlation across all raw mag instances."""
    if len(mag_data["raw"]) < 2:
        return None
    if not thrust_data:
        return None
    if cal_instance_map is None:
        cal_instance_map = {}

    n_instances = len(mag_data["raw"])
    fig, axes = plt.subplots(3, n_instances, figsize=(5 * n_instances, 9),
                             sharex=True)
    if n_instances == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Per-Instance Raw Mag vs Thrust", fontsize=14)

    for col, (mid, mag) in enumerate(sorted(mag_data["raw"].items())):
        mag_t = mag["time_s"]
        hover_mask = (mag_t >= hover_start) & (mag_t <= hover_end)
        if np.sum(hover_mask) < 20:
            continue

        mag_hover_t = mag_t[hover_mask]
        thrust_interp = interpolate_to_mag(
            mag_hover_t, thrust_data["time_s"], thrust_data["thrust"])

        dev_id = mag["device_id"][0] if len(mag["device_id"]) > 0 else 0

        for row, axis in enumerate(["x", "y", "z"]):
            mag_vals = mag[axis][hover_mask]
            ax = axes[row][col]
            ax.scatter(thrust_interp, mag_vals, s=1, alpha=0.3)

            if np.std(thrust_interp) > 1e-6:
                slope, offset = np.polyfit(thrust_interp, mag_vals, 1)
                t_range = np.array([thrust_interp.min(), thrust_interp.max()])
                ax.plot(t_range, slope * t_range + offset, "k--", linewidth=1.5,
                        label=f"slope={slope:.4f}")
                corr = np.corrcoef(thrust_interp, mag_vals)[0, 1]
                ax.set_title(f"inst {mid} (0x{dev_id:08x}) {axis.upper()}  "
                             f"r={corr:+.3f}", fontsize=9)
                ax.legend(fontsize=7)

                ci = cal_instance_map.get(mid, mid)
                pa = {"x": "X", "y": "Y", "z": "Z"}[axis]
                ax.text(0.5, 0.02,
                        f"CAL_MAG{ci}_{pa}COMP = {-slope:.3f}",
                        transform=ax.transAxes, ha="center", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#D8D8D8",
                                  ec="0.5", alpha=0.9))

            ax.set_ylabel(f"Mag {axis.upper()} (Ga)")
            if row == 2:
                ax.set_xlabel("Thrust")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_field_norm(mag_data, thrust_data, battery_data, phases,
                    hover_start, hover_end):
    """Plot mag field norm alongside thrust and current to show disturbance."""
    if mag_data["corrected"]:
        mag = mag_data["corrected"]
        src = "vehicle_magnetometer"
    elif mag_data["raw"]:
        first_id = min(mag_data["raw"].keys())
        mag = mag_data["raw"][first_id]
        src = f"sensor_mag[{first_id}]"
    else:
        return None

    mag_t = mag["time_s"]
    norm = np.sqrt(mag["x"]**2 + mag["y"]**2 + mag["z"]**2)

    has_current = battery_data and "current_a" in battery_data
    n_rows = 3 if has_current else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows + 1), sharex=True)
    fig.suptitle(f"Mag Field Norm vs Thrust/Current\nSource: {src}", fontsize=14)

    # Norm
    axes[0].plot(mag_t, norm, color="#17becf", linewidth=0.5, label="|B| norm")
    axes[0].set_ylabel("Field Norm (Ga)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Thrust
    if thrust_data:
        axes[1].plot(thrust_data["time_s"], thrust_data["thrust"],
                     color="#ff7f0e", linewidth=0.5, label="thrust Z")
    axes[1].set_ylabel("Thrust")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Current
    if has_current:
        axes[2].plot(battery_data["time_s"], battery_data["current_a"],
                     color="#e377c2", linewidth=0.5, label="battery current")
        axes[2].set_ylabel("Current (A)")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    # Hover shading and armed markers
    for ax in axes:
        if hover_start is not None:
            ax.axvspan(hover_start, hover_end, alpha=0.1, color="green")
        if phases["armed_start_s"] is not None:
            ax.axvline(phases["armed_start_s"], color="gray", ls="--",
                       alpha=0.5, linewidth=0.8)
            ax.axvline(phases["armed_end_s"], color="gray", ls="--",
                       alpha=0.5, linewidth=0.8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def generate_summary(phases, params, corr, raw_corr, hover_start, hover_end,
                     cal_instance_map=None):
    """Generate text summary of mag-thrust analysis."""
    if cal_instance_map is None:
        cal_instance_map = {}

    lines = []
    lines.append("=" * 70)
    lines.append("MAGNETOMETER-THRUST COMPENSATION ANALYSIS")
    lines.append("=" * 70)

    # Parameters
    comp_typ = int(params.get("CAL_MAG_COMP_TYP", 0))
    comp_labels = {0: "Disabled", 1: "Throttle", 2: "Current (bat 0)",
                   3: "Current (bat 1)"}
    lines.append(f"\nCAL_MAG_COMP_TYP = {comp_typ} ({comp_labels.get(comp_typ, '?')})")

    for idx in range(4):
        xc = params.get(f"CAL_MAG{idx}_XCOMP")
        yc = params.get(f"CAL_MAG{idx}_YCOMP")
        zc = params.get(f"CAL_MAG{idx}_ZCOMP")
        if xc is not None:
            lines.append(f"  CAL_MAG{idx}_{{X,Y,Z}}COMP = "
                         f"{xc:.4f}, {yc:.4f}, {zc:.4f}")

    lines.append(f"\nArmed: {phases['armed_start_s']:.1f}s - "
                 f"{phases['armed_end_s']:.1f}s")
    lines.append(f"Hover: {hover_start:.1f}s - {hover_end:.1f}s")

    # Corrected mag correlations (post-compensation if enabled)
    if corr and "axes" in corr:
        label = "Post-compensation" if comp_typ != 0 else "Corrected"
        lines.append(f"\n--- {label} correlations "
                     f"({corr.get('mag_source', '?')}) ---")
        lines.append(f"{'Axis':<6} {'Mean (Ga)':>10} {'Std (Ga)':>10} "
                     f"{'Thrust r':>10} {'Slope':>12}")
        lines.append("-" * 55)
        for axis in ["x", "y", "z"]:
            ax = corr["axes"].get(axis, {})
            thrust_r = ax.get("thrust_corr", float("nan"))
            slope = ax.get("thrust_slope", float("nan"))
            lines.append(f"  {axis.upper():<4} {ax.get('mean', 0):>10.4f} "
                         f"{ax.get('std', 0):>10.4f} "
                         f"{thrust_r:>+10.3f} "
                         f"{slope:>12.4f} Ga/unit")

    # Raw per-instance correlations + recommended params
    if raw_corr:
        lines.append(f"\n--- Raw sensor_mag thrust correlations "
                     f"(pre-compensation) ---")
        for mid, axes_data in sorted(raw_corr.items()):
            ci = cal_instance_map.get(mid, mid)
            lines.append(f"\n  Instance {mid} (CAL_MAG{ci}):")
            lines.append(f"  {'Axis':<6} {'Thrust r':>10} {'Raw slope':>12} "
                         f"{'Recommended':>20}")
            lines.append(f"  {'-' * 52}")
            for axis in ["x", "y", "z"]:
                ax = axes_data.get(axis, {})
                r = ax.get("thrust_corr", 0)
                s = ax.get("thrust_slope", 0)
                pa = axis.upper()
                param = f"CAL_MAG{ci}_{pa}COMP"
                lines.append(f"    {pa:<4} {r:>+10.3f} {s:>12.4f}"
                             f"   {param} = {-s:.3f}")

    # Compensation effectiveness evaluation
    lines.append(f"\n--- Assessment ---")

    if corr and "axes" in corr:
        max_corrected_r = max(abs(corr["axes"].get(a, {}).get("thrust_corr", 0))
                              for a in ["x", "y", "z"])

    max_raw_r = 0.0
    if raw_corr:
        for axes_data in raw_corr.values():
            for axis in ["x", "y", "z"]:
                r = abs(axes_data.get(axis, {}).get("thrust_corr", 0))
                max_raw_r = max(max_raw_r, r)

    if comp_typ != 0 and raw_corr and corr and "axes" in corr:
        # Compensation is enabled — evaluate effectiveness
        lines.append(f"Compensation: ENABLED ({comp_labels.get(comp_typ, '?')})")
        lines.append(f"  Raw max |r|:       {max_raw_r:.3f}")
        lines.append(f"  Corrected max |r|: {max_corrected_r:.3f}")

        if max_raw_r > 0.01:
            reduction = (1.0 - max_corrected_r / max_raw_r) * 100
            lines.append(f"  Correlation reduction: {reduction:.0f}%")

        if max_corrected_r < 0.15:
            lines.append("\n  GOOD — compensation is effective.")
        elif max_corrected_r < 0.3:
            lines.append("\n  OK — compensation helps but residual correlation "
                         "remains.")
            lines.append("  Consider updating COMP params to the recommended "
                         "values above.")
        else:
            lines.append("\n  POOR — significant residual correlation despite "
                         "compensation.")
            lines.append("  Update COMP params to the recommended values above.")

    elif comp_typ == 0:
        # Compensation disabled
        if max_raw_r > 0.5:
            lines.append("STRONG thrust-mag correlation detected with "
                         "compensation DISABLED.")
            lines.append("Recommendation: set the recommended params above and "
                         "enable:")
            lines.append("  param set CAL_MAG_COMP_TYP 1")
        elif max_raw_r > 0.2:
            lines.append("Moderate thrust-mag correlation. Compensation may help.")
            lines.append("Recommended params are listed above.")
        else:
            lines.append("Low thrust-mag correlation. Compensation unlikely "
                         "to help.")

    # Always print recommended param commands
    if raw_corr:
        lines.append(f"\n--- Recommended param commands ---")
        lines.append(f"param set CAL_MAG_COMP_TYP 1")
        for mid, axes_data in sorted(raw_corr.items()):
            ci = cal_instance_map.get(mid, mid)
            for axis in ["x", "y", "z"]:
                s = axes_data.get(axis, {}).get("thrust_slope", 0)
                pa = axis.upper()
                lines.append(f"param set CAL_MAG{ci}_{pa}COMP {-s:.3f}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PX4 magnetometer-thrust correlation analyzer")
    parser.add_argument("ulog_file", help="Path to .ulg file")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: same dir as log)")
    args = parser.parse_args()

    if not os.path.isfile(args.ulog_file):
        print(f"Error: file not found: {args.ulog_file}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(
        os.path.abspath(args.ulog_file))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {args.ulog_file}...")
    ulog = ULog(args.ulog_file)
    duration = (ulog.last_timestamp - ulog.start_timestamp) / 1e6
    print(f"  Duration: {duration:.1f} s")
    print(f"  Topics: {len(ulog.data_list)}")

    # Collect parameters
    params = {}
    param_names = ["CAL_MAG_COMP_TYP"]
    for idx in range(4):
        param_names.extend([
            f"CAL_MAG{idx}_ID",
            f"CAL_MAG{idx}_XCOMP", f"CAL_MAG{idx}_YCOMP", f"CAL_MAG{idx}_ZCOMP",
        ])
    for p in param_names:
        val = get_param(ulog, p)
        if val is not None:
            params[p] = val

    comp_typ = int(params.get("CAL_MAG_COMP_TYP", 0))
    comp_labels = {0: "Disabled", 1: "Throttle", 2: "Current (bat 0)",
                   3: "Current (bat 1)"}
    print(f"\nMag compensation: {comp_labels.get(comp_typ, '?')} "
          f"(CAL_MAG_COMP_TYP={comp_typ})")

    # Detect flight phases
    phases = detect_flight_phases(ulog)
    print(f"Armed: {phases['armed_start_s']:.1f}s - {phases['armed_end_s']:.1f}s")
    hover_start, hover_end = detect_hover_segment(phases)
    print(f"Hover: {hover_start:.1f}s - {hover_end:.1f}s")

    # Extract data
    print("\nExtracting data...")
    mag_data = extract_mag_data(ulog)
    thrust_data = extract_thrust(ulog)
    battery_data = extract_battery(ulog)

    n_raw = len(mag_data["raw"])
    has_corrected = bool(mag_data["corrected"])
    print(f"  Raw mag instances: {n_raw}")
    print(f"  Corrected mag: {'yes' if has_corrected else 'no'}")
    print(f"  Thrust data: {'yes' if thrust_data else 'no'}")
    print(f"  Battery current: {'yes' if 'current_a' in battery_data else 'no'}")

    if n_raw == 0 and not has_corrected:
        print("Error: no magnetometer data found", file=sys.stderr)
        sys.exit(1)

    # Map raw sensor device IDs to calibration instance numbers
    cal_instance_map = {}  # {multi_id: cal_instance}
    for mid, mag in mag_data["raw"].items():
        if len(mag["device_id"]) > 0:
            dev_id = int(mag["device_id"][0])
            for j in range(4):
                cal_id = params.get(f"CAL_MAG{j}_ID")
                if cal_id is not None and int(cal_id) == dev_id:
                    cal_instance_map[mid] = j
                    break
            if mid not in cal_instance_map:
                cal_instance_map[mid] = mid

    # Compute correlations
    print("\nComputing correlations...")
    corr = compute_correlations(mag_data, thrust_data, battery_data,
                                hover_start, hover_end)
    raw_corr = compute_raw_correlations(mag_data, thrust_data,
                                        hover_start, hover_end)

    if corr and "axes" in corr:
        print(f"\n  Source: {corr.get('mag_source', '?')}")
        for axis in ["x", "y", "z"]:
            ax = corr["axes"].get(axis, {})
            r = ax.get("thrust_corr", float("nan"))
            print(f"  {axis.upper()}: thrust r={r:+.3f}, "
                  f"std={ax.get('std', 0):.4f} Ga")

    # Generate plots
    print("\nGenerating plots...")
    figures = []

    fig = plot_field_norm(mag_data, thrust_data, battery_data, phases,
                          hover_start, hover_end)
    if fig:
        figures.append(fig)

    fig = plot_detrended_overlay(mag_data, thrust_data, hover_start, hover_end)
    if fig:
        figures.append(fig)

    first_cal = cal_instance_map.get(min(mag_data["raw"].keys()), 0) \
        if mag_data["raw"] else 0

    fig = plot_scatter(mag_data, thrust_data, hover_start, hover_end,
                       cal_instance=first_cal)
    if fig:
        figures.append(fig)

    fig = plot_raw_vs_corrected(mag_data, thrust_data, hover_start, hover_end,
                                cal_instance=first_cal)
    if fig:
        figures.append(fig)

    fig = plot_multi_instance_comparison(mag_data, thrust_data,
                                         hover_start, hover_end,
                                         cal_instance_map=cal_instance_map)
    if fig:
        figures.append(fig)

    # Save PDF
    pdf_path = os.path.join(output_dir, "mag_thrust_analysis.pdf")
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  Saved: {pdf_path}")

    # Generate and save summary
    summary = generate_summary(phases, params, corr, raw_corr,
                               hover_start, hover_end,
                               cal_instance_map=cal_instance_map)
    summary_path = os.path.join(output_dir, "mag_thrust_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")

    print()
    print(summary)


if __name__ == "__main__":
    main()
