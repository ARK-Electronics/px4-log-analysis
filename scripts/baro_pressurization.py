#!/usr/bin/env python3
"""
PX4 barometric pressure bias analyzer.

Investigates thrust-induced baro pressurization, ground effect contamination,
thermal drift, and their impact on EKF height estimation.

Usage:
    python3 baro_pressurization.py <log.ulg> [--output-dir <dir>]

Outputs:
    - baro_altitude_compare.png   Baro vs range sensor vs EKF altitude
    - baro_error_thrust.png       Baro error timeseries with thrust overlay
    - baro_correlation.png        Scatter plots: error vs thrust, error vs AGL
    - baro_ekf_innovation.png     EKF baro innovation and fusion status
    - baro_raw_pressure.png       Raw barometer pressure and temperature
    - analysis.pdf                Combined PDF with all plots
    - summary.txt                 Text summary with findings
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

def extract_baro_data(ulog):
    """Extract barometer altitude and raw pressure/temperature."""
    start_us = ulog.start_timestamp
    result = {}

    vad = get_topic(ulog, "vehicle_air_data")
    if vad is not None:
        result["baro_time_s"] = timestamps_to_seconds(vad.data["timestamp"], start_us)
        result["baro_alt_m"] = vad.data["baro_alt_meter"]
        result["baro_pressure_pa"] = vad.data["baro_pressure_pa"]

    sbaro = get_topic(ulog, "sensor_baro")
    if sbaro is not None:
        result["raw_time_s"] = timestamps_to_seconds(sbaro.data["timestamp"], start_us)
        result["raw_pressure_pa"] = sbaro.data["pressure"]
        result["raw_temperature_c"] = sbaro.data["temperature"]

    return result


def extract_range_sensor(ulog):
    """Extract range sensor (distance_sensor) data."""
    start_us = ulog.start_timestamp
    dist = get_topic(ulog, "distance_sensor")
    if dist is None:
        return {}
    return {
        "time_s": timestamps_to_seconds(dist.data["timestamp"], start_us),
        "distance_m": dist.data["current_distance"],
    }


def extract_ekf_position(ulog):
    """Extract EKF local position (altitude, Vz, dist_bottom)."""
    start_us = ulog.start_timestamp
    lpos = get_topic(ulog, "vehicle_local_position")
    if lpos is None:
        return {}
    result = {
        "time_s": timestamps_to_seconds(lpos.data["timestamp"], start_us),
        "z": lpos.data["z"],
        "vz": lpos.data["vz"],
    }
    if "dist_bottom" in lpos.data:
        result["dist_bottom"] = lpos.data["dist_bottom"]
        result["dist_bottom_valid"] = lpos.data["dist_bottom_valid"]
    return result


def extract_thrust(ulog):
    """Extract thrust setpoint and hover thrust estimate."""
    start_us = ulog.start_timestamp
    result = {}

    thr = get_topic(ulog, "vehicle_thrust_setpoint")
    if thr is not None:
        result["thrust_time_s"] = timestamps_to_seconds(thr.data["timestamp"], start_us)
        result["thrust_z"] = thr.data["xyz[2]"]

    # Also try actuator_motors for finer-grained thrust info
    motors = get_topic(ulog, "actuator_motors")
    if motors is not None:
        ts = timestamps_to_seconds(motors.data["timestamp"], start_us)
        controls = []
        for i in range(12):
            key = f"control[{i}]"
            if key in motors.data:
                c = motors.data[key]
                if np.any(c > 0):
                    controls.append(c)
        if controls:
            result["motor_time_s"] = ts
            result["motor_mean_control"] = np.mean(controls, axis=0)

    hover = get_topic(ulog, "hover_thrust_estimate")
    if hover is not None:
        result["hover_time_s"] = timestamps_to_seconds(hover.data["timestamp"], start_us)
        result["hover_thrust"] = hover.data["hover_thrust"]

    return result


def extract_baro_innovation(ulog):
    """Extract EKF baro height innovation data."""
    start_us = ulog.start_timestamp
    baro_aid = get_topic(ulog, "estimator_aid_src_baro_hgt")
    if baro_aid is None:
        return {}
    return {
        "time_s": timestamps_to_seconds(baro_aid.data["timestamp"], start_us),
        "innovation": baro_aid.data["innovation"],
        "innovation_variance": baro_aid.data["innovation_variance"],
        "test_ratio": baro_aid.data["test_ratio"],
        "fused": baro_aid.data["fused"],
        "innovation_rejected": baro_aid.data["innovation_rejected"],
    }


def extract_gnd_effect_status(ulog):
    """Check if EKF ground effect flag is active."""
    start_us = ulog.start_timestamp
    esf = get_topic(ulog, "estimator_status_flags")
    if esf is None or "cs_gnd_effect" not in esf.data:
        return {}
    return {
        "time_s": timestamps_to_seconds(esf.data["timestamp"], start_us),
        "cs_gnd_effect": esf.data["cs_gnd_effect"],
        "cs_baro_hgt": esf.data.get("cs_baro_hgt"),
        "cs_rng_hgt": esf.data.get("cs_rng_hgt"),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_baro_error(baro_data, range_data, phases):
    """Compute baro altitude error relative to range sensor.

    Returns dict with time arrays and error components.
    """
    if not baro_data or not range_data:
        return {}

    baro_t = baro_data["baro_time_s"]
    baro_alt = baro_data["baro_alt_m"]
    ds_t = range_data["time_s"]
    ds_dist = range_data["distance_m"]

    # Offset baro to zero at arm time
    arm_baro = np.interp(phases["armed_start_s"], baro_t, baro_alt)
    baro_alt_off = baro_alt - arm_baro

    # Interpolate baro onto range sensor timestamps
    baro_interp = np.interp(ds_t, baro_t, baro_alt_off)

    # Error = baro - range sensor (positive means baro reads higher than truth)
    error = baro_interp - ds_dist

    return {
        "time_s": ds_t,
        "baro_alt_offset": baro_interp,
        "range_alt": ds_dist,
        "error": error,
        "baro_full_time_s": baro_t,
        "baro_full_alt_offset": baro_alt_off,
    }


def compute_correlations(baro_error, thrust_data, hover_start, hover_end):
    """Compute thrust-baro and altitude-baro correlations during hover."""
    if not baro_error or not thrust_data:
        return {}

    t = baro_error["time_s"]
    err = baro_error["error"]
    alt = baro_error["range_alt"]

    hov = (t >= hover_start) & (t <= hover_end)
    if hov.sum() < 10:
        return {}

    t_hov = t[hov]
    err_hov = err[hov]
    alt_hov = alt[hov]

    result = {
        "hover_error_mean": float(np.mean(err_hov)),
        "hover_error_std": float(np.std(err_hov)),
        "hover_alt_mean": float(np.mean(alt_hov)),
        "hover_alt_std": float(np.std(alt_hov)),
    }

    # Thrust correlation
    if "thrust_time_s" in thrust_data:
        thrust_interp = np.interp(t_hov, thrust_data["thrust_time_s"],
                                   thrust_data["thrust_z"])
        result["corr_thrust"] = float(np.corrcoef(thrust_interp, err_hov)[0, 1])
        result["thrust_mean"] = float(np.mean(thrust_interp))
        result["thrust_std"] = float(np.std(thrust_interp))
        result["thrust_interp_hover"] = thrust_interp

    # Motor control correlation (if available, higher rate)
    if "motor_time_s" in thrust_data:
        motor_interp = np.interp(t_hov, thrust_data["motor_time_s"],
                                  thrust_data["motor_mean_control"])
        result["corr_motor"] = float(np.corrcoef(motor_interp, err_hov)[0, 1])
        result["motor_interp_hover"] = motor_interp

    # Altitude correlation (ground effect)
    result["corr_altitude"] = float(np.corrcoef(alt_hov, err_hov)[0, 1])

    # Time correlation (drift)
    result["corr_time"] = float(np.corrcoef(t_hov, err_hov)[0, 1])

    # Multivariate regression: error = a*AGL + b*thrust + c
    if "thrust_interp_hover" in result:
        A = np.column_stack([alt_hov, result["thrust_interp_hover"],
                              np.ones(len(alt_hov))])
        coeffs, _, _, _ = np.linalg.lstsq(A, err_hov, rcond=None)
        pred = A @ coeffs
        r2 = 1 - np.var(err_hov - pred) / np.var(err_hov)
        result["multivar_coeff_alt"] = float(coeffs[0])
        result["multivar_coeff_thrust"] = float(coeffs[1])
        result["multivar_intercept"] = float(coeffs[2])
        result["multivar_r2"] = float(r2)

    # Altitude-binned error
    bins = []
    max_alt = np.max(alt_hov)
    for lo in np.arange(0, max_alt + 0.2, 0.2):
        hi = lo + 0.2
        mask = (alt_hov >= lo) & (alt_hov < hi)
        if mask.sum() > 5:
            bins.append({
                "lo": float(lo), "hi": float(hi),
                "mean_error": float(np.mean(err_hov[mask])),
                "std_error": float(np.std(err_hov[mask])),
                "n": int(mask.sum()),
            })
    result["alt_bins"] = bins

    # Store hover slices for plotting
    result["hover_time"] = t_hov
    result["hover_error"] = err_hov
    result["hover_alt"] = alt_hov

    return result


def compute_pressure_trends(baro_data, hover_start, hover_end):
    """Analyze raw pressure and temperature trends."""
    if "raw_time_s" not in baro_data:
        return {}

    t = baro_data["raw_time_s"]
    pres = baro_data["raw_pressure_pa"]
    temp = baro_data["raw_temperature_c"]

    hov = (t >= hover_start) & (t <= hover_end)
    if hov.sum() < 5:
        return {}

    t_hov = t[hov]
    pres_hov = pres[hov]
    temp_hov = temp[hov]

    # Linear fits
    p_slope, p_intercept = np.polyfit(t_hov, pres_hov, 1)
    t_slope, t_intercept = np.polyfit(t_hov, temp_hov, 1)

    # Pressure-temperature correlation
    corr_pt = float(np.corrcoef(pres_hov, temp_hov)[0, 1])

    return {
        "pressure_slope_pa_s": float(p_slope),
        "pressure_range_pa": float(pres_hov.max() - pres_hov.min()),
        "pressure_range_m": float((pres_hov.max() - pres_hov.min()) * 0.083),
        "temp_slope_c_s": float(t_slope),
        "temp_range_c": float(temp_hov.max() - temp_hov.min()),
        "corr_pressure_temp": corr_pt,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_altitude_compare(baro_error, ekf_data, baro_data, phases,
                          save_path):
    """Plot altitude sources comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Baro Altitude vs Range Sensor vs EKF", fontsize=13,
                 fontweight="bold")

    ax = axes[0]
    t = baro_error["time_s"]
    ax.plot(t, baro_error["range_alt"], label="Range sensor (ground truth)",
            color="tab:green", linewidth=1.2)
    ax.plot(baro_error["baro_full_time_s"], baro_error["baro_full_alt_offset"],
            label="Baro alt (offset to 0 at arm)", color="tab:red",
            linewidth=1.2, alpha=0.85)
    if ekf_data and "z" in ekf_data:
        ekf_alt = -ekf_data["z"]
        ekf_alt_off = ekf_alt - ekf_alt[0]
        ax.plot(ekf_data["time_s"], ekf_alt_off, label="EKF fused alt",
                color="tab:blue", linewidth=1.0, alpha=0.8)
    ax.axvspan(phases["armed_start_s"], phases["armed_end_s"],
               alpha=0.05, color="green", label="Armed")
    ax.set_ylabel("Altitude [m]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Error panel
    ax = axes[1]
    ax.plot(t, baro_error["error"], color="tab:red", linewidth=0.8,
            label="Baro error (baro - range)")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Baro Error [m]")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    return fig


def plot_error_with_thrust(baro_error, thrust_data, corr, phases,
                           hover_start, hover_end, save_path):
    """Plot baro error timeseries with thrust overlay."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Baro Error & Thrust Pressurization", fontsize=13,
                 fontweight="bold")

    t = baro_error["time_s"]
    err = baro_error["error"]

    # Panel 1: baro error
    ax = axes[0]
    ax.plot(t, err, color="tab:red", linewidth=0.8, label="Baro error")
    ax.axhline(corr.get("hover_error_mean", 0), color="gray",
               linestyle="--", linewidth=0.7, alpha=0.7,
               label=f'Hover mean: {corr.get("hover_error_mean", 0):.2f} m')
    ax.axvspan(hover_start, hover_end, alpha=0.08, color="blue",
               label="Hover segment")
    ax.set_ylabel("Baro Error [m]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: thrust (negate so positive = upward, matching baro error sign)
    ax = axes[1]
    if "thrust_time_s" in thrust_data:
        ax.plot(thrust_data["thrust_time_s"], -thrust_data["thrust_z"],
                color="tab:orange", linewidth=0.8, label="Upward thrust")
    ax.axvspan(hover_start, hover_end, alpha=0.08, color="blue")
    ax.set_ylabel("Thrust (up)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: detrended error vs detrended thrust (hover only)
    # Use thrust_z directly — it's negative in NED, so negate once to get
    # "upward thrust" which has the same sign as baro error.
    ax = axes[2]
    if "hover_error" in corr and "thrust_interp_hover" in corr:
        err_dt = corr["hover_error"] - np.mean(corr["hover_error"])
        # Negate thrust_z so positive = more upward thrust = more baro error
        thr_up = -corr["thrust_interp_hover"]
        thr_dt = thr_up - np.mean(thr_up)
        thr_scale = np.std(err_dt) / (np.std(thr_dt) + 1e-10)
        r_val = corr.get("corr_thrust", 0)
        ax.plot(corr["hover_time"], err_dt, color="tab:red", linewidth=0.8,
                label="Baro error (detrended)")
        ax.plot(corr["hover_time"], thr_dt * thr_scale,
                color="tab:orange", linewidth=0.8, alpha=0.7,
                label=f"Upward thrust (scaled), |r|={abs(r_val):.2f}")
        ax.set_ylabel("Detrended [m]")
        ax.legend(fontsize=9)
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    return fig


def plot_correlations(corr, save_path):
    """Scatter plots: baro error vs thrust and vs AGL."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Baro Error Correlations (Hover)", fontsize=13,
                 fontweight="bold")

    # Error vs thrust
    ax = axes[0]
    if "thrust_interp_hover" in corr:
        ax.scatter(corr["thrust_interp_hover"], corr["hover_error"],
                   s=3, alpha=0.4, color="tab:orange")
        # Fit line
        z = np.polyfit(corr["thrust_interp_hover"], corr["hover_error"], 1)
        x_fit = np.linspace(corr["thrust_interp_hover"].min(),
                             corr["thrust_interp_hover"].max(), 50)
        ax.plot(x_fit, np.polyval(z, x_fit), "k--", linewidth=1.2)
        ax.set_xlabel("Thrust Z setpoint")
        ax.set_ylabel("Baro Error [m]")
        ax.set_title(f"Thrust Pressurization\nr = {corr.get('corr_thrust', 0):.3f}")
    ax.grid(True, alpha=0.3)

    # Error vs AGL
    ax = axes[1]
    ax.scatter(corr["hover_alt"], corr["hover_error"], s=3, alpha=0.4,
               color="tab:green")
    z = np.polyfit(corr["hover_alt"], corr["hover_error"], 1)
    x_fit = np.linspace(corr["hover_alt"].min(), corr["hover_alt"].max(), 50)
    ax.plot(x_fit, np.polyval(z, x_fit), "k--", linewidth=1.2)
    ax.set_xlabel("AGL [m]")
    ax.set_ylabel("Baro Error [m]")
    ax.set_title(f"Ground Effect\nr = {corr.get('corr_altitude', 0):.3f}")
    ax.grid(True, alpha=0.3)

    # Altitude-binned bar chart
    ax = axes[2]
    if corr.get("alt_bins"):
        bins = corr["alt_bins"]
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
        means = [b["mean_error"] for b in bins]
        stds = [b["std_error"] for b in bins]
        labels = [f'{b["lo"]:.1f}-{b["hi"]:.1f}' for b in bins]
        ax.bar(range(len(bins)), means, yerr=stds, width=0.7,
               color="tab:green", alpha=0.7, capsize=3)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("AGL bin [m]")
        ax.set_ylabel("Mean Baro Error [m]")
        ax.set_title("Altitude-Binned Error")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    return fig


def plot_ekf_innovation(innov_data, gnd_effect, phases, hover_start,
                         hover_end, save_path):
    """Plot EKF baro innovation and ground effect status."""
    n_panels = 3 if gnd_effect else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels),
                              sharex=True)
    fig.suptitle("EKF Baro Height Innovation", fontsize=13, fontweight="bold")

    if innov_data:
        t = innov_data["time_s"]
        innov = innov_data["innovation"]
        fused = innov_data["fused"]
        test_ratio = innov_data["test_ratio"]

        ax = axes[0]
        ax.plot(t, innov, color="tab:blue", linewidth=0.8,
                label="Baro innovation")
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.axvspan(hover_start, hover_end, alpha=0.08, color="blue")
        ax.set_ylabel("Innovation [m]")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t, test_ratio, color="tab:orange", linewidth=0.8,
                label="Test ratio")
        ax.axhline(1.0, color="red", linewidth=0.7, linestyle="--",
                    label="Rejection threshold")
        # Mark rejected points
        rejected = innov_data.get("innovation_rejected")
        if rejected is not None:
            rej_mask = rejected.astype(bool)
            if rej_mask.any():
                ax.scatter(t[rej_mask], test_ratio[rej_mask], s=10,
                           color="red", zorder=5, label="Rejected")
        # Mark fusion status
        fused_mask = fused.astype(bool)
        fuse_pct = np.mean(fused_mask) * 100
        ax.set_title(f"Baro fused: {fuse_pct:.0f}% of samples", fontsize=10)
        ax.axvspan(hover_start, hover_end, alpha=0.08, color="blue")
        ax.set_ylabel("Test Ratio")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if gnd_effect:
        ax = axes[n_panels - 1]
        t = gnd_effect["time_s"]
        gnd = gnd_effect["cs_gnd_effect"].astype(float)
        ax.fill_between(t, 0, gnd, color="tab:red", alpha=0.5,
                         label="cs_gnd_effect")
        if gnd_effect.get("cs_baro_hgt") is not None:
            baro_fuse = gnd_effect["cs_baro_hgt"].astype(float)
            ax.plot(t, baro_fuse * 0.8, color="tab:blue", linewidth=1.0,
                    label="cs_baro_hgt", alpha=0.8)
        if gnd_effect.get("cs_rng_hgt") is not None:
            rng_fuse = gnd_effect["cs_rng_hgt"].astype(float)
            ax.plot(t, rng_fuse * 0.6, color="tab:green", linewidth=1.0,
                    label="cs_rng_hgt", alpha=0.8)
        ax.axvspan(hover_start, hover_end, alpha=0.08, color="blue")
        ax.set_ylabel("Flag")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=9)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    return fig


def plot_raw_pressure(baro_data, phases, hover_start, hover_end, save_path):
    """Plot raw barometer pressure and temperature."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Raw Barometer Pressure & Temperature", fontsize=13,
                 fontweight="bold")

    if "raw_time_s" in baro_data:
        t = baro_data["raw_time_s"]
        pres = baro_data["raw_pressure_pa"]
        temp = baro_data["raw_temperature_c"]

        ax = axes[0]
        ax.plot(t, pres, color="tab:orange", linewidth=0.8)
        ax.axvspan(hover_start, hover_end, alpha=0.08, color="blue",
                    label="Hover")
        pres_range = pres.max() - pres.min()
        ax.set_ylabel("Pressure [Pa]")
        ax.set_title(f"Pressure (span: {pres_range:.1f} Pa = "
                      f"~{pres_range * 0.083:.2f} m equivalent)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t, temp, color="tab:purple", linewidth=0.8)
        ax.axvspan(hover_start, hover_end, alpha=0.08, color="blue")
        ax.set_ylabel("Temperature [C]")
        ax.set_xlabel("Time [s]")
        temp_range = temp.max() - temp.min()
        ax.set_title(f"Temperature (range: {temp_range:.2f} C)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def generate_summary(phases, params, baro_error, corr, press_trends,
                     innov_data, gnd_effect, hover_start, hover_end):
    """Generate text summary of findings."""
    lines = []
    lines.append("=" * 70)
    lines.append("BAROMETRIC PRESSURE BIAS ANALYSIS SUMMARY")
    lines.append("=" * 70)

    dur = phases["armed_end_s"] - phases["armed_start_s"]
    lines.append(f"\nFlight duration:  {dur:.1f} s")
    lines.append(f"Hover window:     {hover_start:.1f}s - {hover_end:.1f}s")

    # Parameters
    lines.append("\nHEIGHT / BARO PARAMETERS")
    hgt_ref_map = {0: "Barometer", 1: "GNSS", 2: "Range sensor", 3: "Vision"}
    hgt_ref = int(params.get("EKF2_HGT_REF", -1))
    lines.append(f"  EKF2_HGT_REF            = {hgt_ref} "
                 f"({hgt_ref_map.get(hgt_ref, 'Unknown')})")
    lines.append(f"  EKF2_BARO_CTRL          = {params.get('EKF2_BARO_CTRL', 'N/A')}")
    lines.append(f"  EKF2_BARO_NOISE         = {params.get('EKF2_BARO_NOISE', 'N/A')}")
    lines.append(f"  EKF2_BARO_GATE          = {params.get('EKF2_BARO_GATE', 'N/A')}")
    lines.append(f"  EKF2_GND_EFF_DZ         = {params.get('EKF2_GND_EFF_DZ', 'N/A')}")
    lines.append(f"  EKF2_GND_MAX_HGT        = {params.get('EKF2_GND_MAX_HGT', 'N/A')}")

    # Baro error stats
    if corr:
        lines.append("\nBARO ERROR vs RANGE SENSOR (hover)")
        lines.append(f"  Static offset (mean):     {corr['hover_error_mean']:+.3f} m")
        lines.append(f"  Variation (std):           {corr['hover_error_std']:.3f} m")
        lines.append(f"  Hover AGL:                 {corr['hover_alt_mean']:.2f} m "
                      f"(std {corr['hover_alt_std']:.2f} m)")

        if abs(corr["hover_error_mean"]) > 1.0:
            lines.append(f"  Assessment: LARGE static offset "
                         f"({abs(corr['hover_error_mean']):.1f} m) — "
                         "thrust-induced pressure depression")

    # Correlations
    if corr:
        lines.append("\nCORRELATION ANALYSIS (hover)")
        ct = corr.get("corr_thrust", 0)
        ca = corr.get("corr_altitude", 0)
        cd = corr.get("corr_time", 0)
        lines.append(f"  Corr(thrust, baro_error):  {ct:+.3f}"
                     f"  {'STRONG' if abs(ct) > 0.7 else 'MODERATE' if abs(ct) > 0.4 else 'WEAK'}")
        lines.append(f"  Corr(AGL, baro_error):     {ca:+.3f}"
                     f"  {'STRONG' if abs(ca) > 0.7 else 'MODERATE' if abs(ca) > 0.4 else 'WEAK'}")
        lines.append(f"  Corr(time, baro_error):    {cd:+.3f}"
                     f"  {'STRONG' if abs(cd) > 0.7 else 'MODERATE' if abs(cd) > 0.4 else 'WEAK'}")

        if "multivar_r2" in corr:
            lines.append(f"\n  Multivariate: error = "
                          f"{corr['multivar_coeff_alt']:.3f}*AGL + "
                          f"{corr['multivar_coeff_thrust']:.3f}*thrust + "
                          f"{corr['multivar_intercept']:.3f}")
            lines.append(f"  R² = {corr['multivar_r2']:.3f}")

        lines.append("\n  Dominant bias source:")
        if abs(ct) > abs(ca) and abs(ct) > 0.5:
            lines.append("  -> THRUST PRESSURIZATION (baro error tracks thrust changes)")
        elif abs(ca) > abs(ct) and abs(ca) > 0.5:
            lines.append("  -> GROUND EFFECT (baro error correlates with AGL)")
        elif abs(ct) > 0.5 and abs(ca) > 0.5:
            lines.append("  -> BOTH thrust pressurization AND ground effect")
        elif abs(cd) > 0.7:
            lines.append("  -> THERMAL DRIFT (monotonic error growth over time)")
        else:
            lines.append("  -> No strong single source identified")

    # Altitude bins
    if corr and corr.get("alt_bins"):
        lines.append("\nALTITUDE-BINNED BARO ERROR")
        for b in corr["alt_bins"]:
            lines.append(f"  {b['lo']:.1f}-{b['hi']:.1f} m AGL: "
                          f"{b['mean_error']:+.3f} m  (n={b['n']})")

    # Pressure trends
    if press_trends:
        lines.append("\nRAW PRESSURE / TEMPERATURE TRENDS (hover)")
        lines.append(f"  Pressure drift:            "
                      f"{press_trends['pressure_slope_pa_s']:+.3f} Pa/s "
                      f"({press_trends['pressure_slope_pa_s'] * 60:+.1f} Pa/min)")
        lines.append(f"  Pressure span:             "
                      f"{press_trends['pressure_range_pa']:.1f} Pa "
                      f"(~{press_trends['pressure_range_m']:.2f} m altitude)")
        lines.append(f"  Temperature drift:         "
                      f"{press_trends['temp_slope_c_s']:+.4f} C/s "
                      f"({press_trends['temp_slope_c_s'] * 60:+.2f} C/min)")
        lines.append(f"  Temp range:                "
                      f"{press_trends['temp_range_c']:.2f} C")
        lines.append(f"  Pressure-temp correlation: "
                      f"{press_trends['corr_pressure_temp']:+.3f}")

    # EKF innovation
    if innov_data:
        t = innov_data["time_s"]
        hov = (t >= hover_start) & (t <= hover_end)
        if hov.any():
            innov_hov = innov_data["innovation"][hov]
            fused_hov = innov_data["fused"][hov].astype(bool)
            rejected_hov = innov_data.get("innovation_rejected",
                                           np.zeros_like(t))[hov].astype(bool)
            lines.append("\nEKF BARO INNOVATION (hover)")
            lines.append(f"  Innovation mean:           {np.mean(innov_hov):+.3f} m")
            lines.append(f"  Innovation std:            {np.std(innov_hov):.3f} m")
            lines.append(f"  Fused:                     {np.mean(fused_hov)*100:.0f}%")
            lines.append(f"  Rejected:                  {np.sum(rejected_hov)} samples")

    # Ground effect status
    if gnd_effect:
        t = gnd_effect["time_s"]
        gnd = gnd_effect["cs_gnd_effect"].astype(bool)
        lines.append(f"\nGROUND EFFECT STATUS")
        lines.append(f"  cs_gnd_effect active:      "
                      f"{'YES' if np.any(gnd) else 'NEVER'} "
                      f"({np.mean(gnd)*100:.1f}% of flight)")
        gnd_max_hgt = params.get("EKF2_GND_MAX_HGT", 0)
        if corr and corr["hover_alt_mean"] < gnd_max_hgt and not np.any(gnd):
            lines.append(f"  WARNING: Hovering at {corr['hover_alt_mean']:.2f} m AGL "
                         f"(below GND_MAX_HGT={gnd_max_hgt}) "
                         f"but ground effect flag is NEVER active!")

    # Assessment
    lines.append("\n" + "=" * 70)
    lines.append("ASSESSMENT")
    lines.append("=" * 70)

    issues = []
    if corr:
        if abs(corr["hover_error_mean"]) > 2.0:
            issues.append(f"SEVERE baro pressurization: {abs(corr['hover_error_mean']):.1f} m "
                          "static offset from prop wash")
        elif abs(corr["hover_error_mean"]) > 0.5:
            issues.append(f"MODERATE baro pressurization: {abs(corr['hover_error_mean']):.1f} m "
                          "static offset")

        if corr.get("corr_thrust") and abs(corr["corr_thrust"]) > 0.6:
            issues.append(f"Thrust-dependent baro bias (r={corr['corr_thrust']:.2f}): "
                          "throttle changes directly modulate baro reading")

        if abs(corr.get("corr_altitude", 0)) > 0.5:
            issues.append(f"Ground effect contamination (r={corr['corr_altitude']:.2f}): "
                          "baro error varies with altitude AGL")

        if corr["hover_error_std"] > 0.3:
            issues.append(f"High baro error variation ({corr['hover_error_std']:.2f} m std): "
                          "will inject noise into EKF height estimate")

    if gnd_effect:
        gnd = gnd_effect["cs_gnd_effect"].astype(bool)
        if corr and corr["hover_alt_mean"] < params.get("EKF2_GND_MAX_HGT", 0):
            if not np.any(gnd):
                issues.append("Ground effect protection NOT active despite low hover altitude")

    if issues:
        for i, issue in enumerate(issues, 1):
            lines.append(f"  {i}. {issue}")
    else:
        lines.append("  No significant baro pressurization issues detected.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PX4 barometric pressure bias analyzer")
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
    param_names = [
        "EKF2_HGT_REF", "EKF2_BARO_CTRL", "EKF2_BARO_NOISE",
        "EKF2_BARO_GATE", "EKF2_BARO_DELAY",
        "EKF2_GND_EFF_DZ", "EKF2_GND_MAX_HGT",
        "SENS_BARO_QNH", "SENS_BARO_RATE",
    ]
    params = {}
    for p in param_names:
        val = get_param(ulog, p)
        if val is not None:
            params[p] = val

    hgt_ref_map = {0: "Barometer", 1: "GNSS", 2: "Range sensor", 3: "Vision"}
    hgt_ref = int(params.get("EKF2_HGT_REF", -1))
    print(f"\nHeight ref: {hgt_ref_map.get(hgt_ref, 'Unknown')} "
          f"(EKF2_HGT_REF={hgt_ref})")
    print(f"Baro ctrl: {params.get('EKF2_BARO_CTRL', 'N/A')}, "
          f"noise: {params.get('EKF2_BARO_NOISE', 'N/A')} m, "
          f"gate: {params.get('EKF2_BARO_GATE', 'N/A')} sigma")
    print(f"Ground effect: deadzone={params.get('EKF2_GND_EFF_DZ', 'N/A')} m, "
          f"max_hgt={params.get('EKF2_GND_MAX_HGT', 'N/A')} m")

    # Detect flight phases
    phases = detect_flight_phases(ulog)
    print(f"\nArmed: {phases['armed_start_s']:.1f}s - {phases['armed_end_s']:.1f}s")
    hover_start, hover_end = detect_hover_segment(phases)
    print(f"Hover: {hover_start:.1f}s - {hover_end:.1f}s")

    # Extract data
    print("\nExtracting data...")
    baro_data = extract_baro_data(ulog)
    range_data = extract_range_sensor(ulog)
    ekf_data = extract_ekf_position(ulog)
    thrust_data = extract_thrust(ulog)
    innov_data = extract_baro_innovation(ulog)
    gnd_effect = extract_gnd_effect_status(ulog)

    has_range = bool(range_data)
    has_baro = "baro_time_s" in baro_data
    print(f"  Baro data: {'yes' if has_baro else 'NO'}")
    print(f"  Range sensor: {'yes' if has_range else 'NO'}")
    print(f"  Thrust data: {'yes' if thrust_data else 'NO'}")
    print(f"  Baro innovation: {'yes' if innov_data else 'NO'}")
    print(f"  Ground effect flags: {'yes' if gnd_effect else 'NO'}")

    if not has_baro:
        print("Error: no barometer data found", file=sys.stderr)
        sys.exit(1)

    # Compute baro error
    baro_error = {}
    corr = {}
    if has_range:
        print("\nComputing baro error...")
        baro_error = compute_baro_error(baro_data, range_data, phases)
        corr = compute_correlations(baro_error, thrust_data,
                                     hover_start, hover_end)
        if corr:
            print(f"  Static offset: {corr['hover_error_mean']:+.2f} m")
            print(f"  Variation: {corr['hover_error_std']:.3f} m")
            if "corr_thrust" in corr:
                print(f"  Thrust correlation: {corr['corr_thrust']:+.3f}")
            print(f"  AGL correlation: {corr['corr_altitude']:+.3f}")
    else:
        print("\nWARNING: No range sensor — cannot compute baro error ground truth")

    # Pressure trends
    press_trends = compute_pressure_trends(baro_data, hover_start, hover_end)

    # Generate plots
    print("\nGenerating plots...")
    figures = []

    if baro_error:
        fig1 = plot_altitude_compare(
            baro_error, ekf_data, baro_data, phases,
            os.path.join(output_dir, "baro_altitude_compare.png"))
        figures.append(fig1)

        fig2 = plot_error_with_thrust(
            baro_error, thrust_data, corr, phases, hover_start, hover_end,
            os.path.join(output_dir, "baro_error_thrust.png"))
        figures.append(fig2)

    if corr and "hover_error" in corr:
        fig3 = plot_correlations(
            corr, os.path.join(output_dir, "baro_correlation.png"))
        figures.append(fig3)

    if innov_data or gnd_effect:
        fig4 = plot_ekf_innovation(
            innov_data, gnd_effect, phases, hover_start, hover_end,
            os.path.join(output_dir, "baro_ekf_innovation.png"))
        figures.append(fig4)

    if "raw_time_s" in baro_data:
        fig5 = plot_raw_pressure(
            baro_data, phases, hover_start, hover_end,
            os.path.join(output_dir, "baro_raw_pressure.png"))
        figures.append(fig5)

    # Combined PDF
    pdf_path = os.path.join(output_dir, "baro_analysis.pdf")
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  Saved: {pdf_path}")

    # Generate and save text summary
    summary = generate_summary(phases, params, baro_error, corr,
                                press_trends, innov_data, gnd_effect,
                                hover_start, hover_end)
    summary_path = os.path.join(output_dir, "baro_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")

    print()
    print(summary)


if __name__ == "__main__":
    main()
