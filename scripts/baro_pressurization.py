#!/usr/bin/env python3
"""
PX4 barometric pressure bias analyzer.

Investigates thrust-induced baro pressurization, ground effect contamination,
thermal drift, and their impact on EKF height estimation.

Usage:
    python3 baro_pressurization.py <log.ulg> [--output-dir <dir>] [--calibrate]

Options:
    --calibrate   Run system identification to find optimal EKF2_PCOEF_THR
                  and EKF2_PCOEF_TTAU values. Requires range sensor data.

Outputs:
    - baro_analysis.pdf           Combined PDF with all plots
    - baro_summary.txt            Text summary with findings
    When --calibrate: recommended EKF2_PCOEF_THR / EKF2_PCOEF_TTAU values
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


def extract_vertical_velocity(ekf_data, range_data, gnd_effect):
    """Extract vertical velocity — EKF Vz if range is fused, else range derivative.

    Returns dict with time_s and vz arrays, plus the source label.
    """
    # Check if range sensor is being fused via cs_rng_hgt flag
    rng_fused = False
    if gnd_effect and gnd_effect.get("cs_rng_hgt") is not None:
        rng_fused = bool(np.any(gnd_effect["cs_rng_hgt"].astype(bool)))

    if rng_fused and ekf_data and "vz" in ekf_data:
        # Range is fused, so EKF Vz already incorporates it — use directly
        return {
            "time_s": ekf_data["time_s"],
            "vz": -ekf_data["vz"],  # negate: NED down-positive -> up-positive
            "source": "EKF Vz (range fused)",
        }

    if range_data and "distance_m" in range_data:
        # Range not fused — compute derivative of raw distance sensor
        ts = range_data["time_s"]
        dist = range_data["distance_m"]
        dt = np.diff(ts)
        dz = np.diff(dist)
        valid = dt > 0.001
        vz = np.zeros_like(dt)
        vz[valid] = dz[valid] / dt[valid]  # positive = climbing (distance increasing)
        # Smooth with 5-point moving average
        if len(vz) > 5:
            kernel = np.ones(5) / 5
            vz = np.convolve(vz, kernel, mode="same")
        return {
            "time_s": (ts[:-1] + ts[1:]) / 2,  # midpoints
            "vz": vz,
            "source": "Range sensor derivative",
        }

    # Fallback: use EKF Vz even without range fusion
    if ekf_data and "vz" in ekf_data:
        return {
            "time_s": ekf_data["time_s"],
            "vz": -ekf_data["vz"],
            "source": "EKF Vz (no range)",
        }

    return {}


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


def first_order_lpf(signal, time_s, tau):
    """Apply a causal first-order low-pass filter with time constant tau.

    Matches the AlphaFilter used in the EKF: alpha = dt / (dt + tau).
    """
    if tau <= 0 or len(signal) < 2:
        return signal.copy()
    out = np.empty_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        dt = time_s[i] - time_s[i - 1]
        if dt <= 0:
            out[i] = out[i - 1]
            continue
        alpha = dt / (dt + tau)
        out[i] = out[i - 1] + alpha * (signal[i] - out[i - 1])
    return out


def calibrate_thrust_compensation(baro_error, thrust_data, hover_start,
                                   hover_end):
    """System identification for thrust-based baro compensation.

    Fits a first-order lag model:  baro_error ≈ K * LPF(thrust, tau) + c

    Returns dict with identified K (EKF2_PCOEF_THR), tau (EKF2_PCOEF_TTAU),
    cross-correlation data, and sweep results for plotting.
    """
    if not baro_error or "thrust_time_s" not in thrust_data:
        return {}

    err_t = baro_error["time_s"]
    err = baro_error["error"]

    # Focus on hover segment
    hov = (err_t >= hover_start) & (err_t <= hover_end)
    if hov.sum() < 20:
        return {}

    err_t_hov = err_t[hov]
    err_hov = err[hov]

    # Interpolate thrust onto baro error timestamps
    thrust_raw = np.interp(err_t_hov, thrust_data["thrust_time_s"],
                            thrust_data["thrust_z"])
    # Convert to magnitude: thrust_z is negative-down, so negate and clamp
    thrust_mag = np.clip(-thrust_raw, 0, 1)

    # Detrend error to remove slow drift (thermal, bias drift)
    err_detrended = err_hov - np.polyval(np.polyfit(err_t_hov, err_hov, 1),
                                          err_t_hov)
    thrust_detrended = thrust_mag - np.mean(thrust_mag)

    result = {}

    # --- Cross-correlation to visualize delay structure ---
    dt_median = np.median(np.diff(err_t_hov))
    if dt_median > 0:
        max_lag_samples = min(int(2.0 / dt_median), len(err_detrended) // 2)
        if max_lag_samples > 5:
            xcorr = np.correlate(err_detrended, thrust_detrended, "full")
            mid = len(thrust_detrended) - 1
            lags = (np.arange(len(xcorr)) - mid) * dt_median
            # Normalise
            norm = np.sqrt(np.sum(err_detrended**2) *
                           np.sum(thrust_detrended**2))
            if norm > 0:
                xcorr = xcorr / norm
            # Keep +/- 2 seconds around zero
            valid = (lags >= -2.0) & (lags <= 2.0)
            result["xcorr_lags_s"] = lags[valid]
            result["xcorr_values"] = xcorr[valid]

            # Peak lag (positive means error lags behind thrust)
            peak_idx = np.argmax(np.abs(xcorr[valid]))
            result["xcorr_peak_lag_s"] = float(lags[valid][peak_idx])
            result["xcorr_peak_value"] = float(xcorr[valid][peak_idx])

    # --- Grid search over time constants ---
    tau_candidates = np.concatenate([
        [0.0],
        np.arange(0.02, 0.2, 0.02),
        np.arange(0.2, 1.01, 0.05),
    ])

    sweep_tau = []
    sweep_r2 = []
    sweep_K = []
    sweep_rmse = []

    err_var = np.var(err_detrended)
    if err_var < 1e-10:
        return {}

    for tau in tau_candidates:
        thrust_filt = first_order_lpf(thrust_mag, err_t_hov, tau)
        thrust_filt_dt = thrust_filt - np.mean(thrust_filt)

        # Least-squares fit: err_detrended = K * thrust_filt_dt
        var_thr = np.var(thrust_filt_dt)
        if var_thr < 1e-10:
            continue

        K = np.sum(err_detrended * thrust_filt_dt) / np.sum(thrust_filt_dt**2)
        residual = err_detrended - K * thrust_filt_dt
        r2 = 1.0 - np.var(residual) / err_var
        rmse = float(np.sqrt(np.mean(residual**2)))

        sweep_tau.append(float(tau))
        sweep_r2.append(float(r2))
        sweep_K.append(float(K))
        sweep_rmse.append(rmse)

    if not sweep_tau:
        return {}

    result["sweep_tau"] = np.array(sweep_tau)
    result["sweep_r2"] = np.array(sweep_r2)
    result["sweep_K"] = np.array(sweep_K)
    result["sweep_rmse"] = np.array(sweep_rmse)

    # Best fit: highest R²
    best_idx = int(np.argmax(sweep_r2))
    result["best_tau"] = sweep_tau[best_idx]
    result["best_K"] = sweep_K[best_idx]
    result["best_r2"] = sweep_r2[best_idx]
    result["best_rmse"] = sweep_rmse[best_idx]

    # Compute the compensated timeseries for the best model (for plotting)
    best_thrust_filt = first_order_lpf(thrust_mag, err_t_hov, result["best_tau"])
    result["best_compensated_error"] = (
        err_hov - result["best_K"] * best_thrust_filt)
    result["best_thrust_filtered"] = best_thrust_filt

    # Also store the unfiltered (tau=0) result for comparison
    K_nolag = sweep_K[0] if sweep_tau[0] == 0.0 else 0.0
    r2_nolag = sweep_r2[0] if sweep_tau[0] == 0.0 else 0.0
    result["nolag_K"] = K_nolag
    result["nolag_r2"] = r2_nolag

    # Store hover slices for plotting
    result["hover_time"] = err_t_hov
    result["hover_error"] = err_hov
    result["thrust_mag"] = thrust_mag

    # Recommended parameters (sign flip: EKF2_PCOEF_THR corrects, so negate K)
    # K here is the measured relationship: error = K * thrust, so to *remove*
    # that error, the EKF should apply correction = -K * thrust.
    # But the EKF adds: baro_alt += pcoef_thr * thrust
    # If K is negative (thrust makes baro read lower), pcoef_thr should be positive.
    result["recommended_pcoef_thr"] = -result["best_K"]
    result["recommended_pcoef_thr_tau"] = result["best_tau"]

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

def plot_altitude_compare(baro_error, ekf_data, baro_data, phases):
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
    return fig


def plot_error_with_thrust(baro_error, thrust_data, corr, vz_data,
                           phases, hover_start, hover_end):
    """Plot baro error timeseries with thrust overlay and vertical velocity."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Baro Error & Thrust Pressurization", fontsize=13,
                 fontweight="bold")

    t = baro_error["time_s"]
    err = baro_error["error"]

    # Panel 1: baro error with scaled thrust and Vz overlays
    ax = axes[0]
    ax.plot(t, err, color="tab:red", linewidth=0.8, label="Baro error")
    if "thrust_time_s" in thrust_data:
        thr_up = -thrust_data["thrust_z"]
        err_min, err_max = np.min(err), np.max(err)
        thr_min, thr_max = np.min(thr_up), np.max(thr_up)
        thr_scaled = (thr_up - thr_min) / (thr_max - thr_min + 1e-10) \
                     * (err_max - err_min) + err_min
        r_val = corr.get("corr_thrust", 0)
        ax.plot(thrust_data["thrust_time_s"], thr_scaled,
                color="tab:orange", linewidth=0.8, alpha=0.7,
                label=f"Upward thrust (scaled), |r|={abs(r_val):.2f}")
    if vz_data and "vz" in vz_data:
        vz = vz_data["vz"]
        vz_min, vz_max = np.min(vz), np.max(vz)
        vz_scaled = (vz - vz_min) / (vz_max - vz_min + 1e-10) \
                    * (err_max - err_min) + err_min
        ax.plot(vz_data["time_s"], vz_scaled, color="tab:blue",
                linewidth=0.8, alpha=0.6,
                label="Vertical vel (scaled)")
    ax.axvspan(hover_start, hover_end, alpha=0.08, color="blue",
               label="Hover segment")
    ax.set_ylabel("Baro Error [m]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: detrended error vs detrended thrust (hover only)
    ax = axes[1]
    if "hover_error" in corr and "thrust_interp_hover" in corr:
        err_dt = corr["hover_error"] - np.mean(corr["hover_error"])
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
    return fig


def plot_correlations(corr, innov_data=None, thrust_data=None,
                      hover_start=None, hover_end=None, pcoef_thr=0.0):
    """Scatter plots: baro error vs thrust, vs AGL, binned error, compensated error."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baro Error Correlations (Hover)", fontsize=13,
                 fontweight="bold")

    # Error vs thrust
    ax = axes[0, 0]
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
        ax.set_title(f"Thrust Pressurization\nr = {corr.get('corr_thrust', 0):.3f}"
                     f",  slope = {z[0]:.1f} m/unit")
    ax.grid(True, alpha=0.3)

    # Error vs AGL
    ax = axes[0, 1]
    ax.scatter(corr["hover_alt"], corr["hover_error"], s=3, alpha=0.4,
               color="tab:green")
    z = np.polyfit(corr["hover_alt"], corr["hover_error"], 1)
    x_fit = np.linspace(corr["hover_alt"].min(), corr["hover_alt"].max(), 50)
    ax.plot(x_fit, np.polyval(z, x_fit), "k--", linewidth=1.2)
    ax.set_xlabel("AGL [m]")
    ax.set_ylabel("Baro Error [m]")
    ax.set_title(f"Ground Effect\nr = {corr.get('corr_altitude', 0):.3f}"
                 f",  slope = {z[0]:.1f} m/m")
    ax.grid(True, alpha=0.3)

    # Altitude-binned bar chart
    ax = axes[1, 0]
    if corr.get("alt_bins"):
        bins = corr["alt_bins"]
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
        means = [b["mean_error"] for b in bins]
        stds = [b["std_error"] for b in bins]
        labels = [f'{b["lo"]:.1f}-{b["hi"]:.1f}' for b in bins]
        ax.bar(range(len(bins)), means, yerr=stds, width=0.7,
               color="tab:green", alpha=0.7, capsize=3)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_xlabel("AGL bin [m]")
        ax.set_ylabel("Mean Baro Error [m]")
        ax.set_title("Altitude-Binned Error")
    ax.grid(True, alpha=0.3, axis="y")

    # Compensated baro error vs thrust
    ax = axes[1, 1]
    if "thrust_interp_hover" in corr:
        thrust_hov = corr["thrust_interp_hover"]
        err_hov = corr["hover_error"]
        # Apply compensation: thrust_mag = -thrust_z (constrained 0-1)
        thrust_mag = np.clip(-thrust_hov, 0, 1)
        comp_error = err_hov + pcoef_thr * thrust_mag
        ax.scatter(thrust_hov, comp_error, s=3, alpha=0.4, color="tab:purple")
        z = np.polyfit(thrust_hov, comp_error, 1)
        x_fit = np.linspace(thrust_hov.min(), thrust_hov.max(), 50)
        ax.plot(x_fit, np.polyval(z, x_fit), "k--", linewidth=1.2)
        r_val = float(np.corrcoef(thrust_hov, comp_error)[0, 1])
        coef_label = f"PCOEF_THR={pcoef_thr:.1f}"
        ax.set_title(f"Compensated Error vs Thrust ({coef_label})\n"
                     f"r = {r_val:.3f},  slope = {z[0]:.1f} m/unit")
    else:
        ax.set_title("Compensated Error vs Thrust\n(no data)")
    ax.set_xlabel("Thrust Z setpoint")
    ax.set_ylabel("Compensated Error [m]")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ekf_innovation(innov_data, gnd_effect, phases, hover_start,
                         hover_end):
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
    return fig


def plot_calibration(calib, hover_start, hover_end):
    """Plot thrust compensation calibration results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Thrust Compensation Calibration", fontsize=13,
                 fontweight="bold")

    # Top-left: Cross-correlation
    ax = axes[0, 0]
    if "xcorr_lags_s" in calib:
        ax.plot(calib["xcorr_lags_s"] * 1000, calib["xcorr_values"],
                color="tab:blue", linewidth=1.0)
        peak_lag = calib["xcorr_peak_lag_s"]
        peak_val = calib["xcorr_peak_value"]
        ax.axvline(peak_lag * 1000, color="tab:red", linestyle="--",
                    linewidth=0.8,
                    label=f"Peak: {peak_lag*1000:.0f} ms (r={peak_val:.2f})")
        ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=9)
    ax.set_xlabel("Lag [ms] (positive = error lags thrust)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Cross-Correlation: Thrust vs Baro Error")
    ax.grid(True, alpha=0.3)

    # Top-right: R² vs time constant
    ax = axes[0, 1]
    if "sweep_tau" in calib:
        ax.plot(calib["sweep_tau"], calib["sweep_r2"],
                "o-", color="tab:green", markersize=3, linewidth=1.0)
        best_tau = calib["best_tau"]
        best_r2 = calib["best_r2"]
        ax.axvline(best_tau, color="tab:red", linestyle="--", linewidth=0.8,
                    label=f"Best: tau={best_tau:.2f}s (R²={best_r2:.3f})")
        ax.legend(fontsize=9)
    ax.set_xlabel("Time constant tau [s]")
    ax.set_ylabel("R²")
    ax.set_title("Model Fit vs Time Constant")
    ax.grid(True, alpha=0.3)

    # Bottom-left: Before/after compensation timeseries
    ax = axes[1, 0]
    if "hover_time" in calib and "best_compensated_error" in calib:
        t = calib["hover_time"]
        ax.plot(t, calib["hover_error"], color="tab:red", linewidth=0.8,
                alpha=0.7, label="Raw baro error")
        ax.plot(t, calib["best_compensated_error"], color="tab:blue",
                linewidth=0.8,
                label=f"After compensation (K={calib['best_K']:.2f},"
                      f" tau={calib['best_tau']:.2f}s)")
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=9)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Baro Error [m]")
    ax.set_title("Compensation Effect")
    ax.grid(True, alpha=0.3)

    # Bottom-right: Recommended parameters text
    ax = axes[1, 1]
    ax.axis("off")
    if "recommended_pcoef_thr" in calib:
        lines = [
            "Recommended Parameters",
            "",
            f"  EKF2_PCOEF_THR  = {calib['recommended_pcoef_thr']:+.2f}  m",
            f"  EKF2_PCOEF_TTAU = {calib['recommended_pcoef_thr_tau']:.2f}  s",
            "",
            f"  Identified gain K  = {calib['best_K']:.3f} m/unit",
            f"  Time constant tau  = {calib['best_tau']:.3f} s",
            f"  Model R²           = {calib['best_r2']:.3f}",
            f"  Residual RMSE      = {calib['best_rmse']:.3f} m",
            "",
            f"  No-lag model R²    = {calib['nolag_r2']:.3f}",
            f"  R² improvement     = {calib['best_r2'] - calib['nolag_r2']:.3f}",
        ]
        if "xcorr_peak_lag_s" in calib:
            lines.append(
                f"  Cross-corr peak    = {calib['xcorr_peak_lag_s']*1000:.0f} ms")
        ax.text(0.1, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=11, verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                          edgecolor="#cccccc"))

    plt.tight_layout()
    return fig


def plot_raw_pressure(baro_data, phases, hover_start, hover_end):
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
    return fig


# ---------------------------------------------------------------------------
# Guide page
# ---------------------------------------------------------------------------

_GUIDE_TEXT = [
    ("Altitude Comparison (p2)",
     "Baro alt (zeroed at arm) vs range sensor vs EKF.\n"
     "Gap between baro and range = pressurization offset.\n"
     "Bottom: baro error timeseries (baro minus range)."),

    ("Thrust Pressurization (p3)",
     "Top: baro error with scaled thrust overlay.\n"
     "Middle: vertical velocity (climb/descend context).\n"
     "Bottom: detrended hover overlay showing correlation."),

    ("Correlations (p4)",
     "Top-left: raw baro error vs thrust (pressurization).\n"
     "Top-right: error vs AGL (ground effect gradient).\n"
     "Bottom-left: altitude-binned mean error with std bars.\n"
     "Bottom-right: compensated error vs thrust (after PCOEF_THR)."),

    ("EKF Innovation (p5)",
     "Top: baro height innovation (EKF prediction - baro).\n"
     "Middle: test ratio (>1 = rejected by gate).\n"
     "Bottom: EKF flags — ground effect, baro/range fuse."),

    ("Calibration (p6, if --calibrate)",
     "Top-left: cross-correlation showing delay structure.\n"
     "Top-right: R² vs time constant tau for model selection.\n"
     "Bottom-left: before/after compensation timeseries.\n"
     "Bottom-right: recommended EKF2_PCOEF_THR and EKF2_PCOEF_TTAU."),

    ("Raw Pressure (p7)",
     "Raw barometer pressure and temperature.\n"
     "Pressure drop at arm = prop wash depression.\n"
     "Temperature trend = thermal drift check."),
]

_HGT_REF_LABELS = {0: "Barometer", 1: "GNSS", 2: "Range sensor", 3: "Vision"}


def render_guide_page(params):
    """Render guide + baro/height params as page 1 of the report."""
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#fafafa")

    fig.text(0.5, 0.97, "Barometric Pressure Bias Report",
             fontsize=18, fontweight="bold", ha="center", va="top",
             family="sans-serif")

    # Left: plot guide
    fig.text(0.06, 0.91, "Plot Guide",
             fontsize=13, fontweight="bold", va="top",
             family="sans-serif", color="#333333")

    colors = ["#d32f2f", "#e65100", "#2e7d32", "#1565c0", "#6a1b9a", "#00838f"]
    y = 0.86
    for i, (title, body) in enumerate(_GUIDE_TEXT):
        color = colors[i]
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

    # Right: parameters
    fig.text(0.55, 0.91, "Height / Baro Parameters",
             fontsize=13, fontweight="bold", va="top",
             family="sans-serif", color="#333333")

    param_groups = [
        ("Height Reference", [
            ("EKF2_HGT_REF", _HGT_REF_LABELS),
            ("EKF2_BARO_CTRL", None),
            ("EKF2_RNG_CTRL", None),
        ]),
        ("Baro Fusion", [
            ("EKF2_BARO_NOISE", None),
            ("EKF2_BARO_GATE", None),
            ("EKF2_BARO_DELAY", None),
            ("SENS_BARO_RATE", None),
        ]),
        ("Thrust Compensation", [
            ("EKF2_PCOEF_THR", None),
            ("EKF2_PCOEF_TTAU", None),
        ]),
        ("Ground Effect", [
            ("EKF2_GND_EFF_DZ", None),
            ("EKF2_GND_MAX_HGT", None),
        ]),
    ]

    y = 0.86
    for group_name, param_list in param_groups:
        fig.text(0.56, y, group_name,
                 fontsize=10, fontweight="bold", va="top",
                 family="sans-serif", color="#555555")
        y -= 0.03
        for pname, labels in param_list:
            val = params.get(pname, "N/A")
            extra = ""
            if labels and val != "N/A":
                extra = f"  ({labels.get(int(val), '?')})"
            fig.text(0.58, y, pname, fontsize=8.5, va="top",
                     family="monospace", color="#444444")
            fig.text(0.82, y, f"{val}{extra}", fontsize=8.5, va="top",
                     family="monospace", color="#111111", fontweight="bold")
            y -= 0.022
        y -= 0.015

    fig.text(0.5, 0.02, "Generated by baro_pressurization.py",
             fontsize=8, ha="center", color="#999999", family="sans-serif")

    return fig


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def generate_summary(phases, params, baro_error, corr, press_trends,
                     innov_data, gnd_effect, hover_start, hover_end,
                     calib=None):
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
    lines.append(f"  EKF2_PCOEF_THR          = {params.get('EKF2_PCOEF_THR', 'N/A')}")
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

    # Calibration
    if calib and "best_K" in calib:
        lines.append("\n" + "=" * 70)
        lines.append("THRUST COMPENSATION CALIBRATION")
        lines.append("=" * 70)
        lines.append(f"\n  Model: baro_error = K * LPF(thrust, tau)")
        lines.append(f"  Identified gain K          = {calib['best_K']:.3f} m/unit")
        lines.append(f"  Identified time constant   = {calib['best_tau']:.3f} s")
        lines.append(f"  Model R²                   = {calib['best_r2']:.3f}")
        lines.append(f"  Residual RMSE              = {calib['best_rmse']:.3f} m")
        if "xcorr_peak_lag_s" in calib:
            lines.append(f"  Cross-corr peak lag        = "
                          f"{calib['xcorr_peak_lag_s']*1000:.0f} ms")
        lines.append(f"\n  No-lag model (tau=0):")
        lines.append(f"    R²                       = {calib['nolag_r2']:.3f}")
        improvement = calib['best_r2'] - calib['nolag_r2']
        lines.append(f"    R² improvement from lag   = {improvement:+.3f}")

        lines.append(f"\n  RECOMMENDED PARAMETERS:")
        lines.append(f"    EKF2_PCOEF_THR  = {calib['recommended_pcoef_thr']:+.2f}")
        lines.append(f"    EKF2_PCOEF_TTAU = {calib['recommended_pcoef_thr_tau']:.2f}")

        if calib['best_r2'] < 0.1:
            lines.append(f"\n  NOTE: Low R² ({calib['best_r2']:.3f}) suggests thrust is not "
                          "the dominant baro error source. Compensation may not help.")
        elif improvement < 0.01:
            lines.append(f"\n  NOTE: Lag filter provides negligible improvement. "
                          "The system delay may be short enough that tau=0 is fine.")

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
    parser.add_argument("--calibrate", action="store_true",
                        help="Run system identification to find optimal "
                             "EKF2_PCOEF_THR and EKF2_PCOEF_TTAU values")
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
        "EKF2_HGT_REF", "EKF2_BARO_CTRL", "EKF2_RNG_CTRL",
        "EKF2_BARO_NOISE", "EKF2_BARO_GATE", "EKF2_BARO_DELAY",
        "EKF2_GND_EFF_DZ", "EKF2_GND_MAX_HGT",
        "EKF2_PCOEF_THR", "EKF2_PCOEF_TTAU",
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
    pcoef = params.get('EKF2_PCOEF_THR')
    if pcoef is not None:
        print(f"Thrust compensation: EKF2_PCOEF_THR={pcoef}"
              f"{' (disabled)' if pcoef == 0 else ''}")

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

    # Vertical velocity (EKF Vz if range fused, else range derivative)
    vz_data = extract_vertical_velocity(ekf_data, range_data, gnd_effect)

    print(f"  Baro data: {'yes' if has_baro else 'NO'}")
    print(f"  Range sensor: {'yes' if has_range else 'NO'}")
    print(f"  Thrust data: {'yes' if thrust_data else 'NO'}")
    print(f"  Baro innovation: {'yes' if innov_data else 'NO'}")
    print(f"  Ground effect flags: {'yes' if gnd_effect else 'NO'}")
    if vz_data:
        print(f"  Vertical velocity: {vz_data['source']}")

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

    # Calibration (system identification)
    calib = {}
    if args.calibrate:
        if baro_error and thrust_data:
            print("\nRunning thrust compensation calibration...")
            calib = calibrate_thrust_compensation(
                baro_error, thrust_data, hover_start, hover_end)
            if calib and "best_K" in calib:
                print(f"  Identified gain K  = {calib['best_K']:.3f} m/unit")
                print(f"  Time constant tau  = {calib['best_tau']:.3f} s")
                print(f"  Model R²           = {calib['best_r2']:.3f}")
                print(f"\n  Recommended parameters:")
                print(f"    EKF2_PCOEF_THR  = {calib['recommended_pcoef_thr']:+.2f}")
                print(f"    EKF2_PCOEF_TTAU = {calib['recommended_pcoef_thr_tau']:.2f}")
            else:
                print("  Calibration failed: insufficient data or variation")
        else:
            print("\nWARNING: --calibrate requires range sensor + thrust data")

    # Generate plots
    print("\nGenerating plots...")
    figures = [render_guide_page(params)]  # page 1: guide + params

    if baro_error:
        figures.append(plot_altitude_compare(
            baro_error, ekf_data, baro_data, phases))
        figures.append(plot_error_with_thrust(
            baro_error, thrust_data, corr, vz_data,
            phases, hover_start, hover_end))

    if corr and "hover_error" in corr:
        pcoef = float(params.get("EKF2_PCOEF_THR", 0))
        figures.append(plot_correlations(corr, innov_data, thrust_data,
                                         hover_start, hover_end,
                                         pcoef_thr=pcoef))

    if innov_data or gnd_effect:
        figures.append(plot_ekf_innovation(
            innov_data, gnd_effect, phases, hover_start, hover_end))

    if calib and "best_K" in calib:
        figures.append(plot_calibration(calib, hover_start, hover_end))

    if "raw_time_s" in baro_data:
        figures.append(plot_raw_pressure(
            baro_data, phases, hover_start, hover_end))

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
                                hover_start, hover_end, calib=calib)
    summary_path = os.path.join(output_dir, "baro_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")

    print()
    print(summary)


if __name__ == "__main__":
    main()
