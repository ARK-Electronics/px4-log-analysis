#!/usr/bin/env python3
"""
PX4 GPS signal quality analyzer.

Profiles a GPS module's signal quality from a PX4 ULog and emits a PDF report
with annotated plots. Supports a two-log compare mode so before/after flights
(e.g. buggy vs. fixed driver) line up on the same axes.

Usage:
    python3 gps_signal_quality.py <log_a.ulg> [log_b.ulg] [-o <dir>]
                                   [--label-a NAME] [--label-b NAME]

Outputs:
    - gps_signal_quality.pdf           (single-log mode)
    - gps_signal_quality_compare.pdf   (compare mode)
    - gps_signal_quality.txt           summary stats as plain text
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

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
# Constants
# ---------------------------------------------------------------------------

FIX_TYPE_LABELS = {
    0: "No GPS", 1: "No fix", 2: "2D", 3: "3D",
    4: "DGPS", 5: "RTK-float", 6: "RTK-fixed", 8: "Extrapolated",
}

JAMMING_STATE_LABELS = {
    0: "Unknown", 1: "OK", 2: "Warning", 3: "Critical",
}

SPOOFING_STATE_LABELS = {
    0: "Unknown", 1: "None", 2: "Indicated", 3: "Multiple",
}

# Per-field upper bounds — values >= these are receiver sentinels meaning
# "no fix" / "unavailable" and get masked to NaN so they don't distort plots.
# Rationale: a real GPS never reports eph > 200 m, s_variance > 50 m/s, etc.
# Anything near these limits is the driver's fallback, e.g. UINT32_MAX (mm) / 1000
# = 4.29e6, 99.99 for DOP, 999 for s_variance, and pi for c_variance.
_SENTINEL_CAP = {
    "eph":             200.0,
    "epv":             200.0,
    "hdop":             50.0,
    "vdop":             50.0,
    "s_variance_m_s":   50.0,
    "c_variance_rad":    3.1,  # pi means "unknown course"
    "heading_accuracy":  3.1,
    "vel_m_s":         100.0,
}

# Aliases: new field name -> (old name, scale to new units). Used when the log
# predates the sensor_gps rename that turned integer-scaled lat/lon/alt into
# float decimal degrees / meters.
_POS_ALIASES = [
    ("latitude_deg",         "lat",            1e-7),
    ("longitude_deg",        "lon",            1e-7),
    ("altitude_msl_m",       "alt",            1e-3),
    ("altitude_ellipsoid_m", "alt_ellipsoid",  1e-3),
]


# ---------------------------------------------------------------------------
# ULog helpers
# ---------------------------------------------------------------------------

def get_topic(ulog, topic_name, multi_id=0):
    """Return the dataset for a topic name and exact multi_id, or None."""
    for d in ulog.data_list:
        if d.name == topic_name and d.multi_id == multi_id:
            return d
    return None


def get_topic_first_populated(ulog, topic_name, prefer_multi_id=None):
    """Return the richest populated dataset for `topic_name`, or None.

    Picks the multi_id with the most samples. Useful for sensor_gps since some
    board configs subscribe multiple multi_ids even with one receiver connected,
    in which case the unused ones are empty. (PlotJuggler renders those as
    `sensor_gps.00` and `sensor_gps.01`; here we just see multiple `data_list`
    entries with the same name.)

    If `prefer_multi_id` is set, picks that multi_id when populated.
    """
    candidates = [d for d in ulog.data_list if d.name == topic_name]
    if not candidates:
        return None

    def nsamples(d):
        ts = d.data.get("timestamp")
        return 0 if ts is None else len(ts)

    # Honor explicit pick if it has data
    if prefer_multi_id is not None:
        for d in candidates:
            if d.multi_id == prefer_multi_id and nsamples(d) > 1:
                return d

    # Otherwise pick the one with the most samples
    best = max(candidates, key=nsamples)
    populated = [d for d in candidates if nsamples(d) > 1]
    if len(populated) > 1:
        ids = ", ".join(f"multi_id={d.multi_id} (n={nsamples(d)})"
                        for d in populated)
        print(f"  NOTE: multiple populated {topic_name} instances: {ids}. "
              f"Using multi_id={best.multi_id}.")
    return best if nsamples(best) > 0 else candidates[0]


def get_param(ulog, name, default=None):
    return ulog.initial_parameters.get(name, default)


def timestamps_to_seconds(ts_us, start_us):
    return (ts_us.astype(np.int64) - np.int64(start_us)) / 1e6


def resolve_position_fields(data):
    """Return a dict with canonical position fields from old-or-new schema."""
    out = {}
    for new_name, old_name, scale in _POS_ALIASES:
        if new_name in data:
            out[new_name] = np.asarray(data[new_name], dtype=float)
        elif old_name in data:
            out[new_name] = np.asarray(data[old_name], dtype=float) * scale
    return out


def mask_sentinel(arr, field_name):
    """Replace receiver sentinels with NaN for plotting / stats.

    Each GPS field has its own sentinel pattern (driver-dependent). Use the
    per-field cap in _SENTINEL_CAP; also mask the large UINT32-derived values
    that show up when the driver never got a valid reading.
    """
    arr = np.asarray(arr, dtype=float)
    cap = _SENTINEL_CAP.get(field_name)
    if cap is not None:
        arr = np.where(arr >= cap, np.nan, arr)
    # Also mask obviously-impossible values (UINT32_MAX scaled)
    arr = np.where(arr > 1e5, np.nan, arr)
    return arr


# ---------------------------------------------------------------------------
# Flight phase detection
# ---------------------------------------------------------------------------

def detect_flight_phases(ulog):
    """Detect armed + stationary + moving segments.

    Returns dict with:
      armed_start_s, armed_end_s : float | None
      stationary_mask_fn(t)      : callable returning bool mask for any t array
      moving_mask_fn(t)          : callable
      stationary_start_s         : best stationary window start (for stats)
      stationary_end_s           : end (for stats); may be None if none found
    """
    start_us = ulog.start_timestamp
    info = {
        "start_us": start_us,
        "armed_start_s": None,
        "armed_end_s": None,
        "stationary_start_s": None,
        "stationary_end_s": None,
    }

    vstatus = get_topic(ulog, "vehicle_status")
    if vstatus is not None and "arming_state" in vstatus.data:
        ts = timestamps_to_seconds(vstatus.data["timestamp"], start_us)
        armed_idx = np.where(vstatus.data["arming_state"] == 2)[0]
        if len(armed_idx) > 0:
            info["armed_start_s"] = float(ts[armed_idx[0]])
            info["armed_end_s"] = float(ts[armed_idx[-1]])

    if info["armed_start_s"] is None:
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
                info["armed_start_s"] = float(ts[active_idx[0]])
                info["armed_end_s"] = float(ts[active_idx[-1]])

    if info["armed_start_s"] is None:
        info["armed_start_s"] = 0.0
        info["armed_end_s"] = float((ulog.last_timestamp - start_us) / 1e6)

    # Find stationary segment: armed + vxy < 0.3 m/s for >= 10 s
    lpos = get_topic(ulog, "vehicle_local_position")
    if lpos is not None and "vx" in lpos.data and "vy" in lpos.data:
        t = timestamps_to_seconds(lpos.data["timestamp"], start_us)
        vxy = np.hypot(lpos.data["vx"], lpos.data["vy"])
        in_window = (t >= info["armed_start_s"]) & (t <= info["armed_end_s"])
        slow = in_window & (vxy < 0.3)
        info["stationary_start_s"], info["stationary_end_s"] = \
            _longest_true_run(t, slow, min_duration_s=10.0)

    # If no stationary segment found, pretend the pre-takeoff armed window was it
    if info["stationary_start_s"] is None and info["armed_end_s"] is not None:
        # Use first 10s of armed if available
        t0 = info["armed_start_s"]
        t1 = min(info["armed_end_s"], t0 + 10.0)
        if t1 - t0 >= 3.0:
            info["stationary_start_s"] = t0
            info["stationary_end_s"] = t1

    return info


def _longest_true_run(t, mask, min_duration_s=10.0):
    """Find the longest contiguous run of True in `mask` lasting >= min_duration_s.

    Returns (t_start, t_end) or (None, None) if none found.
    """
    if len(t) == 0 or not mask.any():
        return None, None
    # Find run starts/ends
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [len(mask)]))

    best = None
    for s, e in zip(starts, ends):
        dur = t[e - 1] - t[s]
        if dur < min_duration_s:
            continue
        if best is None or dur > best[2]:
            best = (float(t[s]), float(t[e - 1]), dur)
    if best is None:
        return None, None
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

@dataclass
class LogData:
    """All extracted data for one log, ready for plotting."""
    path: str
    label: str
    ulog: ULog
    params: dict = field(default_factory=dict)
    duration_s: float = 0.0
    phases: dict = field(default_factory=dict)
    gps: dict = field(default_factory=dict)
    lpos: dict = field(default_factory=dict)
    gnss_pos: dict = field(default_factory=dict)
    gnss_vel: dict = field(default_factory=dict)
    gnss_hgt: dict = field(default_factory=dict)
    gps_status: dict = field(default_factory=dict)
    fusion: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def arm_offset_s(self) -> float:
        v = self.phases.get("armed_start_s")
        return 0.0 if v is None else v

    def tshift(self, t):
        """Convert log-time seconds to seconds-since-armed."""
        return t - self.arm_offset_s


def extract_sensor_gps(ulog):
    """Extract sensor_gps timeseries. Falls back to vehicle_gps_position."""
    start_us = ulog.start_timestamp
    d = get_topic_first_populated(ulog, "sensor_gps")
    if d is None or len(d.data.get("timestamp", [])) < 2:
        d = get_topic_first_populated(ulog, "vehicle_gps_position")
    if d is None:
        return {}

    data = d.data
    out = {
        "time_s": timestamps_to_seconds(data["timestamp"], start_us),
        "multi_id": d.multi_id,
        "topic": d.name,
    }
    # Direct fields (with sentinel masking for the unsigned-integer counters)
    for key, mask in [
        ("satellites_used", False),
        ("fix_type",        False),
        ("eph",             True),
        ("epv",             True),
        ("hdop",            True),
        ("vdop",            True),
        ("s_variance_m_s",  True),
        ("c_variance_rad",  True),
        ("noise_per_ms",    False),
        ("jamming_indicator", False),
        ("jamming_state",   False),
        ("spoofing_state",  False),
        ("automatic_gain_control", False),
        ("heading",         True),
        ("heading_accuracy", True),
        ("vel_m_s",         True),
        ("cog_rad",         True),
    ]:
        if key in data:
            arr = mask_sentinel(data[key], key) if mask else np.asarray(data[key])
            out[key] = arr

    out.update(resolve_position_fields(data))
    return out


def extract_local_position(ulog):
    start_us = ulog.start_timestamp
    d = get_topic(ulog, "vehicle_local_position")
    if d is None:
        return {}
    data = d.data
    out = {
        "time_s": timestamps_to_seconds(data["timestamp"], start_us),
    }
    for k in ("eph", "epv", "evh", "evv", "x", "y", "z", "vx", "vy", "vz",
             "dead_reckoning"):
        if k in data:
            out[k] = np.asarray(data[k])
    return out


def extract_gnss_aid(ulog, suffix):
    """Extract estimator_aid_src_gnss_<suffix> (pos / vel / hgt)."""
    start_us = ulog.start_timestamp
    d = get_topic(ulog, f"estimator_aid_src_gnss_{suffix}")
    if d is None:
        return {}
    data = d.data
    out = {
        "time_s": timestamps_to_seconds(data["timestamp"], start_us),
    }
    for k in ("innovation", "innovation_variance", "test_ratio",
              "innovation_rejected", "fused",
              "innovation[0]", "innovation[1]",
              "test_ratio[0]", "test_ratio[1]"):
        if k in data:
            out[k] = np.asarray(data[k])
    # For pos/vel the innovation is 2D (NE); expose a magnitude
    if "innovation[0]" in out and "innovation[1]" in out:
        out["innovation_mag"] = np.hypot(out["innovation[0]"], out["innovation[1]"])
    if "test_ratio[0]" in out and "test_ratio[1]" in out:
        out["test_ratio_max"] = np.maximum(out["test_ratio[0]"], out["test_ratio[1]"])
    return out


def extract_gps_status(ulog):
    """Extract estimator_gps_status (drift rates, check flags). Absent on older logs."""
    start_us = ulog.start_timestamp
    d = get_topic(ulog, "estimator_gps_status")
    if d is None:
        return {}
    data = d.data
    out = {"time_s": timestamps_to_seconds(data["timestamp"], start_us)}
    for k in ("position_drift_rate_horizontal_m_s",
              "position_drift_rate_vertical_m_s",
              "filtered_horizontal_speed_m_s",
              "checks_passed"):
        if k in data:
            out[k] = np.asarray(data[k])
    return out


def extract_fusion_status(ulog):
    start_us = ulog.start_timestamp
    d = get_topic(ulog, "estimator_status_flags")
    if d is None:
        return {}
    data = d.data
    out = {"time_s": timestamps_to_seconds(data["timestamp"], start_us)}
    for k in ("cs_gnss_pos", "cs_gnss_vel", "cs_gps_hgt", "cs_gnss_yaw",
              "cs_gnss_fault", "cs_inertial_dead_reckoning",
              "cs_baro_hgt", "cs_rng_hgt"):
        if k in data:
            out[k] = np.asarray(data[k])
    return out


def load_log(path, label=None):
    """Load a ULog and extract all data needed for the report."""
    print(f"Loading {path}...")
    ulog = ULog(path)
    ld = LogData(
        path=path,
        label=label or os.path.splitext(os.path.basename(path))[0],
        ulog=ulog,
        duration_s=float((ulog.last_timestamp - ulog.start_timestamp) / 1e6),
    )

    # Collect parameters
    param_names = [
        "EKF2_GPS_CTRL", "EKF2_GPS_CHECK", "EKF2_REQ_GPS_H",
        "EKF2_GPS_POS_X", "EKF2_GPS_POS_Y", "EKF2_GPS_POS_Z",
        "EKF2_GPS_DELAY", "EKF2_GPS_V_NOISE", "EKF2_GPS_P_NOISE",
        "EKF2_HGT_REF", "GPS_UBX_DYNMODEL", "GPS_UBX_CFG_INTF",
        "GPS_1_CONFIG", "GPS_1_PROTOCOL",
    ]
    for p in param_names:
        v = get_param(ulog, p)
        if v is not None:
            ld.params[p] = v

    ld.phases = detect_flight_phases(ulog)
    ld.gps = extract_sensor_gps(ulog)
    ld.lpos = extract_local_position(ulog)
    ld.gnss_pos = extract_gnss_aid(ulog, "pos")
    ld.gnss_vel = extract_gnss_aid(ulog, "vel")
    ld.gnss_hgt = extract_gnss_aid(ulog, "hgt")
    ld.gps_status = extract_gps_status(ulog)
    ld.fusion = extract_fusion_status(ulog)

    print(f"  Duration: {ld.duration_s:.1f}s, "
          f"armed: {ld.phases.get('armed_start_s', 0):.1f}-{ld.phases.get('armed_end_s', 0):.1f}s")
    if ld.phases.get("stationary_start_s") is not None:
        t0 = ld.phases["stationary_start_s"]
        t1 = ld.phases["stationary_end_s"]
        print(f"  Stationary window: {t0:.1f}-{t1:.1f}s ({t1 - t0:.1f}s)")
    else:
        print(f"  Stationary window: not detected")
    print(f"  sensor_gps samples: {len(ld.gps.get('time_s', []))} "
          f"(topic: {ld.gps.get('topic', 'MISSING')})")

    return ld


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _stationary_mask(t, phases):
    t0 = phases.get("stationary_start_s")
    t1 = phases.get("stationary_end_s")
    if t0 is None or t1 is None:
        return np.zeros_like(t, dtype=bool)
    return (t >= t0) & (t <= t1)


def _pct(arr, q):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.percentile(arr, q))


def _summary_triplet(arr):
    return {
        "median": _pct(arr, 50),
        "p05": _pct(arr, 5),
        "p95": _pct(arr, 95),
    }


def compute_summary_stats(ld: LogData):
    """Compute median/p05/p95 for each tracked metric over the stationary segment."""
    stats = {}
    g = ld.gps
    if not g:
        ld.stats = stats
        return

    mask = _stationary_mask(g["time_s"], ld.phases)
    if not mask.any():
        # Fall back to full armed window
        t0 = ld.phases.get("armed_start_s")
        t1 = ld.phases.get("armed_end_s")
        if t0 is not None:
            mask = (g["time_s"] >= t0) & (g["time_s"] <= t1)
            stats["_scope"] = "armed"
    else:
        stats["_scope"] = "stationary"

    for key in ("satellites_used", "fix_type", "eph", "epv", "hdop", "vdop",
                "s_variance_m_s", "c_variance_rad", "noise_per_ms",
                "jamming_indicator", "automatic_gain_control"):
        if key in g and mask.any():
            stats[key] = _summary_triplet(g[key][mask])

    # Local position uncertainties (post-EKF)
    if ld.lpos:
        lt = ld.lpos["time_s"]
        lmask = _stationary_mask(lt, ld.phases)
        if not lmask.any():
            t0 = ld.phases.get("armed_start_s")
            t1 = ld.phases.get("armed_end_s")
            if t0 is not None:
                lmask = (lt >= t0) & (lt <= t1)
        for key in ("eph", "epv", "evh", "evv"):
            if key in ld.lpos and lmask.any():
                stats["lpos_" + key] = _summary_triplet(ld.lpos[key][lmask])

    # Drift rates (if estimator_gps_status available)
    if ld.gps_status:
        st = ld.gps_status["time_s"]
        smask = _stationary_mask(st, ld.phases)
        for key in ("position_drift_rate_horizontal_m_s",
                    "position_drift_rate_vertical_m_s"):
            if key in ld.gps_status and smask.any():
                stats[key] = _summary_triplet(ld.gps_status[key][smask])

    # TTFF: time since armed when fix_type first >= 3
    if "fix_type" in g:
        fix_ge3 = np.where(g["fix_type"] >= 3)[0]
        if len(fix_ge3) > 0:
            ttff_t = g["time_s"][fix_ge3[0]] - ld.arm_offset_s
            stats["ttff_s"] = float(ttff_t)

    ld.stats = stats


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

# Colors / styles per log slot
_LOG_STYLE = [
    {"color": "tab:blue",   "alpha": 0.9},
    {"color": "tab:red",    "alpha": 0.9},
]

_INTERP_TEXT_STYLE = dict(
    fontsize=8.5, color="#333333", family="sans-serif",
    ha="left", va="top", wrap=True,
)


def _shade_stationary(ax, logs, alpha_per_log=0.08):
    """Shade the stationary segment(s) on an axis using seconds-since-armed."""
    for i, ld in enumerate(logs):
        t0 = ld.phases.get("stationary_start_s")
        t1 = ld.phases.get("stationary_end_s")
        if t0 is None or t1 is None:
            continue
        color = _LOG_STYLE[i]["color"]
        ax.axvspan(ld.tshift(t0), ld.tshift(t1), alpha=alpha_per_log,
                   color=color, linewidth=0)


def _legend_label(ld: LogData, base: str, slot: int) -> str:
    if len(_LOG_STYLE) and slot is not None:
        return f"{ld.label}: {base}" if slot == 1 or _compare_mode else base
    return base


def _interp_text(fig, text, y=0.02):
    fig.text(0.06, y, text, **_INTERP_TEXT_STYLE)


def _missing_banner(fig, msg):
    fig.text(0.5, 0.5, msg, fontsize=14, ha="center", va="center",
             color="#999999", fontweight="bold")


def _plot_metric(ax, logs, key, ylabel, *, source="gps", ylim=None):
    """Plot a scalar metric vs. time-since-armed for all logs.

    Skips values that are all-NaN after sentinel masking. Returns True if any
    line was drawn.
    """
    any_plotted = False
    for i, ld in enumerate(logs):
        bag = getattr(ld, source)
        if key not in bag:
            continue
        arr = np.asarray(bag[key], dtype=float)
        if np.all(np.isnan(arr)):
            continue
        t_shift = ld.tshift(bag["time_s"])
        style = _LOG_STYLE[i]
        ax.plot(t_shift, arr, label=ld.label,
                color=style["color"], alpha=style["alpha"], linewidth=0.9)
        any_plotted = True
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if any_plotted:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="#aaaaaa", fontsize=10)
    return any_plotted


# ---------------------------------------------------------------------------
# PDF pages
# ---------------------------------------------------------------------------

def _format_delta(a, b, lower_is_better=True, pct=True):
    """Return an arrow-annotated delta string for log_b vs log_a."""
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return "—"
    if a == 0:
        if b == 0:
            return "= 0"
        return f"{b:+.3g} (from 0)"
    diff_pct = (b - a) / abs(a) * 100
    if pct:
        raw = f"{diff_pct:+.0f}%"
    else:
        raw = f"{b - a:+.3g}"
    improving = (b < a) if lower_is_better else (b > a)
    arrow = "↓" if b < a else ("↑" if b > a else "=")
    tag = " (better)" if improving and b != a else (" (worse)" if b != a else "")
    return f"{arrow} {raw}{tag}"


def _metric_row(key, stats_list, *, label, fmt="{:.2f}",
                lower_is_better=True, pct_delta=True):
    """Build a single summary-table row across logs."""
    vals = []
    for s in stats_list:
        trip = s.get(key)
        v = trip["median"] if trip else None
        vals.append(v)

    cells = [label]
    for v in vals:
        cells.append("—" if v is None or (isinstance(v, float) and np.isnan(v))
                     else fmt.format(v))
    if len(stats_list) == 2:
        cells.append(_format_delta(vals[0], vals[1],
                                   lower_is_better=lower_is_better,
                                   pct=pct_delta))
    return cells


def _summary_table(stats_list):
    """Build list-of-rows summarizing key metrics across logs."""
    rows = [
        _metric_row("satellites_used", stats_list, label="Satellites used",
                    fmt="{:.0f}", lower_is_better=False, pct_delta=False),
        _metric_row("fix_type", stats_list, label="Fix type (median)",
                    fmt="{:.0f}", lower_is_better=False, pct_delta=False),
        _metric_row("eph", stats_list, label="eph (receiver) [m]",
                    fmt="{:.2f}", lower_is_better=True),
        _metric_row("epv", stats_list, label="epv (receiver) [m]",
                    fmt="{:.2f}", lower_is_better=True),
        _metric_row("hdop", stats_list, label="HDOP",
                    fmt="{:.2f}", lower_is_better=True),
        _metric_row("vdop", stats_list, label="VDOP",
                    fmt="{:.2f}", lower_is_better=True),
        _metric_row("s_variance_m_s", stats_list, label="Speed accuracy [m/s]",
                    fmt="{:.3f}", lower_is_better=True),
        _metric_row("noise_per_ms", stats_list, label="Noise/ms",
                    fmt="{:.0f}", lower_is_better=True, pct_delta=False),
        _metric_row("jamming_indicator", stats_list, label="Jamming indicator",
                    fmt="{:.0f}", lower_is_better=True, pct_delta=False),
        _metric_row("lpos_eph", stats_list, label="EKF eph [m]",
                    fmt="{:.2f}", lower_is_better=True),
        _metric_row("lpos_epv", stats_list, label="EKF epv [m]",
                    fmt="{:.2f}", lower_is_better=True),
        _metric_row("lpos_evh", stats_list, label="EKF evh [m/s]",
                    fmt="{:.3f}", lower_is_better=True),
        _metric_row("position_drift_rate_horizontal_m_s", stats_list,
                    label="Horiz drift rate [m/s]",
                    fmt="{:.4f}", lower_is_better=True),
        _metric_row("position_drift_rate_vertical_m_s", stats_list,
                    label="Vert drift rate [m/s]",
                    fmt="{:.4f}", lower_is_better=True),
    ]
    # TTFF
    ttff_row = ["TTFF (fix >=3) [s]"]
    ttffs = [s.get("ttff_s") for s in stats_list]
    for v in ttffs:
        ttff_row.append("—" if v is None else f"{v:.1f}")
    if len(stats_list) == 2:
        ttff_row.append(_format_delta(ttffs[0], ttffs[1],
                                      lower_is_better=True, pct=False))
    rows.insert(1, ttff_row)
    return rows


def plot_cover_page(logs):
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#fafafa")
    is_compare = len(logs) == 2

    title = ("GPS Signal Quality — Comparison Report" if is_compare
             else "GPS Signal Quality Report")
    fig.text(0.5, 0.97, title, fontsize=18, fontweight="bold",
             ha="center", va="top")

    # Left: metadata per log
    y = 0.91
    fig.text(0.06, y, "Logs", fontsize=13, fontweight="bold", va="top",
             color="#333333")
    y -= 0.03
    for i, ld in enumerate(logs):
        color = _LOG_STYLE[i]["color"]
        slot = "A" if i == 0 else "B"
        fig.patches.append(plt.Rectangle(
            (0.05, y - 0.003), 0.004, 0.018,
            transform=fig.transFigure, facecolor=color, clip_on=False))
        fig.text(0.065, y, f"Log {slot}: {ld.label}",
                 fontsize=10, fontweight="bold", va="top", color=color)
        y -= 0.022
        arm0 = ld.phases.get("armed_start_s")
        arm1 = ld.phases.get("armed_end_s")
        arm_txt = (f"armed {arm0:.0f}-{arm1:.0f}s ({arm1 - arm0:.0f}s)"
                   if arm0 is not None else "armed window n/a")
        fig.text(0.075, y, f"  Duration: {ld.duration_s:.0f}s   {arm_txt}",
                 fontsize=8.5, va="top", family="monospace", color="#444444")
        y -= 0.018
        stn0 = ld.phases.get("stationary_start_s")
        stn1 = ld.phases.get("stationary_end_s")
        if stn0 is not None:
            stn_txt = f"stationary {stn0:.0f}-{stn1:.0f}s ({stn1 - stn0:.0f}s)"
        else:
            stn_txt = "stationary window: not detected"
        fig.text(0.075, y, f"  {stn_txt}",
                 fontsize=8.5, va="top", family="monospace", color="#444444")
        y -= 0.018
        fig.text(0.075, y, f"  Path: {ld.path}",
                 fontsize=7.5, va="top", family="monospace", color="#888888")
        y -= 0.028

    # How to read
    y -= 0.01
    fig.text(0.06, y, "How to read this report", fontsize=11,
             fontweight="bold", va="top", color="#333333")
    y -= 0.025
    hint_lines = [
        "• Time axis is seconds-since-armed so both logs align.",
        "• Shaded bands mark the stationary window used for summary stats.",
        "• Tables show medians (p05/p95 implied) over the stationary window.",
    ]
    if is_compare:
        hint_lines.append(
            "• Δ columns compare log B to log A; arrows point to direction of change.")
    for line in hint_lines:
        fig.text(0.075, y, line, fontsize=8.5, va="top", color="#444444")
        y -= 0.020

    # Right: summary table
    stats_list = [ld.stats for ld in logs]
    rows = _summary_table(stats_list)
    headers = ["Metric"] + [ld.label for ld in logs]
    if is_compare:
        headers.append("Δ (B vs A)")

    x_left = 0.52
    y_top = 0.91
    fig.text(x_left, y_top, "Summary (stationary window)",
             fontsize=13, fontweight="bold", va="top", color="#333333")
    y_top -= 0.04

    col_widths = [0.26, 0.08, 0.08, 0.09] if is_compare else [0.30, 0.10, 0.10]
    # Ensure we only use the widths we need
    col_widths = col_widths[:len(headers)]

    # Header
    x = x_left
    for h, w in zip(headers, col_widths):
        fig.text(x, y_top, h, fontsize=9, fontweight="bold", va="top",
                 color="#222222", family="sans-serif")
        x += w
    y_top -= 0.022

    for row in rows:
        x = x_left
        for val, w in zip(row, col_widths):
            fig.text(x, y_top, str(val), fontsize=8.5, va="top",
                     family="monospace", color="#333333")
            x += w
        y_top -= 0.020

    fig.text(0.5, 0.02, "Generated by gps_signal_quality.py",
             fontsize=8, ha="center", color="#999999")
    return fig


def plot_satellites_fix(logs):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Satellites & Fix Type", fontsize=13, fontweight="bold")

    any_plotted = _plot_metric(axes[0], logs, "satellites_used",
                                "Satellites used", source="gps")
    _shade_stationary(axes[0], logs)
    axes[0].set_title("More sats on the same sky = receiver tracking additional signals",
                      fontsize=9, loc="left", color="#666666")

    _plot_metric(axes[1], logs, "fix_type", "Fix type", source="gps")
    _shade_stationary(axes[1], logs)
    axes[1].set_yticks(list(FIX_TYPE_LABELS.keys()))
    axes[1].set_yticklabels([f"{v} {FIX_TYPE_LABELS[v]}" for v in FIX_TYPE_LABELS])
    axes[1].set_xlabel("Time since armed [s]")
    axes[1].set_title("3=3D, 4=DGPS, 5=RTK-float, 6=RTK-fixed",
                      fontsize=9, loc="left", color="#666666")

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    _interp_text(
        fig,
        "Higher satellite count on the same sky means the receiver is tracking "
        "more signals (e.g. adding L5 / E5a / B2a). An earlier upgrade to "
        "fix_type ≥ 3 means faster TTFF; ≥ 5 indicates RTK float.",
    )
    if not any_plotted:
        _missing_banner(fig, "sensor_gps missing")
    return fig


def plot_receiver_accuracy(logs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("Receiver-Reported Accuracy", fontsize=13, fontweight="bold")

    _plot_metric(axes[0, 0], logs, "eph", "eph [m]", source="gps")
    axes[0, 0].set_title("Horizontal position std")
    _plot_metric(axes[0, 1], logs, "epv", "epv [m]", source="gps")
    axes[0, 1].set_title("Vertical position std")
    _plot_metric(axes[1, 0], logs, "hdop", "HDOP", source="gps")
    axes[1, 0].set_title("Horizontal dilution of precision")
    _plot_metric(axes[1, 1], logs, "vdop", "VDOP", source="gps")
    axes[1, 1].set_title("Vertical dilution of precision")
    for ax in axes.flat:
        _shade_stationary(ax, logs)
    axes[1, 0].set_xlabel("Time since armed [s]")
    axes[1, 1].set_xlabel("Time since armed [s]")

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    _interp_text(
        fig,
        "eph/epv are the receiver's own honest error estimates. Multi-band "
        "receivers report smaller eph/epv because dual-frequency correction "
        "removes ionospheric delay. DOPs are geometry-only — they drop when "
        "more sats are tracked across a wider sky footprint.",
    )
    return fig


def plot_velocity_course(logs):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Velocity & Course Accuracy", fontsize=13, fontweight="bold")

    _plot_metric(axes[0], logs, "s_variance_m_s",
                 "Speed accuracy [m/s]", source="gps")
    axes[0].set_title("s_variance_m_s — receiver-reported speed accuracy")
    _plot_metric(axes[1], logs, "c_variance_rad",
                 "Course accuracy [rad]", source="gps")
    axes[1].set_title("c_variance_rad — receiver-reported heading/course accuracy")
    axes[1].set_xlabel("Time since armed [s]")
    for ax in axes:
        _shade_stationary(ax, logs)

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    _interp_text(
        fig,
        "Velocity accuracy drives position-hold feel. On FPV drones this shows "
        "up as how 'locked' Loiter and Position modes feel. A multi-band fix "
        "typically shows a 2-5x drop in s_variance_m_s.",
    )
    return fig


def plot_environment(logs):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Environment / Signal Health", fontsize=13, fontweight="bold")

    _plot_metric(axes[0], logs, "noise_per_ms",
                 "Noise per ms", source="gps")
    axes[0].set_title("Receiver background noise counter — flat trace = quiet RF environment")
    _plot_metric(axes[1], logs, "jamming_indicator",
                 "Jamming indicator", source="gps")
    axes[1].set_title("0 = OK. Climbs if RF interference detected.")
    _plot_metric(axes[2], logs, "automatic_gain_control",
                 "AGC", source="gps")
    axes[2].set_title("Automatic gain control — receiver's attempt to compensate for SNR")
    axes[2].set_xlabel("Time since armed [s]")
    for ax in axes:
        _shade_stationary(ax, logs)

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    _interp_text(
        fig,
        "Sanity check: if these differ between logs, the sky / RF environment "
        "differed too, and other deltas should be interpreted with that caveat. "
        "Ideally the environment is quiet and comparable across both flights.",
    )
    return fig


def plot_ekf_innovations(logs):
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("EKF GNSS Innovations", fontsize=13, fontweight="bold")

    # Rows: pos, vel, hgt
    # Cols: innovation (or magnitude), test_ratio
    any_data = False

    for row, (key, src_attr) in enumerate([
        ("Position", "gnss_pos"),
        ("Velocity", "gnss_vel"),
        ("Height",   "gnss_hgt"),
    ]):
        ax_innov, ax_ratio = axes[row]
        ax_innov.set_ylabel(f"{key} innov")
        ax_ratio.set_ylabel(f"{key} test_ratio")

        plotted_here = False
        for i, ld in enumerate(logs):
            bag = getattr(ld, src_attr)
            if not bag:
                continue
            t = ld.tshift(bag["time_s"])
            style = _LOG_STYLE[i]

            inn_key = ("innovation_mag" if "innovation_mag" in bag
                       else "innovation" if "innovation" in bag else None)
            if inn_key is not None:
                ax_innov.plot(t, bag[inn_key], color=style["color"],
                              alpha=0.7, linewidth=0.7, label=ld.label)
                plotted_here = True

            ratio_key = ("test_ratio_max" if "test_ratio_max" in bag
                         else "test_ratio" if "test_ratio" in bag else None)
            if ratio_key is not None:
                ax_ratio.plot(t, bag[ratio_key], color=style["color"],
                              alpha=0.7, linewidth=0.7, label=ld.label)

            # Mark rejections
            if "innovation_rejected" in bag:
                rej = bag["innovation_rejected"].astype(bool)
                if rej.any() and ratio_key is not None:
                    ax_ratio.scatter(t[rej], bag[ratio_key][rej],
                                     s=6, color=style["color"], zorder=5)
                    plotted_here = True

        if plotted_here:
            any_data = True
            ax_innov.legend(fontsize=8)
            ax_ratio.axhline(1.0, color="red", linewidth=0.6, linestyle="--",
                             alpha=0.6)
            _shade_stationary(ax_innov, logs)
            _shade_stationary(ax_ratio, logs)
        ax_innov.grid(True, alpha=0.3)
        ax_ratio.grid(True, alpha=0.3)

    axes[2, 0].set_xlabel("Time since armed [s]")
    axes[2, 1].set_xlabel("Time since armed [s]")

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    if not any_data:
        _missing_banner(fig, "estimator_aid_src_gnss_* topics missing in this log")
    _interp_text(
        fig,
        "Smaller innovations mean GPS agrees with the IMU. Rejections (dots) "
        "happen when test_ratio > 1.0. A cleaner fix should show fewer spikes "
        "and a tighter distribution around zero. Missing panels = topic absent.",
    )
    return fig


def plot_drift(logs):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Stationary Drift (Money Plot)", fontsize=13, fontweight="bold")

    any_data = False
    for i, ld in enumerate(logs):
        if not ld.gps_status:
            continue
        t = ld.tshift(ld.gps_status["time_s"])
        style = _LOG_STYLE[i]
        if "position_drift_rate_horizontal_m_s" in ld.gps_status:
            axes[0].plot(t, ld.gps_status["position_drift_rate_horizontal_m_s"],
                         color=style["color"], linewidth=0.9, label=ld.label)
            any_data = True
        if "position_drift_rate_vertical_m_s" in ld.gps_status:
            axes[1].plot(t, ld.gps_status["position_drift_rate_vertical_m_s"],
                         color=style["color"], linewidth=0.9, label=ld.label)

    axes[0].set_ylabel("Horiz drift rate [m/s]")
    axes[1].set_ylabel("Vert drift rate [m/s]")
    axes[1].set_xlabel("Time since armed [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        _shade_stationary(ax, logs)
        if any_data:
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    if not any_data:
        _missing_banner(fig, "estimator_gps_status topic missing in this log")
    _interp_text(
        fig,
        "With the aircraft sitting still, this is the rate at which the EKF's "
        "position estimate walks. Fix-pass criterion: horizontal drift rate "
        "should be visibly lower at the 50th percentile, not just at the peaks.",
    )
    return fig


def plot_ekf_uncertainty(logs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("Post-EKF Uncertainty (vehicle_local_position)",
                 fontsize=13, fontweight="bold")
    _plot_metric(axes[0, 0], logs, "eph", "eph [m]", source="lpos")
    axes[0, 0].set_title("Horizontal position std")
    _plot_metric(axes[0, 1], logs, "epv", "epv [m]", source="lpos")
    axes[0, 1].set_title("Vertical position std")
    _plot_metric(axes[1, 0], logs, "evh", "evh [m/s]", source="lpos")
    axes[1, 0].set_title("Horizontal velocity std")
    _plot_metric(axes[1, 1], logs, "evv", "evv [m/s]", source="lpos")
    axes[1, 1].set_title("Vertical velocity std")
    axes[1, 0].set_xlabel("Time since armed [s]")
    axes[1, 1].set_xlabel("Time since armed [s]")
    for ax in axes.flat:
        _shade_stationary(ax, logs)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    _interp_text(
        fig,
        "What the flight controller actually consumes. Position-hold, RTL, "
        "and auto-land gate on these. A cleaner GPS fix flows through to "
        "tighter post-EKF uncertainty, even though the EKF also combines "
        "inertial and other aiding data.",
    )
    return fig


def plot_raw_position_jitter(logs):
    """d(lat)/dt, d(lon)/dt, d(alt)/dt — instantaneous GPS velocity from raw position."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Raw Position Jitter (d/dt of lat/lon/alt)",
                 fontsize=13, fontweight="bold")

    any_plotted = False
    for i, ld in enumerate(logs):
        g = ld.gps
        if not g or "latitude_deg" not in g:
            continue
        t = ld.tshift(g["time_s"])
        dt = np.diff(g["time_s"])
        dt = np.where(dt <= 0, np.nan, dt)
        t_mid = (t[:-1] + t[1:]) / 2

        # Convert deg to meters at local latitude
        lat_rad = np.deg2rad(np.nanmean(g["latitude_deg"]))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(lat_rad)

        dlat_m = np.diff(g["latitude_deg"]) * m_per_deg_lat
        dlon_m = np.diff(g["longitude_deg"]) * m_per_deg_lon
        vn = dlat_m / dt
        ve = dlon_m / dt

        style = _LOG_STYLE[i]
        axes[0].plot(t_mid, vn, color=style["color"], alpha=0.7,
                     linewidth=0.6, label=ld.label)
        axes[1].plot(t_mid, ve, color=style["color"], alpha=0.7,
                     linewidth=0.6, label=ld.label)
        any_plotted = True

        if "altitude_msl_m" in g:
            valt = np.diff(g["altitude_msl_m"]) / dt
            axes[2].plot(t_mid, valt, color=style["color"], alpha=0.7,
                         linewidth=0.6, label=ld.label)

    axes[0].set_ylabel("d(lat)/dt [m/s north]")
    axes[1].set_ylabel("d(lon)/dt [m/s east]")
    axes[2].set_ylabel("d(alt)/dt [m/s up]")
    axes[2].set_xlabel("Time since armed [s]")
    for ax in axes:
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.3)
        _shade_stationary(ax, logs)
        if any_plotted:
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    if not any_plotted:
        _missing_banner(fig, "sensor_gps position missing")
    _interp_text(
        fig,
        "Time derivative of the raw lat/lon/alt stream — instantaneous velocity "
        "noise as seen by the receiver. Multipath and cycle-slip events show up "
        "here as spikes. Clean multi-band fixes give a quieter trace.",
    )
    return fig


def plot_verdict(logs):
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#fafafa")
    is_compare = len(logs) == 2

    fig.text(0.5, 0.97, "Summary & Verdict", fontsize=18, fontweight="bold",
             ha="center", va="top")

    y = 0.90

    if not is_compare:
        ld = logs[0]
        checks = _single_log_checks(ld)
        fig.text(0.08, y, f"Pass/fail checks — {ld.label}",
                 fontsize=12, fontweight="bold", va="top", color="#333333")
        y -= 0.04
        for name, passed, value, threshold in checks:
            color = "#2e7d32" if passed else "#c62828"
            mark = "PASS" if passed else "FAIL"
            line = f"[{mark}] {name}: {value}   (threshold: {threshold})"
            fig.text(0.08, y, line, fontsize=10, va="top",
                     family="monospace", color=color)
            y -= 0.028

        y -= 0.02
        fig.text(0.08, y, "Interpretation", fontsize=12, fontweight="bold",
                 va="top", color="#333333")
        y -= 0.035
        notes = _single_log_notes(ld)
        for note in notes:
            fig.text(0.08, y, f"• {note}", fontsize=9.5, va="top",
                     color="#444444")
            y -= 0.028
        return fig

    # Compare mode
    ld_a, ld_b = logs
    stats_a, stats_b = ld_a.stats, ld_b.stats
    rows = _compare_delta_rows(stats_a, stats_b)

    fig.text(0.08, y, "Δ Summary (log B vs log A)",
             fontsize=12, fontweight="bold", va="top", color="#333333")
    y -= 0.04

    headers = ["Metric", ld_a.label, ld_b.label, "Δ (B vs A)"]
    col_x = [0.08, 0.40, 0.55, 0.70]
    for h, x in zip(headers, col_x):
        fig.text(x, y, h, fontsize=10, fontweight="bold", va="top",
                 color="#222222")
    y -= 0.028

    for row in rows:
        for val, x in zip(row, col_x):
            fig.text(x, y, str(val), fontsize=9, va="top",
                     family="monospace", color="#333333")
        y -= 0.024

    # Verdict
    y -= 0.03
    fig.text(0.08, y, "Verdict", fontsize=12, fontweight="bold",
             va="top", color="#333333")
    y -= 0.035
    verdict = _build_compare_verdict(ld_a, ld_b)
    for line in verdict:
        fig.text(0.08, y, line, fontsize=10, va="top", color="#444444")
        y -= 0.028

    return fig


def _single_log_checks(ld: LogData):
    """Pass/fail bullet items for a single log."""
    s = ld.stats
    checks = []

    ft = s.get("fix_type")
    ft_med = ft["median"] if ft else None
    checks.append((
        "Median fix_type >= 3 (3D)",
        ft_med is not None and ft_med >= 3,
        "n/a" if ft_med is None else f"{ft_med:.0f}",
        ">= 3",
    ))

    eph = s.get("eph")
    eph_med = eph["median"] if eph else None
    checks.append((
        "Median eph < 2.0 m",
        eph_med is not None and not np.isnan(eph_med) and eph_med < 2.0,
        "n/a" if eph_med is None or np.isnan(eph_med) else f"{eph_med:.2f} m",
        "< 2.0 m",
    ))

    sats = s.get("satellites_used")
    sats_med = sats["median"] if sats else None
    checks.append((
        "Median satellites_used >= 10",
        sats_med is not None and sats_med >= 10,
        "n/a" if sats_med is None else f"{sats_med:.0f}",
        ">= 10",
    ))

    drift_h = s.get("position_drift_rate_horizontal_m_s")
    drift_val = drift_h["median"] if drift_h else None
    checks.append((
        "Horiz drift rate < 0.1 m/s (estimator_gps_status)",
        drift_val is not None and drift_val < 0.1,
        "n/a" if drift_val is None else f"{drift_val:.4f} m/s",
        "< 0.1 m/s",
    ))

    ttff = s.get("ttff_s")
    checks.append((
        "TTFF (fix_type >= 3) < 60 s",
        ttff is not None and ttff < 60,
        "n/a" if ttff is None else f"{ttff:.1f} s",
        "< 60 s",
    ))

    return checks


def _single_log_notes(ld: LogData):
    notes = []
    s = ld.stats
    if not s:
        notes.append("No summary stats available — sensor_gps data missing.")
        return notes

    eph = s.get("eph")
    if eph and not np.isnan(eph["median"]):
        notes.append(f"Receiver eph median {eph['median']:.2f} m "
                     f"(p05 {eph['p05']:.2f}, p95 {eph['p95']:.2f}).")
    lpos_eph = s.get("lpos_eph")
    if lpos_eph and not np.isnan(lpos_eph["median"]):
        notes.append(f"EKF eph median {lpos_eph['median']:.2f} m — "
                     "this is what position-hold / RTL actually consume.")
    sats = s.get("satellites_used")
    if sats:
        notes.append(f"Satellites used: median {sats['median']:.0f} "
                     f"(range {sats['p05']:.0f}–{sats['p95']:.0f}).")
    jam = s.get("jamming_indicator")
    if jam and jam["p95"] > 50:
        notes.append(f"Jamming indicator p95={jam['p95']:.0f} — "
                     "elevated, RF environment was not quiet.")
    return notes


def _compare_delta_rows(stats_a, stats_b):
    rows = []
    specs = [
        ("satellites_used", "Satellites used (median)", "{:.0f}", False, False),
        ("ttff_s",          "TTFF (fix>=3) [s]",        "{:.1f}", True, False),
        ("fix_type",        "Fix type (median)",        "{:.0f}", False, False),
        ("eph",             "Receiver eph [m]",         "{:.2f}", True, True),
        ("epv",             "Receiver epv [m]",         "{:.2f}", True, True),
        ("hdop",            "HDOP",                     "{:.2f}", True, True),
        ("vdop",            "VDOP",                     "{:.2f}", True, True),
        ("s_variance_m_s",  "Speed accuracy [m/s]",     "{:.3f}", True, True),
        ("lpos_eph",        "EKF eph [m]",              "{:.2f}", True, True),
        ("lpos_evh",        "EKF evh [m/s]",            "{:.3f}", True, True),
        ("position_drift_rate_horizontal_m_s",
                            "Horiz drift [m/s]",        "{:.4f}", True, True),
        ("position_drift_rate_vertical_m_s",
                            "Vert drift [m/s]",         "{:.4f}", True, True),
        ("noise_per_ms",    "Noise/ms (median)",        "{:.0f}", True, False),
        ("jamming_indicator", "Jamming indicator",      "{:.0f}", True, False),
    ]
    for key, label, fmt, lower_better, pct in specs:
        a = stats_a.get(key)
        b = stats_b.get(key)
        if key == "ttff_s":
            av = a if a is not None else None
            bv = b if b is not None else None
        else:
            av = a["median"] if a else None
            bv = b["median"] if b else None
        cells = [label]
        for v in (av, bv):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("—")
            else:
                cells.append(fmt.format(v))
        cells.append(_format_delta(av, bv, lower_is_better=lower_better, pct=pct))
        rows.append(cells)
    return rows


def _build_compare_verdict(ld_a, ld_b):
    """Pack a few sentences summarizing the comparison."""
    lines = []
    sa, sb = ld_a.stats, ld_b.stats

    def med(s, key):
        trip = s.get(key)
        return trip["median"] if trip else None

    sats_delta = None
    if med(sa, "satellites_used") is not None and med(sb, "satellites_used") is not None:
        sats_delta = med(sb, "satellites_used") - med(sa, "satellites_used")

    drift_delta_pct = None
    da = med(sa, "position_drift_rate_horizontal_m_s")
    db = med(sb, "position_drift_rate_horizontal_m_s")
    if da and db and da > 0:
        drift_delta_pct = (db - da) / da * 100

    eph_delta_pct = None
    ea = med(sa, "eph")
    eb = med(sb, "eph")
    if ea and eb and not np.isnan(ea) and not np.isnan(eb) and ea > 0:
        eph_delta_pct = (eb - ea) / ea * 100

    parts = []
    if sats_delta is not None and abs(sats_delta) >= 1:
        direction = "more" if sats_delta > 0 else "fewer"
        parts.append(f"log B tracked {abs(sats_delta):.0f} {direction} satellites")
    if drift_delta_pct is not None and abs(drift_delta_pct) >= 1:
        direction = "reduction" if drift_delta_pct < 0 else "increase"
        parts.append(f"{abs(drift_delta_pct):.0f}% {direction} in horizontal drift rate")
    if eph_delta_pct is not None and abs(eph_delta_pct) >= 1:
        direction = "reduction" if eph_delta_pct < 0 else "increase"
        parts.append(f"{abs(eph_delta_pct):.0f}% {direction} in receiver eph")

    if parts:
        lines.append("Log B vs A: " + ", ".join(parts) + ".")
    else:
        lines.append("Log B vs A: no material change in headline metrics.")

    # Caveats
    noise_a = med(sa, "noise_per_ms")
    noise_b = med(sb, "noise_per_ms")
    if noise_a and noise_b and noise_a > 0:
        diff = abs(noise_b - noise_a) / noise_a * 100
        if diff > 20:
            lines.append("Caveat: noise_per_ms differs by >20% between flights — "
                         "RF environments were not comparable.")

    jam_a = sa.get("jamming_indicator", {}).get("p95")
    jam_b = sb.get("jamming_indicator", {}).get("p95")
    if (jam_a is not None and jam_a > 100) or (jam_b is not None and jam_b > 100):
        lines.append("Caveat: jamming_indicator p95 > 100 in at least one log — "
                     "interference may have been present.")

    return lines


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def generate_summary_text(logs):
    lines = []
    lines.append("=" * 72)
    lines.append("GPS SIGNAL QUALITY ANALYSIS SUMMARY")
    lines.append("=" * 72)

    for i, ld in enumerate(logs):
        lines.append("")
        lines.append(f"Log {chr(ord('A') + i)}: {ld.label}")
        lines.append(f"  Path: {ld.path}")
        lines.append(f"  Duration: {ld.duration_s:.1f} s")
        t0, t1 = ld.phases.get("armed_start_s"), ld.phases.get("armed_end_s")
        if t0 is not None:
            lines.append(f"  Armed: {t0:.1f}s - {t1:.1f}s")
        s0, s1 = ld.phases.get("stationary_start_s"), ld.phases.get("stationary_end_s")
        if s0 is not None:
            lines.append(f"  Stationary window: {s0:.1f}s - {s1:.1f}s "
                         f"({s1 - s0:.1f}s used for stats)")
        else:
            lines.append("  Stationary window: not detected")
        lines.append(f"  GPS topic: {ld.gps.get('topic', 'MISSING')} "
                     f"(multi_id {ld.gps.get('multi_id', '?')})")

    # Stats table
    lines.append("")
    lines.append("-" * 72)
    lines.append("SUMMARY STATISTICS (stationary window median)")
    lines.append("-" * 72)
    header = "  Metric".ljust(42)
    for ld in logs:
        header += f"{ld.label[:16]:>17}"
    if len(logs) == 2:
        header += "      Δ"
    lines.append(header)

    spec = [
        ("satellites_used",      "Satellites used",      "{:.0f}"),
        ("ttff_s",               "TTFF (fix>=3) [s]",    "{:.1f}"),
        ("fix_type",             "Fix type",             "{:.0f}"),
        ("eph",                  "Receiver eph [m]",     "{:.2f}"),
        ("epv",                  "Receiver epv [m]",     "{:.2f}"),
        ("hdop",                 "HDOP",                 "{:.2f}"),
        ("vdop",                 "VDOP",                 "{:.2f}"),
        ("s_variance_m_s",       "Speed accuracy [m/s]", "{:.3f}"),
        ("c_variance_rad",       "Course accuracy [rad]","{:.4f}"),
        ("noise_per_ms",         "Noise/ms",             "{:.0f}"),
        ("jamming_indicator",    "Jamming indicator",    "{:.0f}"),
        ("automatic_gain_control","AGC",                 "{:.0f}"),
        ("lpos_eph",             "EKF eph [m]",          "{:.2f}"),
        ("lpos_epv",             "EKF epv [m]",          "{:.2f}"),
        ("lpos_evh",             "EKF evh [m/s]",        "{:.3f}"),
        ("lpos_evv",             "EKF evv [m/s]",        "{:.3f}"),
        ("position_drift_rate_horizontal_m_s",
                                  "Horiz drift [m/s]",    "{:.4f}"),
        ("position_drift_rate_vertical_m_s",
                                  "Vert drift [m/s]",     "{:.4f}"),
    ]
    for key, label, fmt in spec:
        row = "  " + label.ljust(40)
        vals = []
        for ld in logs:
            v = ld.stats.get(key)
            if key == "ttff_s":
                vals.append(v)
            else:
                vals.append(v["median"] if v else None)
        for v in vals:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row += "              n/a"
            else:
                row += f"{fmt.format(v):>17}"
        if len(logs) == 2 and vals[0] is not None and vals[1] is not None \
                and not np.isnan(vals[0]) and not np.isnan(vals[1]):
            diff = vals[1] - vals[0]
            row += f"   {diff:+.3g}"
        lines.append(row)

    if len(logs) == 2:
        lines.append("")
        lines.append("-" * 72)
        lines.append("VERDICT")
        lines.append("-" * 72)
        for line in _build_compare_verdict(logs[0], logs[1]):
            lines.append("  " + line)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PX4 GPS signal quality analyzer")
    parser.add_argument("log_a", help="Path to first .ulg file")
    parser.add_argument("log_b", nargs="?", default=None,
                        help="Optional second .ulg file to enable compare mode")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: dir of log_a)")
    parser.add_argument("--label-a", default=None,
                        help="Label for log A (default: log_a filename)")
    parser.add_argument("--label-b", default=None,
                        help="Label for log B (default: log_b filename)")
    args = parser.parse_args()

    if not os.path.isfile(args.log_a):
        print(f"Error: file not found: {args.log_a}", file=sys.stderr)
        sys.exit(1)
    if args.log_b is not None and not os.path.isfile(args.log_b):
        print(f"Error: file not found: {args.log_b}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.log_a))
    os.makedirs(output_dir, exist_ok=True)

    logs = [load_log(args.log_a, args.label_a)]
    if args.log_b:
        logs.append(load_log(args.log_b, args.label_b))

    for ld in logs:
        compute_summary_stats(ld)

    print("\nGenerating PDF...")
    figures = [
        plot_cover_page(logs),
        plot_satellites_fix(logs),
        plot_receiver_accuracy(logs),
        plot_velocity_course(logs),
        plot_environment(logs),
        plot_ekf_innovations(logs),
        plot_drift(logs),
        plot_ekf_uncertainty(logs),
        plot_raw_position_jitter(logs),
        plot_verdict(logs),
    ]

    pdf_name = ("gps_signal_quality_compare.pdf" if len(logs) == 2
                else "gps_signal_quality.pdf")
    pdf_path = os.path.join(output_dir, pdf_name)
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  Saved: {pdf_path}")

    summary = generate_summary_text(logs)
    txt_path = os.path.join(output_dir, "gps_signal_quality.txt")
    with open(txt_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {txt_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
