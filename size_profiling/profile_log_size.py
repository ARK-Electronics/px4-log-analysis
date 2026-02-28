#!/usr/bin/env python3
"""PX4 ULog size profiler — shows which topics consume the most log space."""

import argparse
import os
import sys

from pyulog import ULog

# SDLOG_PROFILE bitmask definitions (PX4 parameter)
SDLOG_PROFILES = {
    0: "Default",
    1: "Estimator replay (EKF2)",
    2: "Thermal calibration",
    3: "System identification",
    4: "High rate",
    5: "Debug",
    6: "Sensor comparison",
    7: "Computer Vision and Avoidance",
}

# Rough topic-to-category mapping for the summary breakdown.
# Topics not matching any prefix fall into "other".
CATEGORY_PREFIXES = {
    "control": [
        "actuator_motors", "actuator_servos", "actuator_outputs",
        "vehicle_angular_velocity", "vehicle_thrust_setpoint",
        "vehicle_torque_setpoint", "vehicle_rates_setpoint",
        "vehicle_attitude_setpoint", "rate_ctrl_status",
        "manual_control_setpoint",
    ],
    "estimator": [
        "estimator_", "vehicle_local_position", "vehicle_global_position",
        "vehicle_attitude",
    ],
    "sensor": [
        "sensor_combined", "sensor_accel", "sensor_gyro", "sensor_mag",
        "sensor_baro", "sensor_gps", "vehicle_imu", "vehicle_magnetometer",
        "vehicle_air_data",
    ],
    "battery": ["battery_status"],
    "system": [
        "cpuload", "logger_status", "heater_status",
        "vehicle_status", "commander_state",
    ],
    "navigation": [
        "position_setpoint_triplet", "vehicle_trajectory_waypoint",
        "home_position", "mission_result",
    ],
    "rc": ["input_rc", "rc_channels"],
    "ekf2": ["ekf2_timestamps"],
}


def categorize_topic(topic_name: str) -> str:
    """Return a category string for the given topic name."""
    for category, prefixes in CATEGORY_PREFIXES.items():
        for prefix in prefixes:
            if prefix.endswith("_"):
                if topic_name.startswith(prefix):
                    return category
            else:
                if topic_name == prefix:
                    return category
    return "other"


def decode_sdlog_profile(value: int) -> list[str]:
    """Decode the SDLOG_PROFILE bitmask into human-readable profile names."""
    active = []
    for bit, name in SDLOG_PROFILES.items():
        if value & (1 << bit):
            active.append(name)
    return active


def profile_ulog(filepath: str) -> None:
    file_size = os.path.getsize(filepath)
    ulog = ULog(filepath)

    duration_us = ulog.last_timestamp - ulog.start_timestamp
    duration_s = duration_us / 1e6

    # --- Header ---
    print(f"File:     {os.path.basename(filepath)}")
    print(f"Size:     {file_size / 1024:.1f} KB ({file_size / (1024*1024):.2f} MB)")
    print(f"Duration: {duration_s:.1f} s ({duration_s/60:.1f} min)")

    # SDLOG_PROFILE parameter
    sdlog_val = ulog.initial_parameters.get("SDLOG_PROFILE")
    if sdlog_val is not None:
        sdlog_val = int(sdlog_val)
        profiles = decode_sdlog_profile(sdlog_val)
        print(f"SDLOG_PROFILE: {sdlog_val} ({', '.join(profiles) if profiles else 'none'})")
    else:
        print("SDLOG_PROFILE: not found in log parameters")

    print()

    # --- Per-topic stats ---
    topic_stats = []
    for d in ulog.data_list:
        num_messages = len(d.data["timestamp"])
        row_bytes = sum(d.data[f].dtype.itemsize for f in d.data)
        total_bytes = row_bytes * num_messages
        rate_hz = num_messages / duration_s if duration_s > 0 else 0
        topic_stats.append({
            "topic": d.name,
            "multi_id": d.multi_id,
            "num_messages": num_messages,
            "rate_hz": rate_hz,
            "bytes_per_msg": row_bytes,
            "total_bytes": total_bytes,
            "category": categorize_topic(d.name),
        })

    topic_stats.sort(key=lambda x: x["total_bytes"], reverse=True)
    total_data_bytes = sum(t["total_bytes"] for t in topic_stats)

    # Print table
    hdr = (f"{'Topic':<45} {'ID':>3} {'Messages':>10} {'Rate Hz':>8} "
           f"{'B/msg':>6} {'Total KB':>9} {'%':>6} {'Cum%':>6}")
    print(hdr)
    print("-" * len(hdr))

    cum_pct = 0.0
    for t in topic_stats:
        pct = t["total_bytes"] / total_data_bytes * 100 if total_data_bytes else 0
        cum_pct += pct
        kb = t["total_bytes"] / 1024
        print(
            f"{t['topic']:<45} {t['multi_id']:>3} {t['num_messages']:>10} "
            f"{t['rate_hz']:>8.1f} {t['bytes_per_msg']:>6} {kb:>9.1f} "
            f"{pct:>5.1f}% {cum_pct:>5.1f}%"
        )

    print()
    print(f"Total topic data: {total_data_bytes/1024:.1f} KB "
          f"(overhead/metadata ≈ {(file_size - total_data_bytes)/1024:.1f} KB)")
    print()

    # --- Category summary ---
    cat_totals: dict[str, int] = {}
    for t in topic_stats:
        cat_totals[t["category"]] = cat_totals.get(t["category"], 0) + t["total_bytes"]

    cat_sorted = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Category':<20} {'KB':>9} {'%':>6}")
    print("-" * 37)
    for cat, nbytes in cat_sorted:
        pct = nbytes / total_data_bytes * 100 if total_data_bytes else 0
        print(f"{cat:<20} {nbytes/1024:>9.1f} {pct:>5.1f}%")

    print(f"{'TOTAL':<20} {total_data_bytes/1024:>9.1f} 100.0%")


def main():
    parser = argparse.ArgumentParser(
        description="Profile PX4 ULog file size by topic."
    )
    parser.add_argument("ulog_file", help="Path to a .ulg file")
    args = parser.parse_args()

    if not os.path.isfile(args.ulog_file):
        print(f"Error: file not found: {args.ulog_file}", file=sys.stderr)
        sys.exit(1)

    profile_ulog(args.ulog_file)


if __name__ == "__main__":
    main()
