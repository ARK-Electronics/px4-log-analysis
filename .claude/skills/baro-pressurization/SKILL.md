---
name: baro-pressurization
description: Analyze barometric pressure bias from a PX4 ULog — thrust pressurization, ground effect, thermal drift, and EKF impact. Use when the user wants to investigate baro-related altitude errors, Z velocity drift, or ground effect issues.
argument-hint: <path-to-ulg-file>
allowed-tools: Bash(python3 *)
---

# Barometric Pressure Bias Analysis

Run the baro pressurization analyzer on the given ULog file and interpret the results.

## Before running

1. Resolve the log file path:
   - First check if it already exists under `logs/` in this repo (by filename match).
   - If not found, check `~/Downloads/`.
   - If found outside the repo, catalogue it: create `logs/<log-name>/`, copy the `.ulg` there.
2. Set the output directory to the log's `logs/<log-name>/` directory.

## Command

```bash
python3 scripts/baro_pressurization.py <resolved-log-path> --output-dir <logs/log-name/>
```

## After running

1. Open the combined PDF for the user (`xdg-open <output-dir>/baro_analysis.pdf`). Then read the individual PNGs yourself with the Read tool so you can interpret them.
2. Interpret the findings:
   - **Static baro offset**: How much the baro reads above ground truth (range sensor). >2m indicates significant prop wash pressurization.
   - **Thrust-baro correlation**: r < -0.6 means throttle changes directly modulate the baro reading. This is thrust pressurization — the propellers create a pressure depression that varies with RPM.
   - **Altitude-baro correlation**: r < -0.4 means the baro error decreases with altitude AGL. This is ground effect — prop wash bouncing off the ground amplifies the pressure disturbance at low altitude.
   - **Baro error variation** (std): >0.3m during hover means significant noise is being injected into the EKF through the baro fusion.
   - **EKF innovation**: Large mean innovation and high std indicate the EKF is receiving contaminated baro data. Check the test_ratio — if always below 1.0, no innovations are being rejected despite the contamination.
   - **Ground effect flag** (`cs_gnd_effect`): Whether the EKF is activating ground effect protection. If the vehicle is hovering below `EKF2_GND_MAX_HGT` but the flag is never active, something is misconfigured.
3. Recommend parameter changes:
   - `EKF2_BARO_CTRL` — set to 0 to disable baro fusion entirely if baro is heavily contaminated and other height sources are available (GNSS, range sensor)
   - `EKF2_BARO_NOISE` — increase to reduce baro weight in the EKF (default 3.5, try 5-10 for contaminated setups)
   - `EKF2_GND_EFF_DZ` — ground effect deadzone in meters added to baro noise when below `GND_MAX_HGT` (default 4.0)
   - `EKF2_GND_MAX_HGT` — maximum AGL at which ground effect protection is active (increase if hovering higher than current value)
   - If baro pressurization is severe and consistent, consider physical mitigations: foam cover over baro sensor, relocating baro away from prop wash, or adding a static port tube.
4. If comparing flights, quantify whether parameter or physical changes reduced the baro contamination.
5. Create or update the `logs/<log-name>/README.md` catalogue entry per the convention in CLAUDE.md.

## Notes

- The script requires `distance_sensor` (range sensor) data as ground truth to compute baro error. Without it, only raw pressure/temperature trends can be analyzed.
- A large static baro offset (e.g. 5-10m) at hover is common on small multirotors — the props create a sustained pressure depression around the airframe. The EKF handles this via its baro bias state. The concern is the *variation* in this offset, not the offset itself.
- Thrust pressurization and ground effect are distinct mechanisms but often co-occur. The multivariate regression separates their contributions.
- 1 Pa of pressure change corresponds to approximately 0.083 m of altitude at sea level.
- All outputs go into the log's catalogue directory under `logs/`.
