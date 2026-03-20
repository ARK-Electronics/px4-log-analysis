# DNF RPM Frequency Offset Parameter

## Problem

Spectrogram analysis shows accel vibration energy consistently biased 3-10% above the ESC-reported motor frequency. The energy tracks RPM (confirming motor origin) but the peak sits above the notch center. Current DNF bandwidth (limited to 30 Hz) is too narrow to reach the offset peaks.

## Evidence

From log_19_UnknownDate analysis:
- 1x motor (320 Hz): peak energy at 329-354 Hz (+3% to +10%)
- 2x motor (640 Hz): peak energy at 662 Hz (+3.4%)
- 3x motor (961 Hz): peak energy at 954 Hz (-0.7%)

## Proposed PX4 Change

Add a parameter like `IMU_ACC_DNF_FREQ_MULT` (default 1.0) that scales the ESC RPM before computing the notch center frequency. A value of 1.05 would shift the notch from 320 Hz to 336 Hz, centering it on the actual energy peak.

Alternative: `IMU_ACC_DNF_FREQ_OFF` as a percentage offset (default 0, set to +5 for this case).

Also: raise `IMU_ACC_DNF_BW` parameter max from 30 Hz to at least 80-100 Hz.

## Status

WIP — discussed 2026-03-19.
