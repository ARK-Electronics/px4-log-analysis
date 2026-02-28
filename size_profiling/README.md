# ULog Size Profiler

Profiles a PX4 `.ulg` log file and reports which topics consume the most space.

## Usage

```bash
python3 profile_log_size.py <path_to.ulg>
```

## Output

1. **Header** — file size, duration, and active `SDLOG_PROFILE` bitmask
2. **Topic table** — every logged topic sorted by total size (descending), with:
   - Topic name and multi_id
   - Message count and logging rate (Hz)
   - Bytes per message and total KB
   - Percentage of total data and cumulative percentage
3. **Category summary** — topics grouped into categories (control, estimator, sensor, system, etc.)

## Example

```
$ python3 profile_log_size.py ../57627df0-ae8e-445c-a643-f3077f6d8cb9.ulg

File:     57627df0-ae8e-445c-a643-f3077f6d8cb9.ulg
Size:     1536.2 KB (1.50 MB)
Duration: 162.3 s (2.7 min)
SDLOG_PROFILE: 3 (Default, Estimator replay (EKF2))

Topic                                          ID   Messages  Rate Hz  B/msg  Total KB      %   Cum%
--------------------------------------------------------------------------------------------------------------
estimator_status                                 0       1623    10.0     280     443.9  28.9%  28.9%
...
```

## Dependencies

- `pyulog` (`pip install pyulog`)
