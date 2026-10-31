# FastF1 data probe (PLAN T6)

Generated 2026-10-31T16:56:48Z by `scripts/probe_fastf1.py` on local (Python 3.12.13, FastF1 3.9.0, pandas 2.3.3). All numbers are `Observed` on the races below.

## Races

| Event | Role | Fold | Load s | Download MB | Laps | Telemetry missing | Dry | O1 keep |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| 2023_barcelona (Spanish Grand Prix) | dev | F3 | 5.3 | 0.0 | 1312 | 0.000 | True | True |
| 2024_bahrain (Bahrain Grand Prix) | dev | F1 | 4.5 | 0.0 | 1129 | 0.000 | True | True |
| 2025_suzuka (Japanese Grand Prix) | holdout | F1 | 4.4 | 0.0 | 1059 | 0.000 | True | True |
| 2024_suzuka (Japanese Grand Prix) | dev | F1 | 28.6 | 118.4 | 907 | 0.000 | True | True |
| 2024_miami (Miami Grand Prix) | dev | F2 | 31.1 | 107.9 | 1111 | 0.000 | True | True |
| 2023_silverstone (British Grand Prix) | dev | F3 | 27.4 | 140.0 | 971 | 0.000 | True | True |
| 2024_spielberg (Austrian Grand Prix) | dev | F3 | 28.2 | 102.6 | 1405 | 0.000 | False | False |
| 2023_monza (Italian Grand Prix) | dev | F4 | 32.7 | 106.1 | 957 | 0.000 | True | True |
| 2023_budapest (Hungarian Grand Prix) | dev | F4 | 28.4 | 109.6 | 1252 | 0.000 | True | True |
| 2023_singapore (Singapore Grand Prix) | dev | F5 | 37.4 | 122.4 | 1088 | 0.000 | True | True |
| 2023_yas_marina (Abu Dhabi Grand Prix) | dev | F6 | 27.8 | 105.0 | 1157 | 0.000 | True | True |

**E7 projection**: mean download 82.9 MB and 23.2 s per race; 70 events in the config give about 5.80 GB and 0.45 h. A download of 0 MB means the cache was already warm for that race.

## Column availability (missing fraction)

### laps

| Column | 2023_barcelona | 2024_bahrain | 2025_suzuka | 2024_suzuka | 2024_miami | 2023_silverstone | 2024_spielberg | 2023_monza | 2023_budapest | 2023_singapore | 2023_yas_marina |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Driver` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `DriverNumber` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `LapTime` | 0.000 | 0.001 | 0.000 | 0.029 | 0.001 | 0.001 | 0.000 | 0.009 | 0.000 | 0.005 | 0.000 |
| `LapNumber` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `Stint` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `Compound` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `TyreLife` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `FreshTyre` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `TrackStatus` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `IsAccurate` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `PitInTime` | 0.967 | 0.962 | 0.980 | 0.939 | 0.975 | 0.973 | 0.967 | 0.973 | 0.969 | 0.977 | 0.967 |
| `PitOutTime` | 0.966 | 0.962 | 0.980 | 0.940 | 0.975 | 0.975 | 0.967 | 0.974 | 0.971 | 0.976 | 0.968 |
| `Deleted` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `LapStartTime` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `Time` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `SpeedI1` | 0.000 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 | 0.000 | 0.009 | 0.000 | 0.004 | 0.000 |
| `SpeedI2` | 0.000 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 | 0.000 | 0.009 | 0.000 | 0.004 | 0.000 |
| `SpeedFL` | 0.033 | 0.038 | 0.020 | 0.063 | 0.026 | 0.028 | 0.032 | 0.037 | 0.031 | 0.027 | 0.033 |
| `SpeedST` | 0.000 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 | 0.000 | 0.009 | 0.000 | 0.005 | 0.000 |

### car

| Column | 2023_barcelona | 2024_bahrain | 2025_suzuka | 2024_suzuka | 2024_miami | 2023_silverstone | 2024_spielberg | 2023_monza | 2023_budapest | 2023_singapore | 2023_yas_marina |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SessionTime` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `Speed` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `Throttle` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `Brake` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### pos

| Column | 2023_barcelona | 2024_bahrain | 2025_suzuka | 2024_suzuka | 2024_miami | 2023_silverstone | 2024_spielberg | 2023_monza | 2023_budapest | 2023_singapore | 2023_yas_marina |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SessionTime` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `X` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `Y` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### weather

| Column | 2023_barcelona | 2024_bahrain | 2025_suzuka | 2024_suzuka | 2024_miami | 2023_silverstone | 2024_spielberg | 2023_monza | 2023_budapest | 2023_singapore | 2023_yas_marina |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Time` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `AirTemp` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `TrackTemp` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `Rainfall` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Exclusions, traffic, pit lane

Traffic shares, pit-lane times, and stint slopes come from development races only; holdout races report exclusion counts and availability.

- **2023_barcelona** (dev): exclusions {'scored': 1021, 'traffic': 177, 'out_lap': 43, 'in_lap': 42, 'race_lap_1': 20, 'deleted': 9}. Share of green timed laps with gap ahead <0.5 s: 0.063, <1.0 s: 0.165, <1.5 s: 0.272, <2.0 s: 0.373. Pit-lane time median 22.58 s, IQR [22.24, 23.37] s over 43 stops. Median raw within-stint slope 0.013 s/lap over 56 stints. Drivers without car data: none.
- **2024_bahrain** (dev): exclusions {'scored': 818, 'traffic': 170, 'out_lap': 42, 'in_lap': 42, 'race_lap_1': 20, 'not_green': 18, 'deleted': 18, 'no_lap_time': 1}. Share of green timed laps with gap ahead <0.5 s: 0.034, <1.0 s: 0.175, <1.5 s: 0.267, <2.0 s: 0.342. Pit-lane time median 24.77 s, IQR [24.51, 25.12] s over 42 stops. Median raw within-stint slope 0.041 s/lap over 50 stints. Drivers without car data: none.
- **2025_suzuka** (holdout): exclusions {'scored': 800, 'traffic': 189, 'out_lap': 21, 'in_lap': 21, 'race_lap_1': 20, 'deleted': 8}. Drivers without car data: none.
- **2024_suzuka** (dev): exclusions {'scored': 613, 'traffic': 148, 'in_lap': 37, 'out_lap': 35, 'no_lap_time': 26, 'not_green': 21, 'race_lap_1': 18, 'deleted': 8, 'inaccurate': 1}. Share of green timed laps with gap ahead <0.5 s: 0.078, <1.0 s: 0.202, <1.5 s: 0.309, <2.0 s: 0.400. Pit-lane time median 23.32 s, IQR [23.11, 23.81] s over 36 stops. Median raw within-stint slope 0.052 s/lap over 45 stints. Drivers without car data: none.
- **2024_miami** (dev): exclusions {'scored': 605, 'traffic': 310, 'not_green': 100, 'in_lap': 28, 'out_lap': 28, 'race_lap_1': 20, 'deleted': 19, 'no_lap_time': 1}. Share of green timed laps with gap ahead <0.5 s: 0.095, <1.0 s: 0.351, <1.5 s: 0.469, <2.0 s: 0.590. Pit-lane time median 22.65 s, IQR [22.33, 23.04] s over 28 stops. Median raw within-stint slope -0.029 s/lap over 39 stints. Drivers without car data: none.
- **2023_silverstone** (dev): exclusions {'scored': 542, 'traffic': 245, 'not_green': 98, 'in_lap': 26, 'out_lap': 24, 'race_lap_1': 20, 'deleted': 15, 'no_lap_time': 1}. Share of green timed laps with gap ahead <0.5 s: 0.056, <1.0 s: 0.322, <1.5 s: 0.456, <2.0 s: 0.557. Pit-lane time median 28.76 s, IQR [28.49, 29.25] s over 24 stops. Median raw within-stint slope -0.027 s/lap over 35 stints. Drivers without car data: none.
- **2024_spielberg** (dev): exclusions {'scored': 1008, 'traffic': 232, 'not_green': 46, 'out_lap': 45, 'in_lap': 44, 'race_lap_1': 20, 'deleted': 10}. Share of green timed laps with gap ahead <0.5 s: 0.041, <1.0 s: 0.200, <1.5 s: 0.283, <2.0 s: 0.378. Pit-lane time median 21.67 s, IQR [21.43, 22.37] s over 45 stops. Median raw within-stint slope 0.049 s/lap over 57 stints. Drivers without car data: none.
- **2023_monza** (dev): exclusions {'scored': 586, 'traffic': 278, 'in_lap': 26, 'out_lap': 25, 'race_lap_1': 19, 'deleted': 14, 'no_lap_time': 9}. Share of green timed laps with gap ahead <0.5 s: 0.094, <1.0 s: 0.330, <1.5 s: 0.425, <2.0 s: 0.496. Pit-lane time median 24.41 s, IQR [24.23, 24.83] s over 25 stops. Median raw within-stint slope 0.022 s/lap over 34 stints. Drivers without car data: none.
- **2023_budapest** (dev): exclusions {'scored': 976, 'traffic': 148, 'in_lap': 38, 'out_lap': 36, 'race_lap_1': 20, 'not_green': 18, 'deleted': 16}. Share of green timed laps with gap ahead <0.5 s: 0.025, <1.0 s: 0.134, <1.5 s: 0.211, <2.0 s: 0.307. Pit-lane time median 21.98 s, IQR [21.64, 22.75] s over 36 stops. Median raw within-stint slope 0.019 s/lap over 48 stints. Drivers without car data: none.
- **2023_singapore** (dev): exclusions {'scored': 554, 'traffic': 275, 'not_green': 178, 'in_lap': 25, 'out_lap': 24, 'race_lap_1': 18, 'deleted': 9, 'no_lap_time': 5}. Share of green timed laps with gap ahead <0.5 s: 0.066, <1.0 s: 0.330, <1.5 s: 0.512, <2.0 s: 0.611. Pit-lane time median 30.57 s, IQR [29.77, 32.40] s over 24 stops. Median raw within-stint slope -0.019 s/lap over 38 stints. Drivers without car data: none.
- **2023_yas_marina** (dev): exclusions {'scored': 811, 'traffic': 221, 'in_lap': 38, 'out_lap': 36, 'deleted': 31, 'race_lap_1': 20}. Share of green timed laps with gap ahead <0.5 s: 0.038, <1.0 s: 0.232, <1.5 s: 0.351, <2.0 s: 0.467. Pit-lane time median 21.91 s, IQR [21.61, 22.19] s over 37 stops. Median raw within-stint slope 0.002 s/lap over 47 stints. Drivers without car data: none.

## Warm-up visibility (S4 on real stints)

Development races only, scored laps after all exclusions, stints with at least 3 scored laps. `y = lap_time - phi_mid * (L - race_lap)`. Stint lap is the 1-based position in the stint over all laps before exclusions, so stint lap 1 is the out-lap (or race lap 1) and is normally excluded. Gain is `y` of the first scored lap minus the minimum `y` of the stint. A dip means the fastest scored lap comes strictly after the first scored lap, at stint lap 2-8, with a gain above 0.10 s, and the last scored lap is slower (PRD S4, PLAN T3 test 3). A stint whose first scored lap is its fastest is plain wear and never counts as a dip. The all-stint gain columns show how much early improvement exists regardless of the dip rule.

| Event | Stints | Dip stints | Dip share | Dip gain median s | Dip gain IQR s | All-stint gain median s | All-stint gain IQR s | Share gain > 0.10 s | First scored stint lap (median; counts) | First scored TyreLife median |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | :--- | ---: | :--- | ---: |
| 2023_barcelona | 58 | 18 | 0.310 | 0.311 | [0.178, 0.522] | 0.000 | [0.000, 0.327] | 0.431 | 2.0; 2: 40, 3: 6, 4: 5, 5: 2, 6: 2, 7: 1, 8: 1, 11: 1 | 2.0 |
| 2024_bahrain | 56 | 19 | 0.339 | 0.473 | [0.219, 0.548] | 0.000 | [0.000, 0.258] | 0.375 | 2.0; 2: 39, 3: 7, 4: 2, 5: 2, 6: 3, 7: 2, 8: 1 | 2.0 |
| 2024_suzuka | 47 | 10 | 0.213 | 0.572 | [0.392, 1.038] | 0.000 | [0.000, 0.058] | 0.234 | 2.0; 2: 29, 3: 9, 4: 4, 5: 3, 6: 1, 7: 1 | 2.0 |
| 2024_miami | 42 | 13 | 0.310 | 0.345 | [0.236, 0.514] | 0.328 | [0.067, 0.742] | 0.690 | 4.0; 2: 16, 3: 3, 4: 5, 5: 3, 7: 3, 8: 1, 9: 1, 10: 2, 12: 4, 13: 1, 14: 1, 15: 1, 17: 1 | 4.0 |
| 2023_silverstone | 40 | 3 | 0.075 | 0.154 | [0.145, 0.413] | 0.056 | [0.000, 0.239] | 0.425 | 5.5; 2: 12, 3: 3, 4: 3, 5: 2, 6: 5, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 1, 13: 1, 18: 1 | 5.5 |
| 2024_spielberg | 57 | 17 | 0.298 | 0.211 | [0.196, 0.352] | 0.000 | [0.000, 0.196] | 0.333 | 2.0; 2: 42, 3: 3, 4: 2, 5: 3, 6: 4, 7: 1, 10: 1, 12: 1 | 2.0 |
| 2023_monza | 37 | 8 | 0.216 | 0.437 | [0.243, 0.548] | 0.000 | [0.000, 0.103] | 0.270 | 2.0; 2: 23, 3: 4, 4: 3, 6: 1, 8: 1, 10: 1, 11: 1, 13: 1, 15: 1, 16: 1 | 5.0 |
| 2023_budapest | 49 | 19 | 0.388 | 0.612 | [0.279, 0.868] | 0.066 | [0.000, 0.495] | 0.469 | 2.0; 2: 37, 3: 3, 4: 2, 6: 2, 7: 2, 8: 1, 9: 1, 11: 1 | 5.0 |
| 2023_singapore | 40 | 8 | 0.200 | 1.258 | [0.915, 1.571] | 0.706 | [0.209, 1.319] | 0.775 | 4.0; 2: 6, 3: 9, 4: 6, 5: 5, 6: 3, 7: 3, 8: 3, 10: 1, 14: 1, 15: 1, 16: 1, 19: 1 | 5.0 |
| 2023_yas_marina | 50 | 16 | 0.320 | 0.441 | [0.254, 0.880] | 0.030 | [0.000, 0.254] | 0.420 | 2.0; 2: 29, 3: 8, 4: 2, 5: 1, 6: 2, 7: 1, 8: 1, 9: 2, 11: 1, 12: 1, 13: 1, 15: 1 | 2.0 |

## O2 heat-load proxy candidates

Scored laps of development events ['2023_barcelona', '2023_budapest', '2023_monza', '2023_silverstone', '2023_singapore', '2023_yas_marina', '2024_bahrain', '2024_miami', '2024_spielberg', '2024_suzuka'] only (holdout races excluded), `phi = 0.06`, traffic threshold 1.0 s. `energy_index` uses positions smoothed within each lap (Savitzky-Golay, 7 samples, cubic) and drops samples above 80 m/s^2 as glitches.

Columns, all over stints with at least 3 scored laps:

- `between share`: share of the proxy's variance over laps that lies between event means (an ICC-like ratio, 0 to 1). A circuit-level heat index should be high here.
- `within std`, `across std`, `l2_ratio`: mean within-stint std, std of per-event means, and their ratio (limit L2 is refuted above 0.1).
- `noise std`: mean within-stint std after removing a linear race-lap trend. `drift/lap`: pooled within-stint slope per race lap.
- `corr`: Pearson correlation of stint-centered proxy and stint-centered `y`. `partial corr`: the same after removing a linear race-lap trend from both. These are leakage warnings, not reasons to pick a proxy: a proxy that moves with lap time (trap speed is higher on faster laps, a slow lap holds more braking time) feeds the target back into the model input. `leak?` is set when |corr| or |partial corr| is above 0.3.

| Proxy | Laps | Stints | Events | between share | within std | across std | l2_ratio | noise std | drift/lap | corr | partial corr | leak? | score |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| `speed_trap` | 7501 | 476 | 10 | 0.935 | 2.287 | 16.637 | 0.137 | 1.941 | 0.0574 | -0.074 | -0.230 | no | 0.865 |
| `energy_index` | 7501 | 476 | 10 | 0.920 | 0.529 | 2.461 | 0.215 | 0.494 | -0.0005 | -0.204 | -0.221 | no | 0.733 |
| `brake_fraction_weighted` | 7501 | 476 | 10 | 0.764 | 0.011 | 0.033 | 0.332 | 0.010 | -0.0001 | 0.143 | 0.191 | no | 0.655 |

Mechanical suggestion: `speed_trap`. Rule: largest `score = between share x (1 - |corr|)`, so proxies are ranked by circuit-level separation and discounted for tracking lap time. This is input to the O2 decision, not the decision. With few dev events the between-event statistics rest on very few means; run `--preset extended` (or add `--race` entries) to tighten them.

## Next steps

- O1: drop races that are not dry or exceed 20% missing telemetry; extend the probe if needed.
- O2: choose the proxy and normalization from the table above and record the reason.
- O3: check `phi` and the traffic threshold against the traffic shares and slopes above.
- O8: compare pit-lane medians with the `per_circuit_median_iqr` plan.
- T5: use the warm-up visibility table to judge whether real stints show the S4 dip before fitting.
- O5: user review before any model results are seen.
