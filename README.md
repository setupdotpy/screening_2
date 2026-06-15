# GP-Guided MPC-CBF Active Sensing USV Simulation

This repository contains a standalone Python simulation for a maritime active-sensing mission with an unmanned surface vehicle (USV), a directional camera, Gaussian Process target-arrival likelihood, sample-based MPC, and CBF obstacle safety constraints.

The purpose of the simulation is to demonstrate one central idea:

```text
choose where the USV moves and where the camera points
to maximize target-detection performance
while guaranteeing safe motion around obstacles
```

The main objective is:

```text
maximize    J_detection = integral P_d(t) dt
subject to  h(p) >= 0
```

where:

- `P_d` is the directional-camera detection probability,
- `h(p)` is the obstacle CBF safety function,
- `h(p) >= 0` means the USV remains outside obstacle safety zones.

Everything in the simulation supports this objective:

- the GP map tells the USV where targets are likely to appear,
- observation poses place the USV where detection probability should be high,
- the directional camera objective keeps the target centered in the field of view,
- MPC selects motion and camera commands that improve detection,
- CBF constraints reject unsafe motions before they are applied.

## Simulation Scenario

The simulation represents a USV maritime surveillance mission in a `10 km x 10 km` operating area:

```text
x in [-5000, 5000] m
y in [0, 10000] m
```

The USV starts near the bottom of the domain:

```text
x = 0 m
y = 500 m
heading psi = pi/2
camera pan theta_c = 0
```

The nominal route is the vertical line:

```text
x = 0
```

The mission is not simple waypoint tracking. The USV must actively position itself and orient its camera so that the accumulated detection probability is high, while all executed motion remains safe with respect to restricted zones.

Scheduled observation windows are included to make the sensing task realistic: the USV cannot simply observe whenever it wants. It must reach useful viewpoints in time, keep the target in view, and maintain safety throughout the transit, observation, and route-return phases.

### Target-Arrival Model

The code first creates synthetic historical target-arrival observations from latent simulation-only arrival sources. These are not mission targets. They are used only to generate historical data.

A Gaussian Process model is fit over the map to estimate a normalized spatial target-arrival likelihood:

```text
lambda_bar(q) in [0, 1]
```

The mission targets are then selected automatically from high-likelihood GP regions. This means the final target locations are outputs of the learned spatial likelihood map, not manually fixed target coordinates.

### Mission Timing

The mission has four ordered targets, `T0` through `T3`, with scheduled observation windows:

```text
T0: 11:00-12:00
T1: 14:00-14:45
T2: 15:30-16:15
T3: 17:00-17:30
```

The vehicle must reach an observation pose and accumulate enough expected observation time during the target's window. These windows are not the main objective by themselves; they create a realistic constraint on the detection objective.

Observation progress is probabilistic:

```text
observed_time_i += P_d(q_i, X) * dt
```

A target is completed when:

```text
observed_time_i >= REQUIRED_OBSERVATION_TIME
```

Thus, the mission success condition is tied directly to detection probability:

```text
higher P_d -> faster observation progress
lower P_d  -> slower progress or missed target
```

### Observation Pose Planning

For each GP-selected target, the code searches for an observation pose on a ring around the target:

```text
distance(p_obs, target) = R_BEST
```

where:

```text
R_BEST = 450 m
```

Candidate poses are rejected if they are outside the valid operating area, inside an obstacle safety zone, or blocked by obstacle safety zones along the route-to-pose segment.

The optimizer prefers poses that:

- stay reasonably close to the nominal route,
- reduce travel distance from and back to the route,
- preserve good viewing geometry,
- maintain clearance from obstacle safety zones.

An observation pose is the planned USV position from which the target should be viewed. Its purpose is to increase detection probability, not merely to place the USV near the target.

The observation pose contributes directly to the detection objective because the detection probability depends on range, camera angle, field-of-view visibility, and sensing range:

```text
P_d =
lambda_spatial
* lambda_temporal
* P_range
* P_angle
* FOV_gate
* range_gate
```

A good observation pose places the USV close to the best sensing distance:

```text
distance(p_obs, target) approx R_BEST
```

This increases the range term:

```text
P_range = exp(-((distance - R_BEST)^2) / (2 * SIGMA_R_BEST^2))
```

It also gives the camera a clear viewing direction so the angular error is small:

```text
beta = theta_goal - theta_view approx 0
```

which increases:

```text
P_angle = exp(-beta^2 / SIGMA_BETA^2)
```

The pose must also keep the target inside the field of view and usable range:

```text
FOV_gate = 1
range_gate = 1
```

Therefore:

```text
poor observation pose  -> low P_d -> slow or failed observation
good observation pose  -> high P_d -> faster target completion
```

The selected observation pose also improves mission feasibility because it must be reachable and safe with respect to obstacle safety zones. In this way, observation pose planning directly connects the two parts of the objective:

```text
maximize detection
while preserving safe motion
```

### Directional Camera Model

The USV has a directional camera with pan angle `theta_c`. The camera view direction is:

```text
theta_view = psi + theta_c
```

The detection probability combines GP arrival likelihood, temporal window likelihood, range quality, angular alignment, and hard field-of-view/range gates:

```text
P_d =
lambda_spatial
* lambda_temporal
* P_range
* P_angle
* FOV_gate
* range_gate
```

where:

- `P_range` is highest near `R_BEST`,
- `P_angle` is highest when the camera points directly at the target,
- `FOV_gate` requires the target to be inside the camera field of view,
- `range_gate` requires the target to be inside maximum sensing range.

### MPC-CBF Active Sensing Controller

The main controller is a sample-based MPC. At every step, candidate controls are rolled out over a short horizon. Unsafe rollouts are rejected using CBF constraints.

The MPC objective is designed to increase detection while respecting safety. It rewards:

- progress toward the observation pose or route rejoin point,
- camera alignment with the active target,
- target detection probability,
- schedule urgency,
- route-following behavior when appropriate.

It penalizes:

- control effort,
- abrupt changes in control.

Obstacle safety is enforced using the CBF condition:

```text
h = ||p - p_obs||^2 - R_safe^2
hdot + alpha h >= 0
```

Candidates that violate this condition are rejected. Therefore, the controller does not merely penalize unsafe behavior; it removes unsafe motion candidates from consideration.

The selected control is the best control among the safe candidates:

```text
u* = argmax safe score(u)
```

where the score includes detection probability, camera alignment, progress, scheduling urgency, route behavior, effort, and smoothness.

### Baseline Comparison Scenario

The comparison script evaluates whether each part of the proposed objective matters:

```text
Ours        GP-guided MPC-CBF active sensing
Baseline 1 greedy nearest observation pose with CBF
Baseline 2 active sensing without CBF
```

Baseline 1 is intentionally simpler:

- it uses greedy nearest observation poses,
- it uses waypoint-following control,
- it keeps CBF safety active,
- it does not use the full GP-guided MPC objective or optimized rejoin behavior.

Baseline 2 keeps active sensing behavior but disables obstacle CBF filtering. This shows what happens when detection is optimized without enforcing safety.

The expected contrast is:

```text
Ours:
  high detection performance
  zero safety violations

Baseline 1:
  safe motion
  weaker detection because sensing and motion are not jointly optimized

Baseline 2:
  high detection can still occur
  but safety is violated because CBF filtering is removed
```

The comparison scenario includes extra comparison-only obstacles. These are added only inside `compare_baselines.py` and do not modify the original simulation file.

## Files

- `usv_active_sensing_mpc_cbf_sim_research_model.py`
  Main GP-guided MPC-CBF simulation.

- `compare_baselines.py`
  Compares three methods:
  - Ours: GP-guided MPC-CBF active sensing
  - Baseline 1: greedy nearest observation pose with CBF
  - Baseline 2: active sensing without CBF

- `validate_method_math.py`
  Generates validation plots and CSV files showing how each component contributes to detection and safety.

- `plot/`
  Output folder for generated figures, CSV files, and MP4 videos.

## Environment

Use the existing conda environment:

```bash
conda activate screening
```

or run commands directly with:

```bash
conda run -n screening python <script_name.py>
```

Required packages:

```text
numpy
matplotlib
scikit-learn
```

The code also works with a fallback if scikit-learn is unavailable, but Gaussian Process regression uses scikit-learn when installed.

## Run Main Simulation

```bash
conda run -n screening python usv_active_sensing_mpc_cbf_sim_research_model.py
```

Main outputs are saved in `plot/`:

```text
trajectory_map.png
gp_lambda_map.png
schedule_status.png
observation_timer.png
detection_probability.png
cbf_h.png
route_active_sensing.mp4   if MP4 generation is enabled
```

To disable or enable the main MP4, edit:

```python
GENERATE_MP4 = True
```

near the top of `usv_active_sensing_mpc_cbf_sim_research_model.py`.

## Run Baseline Comparison

```bash
conda run -n screening python compare_baselines.py
```

This runs:

```text
Ours
Baseline 1
Baseline 2
```

Comparison outputs are saved in `plot/comparison/`:

```text
comparison_metrics.csv
comparison_metrics.png
comparison_trajectories.png
comparison_detection_probability.png
comparison_observation_progress.png
comparison_cbf_h.png
comparison_test.mp4
```

To disable the comparison MP4, edit:

```python
GENERATE_COMPARISON_MP4 = False
```

in `compare_baselines.py`.

## Run Mathematical Validation

```bash
conda run -n screening python validate_method_math.py
```

Validation outputs are saved in `plot/validation/`:

```text
validation_gp_targets.csv
validation_gp_targets.png
validation_observation_pose.csv
validation_observation_pose.png
validation_camera_alignment.csv
validation_camera_alignment.png
validation_mpc_detection.csv
validation_mpc_detection.png
validation_cbf.csv
validation_cbf.png
validation_summary.csv
```

These plots validate:

- GP target selection chooses high-likelihood target areas.
- Optimized observation poses improve expected sensing quality.
- Camera alignment increases detection probability.
- MPC active sensing improves integrated detection over the greedy baseline.
- CBF constraints prevent obstacle safety violations.

## Important Parameters

Most parameters are defined near the top of `usv_active_sensing_mpc_cbf_sim_research_model.py`, including:

```python
X_MIN, X_MAX, Y_MIN, Y_MAX
TARGETS
WINDOWS
OBSTACLES
VMAX
CRUISE_SPEED
OMEGA_MAX
FOV_RAD
CAMERA_RANGE
R_BEST
SAFETY_MARGIN
REQUIRED_OBSERVATION_TIME
```

The comparison script adds extra comparison-only obstacles inside `compare_baselines.py`. These do not modify the original simulation file.

## Typical Workflow

1. Run the main simulation:

   ```bash
   conda run -n screening python usv_active_sensing_mpc_cbf_sim_research_model.py
   ```

2. Run the baseline comparison:

   ```bash
   conda run -n screening python compare_baselines.py
   ```

3. Run mathematical validation:

   ```bash
   conda run -n screening python validate_method_math.py
   ```

4. Inspect results in:

   ```text
   plot/
   plot/validation/
   ```

## Notes

- The MPC is sample-based.
- CBF safety is enforced by rejecting unsafe rollout candidates.
- Observation progress accumulates using detection probability:

```text
observed_time_i += P_d(q_i, X) * dt
```
