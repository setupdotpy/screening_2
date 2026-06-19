# Distribution-Aware GP-Guided Active Sensing with a Short-Range Directional Camera

This repository contains a simulation of an unmanned surface vehicle (USV)
performing active sensing with a short-range directional camera.

The objective is to try to maximize target detection under uncertain
target-arrival location while maintaining safe USV travel.

The proposed framework combines:

- GP/KDE-style target-arrival distribution modeling
- distribution-adaptive observation planning
- active directional-camera control
- Model Predictive Control (MPC)
- Control Barrier Function (CBF) safety filtering

The key idea is that the target is not treated as one fixed point or as a
geometric region that must be covered for its own sake. Instead, historical
target-arrival samples define a spatial likelihood distribution. The USV chooses
viewpoints that observe high-probability and still-unobserved parts of this
distribution, while MPC-CBF keeps the vehicle motion safe around discovered
obstacles.

---

# Simulation Scenario

The simulation is performed in a:

```text
10 km x 10 km
```

operating area:

```text
x in [-5000, 5000] m
y in [0, 10000] m
```

The USV starts at:

```text
(0, 0)
```

and follows a nominal route:

```text
x = 0
```

The mission contains four target-arrival distributions:

```text
T0
T1
T2
T3
```

These are generated from historical target-arrival observations.

The vehicle is equipped with a directional camera:

```text
FOV = 60 deg
Maximum range = 450 m
```

Obstacles are initially unknown and are discovered only when they enter the
camera sensing region.

---

# Pipeline Flow

The current pipeline is:

```text
Historical target-arrival samples
        ->
GP/KDE-style target-arrival likelihood field
        ->
Target spread estimation
        ->
Support samples from high-value arrival distribution
        ->
Candidate observation pose generation
        ->
Visibility and detection scoring
        ->
MPC motion planning
        ->
Online obstacle discovery
        ->
CBF safety filtering
        ->
USV state update and metric evaluation
```

The pipeline is detection-oriented. The USV is not trying to cover a whole
target region just for coverage. The goal is to detect one target whose future
arrival point is uncertain. Coverage is used only as a supporting metric to
measure how much likely target-arrival support has been observed.

---

# Proposed Framework

The proposed method uses historical target-arrival samples to estimate where a
future target is more likely to appear. This likelihood field is then converted
into support samples. Candidate observation poses are evaluated by how much
high-probability, still-unobserved support can be seen from each pose.

In simple terms:

```text
High-probability target-arrival support
        ->
Useful camera viewpoints
        ->
MPC trajectory toward the selected viewpoint
        ->
CBF filter removes unsafe controls
```

Different arrival distributions naturally lead to different sensing behavior:

```text
Compact distribution
        ->
Shorter sensing path

Elongated distribution
        ->
Longer path along the distribution support

Wide distribution
        ->
More observation viewpoints

Irregular or multi-modal distribution
        ->
Adaptive viewpoint sequence
```

The directional camera points toward high-value target-arrival support while
also discovering nearby obstacles. MPC proposes motion commands that balance
progress, detection utility, camera alignment, and smoothness. CBF filtering
rejects unsafe candidate controls after obstacles have been detected.

---

# Detection-Oriented Active Sensing

The objective is not simply to reach the nominal target center.

Instead, the USV seeks observation viewpoints that increase the chance of
detecting a target whose arrival location is uncertain.

Detection performance depends on:

```text
Target-arrival likelihood
Observation distance
Camera orientation
Field-of-view visibility
Camera range
Obstacle visibility
```

Because the camera has limited range and FOV, the USV may need to move around
the target-arrival distribution and observe several high-probability regions.

Coverage is reported only as a supporting metric because it indicates how much
likely target-arrival probability or support has been observed.

---

# Obstacle Discovery and Safety

Obstacles are initially unknown.

The onboard camera discovers obstacles only when they become visible:

```text
Obstacle enters camera view
        ->
Obstacle detected
        ->
CBF safety filter activated
        ->
Unsafe controls rejected
        ->
Safe motion command selected
```

This creates an online active-sensing problem where perception, planning, and
safety are coupled.

---

# Proposed Method

## Ours

```text
Distribution-aware GP/KDE-guided active sensing
Active pan camera
MPC controller
CBF safety filtering
Online obstacle discovery
```

Features:

- observation path adapts to the target-arrival distribution
- camera actively tracks high-value target-arrival support
- candidate poses are selected using visible unobserved probability mass
- obstacles are discovered online
- MPC generates smooth motion toward selected viewpoints
- CBF filtering rejects unsafe controls after obstacle discovery

---

# Baselines

## Baseline 1: Fixed Camera Heading

```text
Distribution-aware observation planning
Fixed camera relative to vehicle heading
MPC controller
CBF safety filtering
```

Features:

- camera is fixed to vehicle heading
- no independent camera steering
- same CBF safety filtering as the proposed method
- evaluates the value of active camera orientation

---

## Baseline 2: Active Sensing Without CBF

```text
Distribution-aware observation planning
Active pan camera
MPC controller
No CBF safety filtering
```

Features:

- active sensing behavior is retained
- candidate controls are not filtered by CBF constraints
- may achieve high raw detection score
- may collide with obstacles or violate safety boundaries
- evaluates the value of the CBF safety layer

---

# Performance Metrics

The methods are evaluated using:

```text
Detection score
Target-arrival distribution coverage
Field-of-view tracking
Obstacle detection
Avoided obstacles
Collisions
Safety violations
Minimum obstacle clearance
Trajectory smoothness
Mission completion
```

Coverage is a supporting metric, not the main objective. The main objective is
target detection under arrival uncertainty while maintaining safe navigation.

---

# Generated Outputs

All outputs are saved in:

```text
plot/
```

Typical outputs include:

```text
trajectory_map.png
target_coverage_area.png
safety_violations.png
fov_success.png
obstacle_avoided.png
turning_radius.png
simulation.mp4
metrics.csv
r95.csv
```

---

# Run

Run the simulation:

```bash
python short_range_camera.py
```

Generate video:

```bash
python short_range_camera.py --video
```

Generate video with a custom name:

```bash
python short_range_camera.py --video paper_demo.mp4
```

Specify output folder:

```bash
set SRC_PLOT_DIR=plot
python short_range_camera.py
```

On Linux or macOS:

```bash
export SRC_PLOT_DIR=plot
python short_range_camera.py
```

---

# Key Contribution

Most active-sensing approaches observe a target from a fixed viewpoint or use a
single predicted target point.

This work instead treats the future target location as a spatial probability
distribution:

```text
Traditional:
Target -> one point

This work:
Target -> target-arrival distribution
```

The USV then selects observation poses based on high-probability parts of this
distribution and uses MPC-CBF to balance detection utility with navigation
safety.

The main contribution is:

```text
Detection-oriented, distribution-aware active sensing
for a short-range directional-camera USV
with MPC-CBF safety filtering.
```
