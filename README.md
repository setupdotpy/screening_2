# Distribution-Aware Active Sensing for a Toy Surveillance USV

This repository contains a toy simulation of a surveillance unmanned surface
vehicle (USV) with a limited-range directional camera. The USV repositions
itself and reorients its camera to try to maximize target detection under
uncertain target-arrival location, while maintaining safe motion around
obstacles.

The project is not a full real-world USV stack. It is a simulation for testing
the idea of distribution-aware active sensing with MPC-CBF safety filtering.

---

# Simulation Scenario

The simulation uses a 2D operating area:

```text
Area: 10 km x 10 km
x in [-5000, 5000] m
y in [0, 10000] m
Start: (0, 0)
Nominal route: x = 0
```

The mission contains four uncertain target-arrival cases:

```text
T0, T1, T2, T3
```

Each target case is generated from historical target-arrival samples. The true
future target location is uncertain, so the target is modeled as a spatial
arrival distribution instead of one fixed point.

The USV sensor is a short-range directional camera:

```text
Field of view: 60 deg
Maximum range: 450 m
Camera mode: pan camera for the proposed method
```

Obstacles are initially unknown. The USV discovers an obstacle only after the
obstacle enters the camera sensing region. After discovery, the obstacle is used
by the CBF safety filter.

---

# Core Objective

The objective is:

```text
try to maximize target detection probability
under uncertain target-arrival location
while maintaining safe USV motion
```

The goal is not to cover a whole target region for its own sake. Coverage is
reported only as a supporting metric that measures how much high-probability
target-arrival support has been observed.

---

# Pipeline

The simulation pipeline is:

```text
Historical target-arrival samples
        ->
GP/KDE-style likelihood field
        ->
Target spread estimation
        ->
Support samples from the arrival distribution
        ->
Candidate observation pose generation
        ->
Camera visibility and detection scoring
        ->
MPC motion planning
        ->
Online obstacle discovery
        ->
CBF safety filtering
        ->
USV state update and metric evaluation
```

In practical terms, the USV searches for viewpoints that can see useful,
high-probability parts of the target-arrival distribution. MPC moves the USV
toward the selected viewpoint, and CBF filtering removes unsafe controls when
detected obstacles are nearby.

---

# Method Summary

## Target-Arrival Distribution

Historical samples are converted into a GP/KDE-style spatial likelihood field.
High likelihood means the target is more likely to arrive at that location.

The planner uses this field to avoid assuming that the future target position is
only the distribution center.

## Distribution-Aware Observation Planning

The likelihood field is sampled into support points. Candidate observation poses
are evaluated by how much visible, high-probability, unobserved support they can
observe.

The selected pose is the one with the largest useful visible probability mass.

## Camera Visibility and Detection

Detection depends on:

```text
target-arrival likelihood
camera range
camera field of view
camera-target bearing error
observation distance
```

A target-support sample is useful only if it is within camera range and inside
the camera field of view.

## MPC Motion Planning

MPC samples possible controls and scores them using progress, detection utility,
camera alignment, and smoothness. The controller then moves the USV toward the
selected observation pose.

## Online Obstacle Discovery and CBF Safety

The camera also discovers obstacles. Once an obstacle is detected, CBF safety
filtering rejects candidate controls that would move the USV into unsafe
clearance.

This couples perception, sensing, motion planning, and safety in one loop.

---

# Compared Methods

## Proposed Method

```text
Distribution-aware observation planning
Active pan camera
MPC motion planning
Online obstacle discovery
CBF safety filtering
```

Purpose: test whether the USV can actively reposition and reorient the camera
to observe likely target-arrival locations while staying safe.

## Baseline 1: Fixed Camera Heading

```text
Same observation-planning framework
Camera fixed to vehicle heading
MPC motion planning
CBF safety filtering
```

Purpose: test the value of independent camera pan control.

## Baseline 2: Active Sensing Without CBF

```text
Same active sensing framework
Active pan camera
MPC motion planning
No CBF safety filtering
```

Purpose: test the value of the CBF safety layer.

---

# Metrics

The simulation reports:

```text
Detection score
Target-arrival distribution coverage
FOV success rate
Avoided obstacles
Collisions
Safety violations
Minimum obstacle clearance
Trajectory smoothness
Mission completion
```

Coverage is not the main objective. It is used to show how much likely
target-arrival support has been observed.

---

# Outputs

All generated outputs are saved in:

```text
plot/
```

Common outputs:

```text
trajectory_map.png
target_coverage_area.png
safety_violations.png
fov_success.png
obstacle_avoided.png
turning_radius.png
metrics.csv
r95.csv
simulation.mp4
```

---

# Run

Run the simulation:

```bash
python short_range_camera.py
```

Generate an MP4 video:

```bash
python short_range_camera.py --video
```

Generate a video with a custom filename:

```bash
python short_range_camera.py --video paper_demo.mp4
```

Set a custom output directory on Windows:

```bash
set SRC_PLOT_DIR=plot
python short_range_camera.py
```

Set a custom output directory on Linux or macOS:

```bash
export SRC_PLOT_DIR=plot
python short_range_camera.py
```

---

# Main Idea

This simulation studies a simple question:

```text
Can a surveillance USV with a limited-range camera
move and point its camera in a distribution-aware way
to improve target detection attempts
while still keeping motion safe?
```

The proposed answer is to combine target-arrival likelihood modeling,
distribution-aware viewpoint selection, MPC motion planning, active camera
orientation, online obstacle discovery, and CBF safety filtering.
