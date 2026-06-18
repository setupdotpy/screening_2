# GP-Guided Active Sensing with a Short-Range Directional Camera

This repository contains a simulation of an unmanned surface vehicle (USV) performing active sensing using a short-range directional camera.

The objective is:

```text
maximize target detection performance
while maintaining safe navigation
```

The proposed framework combines:

- Gaussian Process (GP) target-arrival modeling
- Shape-adaptive observation planning
- Directional camera control
- Model Predictive Control (MPC)
- Control Barrier Functions (CBF)

The key idea is that the USV does not use a fixed observation strategy. Instead, it adapts its sensing trajectory according to the spatial shape of the estimated target-arrival distribution.

---

# Simulation Scenario

The simulation is performed in a:

```text
10 km × 10 km
```

operating area.

```text
x ∈ [-5000, 5000] m
y ∈ [0, 10000] m
```

The USV starts at:

```text
(0, 0)
```

and follows a nominal route:

```text
x = 0
```

The mission contains four target regions:

```text
T0
T1
T2
T3
```

generated automatically from historical target-arrival observations.

The vehicle is equipped with a directional camera:

```text
FOV = 60°
Maximum Range = 450 m
```

and must actively position itself to maximize sensing performance.

---

# Proposed Framework

The proposed framework combines:

- Gaussian Process target-arrival modeling
- Shape-adaptive observation planning
- Active sensing MPC
- Directional camera control
- CBF-based obstacle avoidance

Unlike conventional approaches that use a single observation point or a fixed observation pattern, the proposed method adapts the sensing trajectory according to the estimated shape of the target-arrival distribution.

```text
Historical target arrivals
            ↓
      Gaussian Process
            ↓
Target-arrival distribution
            ↓
Shape-adaptive observation planning
            ↓
Active sensing MPC
            ↓
Directional camera control
            ↓
Obstacle-aware navigation via CBF
```

The GP model estimates where targets are most likely to appear and reveals the spatial structure of the target-arrival distribution.

The observation planner then generates sensing viewpoints that adapt to this structure.

```text
Compact distribution
        ↓
Short observation path

Elongated distribution
        ↓
Elongated observation path

Wide distribution
        ↓
More sensing viewpoints

Irregular distribution
        ↓
Adaptive observation trajectory
```

Instead of forcing every target to use the same observation strategy, the USV automatically modifies its sensing behavior according to the underlying data distribution.

The directional camera continuously points toward high-value uncovered target-arrival regions while also discovering nearby obstacles.

The MPC controller selects motion commands that improve sensing quality, while the CBF guarantees safe navigation around detected obstacles.

The resulting framework jointly optimizes:

```text
USV position
Camera orientation
Observation trajectory
Obstacle awareness
```

to maximize detection performance while preserving safety.

---

# Detection-Oriented Active Sensing

The objective is not simply to reach a target location.

Instead, the USV seeks observation viewpoints that maximize detection quality.

Detection performance depends on:

```text
Target likelihood
Observation distance
Camera orientation
Field-of-view visibility
Obstacle visibility
```

Because the camera has limited range, the USV may need to move around a target region and observe it from multiple viewpoints.

Coverage is therefore used as a mechanism to improve detection performance rather than as the primary objective.

---

# Obstacle Discovery

Obstacles are initially unknown.

The onboard camera discovers obstacles only when they become visible.

```text
Obstacle appears in camera view
            ↓
Obstacle detected
            ↓
CBF activated
            ↓
Safe avoidance maneuver
```

This creates a realistic active-sensing scenario where perception and navigation are tightly coupled.

---

# Proposed Method

## Ours

```text
GP-guided active sensing
Shape-adaptive observation planning
Pan camera
MPC controller
CBF safety constraints
```

Features:

- Observation path adapts to target-distribution shape
- Camera actively tracks high-value target-arrival regions
- Online obstacle discovery
- Smooth trajectory generation
- Guaranteed safety through CBF

---

# Baselines

## Baseline 1

```text
Fixed camera pose-heading
Greedy viewpoint selection
CBF safety constraints
```

Features:

- Camera fixed to vehicle heading
- No active camera steering
- Safe navigation

---

## Baseline 2

```text
Active sensing
No CBF
```

Features:

- Active sensing behavior
- No formal safety guarantees
- May collide with obstacles

---

# Performance Metrics

The methods are evaluated using:

```text
Detection score
Completed targets
Field-of-view tracking
Obstacle detection
Safety violations
Minimum obstacle clearance
Trajectory smoothness
Coverage statistics
```

Coverage is reported as a supporting metric because broader observation of the target-arrival distribution generally leads to improved detection performance.

---

# Expected Outcome

```text
Ours:
- highest detection performance
- best obstacle awareness
- zero safety violations
- smooth trajectory

Baseline 1:
- reduced detection performance
- limited by fixed camera orientation

Baseline 2:
- potentially high detection
- unsafe navigation
```

---

# Generated Outputs

All outputs are saved in:

```text
plot/
```

Typical outputs include:

```text
trajectory_map.png
coverage_by_target.png
obstacle_detection.png
simulation.mp4
```

---

# Run


Run the simulation:

```bash
python short_range_camera.py
```

Generate video:

```bash
python short_range_camera.py --generate-video
```

Specify output folder:

```bash
python short_range_camera.py --plot-dir plot
```

---

# Key Contribution

Most active-sensing approaches observe a target from a single viewpoint or follow a fixed observation pattern.

In contrast, this work treats a target as a spatial probability distribution rather than a single point.

```text
Traditional:
Target → Point

This work:
Target → Probability Distribution
```

Historical observations are used to estimate the target-arrival distribution through a Gaussian Process model.

The USV then adapts its observation trajectory according to the shape of the estimated distribution.

```text
Target-distribution shape
            ↓
Observation-trajectory shape
```

As a result, different target-arrival patterns naturally produce different sensing behaviors.

```text
Compact target region
        ↓
Compact sensing path

Elongated target region
        ↓
Elongated sensing path

Wide target region
        ↓
Multiple sensing viewpoints

Irregular target region
        ↓
Adaptive sensing trajectory
```

The main contribution is therefore:

```text
Detection-oriented active sensing
with shape-adaptive observation planning
for a short-range directional camera
under CBF safety constraints.
```
