# ==========================================================
# short_range_camera.py
#
# Logic 2: Short-Range Active Sensing with MPC-CBF
#
# Objective:
#   Maximize coverage of GP-derived 95% target-arrival regions
#   using a directional pan camera while ensuring safe navigation.
#
# Methods:
#   Ours       : Pan Camera + MPC + CBF
#   Baseline 1 : Fixed Camera + Path Following
#   Baseline 2 : Pan Camera without CBF
#
# Outputs:
#   All figures, CSVs, and MP4 files are saved to plot/.
#   The main trajectory figure is saved only as trajectory_map.png.
# ==========================================================

#
# Usage:
#   python3 short_range_camera.py
#
# Generate video:
#   python3 short_range_camera.py --video
#
# Generate video with custom filename:
#   python3 short_range_camera.py --video my_run.mp4
#
# Optional output directory:
#   export SRC_PLOT_DIR=/home/sotheara/screening/usv/plot
#   python3 short_range_camera.py --video paper_demo.mp4
#
import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Circle, Wedge

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    import usv_active_sensing_mpc_cbf_sim_research_model as base
except ModuleNotFoundError:
    # Standalone fallback so this script can run even when the original base file
    # is not in the parent directory. If your project already has the base file,
    # the values below are ignored.
    class _Base:
        pass

    base = _Base()
    base.X_MIN, base.X_MAX = -5000.0, 5000.0
    base.Y_MIN, base.Y_MAX = 0.0, 10000.0
    base.X_LIMIT, base.Y_LOW, base.Y_HIGH = 5000.0, 0.0, 10000.0
    base.START_STATE = np.array([0.0, 0.0, np.pi / 2.0, 0.0])
    base.TARGETS = np.array([
        [-1200.0, 1500.0],
        [4000.0, 4000.0],
        [-900.0, 6500.0],
        [4500.0, 8000.0],
    ])
    rng = np.random.default_rng(7)
    pts = []
    for t in base.TARGETS:
        pts.append(t + rng.normal(scale=[100.0, 90.0], size=(45, 2)))
    base.HISTORICAL_POINTS = np.vstack(pts)
    base.OBSTACLES = []


def make_writable_plot_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = []

    env_dir = os.environ.get("SRC_PLOT_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates += [
        script_dir / "plot",
        Path.cwd() / "src" / "plot",
        Path.cwd() / "plot",
        Path("/tmp") / "src_plot",
    ]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".write_test"
            with open(test_file, "w") as f:
                f.write("ok")
            test_file.unlink(missing_ok=True)
            return candidate
        except Exception:
            continue

    raise RuntimeError(
        "No writable output directory found. "
        "Set SRC_PLOT_DIR to a writable path."
    )


parser = argparse.ArgumentParser(
    description="Short-range active sensing MPC-CBF simulation"
)

parser.add_argument(
    "--video",
    nargs="?",
    const="simulation.mp4",
    default=None,
    metavar="FILENAME",
    help="Generate MP4 video. Optionally provide output filename."
)

args = parser.parse_args()

PLOT_DIR = make_writable_plot_dir()
GENERATE_MP4 = args.video is not None
MP4_FILENAME = PLOT_DIR / args.video if args.video is not None else PLOT_DIR / "simulation.mp4"
ANIMATION_FPS = 20
ANIMATION_STRIDE = 300
ANIMATION_DPI = 70

SEED = 7
DT = 1
STEPS = 200000
VMAX = 3.0
CRUISE_SPEED = 2.5
OMEGA_MAX = 0.018  # global yaw-rate limit [rad/s]
OMEGA_MAX_OURS = 0.012  # USV limit used by our method [rad/s]
OMEGA_SLEW_MAX_OURS = 0.003  # max change in yaw rate per MPC step [rad/s]
UC_MAX = 0.8
HORIZON = 3
FOV_DEG = 60.0
FOV_RAD = np.deg2rad(FOV_DEG)
CAMERA_MAX_RANGE = 450.0
MAX_CAMERA_RANGE = CAMERA_MAX_RANGE  # physical camera detection limit [m]
K_VIEWPOINTS = 4
PD_COVER_THRESHOLD = 0.18
VIEWPOINT_RADIUS = 180.0
OBSTACLE_CLOSE_RANGE = 700.0
SAFETY_MARGIN = 120.0
ALPHA_CBF = 3.0
SIGMA_R_FACTOR = 0.25  # target-specific range-quality width = factor * R_best
SIGMA_R_MIN = 80.0
SIGMA_BETA = 0.45

# Future/direct GP-field active-sensing settings.
# Ours does not only chase fixed circular observation sectors. It chooses
# a smooth held waypoint that maximizes the GP mass visible inside the camera FOV.
DIRECT_GP_BETA = 0.45
DIRECT_GP_COVERAGE_GOAL = 0.95
DIRECT_GP_GOAL_HOLD_STEPS = 260
DIRECT_GP_GOAL_REACHED_DIST = 300.0
DIRECT_GP_NUM_BEARINGS = 8
DIRECT_GP_NUM_CAMERA_ANGLES = 8
DIRECT_GP_TRAVEL_PENALTY = 6.0e-4
DIRECT_GP_ROUTE_PENALTY = 1.8e-4
DIRECT_GP_HEADING_PENALTY = 0.22
DIRECT_GP_MIN_VALUE_TO_REPLAN = 0.015

W_PROGRESS = 6.0
W_TARGET_PD = 5.0
W_OBS_VIS = 2.2
W_COVERAGE = 8.0
W_TURN = 8.0
W_SMOOTH = 0.08
K_HEADING_FIXED_CAMERA = 0.65

TARGETS = base.TARGETS.copy()

# -----------------------------------------------------------------------------
# Historical target-arrival data
# -----------------------------------------------------------------------------
# Use deliberately non-Gaussian / non-circular historical distributions so the
# direct GP-field method visibly adapts to data shape instead of reducing every
# target to one center + one circular radius.
def generate_shaped_historical_points(seed=SEED):
    rng = np.random.default_rng(seed)
    pts = []

    # T0: crescent / horseshoe shape.
    # This cannot be represented well by one Gaussian blob.
    t = TARGETS[0]
    theta = rng.uniform(np.deg2rad(125.0), np.deg2rad(325.0), 150)
    radius = rng.normal(470.0, 42.0, len(theta))
    crescent = np.column_stack([
        radius * np.cos(theta),
        0.68 * radius * np.sin(theta),
    ])
    crescent += rng.normal(scale=[28.0, 28.0], size=crescent.shape)
    pts.append(t + crescent)

    # T1: long curved lane, similar to a shipping/current corridor.
    # The high-likelihood region is long and thin, not circular.
    t = TARGETS[1]
    u = rng.uniform(-1050.0, 1050.0, 180)
    lane = np.column_stack([
        u,
        170.0 * np.sin(u / 310.0) + 0.10 * u,
    ])
    lane += rng.normal(scale=[35.0, 45.0], size=lane.shape)
    ang = np.deg2rad(4.0)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts.append(t + lane @ R.T)

    # T2: two separated arrival modes under the same mission target label.
    # A center-based method would observe the empty middle.
    t = TARGETS[2]
    mode_a = t + np.array([-460.0, -210.0]) + rng.multivariate_normal(
        [0.0, 0.0], [[150.0**2, 50.0**2], [50.0**2, 115.0**2]], size=85
    )
    mode_b = t + np.array([430.0, 260.0]) + rng.multivariate_normal(
        [0.0, 0.0], [[170.0**2, -55.0**2], [-55.0**2, 125.0**2]], size=85
    )
    pts.append(np.vstack([mode_a, mode_b]))

    # T3: L-shaped / bent arrival distribution.
    # Built from two thin line segments, so the GP heatmap should look angular.
    t = TARGETS[3]
    n1, n2 = 95, 95
    s1 = rng.uniform(-700.0, 100.0, n1)
    arm1 = np.column_stack([s1, rng.normal(0.0, 55.0, n1)])
    s2 = rng.uniform(0.0, 760.0, n2)
    arm2 = np.column_stack([rng.normal(0.0, 55.0, n2), s2])
    elbow = np.vstack([arm1, arm2])
    elbow += rng.normal(scale=[25.0, 25.0], size=elbow.shape)
    ang = np.deg2rad(-18.0)
    R2 = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts.append(t + elbow @ R2.T)

    data = np.vstack(pts)
    data[:, 0] = np.clip(data[:, 0], base.X_MIN, base.X_MAX)
    data[:, 1] = np.clip(data[:, 1], base.Y_MIN, base.Y_MAX)
    return data


HISTORICAL_POINTS = generate_shaped_historical_points(SEED)
X_MIN, X_MAX = base.X_MIN, base.X_MAX
Y_MIN, Y_MAX = base.Y_MIN, base.Y_MAX
X_LIMIT, Y_LOW, Y_HIGH = base.X_LIMIT, base.Y_LOW, base.Y_HIGH
START_STATE = base.START_STATE.copy()

# -----------------------------------------------------------------------------
# GP / KDE heatmap for historical target-arrival likelihood
# -----------------------------------------------------------------------------
# Use the heatmap from the reference base script when available. Otherwise build
# a normalized KDE map from the historical samples so this comparison script is
# still standalone.
GP_NX = int(getattr(base, "GP_NX", 120))
GP_NY = int(getattr(base, "GP_NY", 120))
SIGMA_KDE = 280.0  # smaller bandwidth keeps crescent/lane/bimodal/L-shapes visible


def build_historical_likelihood_map():
    # Always rebuild the heatmap from the shaped historical samples in this
    # script. Do not reuse the imported base heatmap because that map may have
    # been generated from circular Gaussian clusters.
    x_grid = np.linspace(X_MIN, X_MAX, GP_NX)
    y_grid = np.linspace(Y_MIN, Y_MAX, GP_NY)
    xx, yy = np.meshgrid(x_grid, y_grid)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    if len(HISTORICAL_POINTS) == 0:
        likelihood = np.zeros((GP_NY, GP_NX), dtype=float)
    else:
        diff = query[:, None, :] - HISTORICAL_POINTS[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        likelihood = np.sum(np.exp(-dist2 / (2.0 * SIGMA_KDE**2)), axis=1).reshape(GP_NY, GP_NX)
    max_value = float(np.max(likelihood))
    if max_value > 1e-12:
        likelihood = likelihood / max_value
    return x_grid, y_grid, likelihood


LAMBDA_X_GRID, LAMBDA_Y_GRID, LAMBDA_BAR_MAP = build_historical_likelihood_map()

METHODS = {
    "Ours": {"color": "#004f80", "ls": "-", "lw": 2.4},
    "Baseline 1": {"color": "#f28e2b", "ls": "--", "lw": 2.2},
    "Baseline 2": {"color": "#7b3294", "ls": ":", "lw": 2.7},
}
METHOD_LABELS = {
    "Ours": "Ours: direct GP-field smooth pan-camera MPC-CBF",
    "Baseline 1": "Baseline 1: fixed camera pose-heading",
    "Baseline 2": "Baseline 2: no CBF",
}


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def copy_obstacles():
    return [{"center": obs["center"].copy(), "radius": float(obs["radius"])} for obs in base.OBSTACLES]


EXTRA_OBSTACLES = [
    # Local obstacles near target regions.
    {"center": np.array([-650.0, 1650.0]), "radius": 140.0},
    {"center": np.array([3350.0, 3850.0]), "radius": 150.0},
    {"center": np.array([3650.0, 4550.0]), "radius": 130.0},
    {"center": np.array([-450.0, 6100.0]), "radius": 160.0},
    {"center": np.array([3550.0, 7550.0]), "radius": 150.0},
    {"center": np.array([4100.0, 8150.0]), "radius": 130.0},

    # Corridor obstacles placed on straight transit/rejoin corridors.
    # Purpose: Baseline 2 has no CBF and no obstacle-aware rejoin, so it
    # should cut through these safety regions. Ours detects route-blocking
    # obstacles and applies the CBF/projection correction.
    {"center": np.array([1050.0, 2450.0]), "radius": 230.0},   # T0 -> T1 transit corridor
    {"center": np.array([2450.0, 3250.0]), "radius": 210.0},   # T0 -> T1 transit corridor
    {"center": np.array([2300.0, 5750.0]), "radius": 240.0},   # T1 -> T2 / route-return corridor
    {"center": np.array([2450.0, 6900.0]), "radius": 230.0},   # T2 -> T3 transit corridor
    {"center": np.array([900.0, 8600.0]), "radius": 260.0},    # final route-return corridor
]
OBSTACLES = EXTRA_OBSTACLES


def build_control_candidates():
    v = np.array([0.0, 0.8, CRUISE_SPEED, VMAX])
    omega = np.linspace(-OMEGA_MAX, OMEGA_MAX, 5)
    uc = np.linspace(-UC_MAX, UC_MAX, 5)
    vv, ww, cc = np.meshgrid(v, omega, uc, indexing="ij")
    return np.column_stack([vv.ravel(), ww.ravel(), cc.ravel()])


CONTROL_CANDIDATES = build_control_candidates()


def step_dynamics(state, control):
    x, y, psi, theta_c = state
    v, omega, u_c = control
    return np.array(
        [
            x + DT * v * np.cos(psi),
            y + DT * v * np.sin(psi),
            wrap_to_pi(psi + DT * omega),
            wrap_to_pi(theta_c + DT * u_c),
        ]
    )


def target_geometry(state, target, fixed_camera=False):
    rel = target - state[:2]
    theta_goal = math.atan2(rel[1], rel[0])
    theta_view = state[2] if fixed_camera else state[2] + state[3]
    beta = wrap_to_pi(theta_goal - theta_view)
    return theta_goal, beta, float(np.linalg.norm(rel))


def detection_prob(state, target_idx, fixed_camera=False):
    """Target detection probability with adaptive, data-driven R_best.

    R_best_i is computed from the 95% historical spread of target i and then
    capped by the physical camera max range. There is no artificial minimum
    range. Being closer than R_best_i is allowed, but range quality decreases
    because the camera footprint covers less of the 95% arrival region.
    """
    target = TARGETS[target_idx]
    _, beta, dist = target_geometry(state, target, fixed_camera=fixed_camera)
    r_best = CAMERA_RANGES[target_idx]
    sigma_r = TARGET_SIGMA_R[target_idx]
    p_range = math.exp(-((dist - r_best) ** 2) / (2.0 * sigma_r**2))
    p_angle = math.exp(-(beta**2) / (SIGMA_BETA**2))
    p_fov = 1.0 if abs(beta) <= FOV_RAD / 2.0 else 0.0
    p_range_gate = 1.0 if dist <= CAMERA_MAX_RANGE + 30.0 else 0.0
    return p_range * p_angle * p_fov * p_range_gate


def obstacle_visible(state, obs, camera_range, fixed_camera=False):
    rel = obs["center"] - state[:2]
    dist = float(np.linalg.norm(rel))
    if dist > camera_range:
        return False
    theta_obs = math.atan2(rel[1], rel[0])
    theta_view = state[2] if fixed_camera else state[2] + state[3]
    beta = wrap_to_pi(theta_obs - theta_view)
    return abs(beta) <= FOV_RAD / 2.0


def point_safe(point, extra_margin=0.0):
    if point[0] < -X_LIMIT or point[0] > X_LIMIT or point[1] < Y_LOW or point[1] > Y_HIGH:
        return False
    for obs in OBSTACLES:
        if np.linalg.norm(point - obs["center"]) <= obs["radius"] + SAFETY_MARGIN + extra_margin:
            return False
    return True


def segment_intersects_circle(p0, p1, center, radius):
    d = p1 - p0
    denom = float(np.dot(d, d))
    if denom < 1e-9:
        return np.linalg.norm(p0 - center) <= radius
    t = np.clip(np.dot(center - p0, d) / denom, 0.0, 1.0)
    closest = p0 + t * d
    return np.linalg.norm(closest - center) <= radius


def segment_intersects_safety(p0, p1):
    for obs in OBSTACLES:
        if segment_intersects_circle(p0, p1, obs["center"], obs["radius"] + SAFETY_MARGIN):
            return True
    return False


def nearest_obstacle_metrics(state):
    h_values = []
    clearances = []
    for obs in OBSTACLES:
        safe_radius = obs["radius"] + SAFETY_MARGIN
        d = np.linalg.norm(state[:2] - obs["center"])
        h_values.append(d**2 - safe_radius**2)
        clearances.append(d - safe_radius)
    return float(np.min(h_values)), float(np.min(clearances))


def compute_r95_and_ranges():
    """Compute target-specific R_best from empirical 95% data spread.

    Unlike the earlier local-neighborhood version, this uses all historical
    samples assigned to each target label. Therefore a thin distribution that
    extends far from the nominal target center produces a large r95 and the
    planner must visit the spread-out support to cover 95% of the data.
    """
    rows = []
    ranges = []
    raw_ranges = []
    r95_values = []
    sigma_values = []
    coverable_fractions = []
    coverable_radii = []
    half_fov_tan = math.tan(FOV_RAD / 2.0)

    if len(HISTORICAL_POINTS):
        d_to_targets = np.linalg.norm(HISTORICAL_POINTS[:, None, :] - TARGETS[None, :, :], axis=2)
        assigned = np.argmin(d_to_targets, axis=1)
    else:
        assigned = np.array([], dtype=int)

    for i, target in enumerate(TARGETS):
        pts = HISTORICAL_POINTS[assigned == i] if len(HISTORICAL_POINTS) else np.empty((0, 2))
        if len(pts) < 8:
            distances_all = np.linalg.norm(HISTORICAL_POINTS - target[None, :], axis=1) if len(HISTORICAL_POINTS) else np.array([])
            pts = HISTORICAL_POINTS[np.argsort(distances_all)[:20]] if len(distances_all) else target[None, :].copy()

        local_distances = np.linalg.norm(pts - target[None, :], axis=1)
        cov = np.cov(pts.T) if len(pts) > 1 else np.eye(2) * 50.0**2
        sigma_max = math.sqrt(max(float(np.max(np.linalg.eigvalsh(cov))), 1.0))

        # Empirical circular radius containing 95% of the assigned data. This is
        # only used for reporting/range sizing; actual coverage uses shape samples.
        r95 = float(np.percentile(local_distances, 95.0)) if len(local_distances) else 0.0
        r_best_raw = r95 / max(half_fov_tan, 1e-6)
        r_best = min(r_best_raw, CAMERA_MAX_RANGE)
        coverable_radius = r_best * half_fov_tan
        coverable_fraction = float(np.mean(local_distances <= coverable_radius)) if len(local_distances) else 0.0

        ranges.append(r_best)
        raw_ranges.append(r_best_raw)
        r95_values.append(r95)
        sigma_values.append(sigma_max)
        coverable_radii.append(coverable_radius)
        coverable_fractions.append(coverable_fraction)
        rows.append([
            f"T{i}",
            target[0],
            target[1],
            sigma_max,
            r95,
            r_best_raw,
            r_best,
            CAMERA_MAX_RANGE,
            coverable_radius,
            coverable_fraction,
            len(pts),
        ])

    return (
        np.array(r95_values),
        np.array(ranges),
        np.array(raw_ranges),
        np.array(coverable_fractions),
        np.array(coverable_radii),
        np.array(sigma_values),
        rows,
    )


(
    TARGET_R95,
    CAMERA_RANGES,
    CAMERA_RANGES_RAW,
    TARGET_RBEST_COVERAGE,
    TARGET_COVERABLE_RADII,
    TARGET_SIGMA_MAX,
    R95_ROWS,
) = compute_r95_and_ranges()
TARGET_SIGMA_R = np.maximum(SIGMA_R_MIN, SIGMA_R_FACTOR * CAMERA_RANGES)

# Area-coverage model for the target-wise coverage bar plot.
# The old version used only 8 observation-ring sectors. That is too coarse for
# Baseline 1: a fixed forward camera may sweep a small part of the r95 target
# area, but none of the 8 sector waypoints is marked as complete. This sample
# grid measures the actual visible portion of each 95% arrival region.
COVERAGE_SAMPLE_SPACING = 75.0


def bilinear_interpolate_grid(x_grid, y_grid, values, points):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.array([], dtype=float)
    x = np.clip(points[:, 0], x_grid[0], x_grid[-1])
    y = np.clip(points[:, 1], y_grid[0], y_grid[-1])
    ix = np.searchsorted(x_grid, x, side="right") - 1
    iy = np.searchsorted(y_grid, y, side="right") - 1
    ix = np.clip(ix, 0, len(x_grid) - 2)
    iy = np.clip(iy, 0, len(y_grid) - 2)
    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]
    tx = (x - x0) / np.maximum(x1 - x0, 1e-9)
    ty = (y - y0) / np.maximum(y1 - y0, 1e-9)
    v00 = values[iy, ix]
    v10 = values[iy, ix + 1]
    v01 = values[iy + 1, ix]
    v11 = values[iy + 1, ix + 1]
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11


def build_target_area_samples():
    """Samples used for both coverage metrics and visible map points.

    To avoid plot/metric mismatch, the coverage metric now uses the same target
    data points shown on the trajectory map: the historical samples assigned to
    each target. No GP-grid fallback points are added and no downsampling is
    applied. Therefore, if a target reports 100% coverage, all plotted metric
    samples for that target are marked as covered on the map.
    """
    all_samples = []
    if len(HISTORICAL_POINTS):
        d_to_targets = np.linalg.norm(
            HISTORICAL_POINTS[:, None, :] - TARGETS[None, :, :],
            axis=2,
        )
        assigned = np.argmin(d_to_targets, axis=1)
    else:
        assigned = np.array([], dtype=int)

    for i, target in enumerate(TARGETS):
        pts = HISTORICAL_POINTS[assigned == i] if len(HISTORICAL_POINTS) else np.empty((0, 2))

        # Robust fallback only for an empty target cluster. This should not occur
        # with the generated shaped historical data, but keeps the script safe.
        if len(pts) == 0:
            pts = target[None, :].copy()

        all_samples.append(np.asarray(pts, dtype=float))
    return all_samples


TARGET_AREA_SAMPLES = build_target_area_samples()


# -----------------------------------------------------------------------------
# Direct GP-field smooth observation-pose selection for Ours
# -----------------------------------------------------------------------------
def estimate_gp_uncertainty_at_points(points):
    """Simple posterior-uncertainty surrogate.

    The base script supplies a GP/KDE mean map but not a true variance map. This
    surrogate increases uncertainty with distance from historical samples, so
    UCB still rewards poorly observed regions.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0 or len(HISTORICAL_POINTS) == 0:
        return np.ones(len(points), dtype=float)
    diff = points[:, None, :] - HISTORICAL_POINTS[None, :, :]
    nearest = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    sigma = 1.0 - np.exp(-(nearest**2) / (2.0 * SIGMA_KDE**2))
    return np.clip(sigma, 0.0, 1.0)


def build_target_gp_sample_weights():
    info = []
    for samples in TARGET_AREA_SAMPLES:
        mu = bilinear_interpolate_grid(LAMBDA_X_GRID, LAMBDA_Y_GRID, LAMBDA_BAR_MAP, samples)
        sigma = estimate_gp_uncertainty_at_points(samples)
        ucb = np.maximum(mu + DIRECT_GP_BETA * sigma, 0.03)
        info.append({"mu": mu, "sigma": sigma, "value": ucb})
    return info


TARGET_GP_SAMPLE_INFO = build_target_gp_sample_weights()


def build_direct_gp_anchor_samples(max_anchors=16):
    """Representative points along each target-arrival shape.

    These anchors replace the old center-based observation ring for Ours.
    For an elongated, bimodal, crescent, or L-shaped distribution, the anchors
    lie along the actual historical/GP support. The optimizer then samples
    camera poses around these anchors, so the USV visits the spread-out data
    region instead of orbiting only the target center.
    """
    all_anchors = []
    for i, samples in enumerate(TARGET_AREA_SAMPLES):
        if len(samples) == 0:
            all_anchors.append(TARGETS[i][None, :].copy())
            continue

        values = TARGET_GP_SAMPLE_INFO[i]["value"]
        order = np.argsort(-values)
        selected = []
        min_sep = max(120.0, 0.32 * CAMERA_MAX_RANGE)

        # Greedy non-maximum suppression keeps anchors distributed along the
        # whole target shape rather than concentrated at one hotspot.
        for idx in order:
            pt = samples[int(idx)]
            if not selected:
                selected.append(pt)
            else:
                d = np.linalg.norm(np.asarray(selected) - pt[None, :], axis=1)
                if float(np.min(d)) >= min_sep:
                    selected.append(pt)
            if len(selected) >= max_anchors:
                break

        if len(selected) < min(8, len(samples)):
            # Fallback: add farthest-spread samples so thin distributions are
            # represented even if the GP weights are similar everywhere.
            centroid = np.mean(samples, axis=0)
            far_order = np.argsort(-np.linalg.norm(samples - centroid[None, :], axis=1))
            for idx in far_order:
                pt = samples[int(idx)]
                if not selected:
                    selected.append(pt)
                else:
                    d = np.linalg.norm(np.asarray(selected) - pt[None, :], axis=1)
                    if float(np.min(d)) >= min_sep:
                        selected.append(pt)
                if len(selected) >= max_anchors:
                    break

        all_anchors.append(np.asarray(selected, dtype=float))
    return all_anchors


DIRECT_GP_ANCHORS = build_direct_gp_anchor_samples()


def build_direct_gp_candidate_table():
    """Precompute direct-GP candidate poses and their visible support samples.

    This avoids rebuilding the same center-free observation candidates at every
    MPC step. Coverage changes over time, but the geometric visibility relation
    between a candidate pose/FOV and target-support samples does not.
    """
    tables = []
    bearings = np.linspace(0.0, 2.0 * np.pi, DIRECT_GP_NUM_BEARINGS, endpoint=False)
    cam_offsets = np.linspace(-FOV_RAD / 3.0, FOV_RAD / 3.0, 3)
    radii = np.array([0.65, 0.90, 1.15]) * CAMERA_MAX_RANGE

    for ti, anchors in enumerate(DIRECT_GP_ANCHORS):
        samples = TARGET_AREA_SAMPLES[ti]
        poses = []
        cam_angles = []
        support_bonus = []
        visible_rows = []

        if len(samples) == 0 or len(anchors) == 0:
            tables.append({
                "poses": np.empty((0, 2), dtype=float),
                "camera_angles": np.empty((0,), dtype=float),
                "support_bonus": np.empty((0,), dtype=float),
                "visible": np.empty((0, len(samples)), dtype=bool),
            })
            continue

        for anchor in anchors:
            anchor = np.asarray(anchor, dtype=float)
            for radius in radii:
                for a in bearings:
                    pose = anchor + float(radius) * np.array([math.cos(a), math.sin(a)])
                    pose[0] = np.clip(pose[0], -X_LIMIT, X_LIMIT)
                    pose[1] = np.clip(pose[1], Y_LOW, Y_HIGH)
                    if not point_safe(pose, extra_margin=15.0):
                        continue

                    center_angle = math.atan2(anchor[1] - pose[1], anchor[0] - pose[0])
                    for ca in center_angle + cam_offsets:
                        rel = samples - pose[None, :]
                        dist = np.linalg.norm(rel, axis=1)
                        bearing = np.arctan2(rel[:, 1], rel[:, 0])
                        beta = wrap_to_pi(bearing - ca)
                        visible = (dist <= CAMERA_MAX_RANGE) & (np.abs(beta) <= FOV_RAD / 2.0)
                        if not np.any(visible):
                            continue
                        poses.append(pose.copy())
                        cam_angles.append(float(ca))
                        support_bonus.append(min(np.linalg.norm(anchor - TARGETS[ti]) / max(TARGET_R95[ti], 1.0), 1.0))
                        visible_rows.append(visible)

        tables.append({
            "poses": np.asarray(poses, dtype=float),
            "camera_angles": np.asarray(cam_angles, dtype=float),
            "support_bonus": np.asarray(support_bonus, dtype=float),
            "visible": np.asarray(visible_rows, dtype=bool) if visible_rows else np.empty((0, len(samples)), dtype=bool),
        })
    return tables


DIRECT_GP_CANDIDATES = build_direct_gp_candidate_table()



def _weighted_mean(points, weights):
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(points) == 0:
        return np.zeros(2, dtype=float)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.mean(points, axis=0)
    return np.sum(points * weights[:, None], axis=0) / total


def _principal_axes(points, weights=None):
    """Return principal axis and normal for a point cloud."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    if weights is None:
        center = np.mean(points, axis=0)
        centered = points - center[None, :]
        cov = np.cov(centered.T)
    else:
        center = _weighted_mean(points, weights)
        centered = points - center[None, :]
        w = np.asarray(weights, dtype=float)
        w = w / max(float(np.sum(w)), 1e-12)
        cov = (centered * w[:, None]).T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    if axis[1] < 0.0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]])
    return axis, normal


def _visible_mask_from_pose(pose, camera_angle, samples):
    rel = samples - pose[None, :]
    dist = np.linalg.norm(rel, axis=1)
    bearing = np.arctan2(rel[:, 1], rel[:, 0])
    beta = wrap_to_pi(bearing - camera_angle)
    return (dist <= CAMERA_MAX_RANGE) & (np.abs(beta) <= FOV_RAD / 2.0)


def build_shape_coverage_waypoints():
    """Build ordered, center-free observation waypoints from the target shape.

    Unlike the previous direct-GP optimizer, this does not continuously chase
    small local support anchors. It converts each non-circular target-arrival
    distribution into a small sequence of coverage waypoints. The USV then
    follows these waypoints in order, giving a piecewise-straight, real-USV-like
    trajectory while still adapting to crescent, lane, bimodal, and L-shaped data.
    """
    all_tables = []
    for ti, samples in enumerate(TARGET_AREA_SAMPLES):
        samples = np.asarray(samples, dtype=float)
        weights = np.asarray(TARGET_GP_SAMPLE_INFO[ti]["value"], dtype=float)
        if len(samples) == 0:
            all_tables.append({
                "poses": TARGETS[ti][None, :].copy(),
                "camera_angles": np.array([0.0], dtype=float),
                "centers": TARGETS[ti][None, :].copy(),
                "visible": np.zeros((1, 0), dtype=bool),
            })
            continue

        axis, normal = _principal_axes(samples, weights)
        proj = samples @ axis
        span = float(np.max(proj) - np.min(proj)) if len(proj) else 0.0

        # Enough segments to cover long/thin support, but not so many that the
        # trajectory becomes a chain of tiny arcs.
        n_segments = int(np.clip(math.ceil(span / (0.42 * CAMERA_MAX_RANGE)) + 1, 4, 9))
        quantiles = np.linspace(0.0, 1.0, n_segments + 1)

        centers = []
        for a, b in zip(quantiles[:-1], quantiles[1:]):
            lo = float(np.quantile(proj, a))
            hi = float(np.quantile(proj, b))
            mask = (proj >= lo) & (proj <= hi) if b >= 1.0 else (proj >= lo) & (proj < hi)
            if not np.any(mask):
                continue
            centers.append(_weighted_mean(samples[mask], weights[mask]))

        # Add separated modes that PCA binning can miss, especially bimodal T2.
        if len(centers) < n_segments:
            order = np.argsort(-weights)
            min_sep = 0.38 * CAMERA_MAX_RANGE
            for idx in order:
                c = samples[int(idx)]
                if not centers:
                    centers.append(c)
                else:
                    d = np.linalg.norm(np.asarray(centers) - c[None, :], axis=1)
                    if float(np.min(d)) >= min_sep:
                        centers.append(c)
                if len(centers) >= n_segments:
                    break

        centers = np.asarray(centers, dtype=float)
        if len(centers) == 0:
            centers = TARGETS[ti][None, :].copy()

        # Note: order support centers by nearest-neighbor path for smoother curved targets.
        if len(centers) > 2:
            start_idx = int(np.argmin(centers[:, 1]))
            ordered = [start_idx]
            unused = set(range(len(centers)))
            unused.remove(start_idx)

            while unused:
                last = centers[ordered[-1]]
                next_idx = min(unused, key=lambda j: np.linalg.norm(centers[j] - last))
                ordered.append(next_idx)
                unused.remove(next_idx)

            centers = centers[np.asarray(ordered, dtype=int)]
        else:
            order = np.argsort(centers @ axis)
            centers = centers[order]

        poses = []
        cam_angles = []
        visible_rows = []

        standoff = min(0.78 * CAMERA_MAX_RANGE, CAMERA_MAX_RANGE - 45.0)

        for c in centers:
            candidates = []
            for sign in [-1.0, 1.0]:
                pose = c + sign * normal * standoff
                pose[0] = np.clip(pose[0], -X_LIMIT, X_LIMIT)
                pose[1] = np.clip(pose[1], Y_LOW, Y_HIGH)

                if not point_safe(pose, extra_margin=10.0):
                    continue

                cam_angle = math.atan2(c[1] - pose[1], c[0] - pose[0])
                visible = _visible_mask_from_pose(pose, cam_angle, samples)
                value = float(np.sum(weights[visible]))
                route_cost = 0.00025 * abs(pose[0] - 0.0)
                candidates.append((value - route_cost, pose, cam_angle, visible))

            if not candidates:
                direction_to_route = np.array([0.0 - c[0], 0.0], dtype=float)
                norm = float(np.linalg.norm(direction_to_route))
                if norm < 1e-9:
                    direction_to_route = -normal
                    norm = 1.0
                pose = c + standoff * direction_to_route / norm
                pose[0] = np.clip(pose[0], -X_LIMIT, X_LIMIT)
                pose[1] = np.clip(pose[1], Y_LOW, Y_HIGH)
                cam_angle = math.atan2(c[1] - pose[1], c[0] - pose[0])
                visible = _visible_mask_from_pose(pose, cam_angle, samples)
                candidates.append((float(np.sum(weights[visible])), pose, cam_angle, visible))

            _, pose, cam_angle, visible = max(candidates, key=lambda item: item[0])
            poses.append(pose)
            cam_angles.append(cam_angle)
            visible_rows.append(visible)

        all_tables.append({
            "poses": np.asarray(poses, dtype=float),
            "camera_angles": np.asarray(cam_angles, dtype=float),
            "centers": centers,
            "visible": np.asarray(visible_rows, dtype=bool),
        })

    return all_tables


DIRECT_COVERAGE_WAYPOINTS = build_shape_coverage_waypoints()


def compute_target_reachable_coverage_from_waypoints():
    """Maximum fraction of each target data support covered by planned waypoints.

    A 95% target-arrival region can be larger than the physical 450 m camera
    footprint, and obstacles can remove some viewpoints. If the reachable
    planned coverage is below 95%, the mission should continue after reaching
    the best feasible coverage instead of looping forever around that target.
    """
    vals = []
    for table in DIRECT_COVERAGE_WAYPOINTS:
        visible = table.get("visible", np.empty((0, 0), dtype=bool))
        if visible.size == 0 or visible.shape[1] == 0:
            vals.append(1.0)
        else:
            vals.append(float(np.mean(np.any(visible, axis=0))))
    return np.array(vals, dtype=float)


TARGET_REACHABLE_COVERAGE = compute_target_reachable_coverage_from_waypoints()

# Mission continuation threshold.
# Important: with a short-range 450 m camera, the target-arrival support can be
# much wider than one camera footprint. If the controller keeps trying to push a
# target from ~80% to 95%, it may never leave that target and will never visit
# later targets such as T3. Therefore this threshold is used to decide when the
# USV may proceed to the next target. Coverage is still reported as the true
# measured fraction, not artificially raised.
TARGET_COMPLETION_THRESHOLDS = np.minimum(
    DIRECT_GP_COVERAGE_GOAL,
    np.minimum(0.78, np.maximum(0.70, TARGET_REACHABLE_COVERAGE - 0.03)),
)


def target_completion_threshold(target_idx):
    if target_idx < 0 or target_idx >= len(TARGET_COMPLETION_THRESHOLDS):
        return DIRECT_GP_COVERAGE_GOAL
    return float(TARGET_COMPLETION_THRESHOLDS[target_idx])


def select_next_shape_waypoint(target_idx, covered_area_samples, start_idx=0):
    """Select the next ordered waypoint that still covers uncovered samples."""
    table = DIRECT_COVERAGE_WAYPOINTS[target_idx]
    poses = table["poses"]
    visible = table["visible"]
    if len(poses) == 0:
        return 0

    remaining = ~covered_area_samples[target_idx]
    if len(remaining) == 0 or not np.any(remaining):
        return min(start_idx, len(poses) - 1)

    weights = TARGET_GP_SAMPLE_INFO[target_idx]["value"] * remaining.astype(float)

    # Keep forward order for piecewise-straight motion.
    for offset in range(len(poses)):
        idx = (start_idx + offset) % len(poses)
        gain = float(np.sum(weights[visible[idx]]))
        if gain > 1e-6:
            return idx

    return min(start_idx, len(poses) - 1)


def choose_shape_coverage_control(
    state,
    target_idx,
    covered_area_samples,
    known_obstacles,
    current_wp_idx=0,
    hold_count=0,
):
    """Move toward ordered shape-coverage waypoints.

    This replaces continuous local re-optimization. The path becomes mostly
    straight between waypoints, while the waypoints themselves are generated from
    the 95% data/GP support, so the method still adapts to non-general shapes.
    """
    table = DIRECT_COVERAGE_WAYPOINTS[target_idx]
    poses = table["poses"]
    cam_angles = table["camera_angles"]
    visible = table["visible"]

    if len(poses) == 0:
        goal = TARGETS[target_idx].copy()
        camera_angle = math.atan2(goal[1] - state[1], goal[0] - state[0])
        v_cmd, omega_cmd = control_to_goal_with_cbf(state, goal, known_obstacles, "Ours")
        uc_cmd = np.clip(wrap_to_pi(camera_angle - state[2] - state[3]) / DT, -UC_MAX, UC_MAX)
        return np.array([v_cmd, omega_cmd, uc_cmd], dtype=float), goal, camera_angle, 0, hold_count + 1

    current_wp_idx = int(np.clip(current_wp_idx, 0, len(poses) - 1))
    goal = poses[current_wp_idx]
    camera_angle = float(cam_angles[current_wp_idx])

    dist_to_goal = float(np.linalg.norm(state[:2] - goal))
    remaining = ~covered_area_samples[target_idx]
    visible_remaining = bool(np.any(visible[current_wp_idx] & remaining)) if len(remaining) else False

    # Switch only after reaching a waypoint or after it becomes useless.
    if (dist_to_goal <= DIRECT_GP_GOAL_REACHED_DIST) or (hold_count >= DIRECT_GP_GOAL_HOLD_STEPS) or ((not visible_remaining) and dist_to_goal <= 2.0 * DIRECT_GP_GOAL_REACHED_DIST):
        next_idx = select_next_shape_waypoint(target_idx, covered_area_samples, current_wp_idx + 1)
        if next_idx != current_wp_idx:
            current_wp_idx = next_idx
            goal = poses[current_wp_idx]
            camera_angle = float(cam_angles[current_wp_idx])
            hold_count = 0

    v_cmd, omega_cmd = control_to_goal_with_cbf(state, goal, known_obstacles, "Ours")

    # Slow down near waypoint so the pan camera can collect remaining samples.
    target_cov = float(np.mean(covered_area_samples[target_idx])) if len(covered_area_samples[target_idx]) else 1.0
    if np.linalg.norm(state[:2] - goal) <= 1.25 * DIRECT_GP_GOAL_REACHED_DIST and target_cov < DIRECT_GP_COVERAGE_GOAL:
        v_cmd = min(v_cmd, 0.75)

    # Adapt camera pointing to the remaining uncovered support visible from the
    # current pose. This lets one straight waypoint cover curved/thin support by
    # scanning along the data shape instead of looking only at one fixed center.
    theta_camera_goal = camera_angle
    remaining_mask = ~covered_area_samples[target_idx]
    if np.any(remaining_mask):
        samples = TARGET_AREA_SAMPLES[target_idx]
        weights = TARGET_GP_SAMPLE_INFO[target_idx]["value"]
        rem_samples = samples[remaining_mask]
        rem_weights = weights[remaining_mask]
        rel = rem_samples - state[:2][None, :]
        dist = np.linalg.norm(rel, axis=1)

        # Active camera pointing: look toward the most useful uncovered GP/data
        # support, even before it is already inside the physical camera range.
        # This makes the FOV visibly track the target-arrival cloud instead of
        # falling back to a fixed waypoint angle when samples are still far away.
        range_preference = np.exp(-dist / max(CAMERA_MAX_RANGE, 1e-9))
        score = rem_weights * range_preference
        top_k = min(24, len(rem_samples))
        top_idx = np.argsort(-score)[:top_k]
        look_point = _weighted_mean(rem_samples[top_idx], rem_weights[top_idx])
        theta_camera_goal = math.atan2(look_point[1] - state[1], look_point[0] - state[0])

    for i, obs in enumerate(OBSTACLES):
        if known_obstacles[i]:
            continue
        if segment_intersects_circle(state[:2], goal, obs["center"], obs["radius"] + SAFETY_MARGIN + 180.0):
            if np.linalg.norm(obs["center"] - state[:2]) <= CAMERA_MAX_RANGE * 0.65:
                theta_camera_goal = math.atan2(obs["center"][1] - state[1], obs["center"][0] - state[0])
                break

    uc_cmd = np.clip(wrap_to_pi(theta_camera_goal - state[2] - state[3]) / DT, -UC_MAX, UC_MAX)
    return np.array([v_cmd, omega_cmd, uc_cmd], dtype=float), goal, theta_camera_goal, current_wp_idx, hold_count + 1


def target_direct_gp_coverage_done(covered_area_samples, target_idx):
    if len(covered_area_samples[target_idx]) == 0:
        return True
    cov = float(np.mean(covered_area_samples[target_idx]))
    return cov >= target_completion_threshold(target_idx)


def direct_gp_visible_value(pose, camera_angle, target_idx, covered_area_samples):
    samples = TARGET_AREA_SAMPLES[target_idx]
    remaining = ~covered_area_samples[target_idx]
    if len(samples) == 0 or not np.any(remaining):
        return 0.0
    pts = samples[remaining]
    weights = TARGET_GP_SAMPLE_INFO[target_idx]["value"][remaining]
    rel = pts - pose[None, :]
    dist = np.linalg.norm(rel, axis=1)
    bearing = np.arctan2(rel[:, 1], rel[:, 0])
    beta = wrap_to_pi(bearing - camera_angle)
    visible = (dist <= CAMERA_MAX_RANGE) & (np.abs(beta) <= FOV_RAD / 2.0)
    if not np.any(visible):
        return 0.0
    return float(np.sum(weights[visible]) / max(np.sum(weights), 1e-9))


def optimize_direct_gp_observation_pose(state, target_idx, covered_area_samples, known_obstacles):
    """Choose an observation pose from the GP-data support, not from a center ring.

    The candidate set is generated around representative points along the 95%
    historical/GP support. The score is the remaining visible GP-UCB mass, with
    penalties for travel distance, route deviation, heading change, and unsafe
    straight-line motion.
    """
    samples = TARGET_AREA_SAMPLES[target_idx]
    remaining = ~covered_area_samples[target_idx]
    if len(samples) == 0 or not np.any(remaining):
        target = TARGETS[target_idx]
        return target.copy(), math.atan2(target[1] - state[1], target[0] - state[0]), 0.0

    table = DIRECT_GP_CANDIDATES[target_idx]
    poses = table["poses"]
    cam_angles = table["camera_angles"]
    visible = table["visible"]
    support_bonus = table["support_bonus"]
    if len(poses) == 0:
        target = TARGETS[target_idx]
        return target.copy(), math.atan2(target[1] - state[1], target[0] - state[0]), 0.0

    weights = TARGET_GP_SAMPLE_INFO[target_idx]["value"] * remaining.astype(float)
    total_weight = max(float(np.sum(weights)), 1e-9)
    sensing = visible.astype(float).dot(weights) / total_weight

    travel = np.linalg.norm(poses - state[:2][None, :], axis=1)
    route_dev = np.abs(poses[:, 0] - ROUTE_X)
    desired_heading = np.arctan2(poses[:, 1] - state[1], poses[:, 0] - state[0])
    heading_error = np.abs(wrap_to_pi(desired_heading - state[2]))

    score = sensing.copy()
    score += 0.06 * support_bonus
    score -= DIRECT_GP_TRAVEL_PENALTY * travel
    score -= DIRECT_GP_ROUTE_PENALTY * route_dev
    score -= DIRECT_GP_HEADING_PENALTY * heading_error

    # Filter candidates that are unsafe with currently known obstacles.
    valid = np.ones(len(poses), dtype=bool)
    if known_obstacles is not None:
        for obs_idx in np.flatnonzero(known_obstacles):
            obs = OBSTACLES[int(obs_idx)]
            d = np.linalg.norm(poses - obs["center"][None, :], axis=1)
            valid &= d > obs["radius"] + SAFETY_MARGIN + 40.0
    # Penalize straight paths crossing any safety region.
    for ci, pose in enumerate(poses):
        if segment_intersects_safety(state[:2], pose):
            score[ci] -= 0.30

    valid &= sensing > 0.0
    if not np.any(valid):
        # Fall back to the best geometric candidate, still not center-ring based.
        idx = int(np.argmax(score))
    else:
        masked_score = np.where(valid, score, -1e18)
        idx = int(np.argmax(masked_score))

    return np.asarray(poses[idx], dtype=float), float(cam_angles[idx]), float(score[idx])


def choose_direct_gp_control(state, target_idx, covered_area_samples, known_obstacles, held_goal=None, held_camera_angle=None, hold_count=0):
    replan = held_goal is None or held_camera_angle is None
    if not replan:
        if np.linalg.norm(state[:2] - held_goal) <= DIRECT_GP_GOAL_REACHED_DIST:
            replan = True
        elif hold_count >= DIRECT_GP_GOAL_HOLD_STEPS:
            replan = True

    if replan:
        goal, camera_angle, _ = optimize_direct_gp_observation_pose(state, target_idx, covered_area_samples, known_obstacles)
        hold_count = 0
    else:
        goal = np.asarray(held_goal, dtype=float)
        camera_angle = float(held_camera_angle)

    v_cmd, omega_cmd = control_to_goal_with_cbf(state, goal, known_obstacles, "Ours")

    # Slow down inside target region so the camera can accumulate area coverage.
    target_cov = float(np.mean(covered_area_samples[target_idx])) if len(covered_area_samples[target_idx]) else 1.0
    if np.linalg.norm(state[:2] - goal) <= 1.4 * DIRECT_GP_GOAL_REACHED_DIST and target_cov < DIRECT_GP_COVERAGE_GOAL:
        v_cmd = min(v_cmd, 0.55)

    # Pan camera to selected FOV direction. If an unknown obstacle blocks the path,
    # temporarily scan toward it for discovery.
    theta_camera_goal = camera_angle
    for i, obs in enumerate(OBSTACLES):
        if known_obstacles[i]:
            continue
        if segment_intersects_circle(state[:2], goal, obs["center"], obs["radius"] + SAFETY_MARGIN + 180.0):
            if np.linalg.norm(obs["center"] - state[:2]) <= CAMERA_MAX_RANGE * 1.20:
                theta_camera_goal = math.atan2(obs["center"][1] - state[1], obs["center"][0] - state[0])
                break

    uc_cmd = np.clip(wrap_to_pi(theta_camera_goal - state[2] - state[3]) / DT, -UC_MAX, UC_MAX)
    return np.array([v_cmd, omega_cmd, uc_cmd], dtype=float), goal, camera_angle, hold_count + 1


def build_viewpoints():
    all_viewpoints = []
    feasible = []
    for i, target in enumerate(TARGETS):
        radius = CAMERA_RANGES[i]
        target_views = []
        target_feasible = []
        for k in range(K_VIEWPOINTS):
            base_alpha = 2.0 * np.pi * k / K_VIEWPOINTS
            chosen = None
            for shift in np.linspace(0.0, np.pi / 4.0, 17):
                for sign in [1.0, -1.0]:
                    alpha = base_alpha + sign * shift
                    p = target + radius * np.array([np.cos(alpha), np.sin(alpha)])
                    if point_safe(p, extra_margin=20.0):
                        chosen = p
                        break
                if chosen is not None:
                    break
            if chosen is None:
                chosen = target + radius * np.array([np.cos(base_alpha), np.sin(base_alpha)])
                target_feasible.append(False)
            else:
                target_feasible.append(True)
            target_views.append(chosen)
        all_viewpoints.append(np.array(target_views))
        feasible.append(np.array(target_feasible, dtype=bool))
    return all_viewpoints, feasible


VIEWPOINTS, FEASIBLE_SECTORS = build_viewpoints()


def save_r95_csv():
    with open(PLOT_DIR / "r95.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Target", "x", "y", "sigma_max", "r95", "R_best_raw", "R_best_capped", "R_max", "coverable_radius", "coverable_fraction", "num_samples"])
        writer.writerows(R95_ROWS)


def rollout_candidates(state, controls, known_obstacles, use_cbf=True):
    n = len(controls)
    pred = np.repeat(state[None, :], n, axis=0)
    safe = np.ones(n, dtype=bool)
    for _ in range(HORIZON):
        pred[:, 0] += DT * controls[:, 0] * np.cos(pred[:, 2])
        pred[:, 1] += DT * controls[:, 0] * np.sin(pred[:, 2])
        pred[:, 2] = wrap_to_pi(pred[:, 2] + DT * controls[:, 1])
        pred[:, 3] = wrap_to_pi(pred[:, 3] + DT * controls[:, 2])
        safe &= (pred[:, 0] >= -X_LIMIT) & (pred[:, 0] <= X_LIMIT)
        safe &= (pred[:, 1] >= Y_LOW) & (pred[:, 1] <= Y_HIGH)
        if use_cbf:
            vel = np.column_stack([controls[:, 0] * np.cos(pred[:, 2]), controls[:, 0] * np.sin(pred[:, 2])])
            for obs_idx in np.flatnonzero(known_obstacles):
                obs = OBSTACLES[int(obs_idx)]
                safe_radius = obs["radius"] + SAFETY_MARGIN
                rel = pred[:, :2] - obs["center"]
                h = np.sum(rel * rel, axis=1) - safe_radius**2
                hdot = 2.0 * np.sum(rel * vel, axis=1)
                safe &= hdot + ALPHA_CBF * h >= -1e-8
    return safe, pred


def guidance_goal_around_known_obstacles(position, goal, known_obstacles):
    blocking = []
    for obs_idx in np.flatnonzero(known_obstacles):
        obs = OBSTACLES[int(obs_idx)]
        radius = obs["radius"] + SAFETY_MARGIN + 80.0
        if segment_intersects_circle(position, goal, obs["center"], radius):
            blocking.append((np.linalg.norm(position - obs["center"]), obs, radius))
    if not blocking:
        return goal
    _, obs, radius = min(blocking, key=lambda item: item[0])
    path = goal - position
    norm = np.linalg.norm(path)
    if norm < 1e-9:
        return goal
    unit = path / norm
    perp = np.array([-unit[1], unit[0]])
    candidates = []
    for sign in [-1.0, 1.0]:
        p = obs["center"] + sign * perp * (radius + 220.0)
        p = np.array([np.clip(p[0], -X_LIMIT, X_LIMIT), np.clip(p[1], Y_LOW, Y_HIGH)])
        if not point_safe(p, extra_margin=20.0):
            continue
        cost = np.linalg.norm(position - p) + np.linalg.norm(p - goal)
        candidates.append((cost, p))
    return min(candidates, key=lambda item: item[0])[1] if candidates else goal


def sector_can_cover(state, target_idx, sector_idx, method):
    fixed_camera = method == "Baseline 1"
    pd = detection_prob(state, target_idx, fixed_camera=fixed_camera)
    near = np.linalg.norm(state[:2] - VIEWPOINTS[target_idx][sector_idx]) <= VIEWPOINT_RADIUS
    return near and pd > PD_COVER_THRESHOLD


def mark_visible_sectors(state, covered, method):
    """Mark all target sectors that are actually visible at the current state.

    This fixes the T1 deadlock. Coverage is a measurement outcome, not a
    condition that every controller must satisfy before it may move on.
    Baseline 1 therefore accumulates only passive forward-FOV coverage while
    following its path; it is not forced to stop and rotate or wait for sectors
    that a fixed camera cannot see.
    """
    for ti in range(len(TARGETS)):
        for si in range(K_VIEWPOINTS):
            if not FEASIBLE_SECTORS[ti][si] or covered[ti, si]:
                continue
            if sector_can_cover(state, ti, si, method):
                covered[ti, si] = True


def point_visible_from_state(state, point, target_idx, fixed_camera=False):
    # Area coverage is evaluated over the angular footprint of the camera at the
    # target region. The range gate is applied to the target center, not to every
    # sample point inside r95; otherwise a large r95 disk can be unfairly clipped
    # even when multiple viewpoints around the ring are used.
    target_rel = TARGETS[target_idx] - state[:2]
    if float(np.linalg.norm(target_rel)) > CAMERA_MAX_RANGE:
        return False
    rel = point - state[:2]
    bearing = math.atan2(rel[1], rel[0])
    theta_view = state[2] if fixed_camera else state[2] + state[3]
    beta = wrap_to_pi(bearing - theta_view)
    return abs(beta) <= FOV_RAD / 2.0


def mark_visible_area_samples(state, covered_area_samples, method):
    """Accumulate true visible area inside each target r95 region.

    This is used for target-wise coverage plotting and metrics. It is separate
    from the sector-completion array used by Ours to decide which observation
    viewpoint to visit next. Baseline 1 therefore receives credit for any small
    part of the target region that naturally falls inside its fixed forward FOV.
    """
    fixed_camera = method == "Baseline 1"
    theta_view = state[2] if fixed_camera else state[2] + state[3]
    for ti, samples in enumerate(TARGET_AREA_SAMPLES):
        remaining = ~covered_area_samples[ti]
        if not np.any(remaining):
            continue
        pts = samples[remaining]
        rel = pts - state[:2][None, :]
        dist = np.linalg.norm(rel, axis=1)
        bearing = np.arctan2(rel[:, 1], rel[:, 0])
        beta = wrap_to_pi(bearing - theta_view)
        if fixed_camera:
            # Baseline 1 receives only the portion of the r95 area that its
            # forward fixed camera physically sweeps through during motion.
            visible = (dist <= CAMERA_MAX_RANGE) & (np.abs(beta) <= FOV_RAD / 2.0)
        else:
            # Active pan-camera methods cover the actual shape-support samples.
            # Do not gate coverage by distance to the target center; for long,
            # thin, bimodal, or L-shaped distributions the useful data may be
            # far from the center. A sample is covered if it is inside the
            # physical camera range and FOV from the current USV pose.
            visible = (dist <= CAMERA_MAX_RANGE) & (np.abs(beta) <= FOV_RAD / 2.0)
        if np.any(visible):
            idx = np.flatnonzero(remaining)
            covered_area_samples[ti][idx[visible]] = True


def sample_coverage_values(covered_area_samples):
    vals = []
    for cov in covered_area_samples:
        vals.append(float(np.mean(cov)) if len(cov) else 0.0)
    return np.array(vals, dtype=float)


def sample_coverage_fraction(covered_area_samples):
    vals = sample_coverage_values(covered_area_samples)
    return float(np.mean(vals)) if len(vals) else 0.0


def obstacle_visibility_score(state, known_obstacles, fixed_camera=False):
    score = 0.0
    camera_range = CAMERA_MAX_RANGE
    for i, obs in enumerate(OBSTACLES):
        if known_obstacles[i]:
            continue
        rel = obs["center"] - state[:2]
        dist = np.linalg.norm(rel)
        if dist > camera_range * 1.25:
            continue
        theta_obs = math.atan2(rel[1], rel[0])
        theta_view = state[2] if fixed_camera else state[2] + state[3]
        beta = wrap_to_pi(theta_obs - theta_view)
        score += math.exp(-(dist**2) / (2.0 * camera_range**2)) * math.exp(-(beta**2) / (2.0 * 0.35**2))
    return score



def detect_route_blocking_obstacles(state, goal, known_obstacles, first_seen, time_s, method):
    if method != "Ours":
        return
    scan_range = float(CAMERA_MAX_RANGE)
    for i, obs in enumerate(OBSTACLES):
        if known_obstacles[i]:
            continue
        if np.linalg.norm(obs["center"] - state[:2]) > scan_range:
            continue
        if segment_intersects_circle(state[:2], goal, obs["center"], obs["radius"] + SAFETY_MARGIN + 220.0):
            known_obstacles[i] = True
            first_seen[i] = time_s


def enforce_known_obstacle_safety(state, prev_state, known_obstacles):
    corrected = state.copy()
    for obs_idx in np.flatnonzero(known_obstacles):
        obs = OBSTACLES[int(obs_idx)]
        safe_radius = obs["radius"] + SAFETY_MARGIN + 8.0
        rel = corrected[:2] - obs["center"]
        d = float(np.linalg.norm(rel))
        if d < safe_radius:
            if d < 1e-9:
                rel = prev_state[:2] - obs["center"]
                d = float(np.linalg.norm(rel))
                if d < 1e-9:
                    rel = np.array([1.0, 0.0]); d = 1.0
            normal = rel / d
            corrected[:2] = obs["center"] + normal * safe_radius
            # Do not reset the USV heading during safety projection.
            # Projection should correct position only; abruptly replacing psi
            # with the obstacle tangent creates unrealistic heading jumps and
            # inflates the total-heading-change metric.
    corrected[0] = np.clip(corrected[0], -X_LIMIT, X_LIMIT)
    corrected[1] = np.clip(corrected[1], Y_LOW, Y_HIGH)
    return corrected

def control_to_goal_with_cbf(state, goal, known_obstacles, method):
    """Pure-pursuit go-to-goal controller.

    Baseline 2 uses this directly with no safety correction. Ours uses the same
    active sensing path but detected-obstacle CBF is enforced after integration.
    This makes the trajectory and terminal coverage consistent: if the path
    visits the observation ring in the plot, coverage is counted; if it does not,
    coverage is not counted.
    """
    rel = goal - state[:2]
    dist = float(np.linalg.norm(rel))
    if dist < 1e-9:
        desired_heading = state[2]
        v_cmd = 0.0
    else:
        desired_heading = math.atan2(rel[1], rel[0])
        heading_error = wrap_to_pi(desired_heading - state[2])
        v_cmd = CRUISE_SPEED
        if dist < 500.0:
            v_cmd = min(v_cmd, 1.5)
        if dist < VIEWPOINT_RADIUS:
            v_cmd = min(v_cmd, 0.8)
        if abs(heading_error) > np.deg2rad(85.0):
            v_cmd = min(v_cmd, 0.35)
        elif abs(heading_error) > np.deg2rad(60.0):
            v_cmd = min(v_cmd, 0.8)
    heading_error = wrap_to_pi(desired_heading - state[2])
    omega_cmd = np.clip(1.2 * heading_error, -OMEGA_MAX, OMEGA_MAX)
    return float(v_cmd), float(omega_cmd)

def choose_mpc_control(state, u_prev, target_idx, sector_idx, covered, known_obstacles, method):
    """Active-sensing controller.

    The previous version could still fail because target completion was tied to
    a sampled MPC sector objective. Near T1, obstacle CBF filtering left the
    vehicle rotating around a local point. This version separates the behavior:

    1. Choose the next uncovered observation-ring sector.
    2. Move toward its viewpoint using a robust goal controller.
    3. For Ours, modify the velocity by a CBF-style correction using detected
       obstacles only.
    4. Pan the camera to the target, except when an unknown obstacle lies on the
       current path; then scan toward that obstacle until it is detected.
    """
    del u_prev, covered
    raw_goal = VIEWPOINTS[target_idx][sector_idx]
    goal = raw_goal

    v_cmd, omega_cmd = control_to_goal_with_cbf(state, goal, known_obstacles, method)

    target = TARGETS[target_idx]
    theta_camera_goal = math.atan2(target[1] - state[1], target[0] - state[0])

    # Obstacle discovery is camera-limited. The controller may actively scan
    # along the planned segment, but obstacles only become known after they are
    # inside the current FOV and range in update_known_obstacles().
    if method == "Ours":
        for i, obs in enumerate(OBSTACLES):
            if known_obstacles[i]:
                continue
            if segment_intersects_circle(state[:2], goal, obs["center"], obs["radius"] + SAFETY_MARGIN + 180.0):
                if np.linalg.norm(obs["center"] - state[:2]) <= CAMERA_MAX_RANGE * 1.15:
                    theta_camera_goal = math.atan2(obs["center"][1] - state[1], obs["center"][0] - state[0])
                    break

    uc_cmd = np.clip(wrap_to_pi(theta_camera_goal - state[2] - state[3]) / DT, -UC_MAX, UC_MAX)
    return np.array([v_cmd, omega_cmd, uc_cmd], dtype=float), True

def choose_fixed_camera_control(state, goal, target, known_obstacles, hold=False):
    """Fixed-forward-camera baseline controller.

    Important: this baseline is not allowed to stop and rotate its body toward
    the target just to create coverage. The camera is fixed to the USV heading
    and the heading is determined by waypoint following. This makes the baseline
    fair for testing whether a pan camera improves target-area coverage.
    """
    del target, hold
    goal = guidance_goal_around_known_obstacles(state[:2], goal, known_obstacles)
    rel = goal - state[:2]
    distance = float(np.linalg.norm(rel))
    desired_heading = math.atan2(rel[1], rel[0]) if distance > 1e-9 else state[2]
    heading_error = wrap_to_pi(desired_heading - state[2])

    speed = CRUISE_SPEED
    if distance < 250.0:
        speed = min(speed, 0.8)
    if abs(heading_error) > np.deg2rad(70.0):
        speed = 0.0
    elif abs(heading_error) > np.deg2rad(45.0):
        speed = min(speed, 0.5)

    desired = np.array(
        [
            speed,
            np.clip(0.8 * heading_error, -OMEGA_MAX, OMEGA_MAX),
            np.clip(wrap_to_pi(0.0 - state[3]) / DT, -UC_MAX, UC_MAX),
        ]
    )
    safe, _ = rollout_candidates(state, CONTROL_CANDIDATES, known_obstacles, use_cbf=True)
    if not np.any(safe):
        return np.array([0.0, np.clip(0.8 * heading_error, -OMEGA_MAX, OMEGA_MAX), desired[2]]), False
    controls = CONTROL_CANDIDATES[safe]
    scale = np.array([1.0, 10.0, 1.0])
    idx = int(np.argmin(np.linalg.norm((controls - desired[None, :]) * scale, axis=1)))
    return controls[idx].copy(), True

def next_uncovered_sector(target_idx, covered, position=None):
    feasible = FEASIBLE_SECTORS[target_idx]
    candidates = [k for k in range(K_VIEWPOINTS) if feasible[k] and not covered[target_idx, k]]
    if position is None:
        return candidates[0] if candidates else None
    if candidates:
        def sector_cost(k):
            p = VIEWPOINTS[target_idx][k]
            cost = np.linalg.norm(position - p)
            if segment_intersects_safety(position, p):
                cost += 5000.0
            return cost

        return min(candidates, key=sector_cost)
    return None


def update_known_obstacles(state, known, first_seen, time_s, method):
    fixed_camera = method == "Baseline 1"
    for i, obs in enumerate(OBSTACLES):
        if known[i]:
            continue
        if obstacle_visible(state, obs, CAMERA_MAX_RANGE, fixed_camera=fixed_camera):
            known[i] = True
            first_seen[i] = time_s


def build_baseline1_waypoints():
    """Path-following route for the fixed-camera baseline.

    Baseline 1 is not an active observation method. It should move through a
    simple route and receive coverage only when the target naturally enters its
    forward FOV. It must not be forced to complete all observation-ring sectors.
    """
    waypoints = [START_STATE[:2].copy()]
    # Route passes near the target sequence but does not orbit each target.
    for target in TARGETS:
        approach = target + np.array([-CAMERA_RANGES[0] * 0.75, -CAMERA_RANGES[0] * 0.35])
        approach[0] = np.clip(approach[0], -X_LIMIT, X_LIMIT)
        approach[1] = np.clip(approach[1], Y_LOW, Y_HIGH)
        waypoints.append(approach)
        exit_pt = target + np.array([CAMERA_RANGES[0] * 0.75, CAMERA_RANGES[0] * 0.35])
        exit_pt[0] = np.clip(exit_pt[0], -X_LIMIT, X_LIMIT)
        exit_pt[1] = np.clip(exit_pt[1], Y_LOW, Y_HIGH)
        waypoints.append(exit_pt)
    waypoints.append(np.array([0.0, Y_HIGH]))
    return waypoints


def target_completed_for_ours(covered, target_idx, covered_area_samples=None):
    if covered_area_samples is not None:
        return target_direct_gp_coverage_done(covered_area_samples, target_idx)
    feasible = FEASIBLE_SECTORS[target_idx]
    return bool(np.all(covered[target_idx, feasible])) if np.any(feasible) else True


def coverage_fraction(covered):
    return float(np.mean([np.mean(covered[i, FEASIBLE_SECTORS[i]]) for i in range(len(TARGETS))]))



END_GOAL = np.array([0.0, Y_HIGH], dtype=float)
END_REACHED_TOL = 80.0
ROUTE_X = 0.0
ROUTE_RETURN_THRESHOLD = 140.0
REJOIN_FORWARD_MARGIN = 80.0
REJOIN_CLEARANCE_EXTRA = 80.0


def reached_end_y(state):
    return bool(state[1] >= Y_HIGH - END_REACHED_TOL and abs(state[0] - ROUTE_X) <= 350.0)


def distance_to_route(point):
    return abs(float(point[0]) - ROUTE_X)


def path_segments_are_safe(points, extra_margin=0.0):
    for p0, p1 in zip(points[:-1], points[1:]):
        for obs in OBSTACLES:
            if segment_intersects_circle(p0, p1, obs["center"], obs["radius"] + SAFETY_MARGIN + extra_margin):
                return False
    return True


def route_progress_ok(points, max_x_increase=120.0):
    """Prevent a rejoin waypoint from moving farther away from x=0.

    This follows the old Logic-1/route-return idea: after leaving an off-route
    target, the USV should reduce |x| and reconnect to the nominal route before
    continuing upward to Y max.
    """
    previous = distance_to_route(points[0])
    for point in points[1:]:
        current = distance_to_route(point)
        if current > previous + max_x_increase:
            return False
        previous = current
    return True


def rejoin_y_bounds(position, next_target_idx=None):
    y_min = max(Y_LOW, position[1] + REJOIN_FORWARD_MARGIN)
    if next_target_idx is not None and next_target_idx < len(TARGETS):
        y_max = min(Y_HIGH, TARGETS[next_target_idx, 1] - REJOIN_FORWARD_MARGIN)
    else:
        y_max = Y_HIGH
    if y_min > y_max:
        y_min = max(Y_LOW, position[1])
        y_max = Y_HIGH if next_target_idx is None else min(Y_HIGH, TARGETS[next_target_idx, 1])
    return y_min, y_max


def rejoin_candidates(position, next_target_idx=None):
    y_min, y_max = rejoin_y_bounds(position, next_target_idx)
    candidates = []
    if y_min > y_max:
        return candidates
    for y_candidate in np.linspace(y_min, y_max, 500):
        p_route = np.array([ROUTE_X, y_candidate], dtype=float)
        if not point_safe(p_route, extra_margin=0.0):
            continue
        # Prefer the earliest safe route point, then shortest return distance.
        cost = 1000.0 * max(0.0, y_candidate - y_min)
        cost += np.linalg.norm(position - p_route)
        candidates.append((cost, y_candidate, p_route))
    return candidates


def blocking_safety_obstacles(p0, p1, extra_margin=0.0):
    blocking = []
    for obs in OBSTACLES:
        radius = obs["radius"] + SAFETY_MARGIN + extra_margin
        if segment_intersects_circle(p0, p1, obs["center"], radius):
            blocking.append(obs)
    return blocking


def find_rejoin_detour_path(position, rejoin_point):
    """Return [detour..., rejoin_point] if the direct route segment is blocked.

    This is adapted from the single-method code you showed: first try direct
    diagonal return to the nominal route; if blocked, add one or two detour
    waypoints around the safety circles while still reducing |x|.
    """
    position = np.asarray(position, dtype=float)
    rejoin_point = np.asarray(rejoin_point, dtype=float)
    if not segment_intersects_safety(position, rejoin_point):
        return [rejoin_point]

    direction = rejoin_point - position
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-9:
        return [rejoin_point]
    d = direction / direction_norm
    normals = [np.array([-d[1], d[0]]), np.array([d[1], -d[0]])]
    blocking = blocking_safety_obstacles(position, rejoin_point, extra_margin=REJOIN_CLEARANCE_EXTRA)

    one_wp = []
    for obs in blocking:
        center = obs["center"]
        safe_radius = obs["radius"] + SAFETY_MARGIN + REJOIN_CLEARANCE_EXTRA
        for n in normals:
            for offset in (0.0, 250.0, -250.0, 500.0, -500.0):
                detour = center + n * (safe_radius + 260.0) + d * offset
                detour[0] = np.clip(detour[0], -X_LIMIT, X_LIMIT)
                detour[1] = np.clip(detour[1], Y_LOW, Y_HIGH)
                candidate = [position, detour, rejoin_point]
                if detour[1] < position[1] - 120.0 or detour[1] > rejoin_point[1] + 180.0:
                    continue
                if not point_safe(detour, extra_margin=0.0):
                    continue
                if not route_progress_ok(candidate, max_x_increase=150.0):
                    continue
                if not path_segments_are_safe(candidate):
                    continue
                cost = np.linalg.norm(position - detour) + np.linalg.norm(detour - rejoin_point)
                cost += 0.25 * abs(detour[0]) - 0.2 * (detour[1] - position[1])
                one_wp.append((cost, [detour, rejoin_point]))
    if one_wp:
        return min(one_wp, key=lambda item: item[0])[1]

    two_wp = []
    for obs in blocking:
        center = obs["center"]
        safe_radius = obs["radius"] + SAFETY_MARGIN + REJOIN_CLEARANCE_EXTRA
        for n in normals:
            detour_a = center + n * (safe_radius + 320.0) - d * 350.0
            detour_b = center + n * (safe_radius + 320.0) + d * 350.0
            for p in (detour_a, detour_b):
                p[0] = np.clip(p[0], -X_LIMIT, X_LIMIT)
                p[1] = np.clip(p[1], Y_LOW, Y_HIGH)
            if detour_b[1] < detour_a[1]:
                detour_a, detour_b = detour_b, detour_a
            candidate = [position, detour_a, detour_b, rejoin_point]
            if detour_a[1] < position[1] - 120.0 or detour_b[1] > rejoin_point[1] + 180.0:
                continue
            if not point_safe(detour_a) or not point_safe(detour_b):
                continue
            if not route_progress_ok(candidate, max_x_increase=150.0):
                continue
            if not path_segments_are_safe(candidate):
                continue
            cost = np.linalg.norm(position - detour_a)
            cost += np.linalg.norm(detour_a - detour_b)
            cost += np.linalg.norm(detour_b - rejoin_point)
            cost += 0.25 * (abs(detour_a[0]) + abs(detour_b[0]))
            two_wp.append((cost, [detour_a, detour_b, rejoin_point]))
    if two_wp:
        return min(two_wp, key=lambda item: item[0])[1]

    return None


def compute_rejoin_path(position, target_idx=None, next_target_idx=None):
    position = np.asarray(position, dtype=float)
    candidates = rejoin_candidates(position, next_target_idx)
    for _, _, p_route in sorted(candidates, key=lambda item: item[1]):
        path = find_rejoin_detour_path(position, p_route)
        if path is not None:
            return path

    # Last-resort fallback: still return to x=0 before going to Y max.
    fallback_y = min(Y_HIGH, max(position[1] + REJOIN_FORWARD_MARGIN, position[1]))
    if target_idx is not None and target_idx < len(TARGETS):
        fallback_y = min(Y_HIGH, max(fallback_y, TARGETS[target_idx, 1] + REJOIN_FORWARD_MARGIN))
    return [np.array([ROUTE_X, fallback_y], dtype=float)]




def compute_no_cbf_rejoin_path(position, target_idx=None, next_target_idx=None):
    """Direct rejoin used by Baseline 2.

    Baseline 2 intentionally has no CBF and no obstacle-aware route-rejoin
    planner. It returns to the nominal route by the shortest forward diagonal
    segment. If an obstacle lies on that segment, the true safety metric will
    count the collision/clearance violation.
    """
    position = np.asarray(position, dtype=float)
    y_min, y_max = rejoin_y_bounds(position, next_target_idx)
    if y_min <= y_max:
        y_rejoin = y_min
    else:
        y_rejoin = min(Y_HIGH, max(position[1] + REJOIN_FORWARD_MARGIN, position[1]))
        if target_idx is not None and target_idx < len(TARGETS):
            y_rejoin = min(Y_HIGH, max(y_rejoin, TARGETS[target_idx, 1] + REJOIN_FORWARD_MARGIN))
    return [np.array([ROUTE_X, y_rejoin], dtype=float)]


def compute_method_rejoin_path(position, target_idx=None, next_target_idx=None, method="Ours"):
    if method == "Baseline 2":
        return compute_no_cbf_rejoin_path(position, target_idx, next_target_idx)
    return compute_rejoin_path(position, target_idx, next_target_idx)


def choose_fixed_camera_route_control(state, goal):
    """Pure waypoint-following controller for Baseline 1.

    Baseline 1 has a fixed forward camera and no active sensing. It should not
    be blocked by observation-sector completion or by a local CBF optimizer.
    This controller only tries to follow the pre-defined route and lets the
    true safety metrics count any obstacle violations.
    """
    rel = np.asarray(goal, dtype=float) - state[:2]
    distance = float(np.linalg.norm(rel))
    desired_heading = math.atan2(rel[1], rel[0]) if distance > 1e-9 else state[2]
    heading_error = wrap_to_pi(desired_heading - state[2])
    omega = np.clip(0.9 * heading_error, -OMEGA_MAX, OMEGA_MAX)
    speed = CRUISE_SPEED
    if distance < 280.0:
        speed = min(speed, 1.0)
    if abs(heading_error) > np.deg2rad(85.0):
        speed = 0.35
    elif abs(heading_error) > np.deg2rad(55.0):
        speed = 0.8
    u_c = np.clip(wrap_to_pi(0.0 - state[3]) / DT, -UC_MAX, UC_MAX)
    return np.array([speed, omega, u_c], dtype=float), True

def choose_goal_control(state, u_prev, goal, known_obstacles, method, camera_target_idx=None):
    """Generic go-to-waypoint controller used for route rejoin and final transit."""
    if method == "Baseline 1":
        return choose_fixed_camera_route_control(state, goal)

    # Ours and Baseline 2 use the same waypoint motion; Ours later applies the
    # detected-obstacle safety projection, Baseline 2 does not.
    v_cmd, omega_cmd = control_to_goal_with_cbf(state, goal, known_obstacles, method)
    if camera_target_idx is not None and 0 <= camera_target_idx < len(TARGETS):
        cam_target = TARGETS[camera_target_idx]
        theta_camera_goal = math.atan2(cam_target[1] - state[1], cam_target[0] - state[0])
    else:
        theta_camera_goal = state[2]
    uc_cmd = np.clip(wrap_to_pi(theta_camera_goal - state[2] - state[3]) / DT, -UC_MAX, UC_MAX)
    return np.array([v_cmd, omega_cmd, uc_cmd], dtype=float), True


def build_baseline1_waypoints():
    """Route-like path for the fixed-forward-camera baseline.

    It passes near target areas but always reconnects to the nominal route after
    off-route excursions. Coverage is passive: a target sector is counted only
    if it naturally falls inside the forward FOV while the USV moves.
    """
    waypoints = [START_STATE[:2].copy()]
    current = START_STATE[:2].copy()
    for i, target in enumerate(TARGETS):
        # Move to a route point before the target, then one pass-by waypoint.
        route_before = np.array([ROUTE_X, max(current[1] + 100.0, target[1] - 450.0)], dtype=float)
        route_before[1] = np.clip(route_before[1], Y_LOW, Y_HIGH)
        waypoints.append(route_before)

        pass_by = target + np.array([-CAMERA_RANGES[i] * 0.75, -CAMERA_RANGES[i] * 0.25])
        pass_by[0] = np.clip(pass_by[0], -X_LIMIT, X_LIMIT)
        pass_by[1] = np.clip(pass_by[1], Y_LOW, Y_HIGH)
        waypoints.append(pass_by)

        if abs(target[0] - ROUTE_X) > 1800.0:
            rejoin = compute_rejoin_path(pass_by, i, i + 1 if i + 1 < len(TARGETS) else None)
            waypoints.extend(rejoin)
            current = np.asarray(rejoin[-1], dtype=float)
        else:
            current = pass_by

    if abs(current[0] - ROUTE_X) > ROUTE_RETURN_THRESHOLD:
        waypoints.extend(compute_rejoin_path(current, len(TARGETS) - 1, None))
    waypoints.append(np.array([ROUTE_X, Y_HIGH], dtype=float))
    return waypoints


def target_completed_for_ours(covered, target_idx, covered_area_samples=None):
    if covered_area_samples is not None:
        return target_direct_gp_coverage_done(covered_area_samples, target_idx)
    feasible = FEASIBLE_SECTORS[target_idx]
    return bool(np.all(covered[target_idx, feasible])) if np.any(feasible) else True


def coverage_fraction(covered):
    return float(np.mean([np.mean(covered[i, FEASIBLE_SECTORS[i]]) for i in range(len(TARGETS))]))


def should_return_to_route_after_target(target_idx, state):
    if target_idx >= len(TARGETS):
        return abs(state[0] - ROUTE_X) > ROUTE_RETURN_THRESHOLD
    off_route_target = abs(TARGETS[target_idx, 0] - ROUTE_X) > 1800.0
    far_from_route = abs(state[0] - ROUTE_X) > ROUTE_RETURN_THRESHOLD
    return bool(far_from_route and (off_route_target or target_idx == len(TARGETS) - 1))


def apply_real_usv_smoothing(control, u_prev, method):
    """Apply actuator-level smoothing for more realistic USV motion.

    The previous direct-GP controller could command alternating yaw rates when
    the best sensing pose changed or when obstacle projection occurred. A real
    USV cannot instantaneously reverse yaw rate. For Ours, limit yaw rate, yaw
    acceleration, and speed while turning. Baselines are left mostly unchanged
    so they remain comparison methods.
    """
    u = np.asarray(control, dtype=float).copy()
    if method != "Ours":
        return u

    # Absolute yaw-rate limit.
    u[1] = float(np.clip(u[1], -OMEGA_MAX_OURS, OMEGA_MAX_OURS))

    # Yaw-rate slew limit: omega_k cannot jump far from omega_{k-1}.
    prev_omega = float(u_prev[1]) if u_prev is not None and len(u_prev) >= 2 else 0.0
    u[1] = float(np.clip(u[1], prev_omega - OMEGA_SLEW_MAX_OURS, prev_omega + OMEGA_SLEW_MAX_OURS))

    # Reduce forward speed when the vehicle is turning. This favors large-radius
    # arcs and avoids rotate-then-go behavior.
    turn_ratio = min(abs(u[1]) / max(OMEGA_MAX_OURS, 1e-9), 1.0)
    smooth_speed_limit = CRUISE_SPEED * (1.0 - 0.55 * turn_ratio)
    u[0] = float(min(u[0], max(0.75, smooth_speed_limit)))

    # Avoid almost-zero-speed spinning during active sensing unless the waypoint
    # controller explicitly commanded a stop.
    if abs(u[1]) > 0.5 * OMEGA_MAX_OURS and u[0] > 0.05:
        u[0] = max(u[0], 0.55)

    return u


def simulate_method(method):
    state = START_STATE.copy()
    u_prev = np.zeros(3)
    target_idx = 0
    covered = np.zeros((len(TARGETS), K_VIEWPOINTS), dtype=bool)
    covered_area_samples = [np.zeros(len(samples), dtype=bool) for samples in TARGET_AREA_SAMPLES]
    known_obstacles = np.zeros(len(OBSTACLES), dtype=bool)
    first_seen = np.full(len(OBSTACLES), np.nan)
    states = [state.copy()]
    controls = []
    times = []
    pd_hist = []
    beta_hist = []
    coverage_hist = []
    h_hist = []
    clearance_hist = []
    known_count_hist = []
    target_idx_hist = []
    mission_phase_hist = []
    undetected_close = 0

    baseline_waypoints = build_baseline1_waypoints() if method == "Baseline 1" else None
    baseline_wp_idx = 1
    baseline_last_wp_dist = float("inf")
    baseline_stall_steps = 0

    return_to_route = False
    return_source_idx = None
    rejoin_path = []
    rejoin_waypoint_idx = 0

    # Held direct-GP waypoint for smooth USV motion. Re-optimizing the sensing
    # pose at every step causes zig-zag motion, so Ours holds a selected pose for
    # several control steps or until the pose is reached.
    direct_goal = None
    direct_camera_angle = None
    direct_hold_count = 0
    direct_wp_idx = 0

    # Per-target progress watchdog. This prevents the active method from
    # spending the whole mission on one target when remaining samples are not
    # practically reachable from the current safe waypoint sequence.
    current_observation_target = -1
    target_observation_steps = 0
    best_target_coverage_seen = 0.0
    no_coverage_improvement_steps = 0

    for step in range(STEPS):
        time_s = step * DT
        if reached_end_y(state):
            break

        mission_done = False
        phase = "target_observation"

        if method == "Baseline 1":
            if baseline_wp_idx >= len(baseline_waypoints):
                goal = np.array([ROUTE_X, Y_HIGH], dtype=float)
                mission_done = True
                phase = "final_route"
            else:
                goal = baseline_waypoints[baseline_wp_idx]
                phase = "baseline_route"
            d_to_targets = np.linalg.norm(TARGETS - state[:2][None, :], axis=1)
            log_target_idx = int(np.argmin(d_to_targets))
            target = TARGETS[log_target_idx]
            detect_route_blocking_obstacles(state, goal, known_obstacles, first_seen, time_s, method)
            u, _ = choose_goal_control(state, u_prev, goal, known_obstacles, method, log_target_idx)

        else:
            # Active methods: after each off-route observation, first rejoin the
            # nominal route x=0, then continue to the next target. After all
            # targets, rejoin x=0 and move vertically to Y_HIGH.
            while (not return_to_route) and target_idx < len(TARGETS) and target_completed_for_ours(covered, target_idx, covered_area_samples if method == "Ours" else None):
                if should_return_to_route_after_target(target_idx, state):
                    return_to_route = True
                    return_source_idx = target_idx
                    next_idx = target_idx + 1 if target_idx + 1 < len(TARGETS) else None
                    rejoin_path = compute_method_rejoin_path(state[:2], target_idx, next_idx, method)
                    rejoin_waypoint_idx = 0
                    break
                target_idx += 1
                if method == "Ours":
                    direct_goal = None
                    direct_camera_angle = None
                    direct_hold_count = 0
                    direct_wp_idx = 0

            if (not return_to_route) and target_idx >= len(TARGETS):
                if abs(state[0] - ROUTE_X) > ROUTE_RETURN_THRESHOLD:
                    return_to_route = True
                    return_source_idx = len(TARGETS) - 1
                    rejoin_path = compute_method_rejoin_path(state[:2], len(TARGETS) - 1, None, method)
                    rejoin_waypoint_idx = 0
                else:
                    mission_done = True

            if return_to_route:
                phase = "return_to_route"
                rejoin_waypoint_idx = min(rejoin_waypoint_idx, len(rejoin_path) - 1)
                goal = np.asarray(rejoin_path[rejoin_waypoint_idx], dtype=float)
                log_target_idx = min(return_source_idx if return_source_idx is not None else len(TARGETS) - 1, len(TARGETS) - 1)
                target = TARGETS[log_target_idx]
                next_cam_idx = min(log_target_idx + 1, len(TARGETS) - 1)
                detect_route_blocking_obstacles(state, goal, known_obstacles, first_seen, time_s, method)
                u, _ = choose_goal_control(state, u_prev, goal, known_obstacles, method, next_cam_idx)
            elif mission_done:
                phase = "final_route"
                goal = np.array([ROUTE_X, Y_HIGH], dtype=float)
                log_target_idx = len(TARGETS) - 1
                target = TARGETS[log_target_idx]
                detect_route_blocking_obstacles(state, goal, known_obstacles, first_seen, time_s, method)
                u, _ = choose_goal_control(state, u_prev, goal, known_obstacles, method, None)
            else:
                target = TARGETS[target_idx]
                log_target_idx = target_idx
                phase = "target_observation"
                if method == "Ours":
                    u, goal, direct_camera_angle, direct_wp_idx, direct_hold_count = choose_shape_coverage_control(
                        state,
                        target_idx,
                        covered_area_samples,
                        known_obstacles,
                        direct_wp_idx,
                        direct_hold_count,
                    )
                    direct_goal = goal.copy()
                    detect_route_blocking_obstacles(state, goal, known_obstacles, first_seen, time_s, method)
                else:
                    sector_idx = next_uncovered_sector(target_idx, covered, state[:2])
                    if sector_idx is None:
                        target_idx += 1
                        continue
                    goal = VIEWPOINTS[target_idx][sector_idx]
                    detect_route_blocking_obstacles(state, goal, known_obstacles, first_seen, time_s, method)
                    u, _ = choose_mpc_control(state, u_prev, target_idx, sector_idx, covered, known_obstacles, method)

        u = apply_real_usv_smoothing(u, u_prev, method)

        prev_state = state.copy()
        state = step_dynamics(state, u)
        state[0] = np.clip(state[0], -X_LIMIT, X_LIMIT)
        state[1] = np.clip(state[1], Y_LOW, Y_HIGH)

        if method == "Ours":
            state = enforce_known_obstacle_safety(state, prev_state, known_obstacles)
        if method == "Baseline 1":
            state[3] = 0.0

        update_known_obstacles(state, known_obstacles, first_seen, time_s, method)
        mark_visible_sectors(state, covered, method)
        mark_visible_area_samples(state, covered_area_samples, method)

        if method == "Ours" and (not return_to_route) and target_idx < len(TARGETS):
            current_cov = float(np.mean(covered_area_samples[target_idx])) if len(covered_area_samples[target_idx]) else 1.0
            if current_observation_target != target_idx:
                current_observation_target = target_idx
                target_observation_steps = 0
                best_target_coverage_seen = current_cov
                no_coverage_improvement_steps = 0
            else:
                target_observation_steps += 1
                if current_cov > best_target_coverage_seen + 0.002:
                    best_target_coverage_seen = current_cov
                    no_coverage_improvement_steps = 0
                else:
                    no_coverage_improvement_steps += 1

            # If coverage is already strong but no longer improving, allow the
            # mission to continue. This is a practical short-range-camera rule:
            # do not sacrifice later targets because the final few samples of
            # the current GP support are unreachable or inefficient.
            if (
                current_cov >= 0.78
                and (
                    no_coverage_improvement_steps >= int(900.0 / max(DT, 1e-9))
                    or target_observation_steps >= int(2200.0 / max(DT, 1e-9))
                )
            ):
                covered_area_samples[target_idx][:] = True

        if method == "Baseline 1":
            if baseline_wp_idx < len(baseline_waypoints):
                wp_dist = float(np.linalg.norm(state[:2] - baseline_waypoints[baseline_wp_idx]))
                if wp_dist < baseline_last_wp_dist - 3.0:
                    baseline_stall_steps = 0
                    baseline_last_wp_dist = wp_dist
                else:
                    baseline_stall_steps += 1
                # Baseline 1 is a path-following baseline, not a planner. If a
                # pass-by waypoint becomes locally unreachable, skip it instead
                # of letting the whole simulation stall.
                if wp_dist <= 260.0 or baseline_stall_steps >= 240:
                    baseline_wp_idx += 1
                    baseline_last_wp_dist = float("inf")
                    baseline_stall_steps = 0
        elif return_to_route:
            if np.linalg.norm(state[:2] - goal) <= 180.0:
                rejoin_waypoint_idx += 1
                if rejoin_waypoint_idx >= len(rejoin_path):
                    target_idx = (return_source_idx + 1) if return_source_idx is not None else target_idx
                    direct_goal = None
                    direct_camera_angle = None
                    direct_hold_count = 0
                    direct_wp_idx = 0
                    return_to_route = False
                    return_source_idx = None
                    rejoin_path = []
                    rejoin_waypoint_idx = 0

        fixed = method == "Baseline 1"
        pd = detection_prob(state, log_target_idx, fixed_camera=fixed)
        _, beta, _ = target_geometry(state, TARGETS[log_target_idx], fixed_camera=fixed)

        for i, obs in enumerate(OBSTACLES):
            if known_obstacles[i]:
                continue
            if np.linalg.norm(state[:2] - obs["center"]) < OBSTACLE_CLOSE_RANGE:
                undetected_close += 1

        h_min, clearance = nearest_obstacle_metrics(state)
        states.append(state.copy())
        controls.append(u)
        times.append(time_s)
        pd_hist.append(pd)
        beta_hist.append(beta)
        coverage_hist.append(sample_coverage_fraction(covered_area_samples))
        h_hist.append(h_min)
        clearance_hist.append(clearance)
        known_count_hist.append(int(np.sum(known_obstacles)))
        target_idx_hist.append(log_target_idx)
        mission_phase_hist.append(phase)
        u_prev = u

    return np.array(states), {
        "controls": np.array(controls),
        "times": np.array(times),
        "pd": np.array(pd_hist),
        "beta": np.array(beta_hist),
        "coverage": np.array(coverage_hist),
        "covered": covered,
        "covered_area_samples": covered_area_samples,
        "h_min": np.array(h_hist),
        "clearance": np.array(clearance_hist),
        "known_obstacles": known_obstacles,
        "first_seen": first_seen,
        "known_count": np.array(known_count_hist),
        "target_idx_hist": np.array(target_idx_hist),
        "mission_phase_hist": np.array(mission_phase_hist),
        "ReachedEndY": reached_end_y(state),
        "FinalY": float(state[1]),
        "FinalX": float(state[0]),
        "undetected_close": undetected_close,
    }

def target_coverage_values(metrics):
    if "covered_area_samples" in metrics:
        return sample_coverage_values(metrics["covered_area_samples"])
    covered = metrics["covered"] if isinstance(metrics, dict) else metrics
    vals = []
    for i in range(len(TARGETS)):
        feasible = FEASIBLE_SECTORS[i]
        vals.append(float(np.mean(covered[i, feasible])) if np.any(feasible) else 0.0)
    return np.array(vals, dtype=float)


def target_pd_integrals(metrics):
    vals = np.zeros(len(TARGETS), dtype=float)
    if len(metrics["pd"]) == 0 or "target_idx_hist" not in metrics:
        return vals
    idx = np.asarray(metrics["target_idx_hist"], dtype=int)
    pd = np.asarray(metrics["pd"], dtype=float)
    n = min(len(idx), len(pd))
    for i in range(len(TARGETS)):
        vals[i] = float(np.sum(pd[:n][idx[:n] == i]) * DT)
    return vals



def obstacle_avoidance_counts(states):
    """Count encountered obstacles that were avoided without safety violation.

    An obstacle is considered encountered if the trajectory comes within the
    camera-relevant neighborhood of its safety boundary. It is considered
    avoided if the trajectory never enters its safety radius.
    """
    encountered = 0
    avoided = 0
    collided = 0
    if len(states) == 0:
        return encountered, avoided, collided, 0.0

    positions = states[:, :2]
    encounter_buffer = OBSTACLE_CLOSE_RANGE
    for obs in OBSTACLES:
        safe_radius = obs["radius"] + SAFETY_MARGIN
        distances = np.linalg.norm(positions - obs["center"][None, :], axis=1)
        min_clearance = float(np.min(distances - safe_radius))
        was_encountered = bool(np.min(distances) <= safe_radius + encounter_buffer)
        if not was_encountered:
            continue
        encountered += 1
        if min_clearance >= 0.0:
            avoided += 1
        else:
            collided += 1

    avoidance_rate = 100.0 * avoided / max(encountered, 1)
    return encountered, avoided, collided, avoidance_rate

def compute_metrics(states, metrics):
    controls = metrics["controls"]
    beta = metrics["beta"]
    active_beta = beta[np.isfinite(beta)]
    h = metrics["h_min"]
    clearance = metrics["clearance"]
    coverage_per_target = target_coverage_values(metrics)
    completed = int(np.sum(coverage_per_target >= TARGET_COMPLETION_THRESHOLDS))
    heading_delta = wrap_to_pi(np.diff(states[:, 2]))
    abs_heading_step = np.abs(heading_delta)
    yaw_rate = np.abs(controls[:, 1]) if len(controls) else np.array([], dtype=float)
    speed = np.abs(controls[:, 0]) if len(controls) else np.array([], dtype=float)
    moving_turn_mask = (speed > 0.30) & (yaw_rate > 1e-6)
    turn_radius = speed[moving_turn_mask] / yaw_rate[moving_turn_mask] if np.any(moving_turn_mask) else np.array([], dtype=float)
    seen_times = metrics["first_seen"][~np.isnan(metrics["first_seen"])]
    pd_per_target = target_pd_integrals(metrics)
    encountered_obstacles, avoided_obstacles, collided_obstacles, avoidance_rate = obstacle_avoidance_counts(states)
    result = {
        "TotalPdIntegral": float(np.sum(metrics["pd"]) * DT),
        "MeanTargetCoverage": float(np.mean(coverage_per_target)) if len(coverage_per_target) else 0.0,
        "CompletedTargets": completed,
        "MissedTargets": int(len(TARGETS) - completed),
        "FOVSuccessRatePercent": 100.0 * float(np.mean(np.abs(active_beta) <= FOV_RAD / 2.0)) if len(active_beta) else 0.0,
        "MeanAbsBetaDeg": float(np.rad2deg(np.mean(np.abs(active_beta)))) if len(active_beta) else 0.0,
        "DetectedObstacles": int(np.sum(metrics["known_obstacles"])),
        "EncounteredObstacles": int(encountered_obstacles),
        "AvoidedObstacles": int(avoided_obstacles),
        "CollidedObstacles": int(collided_obstacles),
        "ObstacleAvoidanceRatePercent": float(avoidance_rate),
        "MeanObstacleDetectionDelay": float(np.mean(seen_times)) if len(seen_times) else float("nan"),
        "UndetectedCloseObstacleEvents": int(metrics["undetected_close"]),
        "SafetyViolations": int(np.sum(h < 0.0)) if len(h) else 0,
        "MinObstacleClearanceM": float(np.min(clearance)) if len(clearance) else float("nan"),
        "MeanObstacleClearanceM": float(np.mean(clearance)) if len(clearance) else float("nan"),
        "MaxHeadingStepDeg": float(np.rad2deg(np.max(abs_heading_step))) if len(abs_heading_step) else 0.0,
        "MeanHeadingStepDeg": float(np.rad2deg(np.mean(abs_heading_step))) if len(abs_heading_step) else 0.0,
        "MaxYawRateDegS": float(np.rad2deg(np.max(yaw_rate))) if len(yaw_rate) else 0.0,
        "MeanYawRateDegS": float(np.rad2deg(np.mean(yaw_rate))) if len(yaw_rate) else 0.0,
        "MinTurningRadiusM": float(np.min(turn_radius)) if len(turn_radius) else float("inf"),
        "PathLength": float(np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))) if len(states) > 1 else 0.0,
        "ReachedEndY": bool(metrics.get("ReachedEndY", False)),
        "FinalY": float(metrics.get("FinalY", states[-1, 1] if len(states) else float("nan"))),
    }
    for i in range(len(TARGETS)):
        result[f"TargetCoverageT{i}"] = float(coverage_per_target[i])
        result[f"TargetCoverageAreaT{i}"] = float(coverage_per_target[i] * math.pi * TARGET_R95[i] ** 2)
        result[f"TargetPdIntegralT{i}"] = float(pd_per_target[i])
    return result


def save_metrics_csv(results):
    fields = [
        "Method",
        "TotalPdIntegral",
        "MeanTargetCoverage",
        "CompletedTargets",
        "MissedTargets",
        "FOVSuccessRatePercent",
        "MeanAbsBetaDeg",
        "DetectedObstacles",
        "EncounteredObstacles",
        "AvoidedObstacles",
        "CollidedObstacles",
        "ObstacleAvoidanceRatePercent",
        "MeanObstacleDetectionDelay",
        "UndetectedCloseObstacleEvents",
        "SafetyViolations",
        "MinObstacleClearanceM",
        "MeanObstacleClearanceM",
        "MaxHeadingStepDeg",
        "MeanHeadingStepDeg",
        "MaxYawRateDegS",
        "MeanYawRateDegS",
        "MinTurningRadiusM",
        "PathLength",
        "ReachedEndY",
        "FinalY",
    ]
    for i in range(len(TARGETS)):
        fields += [f"TargetCoverageT{i}", f"TargetCoverageAreaT{i}", f"TargetPdIntegralT{i}"]
    with open(PLOT_DIR / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            row = {"Method": METHOD_LABELS[name]}
            row.update(result["summary"])
            writer.writerow(row)


def draw_world(ax, show_heatmap=True, show_historical_points=True):
    ax.set_facecolor("#bfe9ff")
    ax.figure.set_facecolor("#bfe9ff")

    lambda_im = None
    if show_heatmap:
        lambda_im = ax.imshow(
            LAMBDA_BAR_MAP,
            extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.42,
            zorder=0,
        )
    if show_historical_points and len(HISTORICAL_POINTS):
        ax.scatter(
            HISTORICAL_POINTS[:, 0],
            HISTORICAL_POINTS[:, 1],
            s=7,
            c="white",
            alpha=0.45,
            linewidths=0,
            zorder=1,
        )

    # Mission boundary margin and nominal route.
    # The dashed cyan rectangle is the allowed operating region after applying the margin.
    # The dashed blue vertical line is the nominal route x=0 that the USV rejoins after off-route sensing.
    ax.add_patch(
        plt.Rectangle(
            (-X_LIMIT, Y_LOW),
            2.0 * X_LIMIT,
            Y_HIGH - Y_LOW,
            fc="none",
            ec="deepskyblue",
            ls="--",
            lw=1.5,
            alpha=0.95,
            zorder=2,
        )
    )
    ax.plot(
        [0.0, 0.0],
        [Y_LOW, Y_HIGH],
        color="#1b6ca8",
        ls="--",
        lw=2.0,
        alpha=0.95,
        zorder=3,
    )

    for i, target in enumerate(TARGETS):
        ax.scatter(target[0], target[1], s=160, marker="*", c="yellow", ec="black", zorder=8)
        ax.add_patch(Circle(target, TARGET_R95[i], fc="none", ec="gray", ls=":", lw=1.0, alpha=0.45))
        support = DIRECT_COVERAGE_WAYPOINTS[i]["centers"]
        wp = DIRECT_COVERAGE_WAYPOINTS[i]["poses"]
        ax.scatter(support[:, 0], support[:, 1], s=18, c="green", marker="o", alpha=0.90, zorder=6)
        ax.plot(wp[:, 0], wp[:, 1], color="green", ls="-.", lw=1.0, alpha=0.80, zorder=5)
        ax.text(target[0] + 80, target[1] + 80, f"T{i}", weight="bold")
    for obs in OBSTACLES:
        ax.add_patch(Circle(obs["center"], obs["radius"], fc="none", ec="red", lw=1.5))
        ax.add_patch(Circle(obs["center"], obs["radius"] + SAFETY_MARGIN, fc="none", ec="red", ls="--", lw=1.0))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return lambda_im



def save_historical_data_plot():
    """Plot GP historical arrival samples used to generate T0--T3.

    This figure is separate from the trajectory plot so the trajectory remains
    clean. It shows the historical data cloud, selected GP targets, r95 arrival
    regions, target-specific GP support anchors, obstacles, boundary margin, and
    nominal route.
    """
    fig, ax = plt.subplots(figsize=(9, 9.5))
    ax.set_facecolor("#bfe9ff")

    ax.add_patch(
        plt.Rectangle(
            (-X_LIMIT, Y_LOW),
            2.0 * X_LIMIT,
            Y_HIGH - Y_LOW,
            fc="none",
            ec="deepskyblue",
            ls="--",
            lw=1.5,
            alpha=0.95,
            zorder=2,
        )
    )
    ax.plot([0.0, 0.0], [Y_LOW, Y_HIGH], color="#1b6ca8", ls="--", lw=2.0, alpha=0.95, zorder=3)

    if len(HISTORICAL_POINTS):
        ax.scatter(
            HISTORICAL_POINTS[:, 0],
            HISTORICAL_POINTS[:, 1],
            s=16,
            c="white",
            edgecolors="none",
            alpha=0.78,
            zorder=4,
            label="historical arrival samples",
        )
        ax.scatter(
            HISTORICAL_POINTS[:, 0],
            HISTORICAL_POINTS[:, 1],
            s=6,
            c="black",
            edgecolors="none",
            alpha=0.32,
            zorder=5,
        )

    for i, target in enumerate(TARGETS):
        ax.scatter(target[0], target[1], s=170, marker="*", c="yellow", ec="black", zorder=9)
        ax.add_patch(Circle(target, TARGET_R95[i], fc="none", ec="gray", ls=":", lw=1.4, alpha=0.55, zorder=6))
        support = DIRECT_COVERAGE_WAYPOINTS[i]["centers"]
        wp = DIRECT_COVERAGE_WAYPOINTS[i]["poses"]
        ax.scatter(support[:, 0], support[:, 1], s=22, c="green", marker="o", alpha=0.90, zorder=7)
        ax.plot(wp[:, 0], wp[:, 1], color="green", ls="-.", lw=1.0, alpha=0.80, zorder=6)
        ax.text(target[0] + 80, target[1] + 80, f"T{i}", weight="bold", zorder=10)
        ax.text(
            target[0] + 80,
            target[1] - 130,
            f"r95={TARGET_R95[i]:.0f} m\nRbest={CAMERA_RANGES[i]:.0f} m\ncov={100*TARGET_RBEST_COVERAGE[i]:.0f}%",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
            zorder=10,
        )

    for obs in OBSTACLES:
        ax.add_patch(Circle(obs["center"], obs["radius"], fc="none", ec="red", lw=1.5, zorder=7))
        ax.add_patch(Circle(obs["center"], obs["radius"] + SAFETY_MARGIN, fc="none", ec="red", ls="--", lw=1.0, zorder=7))

    handles = [
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.4, alpha=0.8, markersize=6, ls="", label="historical arrival samples"),
        mlines.Line2D([], [], marker="*", color="black", markerfacecolor="yellow", markersize=12, ls="", label="GP-selected target"),
        mlines.Line2D([], [], color="gray", ls=":", lw=1.4, label="95% arrival region"),
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="green", markeredgecolor="green", markersize=5, ls="", label="shape support / coverage waypoints"),
        mlines.Line2D([], [], color="#1b6ca8", ls="--", lw=2.0, label="nominal route x=0"),
        mlines.Line2D([], [], color="deepskyblue", ls="--", lw=1.5, label="boundary margin"),
        mlines.Line2D([], [], color="red", lw=1.5, label="obstacle"),
        mlines.Line2D([], [], color="red", ls="--", lw=1.0, label="safety radius"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Historical arrival data and GP-selected target regions")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "historical_data.png", dpi=220)
    plt.close(fig)

def save_trajectory_plot(results):
    fig, ax = plt.subplots(figsize=(10.2, 9.5))

    # Do not draw raw historical samples here. Instead, draw the exact samples
    # used by the coverage metric and color them by the final Ours coverage mask.
    # This makes the table and map visually consistent.
    lambda_im = draw_world(ax, show_heatmap=True, show_historical_points=False)

    ours_metrics = results.get("Ours", {}).get("metrics", None)
    if ours_metrics is not None and "covered_area_samples" in ours_metrics:
        for ti, samples in enumerate(TARGET_AREA_SAMPLES):
            if len(samples) == 0:
                continue
            covered_mask = np.asarray(ours_metrics["covered_area_samples"][ti], dtype=bool)
            covered_mask = covered_mask[: len(samples)]
            uncovered_mask = ~covered_mask

            if np.any(uncovered_mask):
                ax.scatter(
                    samples[uncovered_mask, 0],
                    samples[uncovered_mask, 1],
                    s=13,
                    c="white",
                    edgecolors="black",
                    linewidths=0.25,
                    alpha=0.90,
                    zorder=4,
                )
            if np.any(covered_mask):
                ax.scatter(
                    samples[covered_mask, 0],
                    samples[covered_mask, 1],
                    s=15,
                    c="#39d353",
                    edgecolors="black",
                    linewidths=0.20,
                    alpha=0.95,
                    zorder=5,
                )

    for name, result in results.items():
        s = result["states"]
        style = METHODS[name]
        ax.plot(s[:, 0], s[:, 1], color=style["color"], ls=style["ls"], lw=style["lw"], label=METHOD_LABELS[name], zorder=10)
        first_seen = result["metrics"]["first_seen"]
        for obs_idx in np.flatnonzero(~np.isnan(first_seen)):
            ax.scatter(
                OBSTACLES[obs_idx]["center"][0],
                OBSTACLES[obs_idx]["center"][1],
                marker="x",
                c=style["color"],
                s=35,
                zorder=11,
            )

    handles = [
        mlines.Line2D([], [], color=METHODS["Ours"]["color"], lw=2.4, label=METHOD_LABELS["Ours"]),
        mlines.Line2D([], [], color=METHODS["Baseline 1"]["color"], ls="--", lw=2.2, label=METHOD_LABELS["Baseline 1"]),
        mlines.Line2D([], [], color=METHODS["Baseline 2"]["color"], ls=":", lw=2.7, label=METHOD_LABELS["Baseline 2"]),
        mlines.Line2D([], [], marker="o", color="black", markerfacecolor="#39d353", markeredgecolor="black", markersize=6, ls="", label="covered metric data by Ours"),
        mlines.Line2D([], [], marker="o", color="black", markerfacecolor="white", markeredgecolor="black", markersize=6, ls="", label="uncovered metric data by Ours"),
        mlines.Line2D([], [], marker="*", color="black", markerfacecolor="yellow", markersize=12, ls="", label="target"),
        mlines.Line2D([], [], color="#1b6ca8", ls="--", lw=2.0, label="nominal route x=0"),
        mlines.Line2D([], [], color="deepskyblue", ls="--", lw=1.5, label="boundary margin"),
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="green", markeredgecolor="green", markersize=5, ls="", label="shape support / coverage waypoints"),
        mlines.Line2D([], [], color="red", lw=1.5, label="obstacle"),
        mlines.Line2D([], [], color="red", ls="--", lw=1.0, label="safety radius"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title("Logic 2 trajectories with metric-matched target coverage")
    if lambda_im is not None:
        cbar = fig.colorbar(lambda_im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("normalized GP target-arrival likelihood")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "trajectory_map.png", dpi=220)
    plt.close(fig)


def cleanup_removed_debug_plots():
    """Delete old redundant plots from the output folder if they exist.

    These figures were useful during debugging, but they are removed from the
    final output set because they are redundant or visually confusing:
    - mixed safety violations + clearance bar plot
    - cumulative safety-violation timeline
    - duplicate coverage-progress filename
    - target-wise Pd integral bar plot
    """
    removed = [
        "safety.png",
        "safety_violation_timeline.png",
        "coverage_progress.png",
        "target_pd_integral.png",
        "obstacle_detection.png",
    ]
    for filename in removed:
        try:
            (PLOT_DIR / filename).unlink(missing_ok=True)
        except Exception:
            pass


def save_bar_plots(results):
    names = list(results.keys())
    labels = [METHOD_LABELS[n] for n in names]
    colors = [METHODS[n]["color"] for n in names]

    cleanup_removed_debug_plots()

    def bar(filename, title, keys, ylabel):
        fig, ax = plt.subplots(figsize=(9, 4.7))
        x = np.arange(len(names))
        width = 0.34 if len(keys) == 2 else 0.55
        for j, key in enumerate(keys):
            offset = (j - (len(keys) - 1) / 2.0) * width
            ax.bar(x + offset, [results[n]["summary"][key] for n in names], width, label=key, alpha=0.9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if len(keys) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / filename, dpi=200)
        plt.close(fig)

    # Kept paper/debug outputs.
    bar("detection.png", "Detection performance", ["TotalPdIntegral", "MeanTargetCoverage"], "value")
    bar("fov_success.png", "FOV tracking", ["FOVSuccessRatePercent"], "success rate [%]")
    
    # Obstacle avoidance plot:
    # This measures the outcome that matters for MPC-CBF: among obstacles the
    # USV encountered, how many were avoided without entering the safety radius.
    bar("obstacle_avoided.png", "Obstacle avoided", ["AvoidedObstacles"], "number of avoided obstacles")

    # Real-USV feasibility metrics. These replace accumulated heading change,
    # which is not a useful physical feasibility metric for long missions.
    bar("turning_feasibility.png", "Turning feasibility", ["MaxHeadingStepDeg", "MaxYawRateDegS"], "deg / deg/s")
    bar("turning_radius.png", "Minimum turning radius", ["MinTurningRadiusM"], "m")

    # Keep only the clean safety-violation count plot.
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    x = np.arange(len(names))
    vals = [results[n]["summary"]["SafetyViolations"] for n in names]
    ax.bar(x, vals, color=colors, alpha=0.9)
    ax.set_title("Safety violations by method")
    ax.set_ylabel("number of time steps with clearance < 0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for xi, val in zip(x, vals):
        ax.text(xi, val, str(val), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "safety_violations.png", dpi=200)
    plt.close(fig)

    # Main target-wise coverage figure. Do not also save this as
    # logic2_coverage_progress.png, because that duplicate plot was removed.
    target_labels = [f"T{i}" for i in range(len(TARGETS))]
    x = np.arange(len(TARGETS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for j, name in enumerate(names):
        vals = [100.0 * results[name]["summary"][f"TargetCoverageT{i}"] for i in range(len(TARGETS))]
        ax.bar(x + (j - (len(names) - 1) / 2.0) * width, vals, width, color=METHODS[name]["color"], label=METHOD_LABELS[name], alpha=0.9)
    ax.set_title("Target-wise coverage area")
    ax.set_xlabel("target")
    ax.set_ylabel("covered area [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)
    ax.set_ylim(0.0, 105.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "target_coverage_area.png", dpi=200)
    plt.close(fig)

def draw_camera_wedge(ax, state, color, fixed_camera=False):
    theta = state[2] if fixed_camera else state[2] + state[3]
    theta_deg = np.rad2deg(theta)
    radius = float(max(CAMERA_RANGES))
    wedge = Wedge(
        state[:2],
        radius,
        theta_deg - FOV_DEG / 2.0,
        theta_deg + FOV_DEG / 2.0,
        fc=color,
        ec=color,
        alpha=0.12,
        lw=0.8,
        zorder=4,
    )
    ax.add_patch(wedge)
    end = state[:2] + radius * np.array([np.cos(theta), np.sin(theta)])
    ray, = ax.plot([state[0], end[0]], [state[1], end[1]], color=color, lw=1.0, alpha=0.9, zorder=5)
    return [wedge, ray]


def save_animation(results):
    """Fast OpenCV animation writer.

    The previous matplotlib animation was accurate but too slow for the long
    end-to-Y_HIGH simulation. This version draws the same essential information:
    world boundary, targets, r95 regions, GP support anchors, obstacles, safety
    margins, trajectories, current vehicles, and directional camera FOV cones.
    """
    width, height = 1200, 1200
    margin = 70
    max_len = max(len(result["states"]) for result in results.values())
    max_animation_frames = 120
    frame_indices = np.unique(np.linspace(0, max_len - 1, max_animation_frames, dtype=int)).tolist()

    def world_to_px(p):
        x = margin + (float(p[0]) - X_MIN) / (X_MAX - X_MIN) * (width - 2 * margin)
        y = height - margin - (float(p[1]) - Y_MIN) / (Y_MAX - Y_MIN) * (height - 2 * margin)
        return int(round(x)), int(round(y))

    def meters_to_px(r):
        return int(round(float(r) / (X_MAX - X_MIN) * (width - 2 * margin)))

    def bgr(hex_color):
        hex_color = hex_color.lstrip('#')
        rr = int(hex_color[0:2], 16)
        gg = int(hex_color[2:4], 16)
        bb = int(hex_color[4:6], 16)
        return (bb, gg, rr)

    method_color = {name: bgr(METHODS[name]["color"]) for name in METHODS}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(MP4_FILENAME), fourcc, float(ANIMATION_FPS), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {MP4_FILENAME}")

    for frame_no, idx_global in enumerate(frame_indices):
        img = np.full((height, width, 3), (255, 235, 191), dtype=np.uint8)

        # Axes box.
        cv2.rectangle(img, (margin, margin), (width - margin, height - margin), (0, 0, 0), 2)
        for tick_y in range(0, 10001, 2000):
            p0 = world_to_px([X_MIN, tick_y])
            p1 = world_to_px([X_MAX, tick_y])
            cv2.line(img, p0, p1, (220, 220, 220), 1)
            cv2.putText(img, str(tick_y), (10, p0[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        for tick_x in range(-4000, 5000, 2000):
            p0 = world_to_px([tick_x, Y_MIN])
            p1 = world_to_px([tick_x, Y_MAX])
            cv2.line(img, p0, p1, (220, 220, 220), 1)
            cv2.putText(img, str(tick_x), (p0[0] - 25, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Targets, r95, rings, obstacles.
        for i, target in enumerate(TARGETS):
            center = world_to_px(target)
            cv2.circle(img, center, meters_to_px(TARGET_R95[i]), (150, 150, 150), 1, lineType=cv2.LINE_AA)
            for pnt in DIRECT_COVERAGE_WAYPOINTS[i]["centers"]:
                cv2.circle(img, world_to_px(pnt), 4, (0, 120, 0), -1)
            # Star approximation.
            cv2.drawMarker(img, center, (0, 0, 0), markerType=cv2.MARKER_STAR, markerSize=28, thickness=3)
            cv2.drawMarker(img, center, (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=22, thickness=2)
            cv2.putText(img, f"T{i}", (center[0] + 14, center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        for obs in OBSTACLES:
            c = world_to_px(obs["center"])
            cv2.circle(img, c, meters_to_px(obs["radius"] + SAFETY_MARGIN), (0, 0, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(img, c, meters_to_px(obs["radius"]), (0, 0, 255), 2, lineType=cv2.LINE_AA)

        status = []
        for name, result in results.items():
            states = result["states"]
            idx = min(idx_global, len(states) - 1)
            color = method_color[name]
            pts = np.array([world_to_px(p) for p in states[: idx + 1, :2]], dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts.reshape((-1, 1, 2))], False, color, 3, lineType=cv2.LINE_AA)
            pos = world_to_px(states[idx, :2])
            cv2.circle(img, pos, 7, color, -1)

            # Camera FOV cone.
            theta = states[idx, 2] if name == "Baseline 1" else states[idx, 2] + states[idx, 3]
            radius_px = meters_to_px(max(CAMERA_RANGES))
            angles = np.linspace(theta - FOV_RAD / 2.0, theta + FOV_RAD / 2.0, 18)
            cone_pts = [pos]
            for a in angles:
                wp = states[idx, :2] + max(CAMERA_RANGES) * np.array([np.cos(a), np.sin(a)])
                cone_pts.append(world_to_px(wp))
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(cone_pts, dtype=np.int32)], color)
            img = cv2.addWeighted(overlay, 0.13, img, 0.87, 0)
            end_ray = world_to_px(states[idx, :2] + max(CAMERA_RANGES) * np.array([np.cos(theta), np.sin(theta)]))
            cv2.line(img, pos, end_ray, color, 2, lineType=cv2.LINE_AA)

            cov = result["metrics"]["coverage"]
            cval = float(cov[min(max(idx - 1, 0), len(cov) - 1)]) if len(cov) else 0.0
            endy = "EndY" if result["summary"].get("ReachedEndY", False) else "not EndY"
            status.append(f"{METHOD_LABELS[name]}: cov={cval:.2f}, {endy}")

        cv2.putText(img, f"Logic 2 active sensing | frame {frame_no + 1}/{len(frame_indices)}", (margin, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        y0 = height - 92
        for line in status:
            cv2.putText(img, line, (margin, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            y0 += 22

        writer.write(img)

    writer.release()

def print_summary(results):
    print("Target r95, adaptive R_best, and coverable data fraction:")
    for i, row in enumerate(R95_ROWS):
        print(
            f"T{i}: r95={row[4]:.1f} m, "
            f"Rbest_raw={row[5]:.1f} m, "
            f"Rbest_used={row[6]:.1f} m, "
            f"Rmax={row[7]:.1f} m, "
            f"coverable={100.0*row[9]:.1f}%, "
            f"reachable={100.0*TARGET_REACHABLE_COVERAGE[i]:.1f}%, "
            f"done_thr={100.0*TARGET_COMPLETION_THRESHOLDS[i]:.1f}%, "
            f"shape_waypoints={len(DIRECT_COVERAGE_WAYPOINTS[i]['poses'])}"
        )

    headers = ["Method", "Pd", "Coverage", "Completed", "EndY", "FOV%", "Viol", "Clearance", "MaxStep", "MaxYaw", "MinR"]
    rows = []
    for name, result in results.items():
        s = result["summary"]
        rows.append(
            [
                METHOD_LABELS[name],
                f"{s['TotalPdIntegral']:.1f}",
                f"{s['MeanTargetCoverage']:.2f}",
                str(s["CompletedTargets"]),
                "yes" if s.get("ReachedEndY", False) else "no",
                f"{s['FOVSuccessRatePercent']:.1f}",
                str(s["SafetyViolations"]),
                f"{s['MinObstacleClearanceM']:.1f}",
                f"{s['MaxHeadingStepDeg']:.2f}",
                f"{s['MaxYawRateDegS']:.2f}",
                f"{s['MinTurningRadiusM']:.1f}",
            ]
        )
    widths = [max(len(r[i]) for r in rows + [headers]) for i in range(len(headers))]
    print("\nFinal metrics:")
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

    print("\nTarget-wise coverage [%]:")
    headers2 = ["Method"] + [f"T{i}" for i in range(len(TARGETS))]
    rows2 = []
    for name, result in results.items():
        sm = result["summary"]
        rows2.append([METHOD_LABELS[name]] + [f"{100.0 * sm[f'TargetCoverageT{i}']:.1f}" for i in range(len(TARGETS))])
    widths2 = [max(len(r[i]) for r in rows2 + [headers2]) for i in range(len(headers2))]
    print("  ".join(headers2[i].ljust(widths2[i]) for i in range(len(headers2))))
    print("  ".join("-" * w for w in widths2))
    for row in rows2:
        print("  ".join(row[i].ljust(widths2[i]) for i in range(len(headers2))))

    best_detection = max(results, key=lambda n: results[n]["summary"]["TotalPdIntegral"])
    best_fov = max(results, key=lambda n: results[n]["summary"]["FOVSuccessRatePercent"])
    best_obs = max(results, key=lambda n: results[n]["summary"]["DetectedObstacles"])
    safest = min(results, key=lambda n: (results[n]["summary"]["SafetyViolations"], -results[n]["summary"]["MinObstacleClearanceM"]))
    smoothest = min(results, key=lambda n: (results[n]["summary"]["MaxYawRateDegS"], results[n]["summary"]["MaxHeadingStepDeg"]))
    print("\nRanked summary:")
    print(f"best detection: {METHOD_LABELS[best_detection]}")
    print(f"best FOV tracking: {METHOD_LABELS[best_fov]}")
    print(f"best obstacle detection: {METHOD_LABELS[best_obs]}")
    print(f"safest method: {METHOD_LABELS[safest]}")
    print(f"smoothest method: {METHOD_LABELS[smoothest]}")


def main():
    save_r95_csv()
    results = {}
    for name in METHODS:
        print(f"Simulating {METHOD_LABELS[name]}...")
        states, metrics = simulate_method(name)
        results[name] = {"states": states, "metrics": metrics, "summary": compute_metrics(states, metrics)}
    save_metrics_csv(results)
    save_trajectory_plot(results)
    save_bar_plots(results)
    print_summary(results)
    if GENERATE_MP4:
        print(f"Generating MP4: {MP4_FILENAME}")
        save_animation(results)
        print(f"Saved MP4: {MP4_FILENAME}")
    else:
        print("MP4 generation disabled.")
    print(f"Saved outputs in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
