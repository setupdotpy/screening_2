import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Wedge
from pathlib import Path

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    SKLEARN_AVAILABLE = True
except ImportError:
    GaussianProcessRegressor = None
    ConstantKernel = None
    RBF = None
    WhiteKernel = None
    SKLEARN_AVAILABLE = False


SEED = 7
GENERATE_MP4 = True
PLOT_DIR = Path("plot")
PLOT_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    }
)
X_MIN, X_MAX = -5000.0, 5000.0
Y_MIN, Y_MAX = 0.0, 10000.0
BOUNDARY_MARGIN = 300.0
X_LIMIT = X_MAX - BOUNDARY_MARGIN
Y_LOW = Y_MIN + BOUNDARY_MARGIN
Y_HIGH = Y_MAX - BOUNDARY_MARGIN
ROUTE_X = 0.0
ROUTE_Y_START = Y_LOW
ROUTE_Y_END = Y_HIGH
FINAL_Y = Y_HIGH - 200.0

DT = 5.0
STEPS = 6500
START_CLOCK = 10 * 3600
START_STATE = np.array([0.0, 500.0, np.pi / 2.0, 0.0])

VMAX = 3.0
CRUISE_SPEED = 2.0
OMEGA_MAX = 0.03
UC_MAX = 0.8
HORIZON = 8
ALPHA_CBF = 2.0

FOV_DEG = 60.0
FOV_RAD = np.deg2rad(FOV_DEG)
CAMERA_RANGE = 800.0
MAX_DETECTION_RANGE = 2200.0
CLEAR_OBSERVATION_RANGE = 450.0
OBSERVATION_RANGE = 450.0
REQUIRED_OBSERVATION_TIME = 300.0
R_BEST = 450.0
R_OBS = R_BEST
OBSERVATION_GATE_BUFFER = 25.0
SIGMA_ALIGN = 0.25
SIGMA_R_BEST = 250.0
SIGMA_BETA = 0.45

SAFETY_MARGIN = 250.0
CBF_BUFFER = 5.0
OBSTACLES = [
    {"center": np.array([0.0, 2500.0]), "radius": 400.0},
    {"center": np.array([800.0, 3500.0]), "radius": 450.0},
    {"center": np.array([1800.0, 4200.0]), "radius": 400.0},
    {"center": np.array([3250.0, 4000.0]), "radius": 100.0},
    {"center": np.array([-500.0, 7000.0]), "radius": 350.0},
    {"center": np.array([1200.0, 7800.0]), "radius": 450.0},
]

# Synthetic arrival sources used only to generate historical observations.
# These are simulation-only latent arrival sources and are not mission targets.
SYNTHETIC_ARRIVAL_SOURCES = np.array(
    [
        [-1200.0, 1500.0],
        [4000.0, 4000.0],
        [-900.0, 6500.0],
        [4500.0, 8000.0],
    ]
)
W_REJOIN_RETURN = 1.0
W_REJOIN_FORWARD = 0.1
HIST_POINTS_PER_TARGET = 80
HIST_SIGMA = 700.0
BACKGROUND_POINTS = 300
GP_NX = 120
GP_NY = 120
SIGMA_KDE = 900.0
NUM_TARGETS = 4
TARGET_PEAK_MIN_DISTANCE = 1500.0
TARGET_PEAK_THRESHOLD = 0.45
TARGET_PEAK_EXCLUSION_RADIUS = 1200.0
WINDOWS = np.array(
    [
        [11 * 3600, 12 * 3600],
        [14 * 3600, 14 * 3600 + 45 * 60],
        [15 * 3600 + 30 * 60, 16 * 3600 + 15 * 60],
        [17 * 3600, 17 * 3600 + 30 * 60],
    ],
    dtype=float,
)

LAMBDA_U = 0.03
LAMBDA_S = 0.15
W_PROGRESS = 30.0
W_CAM = 2.0
W_DETECT = 1.0
W_ROUTE_TARGET = {"route_visible": 0.3, "off_route": 0.03}
W_ROUTE_RETURN = 1.0
W_RETURN_PROGRESS = 80.0
W_RETURN_HEADING = 20.0
PROGRESS_SCALE = 100.0
URGENCY_SCALE = 1800.0
ROUTE_RETURN_THRESHOLD = 80.0
POSE_HOLD_RADIUS = 25.0
SLOWDOWN_RADIUS = 250.0
APPROACH_SPEED_LIMIT = 1.0
FINAL_APPROACH_RADIUS = 600.0
FINAL_STOP_RADIUS = 80.0
FINAL_DWELL_STEPS = 24
FINISH_TOL = 5.0

ANIMATION_FILENAME = PLOT_DIR / "route_active_sensing.mp4"
ANIMATION_FPS = 20
ANIMATION_DPI = 140


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def seconds_to_hms(seconds):
    seconds = int(seconds) % (24 * 3600)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def point_is_safe(point, extra_margin=0.0):
    if point[0] < -X_LIMIT or point[0] > X_LIMIT or point[1] < Y_LOW or point[1] > Y_HIGH:
        return False
    for obs in OBSTACLES:
        safe_radius = obs["radius"] + SAFETY_MARGIN + extra_margin
        if np.linalg.norm(point - obs["center"]) <= safe_radius:
            return False
    return True


def segment_intersects_circle(p1, p2, center, radius):
    seg = p2 - p1
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 < 1e-9:
        closest = p1
    else:
        t = np.clip(np.dot(center - p1, seg) / seg_len2, 0.0, 1.0)
        closest = p1 + t * seg
    return np.linalg.norm(closest - center) <= radius


def segment_intersects_safety(p0, p1):
    for obs in OBSTACLES:
        safe_radius = obs["radius"] + SAFETY_MARGIN
        if segment_intersects_circle(p0, p1, obs["center"], safe_radius):
            return True
    return False


def min_obstacle_clearance(point):
    clearances = []
    for obs in OBSTACLES:
        safe_radius = obs["radius"] + SAFETY_MARGIN
        clearances.append(np.linalg.norm(point - obs["center"]) - safe_radius)
    return float(np.min(clearances))


def generate_historical_arrivals():
    rng = np.random.default_rng(SEED)
    points = []
    for center in SYNTHETIC_ARRIVAL_SOURCES:
        samples = center + rng.normal(0.0, HIST_SIGMA, size=(HIST_POINTS_PER_TARGET, 2))
        samples[:, 0] = np.clip(samples[:, 0], -X_LIMIT, X_LIMIT)
        samples[:, 1] = np.clip(samples[:, 1], Y_LOW, Y_HIGH)
        points.append(samples)
    return np.vstack(points)


def build_lambda_bar_map():
    historical_points = generate_historical_arrivals()
    x_grid = np.linspace(X_MIN, X_MAX, GP_NX)
    y_grid = np.linspace(Y_MIN, Y_MAX, GP_NY)
    xx, yy = np.meshgrid(x_grid, y_grid)
    query = np.column_stack([xx.ravel(), yy.ravel()])

    if SKLEARN_AVAILABLE:
        rng = np.random.default_rng(SEED + 1)
        background = np.column_stack(
            [
                rng.uniform(-X_LIMIT, X_LIMIT, BACKGROUND_POINTS),
                rng.uniform(Y_LOW, Y_HIGH, BACKGROUND_POINTS),
            ]
        )
        x_train = np.vstack([historical_points, background])
        y_train = np.concatenate([np.ones(len(historical_points)), np.zeros(BACKGROUND_POINTS)])
        kernel = ConstantKernel(1.0) * RBF(length_scale=1200.0) + WhiteKernel(noise_level=0.05)
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
        gp.fit(x_train, y_train)
        predicted = gp.predict(query).reshape(GP_NY, GP_NX)
    else:
        diff = query[:, None, :] - historical_points[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        predicted = np.sum(np.exp(-dist2 / (2.0 * SIGMA_KDE**2)), axis=1).reshape(GP_NY, GP_NX)

    lambda_bar_map = np.clip(predicted, 0.0, None)
    max_value = float(np.max(lambda_bar_map))
    if max_value > 1e-12:
        lambda_bar_map = lambda_bar_map / max_value
    return historical_points, x_grid, y_grid, lambda_bar_map


HISTORICAL_POINTS, LAMBDA_X_GRID, LAMBDA_Y_GRID, LAMBDA_BAR_MAP = build_lambda_bar_map()


def select_targets_from_gp_map(num_targets=NUM_TARGETS):
    score_map = LAMBDA_BAR_MAP.copy()
    xx, yy = np.meshgrid(LAMBDA_X_GRID, LAMBDA_Y_GRID)

    for iy in range(GP_NY):
        for ix in range(GP_NX):
            p = np.array([xx[iy, ix], yy[iy, ix]])
            if not point_is_safe(p):
                score_map[iy, ix] = -np.inf

    selected = []
    warned_threshold = False
    while len(selected) < num_targets:
        flat_idx = int(np.argmax(score_map))
        peak_value = float(score_map.ravel()[flat_idx])
        if not np.isfinite(peak_value):
            break
        if peak_value < TARGET_PEAK_THRESHOLD and not warned_threshold:
            print("Warning: fewer GP peaks above threshold; selecting best remaining safe peaks.")
            warned_threshold = True

        iy, ix = np.unravel_index(flat_idx, score_map.shape)
        candidate = np.array([LAMBDA_X_GRID[ix], LAMBDA_Y_GRID[iy]])
        far_enough = all(np.linalg.norm(candidate - target) >= TARGET_PEAK_MIN_DISTANCE for target in selected)
        dist2 = (xx - candidate[0]) ** 2 + (yy - candidate[1]) ** 2
        score_map[dist2 <= TARGET_PEAK_EXCLUSION_RADIUS**2] = -np.inf
        if far_enough:
            selected.append(candidate)

    if len(selected) < num_targets:
        print(f"Warning: selected only {len(selected)} GP targets; filling from remaining safe cells.")
        while len(selected) < num_targets:
            flat_idx = int(np.argmax(score_map))
            peak_value = float(score_map.ravel()[flat_idx])
            if not np.isfinite(peak_value):
                break
            iy, ix = np.unravel_index(flat_idx, score_map.shape)
            candidate = np.array([LAMBDA_X_GRID[ix], LAMBDA_Y_GRID[iy]])
            selected.append(candidate)
            dist2 = (xx - candidate[0]) ** 2 + (yy - candidate[1]) ** 2
            score_map[dist2 <= TARGET_PEAK_EXCLUSION_RADIUS**2] = -np.inf

    selected_targets = np.array(selected[:num_targets])
    return selected_targets[np.argsort(selected_targets[:, 1])]


TARGETS = select_targets_from_gp_map(NUM_TARGETS)
TARGET_TYPES = ["route_visible" if abs(target[0] - ROUTE_X) <= 1800.0 else "off_route" for target in TARGETS]


def lambda_bar_at_position(q):
    x, y = float(q[0]), float(q[1])
    if x < X_MIN or x > X_MAX or y < Y_MIN or y > Y_MAX:
        return 0.0

    tx = (x - X_MIN) / (X_MAX - X_MIN) * (GP_NX - 1)
    ty = (y - Y_MIN) / (Y_MAX - Y_MIN) * (GP_NY - 1)
    ix = int(np.clip(np.floor(tx), 0, GP_NX - 2))
    iy = int(np.clip(np.floor(ty), 0, GP_NY - 2))
    fx = tx - ix
    fy = ty - iy

    v00 = LAMBDA_BAR_MAP[iy, ix]
    v10 = LAMBDA_BAR_MAP[iy, ix + 1]
    v01 = LAMBDA_BAR_MAP[iy + 1, ix]
    v11 = LAMBDA_BAR_MAP[iy + 1, ix + 1]
    value = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10
    value += (1.0 - fx) * fy * v01 + fx * fy * v11
    return float(np.clip(value, 0.0, 1.0))


def temporal_window_weight(target_idx, current_time):
    if target_idx is None or current_time is None or target_idx >= len(WINDOWS):
        return 1.0
    center = 0.5 * (WINDOWS[target_idx, 0] + WINDOWS[target_idx, 1])
    sigma_t = 0.5 * (WINDOWS[target_idx, 1] - WINDOWS[target_idx, 0])
    if sigma_t <= 1e-9:
        return 1.0
    value = math.exp(-((current_time - center) ** 2) / (2.0 * sigma_t**2))
    return float(np.clip(value, 0.0, 1.0))


def optimize_observation_pose(target):
    w_route = 1.0
    w_travel = 1.0
    w_orientation = 20.0
    w_clear = 1000.0
    route_point = np.array([ROUTE_X, target[1]])
    rejected_all = []
    fallback_candidates = []

    for radius in [R_BEST, 600.0, 800.0]:
        valid = []
        rejected = []
        for alpha in np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False):
            p = target + radius * np.array([np.cos(alpha), np.sin(alpha)])
            theta_view_required = math.atan2(target[1] - p[1], target[0] - p[0])
            min_clearance = min_obstacle_clearance(p)
            cost = w_route * abs(p[0])
            cost += w_travel * (np.linalg.norm(p - route_point) + np.linalg.norm(p - route_point))
            cost += w_orientation * abs(wrap_to_pi(theta_view_required))
            if min_clearance > 0.0:
                cost += w_clear / (min_clearance + 1.0)

            if point_is_safe(p):
                fallback_candidates.append((abs(radius - R_BEST), cost, -min_clearance, p, radius))

            if (
                not point_is_safe(p)
                or segment_intersects_safety(route_point, p)
                or segment_intersects_safety(p, route_point)
            ):
                rejected.append(p)
                continue

            valid.append((cost, p))

        rejected_all.extend(rejected)
        if valid:
            return min(valid, key=lambda item: item[0])[1], np.array(rejected_all), radius

    if fallback_candidates:
        _, _, _, pose, radius = min(fallback_candidates, key=lambda item: item[:3])
        print(f"Warning: no unobstructed route segment for target {target}; using safest candidate {pose}")
        return pose, np.array(rejected_all), radius

    manual = np.clip(target + np.array([-R_BEST, 0.0]), [-X_LIMIT, Y_LOW], [X_LIMIT, Y_HIGH])
    print(f"Warning: no safe observation pose found near target {target}; using clipped fallback {manual}")
    return manual, np.array(rejected_all), float(np.linalg.norm(manual - target))


def observation_poses(targets):
    poses = []
    rejected = []
    radii = []
    for target in targets:
        pose, rejected_points, radius = optimize_observation_pose(target)
        poses.append(pose)
        rejected.append(rejected_points)
        radii.append(radius)
    return np.array(poses), rejected, np.array(radii)


OBS_POSES, REJECTED_OBS_POSES, OBS_RADII = observation_poses(TARGETS)


def build_control_candidates():
    v_candidates = np.array([0.0, 1.0, CRUISE_SPEED, VMAX])
    omega_candidates = np.linspace(-OMEGA_MAX, OMEGA_MAX, 9)
    uc_candidates = np.linspace(-UC_MAX, UC_MAX, 9)
    vv, ww, cc = np.meshgrid(v_candidates, omega_candidates, uc_candidates, indexing="ij")
    return np.column_stack([vv.ravel(), ww.ravel(), cc.ravel()])


CONTROL_CANDIDATES = build_control_candidates()


def step_dynamics(state, control, dt=DT):
    x, y, psi, theta_c = state
    v, omega, u_c = control
    return np.array(
        [
            x + dt * v * np.cos(psi),
            y + dt * v * np.sin(psi),
            wrap_to_pi(psi + dt * omega),
            wrap_to_pi(theta_c + dt * u_c),
        ]
    )


def target_geometry(state, target):
    rel = target - state[:2]
    theta_goal = math.atan2(rel[1], rel[0])
    theta_view = state[2] + state[3]
    beta = wrap_to_pi(theta_goal - theta_view)
    return theta_goal, beta, float(np.linalg.norm(rel))


def target_geometry_many(states, target):
    rel = target[None, :] - states[:, :2]
    theta_goal = np.arctan2(rel[:, 1], rel[:, 0])
    theta_view = states[:, 2] + states[:, 3]
    beta = wrap_to_pi(theta_goal - theta_view)
    dist = np.linalg.norm(rel, axis=1)
    return theta_goal, beta, dist


def detection_prob_to_target(state, target, target_idx=None, current_time=None):
    # P_d = lambda_spatial * lambda_temporal * P_range * P_angle * FOV_gate * range_gate.
    # P_range peaks at R_BEST, P_angle peaks on the optical axis, FOV_gate
    # forces the target inside the camera FOV, and range_gate forces it inside
    # the maximum usable sensing range.
    _, beta, dist = target_geometry(state, target)
    lambda_spatial = lambda_bar_at_position(target)
    lambda_temporal = temporal_window_weight(target_idx, current_time)
    p_range = math.exp(-((dist - R_BEST) ** 2) / (2.0 * SIGMA_R_BEST**2))
    p_angle = math.exp(-(beta**2) / SIGMA_BETA**2)
    p_fov = 1.0 if abs(beta) <= FOV_RAD / 2.0 else 0.0
    p_range_gate = 1.0 if dist <= MAX_DETECTION_RANGE else 0.0
    return lambda_spatial * lambda_temporal * p_range * p_angle * p_fov * p_range_gate


def detection_terms_to_target(state, target, target_idx=None, current_time=None):
    _, beta, dist = target_geometry(state, target)
    lambda_spatial = lambda_bar_at_position(target)
    lambda_temporal = temporal_window_weight(target_idx, current_time)
    lambda_total = lambda_spatial * lambda_temporal
    p_range = math.exp(-((dist - R_BEST) ** 2) / (2.0 * SIGMA_R_BEST**2))
    p_angle = math.exp(-(beta**2) / SIGMA_BETA**2)
    p_fov = 1.0 if abs(beta) <= FOV_RAD / 2.0 else 0.0
    p_range_gate = 1.0 if dist <= MAX_DETECTION_RANGE else 0.0
    pd = lambda_total * p_range * p_angle * p_fov * p_range_gate
    return pd, p_range, p_angle, p_fov, p_range_gate, lambda_spatial, lambda_temporal, lambda_total


def detection_prob_many(states, target, target_idx=None, current_time=None):
    _, beta, dist = target_geometry_many(states, target)
    lambda_spatial = lambda_bar_at_position(target)
    lambda_temporal = temporal_window_weight(target_idx, current_time)
    lambda_total = lambda_spatial * lambda_temporal
    p_range = np.exp(-((dist - R_BEST) ** 2) / (2.0 * SIGMA_R_BEST**2))
    p_angle = np.exp(-(beta**2) / SIGMA_BETA**2)
    p_fov = (np.abs(beta) <= FOV_RAD / 2.0).astype(float)
    p_range_gate = (dist <= MAX_DETECTION_RANGE).astype(float)
    return lambda_total * p_range * p_angle * p_fov * p_range_gate


def rollout_candidates(state, controls):
    n = len(controls)
    pred = np.repeat(state[None, :], n, axis=0)
    safe = np.ones(n, dtype=bool)
    history = []
    v = controls[:, 0]
    omega = controls[:, 1]
    u_c = controls[:, 2]

    for _ in range(HORIZON):
        pred[:, 0] += DT * v * np.cos(pred[:, 2])
        pred[:, 1] += DT * v * np.sin(pred[:, 2])
        pred[:, 2] = wrap_to_pi(pred[:, 2] + DT * omega)
        pred[:, 3] = wrap_to_pi(pred[:, 3] + DT * u_c)
        history.append(pred.copy())

        safe &= (pred[:, 0] >= -X_LIMIT) & (pred[:, 0] <= X_LIMIT)
        safe &= (pred[:, 1] >= Y_LOW) & (pred[:, 1] <= Y_HIGH)

        vel = np.column_stack([v * np.cos(pred[:, 2]), v * np.sin(pred[:, 2])])
        for obs in OBSTACLES:
            safe_radius = obs["radius"] + SAFETY_MARGIN + CBF_BUFFER
            rel = pred[:, :2] - obs["center"]
            h = np.sum(rel * rel, axis=1) - safe_radius**2
            hdot = 2.0 * np.sum(rel * vel, axis=1)
            safe &= hdot + ALPHA_CBF * h >= -1e-8

    return safe, pred, np.stack(history, axis=0)


def nearest_obstacle_data(state):
    hs = []
    distances = []
    for obs in OBSTACLES:
        safe_radius = obs["radius"] + SAFETY_MARGIN
        d = np.linalg.norm(state[:2] - obs["center"])
        hs.append(d**2 - safe_radius**2)
        distances.append(d - safe_radius)
    return float(np.min(hs)), float(np.min(distances))


def fallback_control(state):
    nearest = min(OBSTACLES, key=lambda obs: np.linalg.norm(state[:2] - obs["center"]))
    away = state[:2] - nearest["center"]
    desired = math.atan2(away[1], away[0]) if np.linalg.norm(away) > 1e-6 else np.pi / 2.0
    heading_error = wrap_to_pi(desired - state[2])
    return np.array([0.0, np.clip(0.8 * heading_error, -OMEGA_MAX, OMEGA_MAX), 0.0])


def final_transit_control(state):
    goal = np.array([ROUTE_X, FINAL_Y])
    distance_to_goal = np.linalg.norm(state[:2] - goal)
    u_c = np.clip(wrap_to_pi(-state[3]) / DT, -UC_MAX, UC_MAX)
    if distance_to_goal <= FINAL_STOP_RADIUS or state[1] >= FINAL_Y - FINAL_STOP_RADIUS:
        return np.array([0.0, 0.0, u_c])

    desired_heading = math.atan2(goal[1] - state[1], goal[0] - state[0])
    heading_error = wrap_to_pi(desired_heading - state[2])
    omega = np.clip(0.45 * heading_error, -OMEGA_MAX, OMEGA_MAX)
    speed = APPROACH_SPEED_LIMIT if distance_to_goal <= FINAL_APPROACH_RADIUS else CRUISE_SPEED
    if abs(heading_error) > np.deg2rad(55.0):
        speed = min(speed, APPROACH_SPEED_LIMIT)
    return np.array([speed, omega, u_c])


def compute_rejoin_point(position, target_idx, next_target_idx=None):
    if next_target_idx is not None and next_target_idx < len(TARGETS):
        y_min = max(Y_LOW, TARGETS[target_idx, 1])
        y_max = min(Y_HIGH, TARGETS[next_target_idx, 1])
    else:
        y_min = max(Y_LOW, position[1])
        y_max = min(FINAL_Y, Y_HIGH)

    if y_min > y_max:
        y_min, y_max = y_max, y_min

    valid = []
    for y_candidate in np.linspace(y_min, y_max, 300):
        p_route = np.array([ROUTE_X, y_candidate])
        if not point_is_safe(p_route):
            continue
        if segment_intersects_safety(position, p_route):
            continue

        cost = W_REJOIN_RETURN * np.linalg.norm(position - p_route)
        if next_target_idx is not None and next_target_idx < len(TARGETS):
            cost += W_REJOIN_FORWARD * (y_candidate - TARGETS[target_idx, 1])
        valid.append((cost, p_route))

    if valid:
        return min(valid, key=lambda item: item[0])[1]

    if next_target_idx is not None and next_target_idx < len(TARGETS):
        fallback_y = y_max
    else:
        fallback_y = min(FINAL_Y, max(position[1], Y_LOW))
    fallback = np.array([ROUTE_X, fallback_y])
    print(f"Warning: no collision-free rejoin point found after T{target_idx}; using fallback {fallback}")
    return fallback

def mission_phase(target_idx, observed, return_to_route):
    if target_idx >= len(TARGETS):
        return "transit_final"
    if return_to_route:
        return "return_route"
    if observed[target_idx]:
        return "advance"
    return "observe"


def score_controls(state, u_prev, target_idx, observed, return_to_route, current_time, return_goal=None):
    controls = CONTROL_CANDIDATES
    safe, final_states, rollout_states = rollout_candidates(state, controls)
    if not np.any(safe):
        return fallback_control(state), False

    camera_target_idx = target_idx
    if target_idx >= len(TARGETS):
        goal_pose = np.array([ROUTE_X, FINAL_Y])
        target = goal_pose
        target_type = "route_visible"
    else:
        goal_pose = OBS_POSES[target_idx]
        target = TARGETS[target_idx]
        target_type = TARGET_TYPES[target_idx]
        if not return_to_route:
            pose_dist_now = np.linalg.norm(state[:2] - goal_pose)
            target_dist_now = np.linalg.norm(state[:2] - target)
            if pose_dist_now < 120.0 and target_dist_now > OBSERVATION_RANGE - OBSERVATION_GATE_BUFFER:
                direction = goal_pose - target
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    goal_pose = target + direction / direction_norm * (OBSERVATION_RANGE - OBSERVATION_GATE_BUFFER)
    if return_to_route:
        goal_pose = return_goal if return_goal is not None else np.array([ROUTE_X, min(Y_HIGH, state[1] + 3000.0)])
        if target_idx + 1 < len(TARGETS):
            camera_target_idx = target_idx + 1
            target = TARGETS[camera_target_idx]
            target_type = TARGET_TYPES[camera_target_idx]

    _, beta_final, dist_target = target_geometry_many(final_states, target)
    d_now = np.linalg.norm(state[:2] - goal_pose)
    d_pred = np.linalg.norm(final_states[:, :2] - goal_pose[None, :], axis=1)
    j_progress = (d_now - d_pred) / PROGRESS_SCALE
    desired_heading = np.arctan2(goal_pose[1] - final_states[:, 1], goal_pose[0] - final_states[:, 0])
    heading_error = wrap_to_pi(desired_heading - final_states[:, 2])
    j_return_heading = np.exp(-(heading_error**2) / (2.0 * 0.45**2))
    if target_idx < len(TARGETS) and current_time < WINDOWS[target_idx, 0] and d_now <= POSE_HOLD_RADIUS:
        j_progress = -d_pred / PROGRESS_SCALE

    time_to_window = max(WINDOWS[target_idx, 0] - current_time, 1.0) if target_idx < len(TARGETS) else 1.0
    urgency = URGENCY_SCALE / max(time_to_window, 1.0)
    urgency = float(np.clip(urgency, 0.25, 4.0))

    rollout_flat = rollout_states.reshape(-1, 4)
    theta_goal_roll, beta_roll, dist_roll = target_geometry_many(rollout_flat, target)
    theta_goal_roll = theta_goal_roll.reshape(HORIZON, -1)
    beta_roll = beta_roll.reshape(HORIZON, -1)
    dist_roll = dist_roll.reshape(HORIZON, -1)
    theta_c_des = wrap_to_pi(theta_goal_roll - rollout_states[:, :, 2])
    camera_error = wrap_to_pi(rollout_states[:, :, 3] - theta_c_des)

    j_cam = np.mean(np.exp(-(beta_roll**2) / (2.0 * SIGMA_ALIGN**2)), axis=0)
    p_detect = detection_prob_many(final_states, target, camera_target_idx, current_time)

    j_route = -np.abs(final_states[:, 0] - ROUTE_X)
    if return_to_route:
        dist_to_rejoin = np.linalg.norm(state[:2] - goal_pose)
        w_route = W_ROUTE_RETURN if dist_to_rejoin < 500.0 else 0.2
    else:
        w_route = W_ROUTE_TARGET.get(target_type, 0.3)

    if target_idx >= len(TARGETS):
        j_cam *= 0.0
        p_detect *= 0.0

    effort = LAMBDA_U * (controls[:, 0] ** 2 + 0.2 * controls[:, 1] ** 2 + 0.1 * controls[:, 2] ** 2)
    smooth = LAMBDA_S * np.sum((controls - u_prev[None, :]) ** 2, axis=1)
    progress_weight = W_RETURN_PROGRESS if return_to_route else W_PROGRESS * urgency
    scores = progress_weight * j_progress
    if return_to_route:
        scores += W_RETURN_HEADING * j_return_heading
    scores += W_CAM * j_cam + W_DETECT * p_detect + w_route * j_route / 1000.0
    if return_to_route:
        smooth *= 0.2
    scores -= effort + smooth
    if target_idx < len(TARGETS) and not return_to_route and d_now < SLOWDOWN_RADIUS:
        scores[controls[:, 0] > APPROACH_SPEED_LIMIT] = -np.inf
    scores[~safe] = -np.inf
    best = controls[int(np.argmax(scores))].copy()
    if (
        not return_to_route
        and target_idx < len(TARGETS)
        and TARGET_TYPES[target_idx] == "off_route"
        and d_now > 1500.0
        and abs(best[0]) < 1e-9
    ):
        desired_now = math.atan2(goal_pose[1] - state[1], goal_pose[0] - state[0])
        heading_error_now = wrap_to_pi(desired_now - state[2])
        if abs(heading_error_now) > 0.35:
            best[1] = np.clip(0.6 * heading_error_now, -OMEGA_MAX, OMEGA_MAX)
    return best, True


def update_observation(state, target_idx, observed_time, current_time):
    if target_idx >= len(TARGETS):
        return 0.0
    start, end = WINDOWS[target_idx]
    if not (start <= current_time <= end):
        return 0.0
    pd = detection_prob_to_target(state, TARGETS[target_idx], target_idx, current_time)
    observed_time[target_idx] = min(REQUIRED_OBSERVATION_TIME, observed_time[target_idx] + DT * pd)
    return pd


def simulate():
    state = START_STATE.copy()
    u_prev = np.zeros(3)
    target_idx = 0
    return_to_route = False
    return_source_idx = None
    rejoin_goal = None
    observed_time = np.zeros(len(TARGETS))
    observed = np.zeros(len(TARGETS), dtype=bool)
    missed = np.zeros(len(TARGETS), dtype=bool)

    states = [state.copy()]
    controls = []
    times = []
    target_hist = []
    phase_hist = []
    obs_frac_hist = [observed_time / REQUIRED_OBSERVATION_TIME]
    beta_hist = []
    distance_hist = []
    pose_distance_hist = []
    route_error_hist = []
    h_hist = []
    safety_dist_hist = []
    safe_hist = []
    observing_hist = []
    holding_hist = []
    pd_hist = []
    range_term_hist = []
    angle_term_hist = []
    fov_gate_hist = []
    range_gate_hist = []
    lambda_spatial_hist = []
    lambda_temporal_hist = []
    lambda_total_hist = []
    arrival_delay = np.zeros(len(TARGETS))
    arrival_recorded = np.zeros(len(TARGETS), dtype=bool)
    final_dwell_steps = 0

    for step in range(STEPS):
        current_time = START_CLOCK + step * DT
        if (
            target_idx >= len(TARGETS)
            and state[1] >= FINAL_Y - FINAL_STOP_RADIUS
            and abs(state[3]) < 1e-3
            and final_dwell_steps >= FINAL_DWELL_STEPS
        ):
            break

        if target_idx < len(TARGETS):
            pose_dist = np.linalg.norm(state[:2] - OBS_POSES[target_idx])
            if pose_dist <= POSE_HOLD_RADIUS and not arrival_recorded[target_idx]:
                arrival_delay[target_idx] = current_time - WINDOWS[target_idx, 0]
                arrival_recorded[target_idx] = True
            if observed_time[target_idx] >= REQUIRED_OBSERVATION_TIME:
                observed[target_idx] = True
                if TARGET_TYPES[target_idx] == "off_route" and not return_to_route:
                    return_to_route = True
                    return_source_idx = target_idx
                    next_target_idx = return_source_idx + 1 if return_source_idx + 1 < len(TARGETS) else None
                    rejoin_goal = compute_rejoin_point(state[:2], return_source_idx, next_target_idx)
            if current_time > WINDOWS[target_idx, 1] and not observed[target_idx]:
                missed[target_idx] = True
                observed[target_idx] = True
                if TARGET_TYPES[target_idx] == "off_route":
                    return_to_route = True
                    return_source_idx = target_idx
                    next_target_idx = return_source_idx + 1 if return_source_idx + 1 < len(TARGETS) else None
                    rejoin_goal = compute_rejoin_point(state[:2], return_source_idx, next_target_idx)

        return_goal = None
        if return_to_route:
            if rejoin_goal is None and return_source_idx is not None:
                next_target_idx = return_source_idx + 1 if return_source_idx + 1 < len(TARGETS) else None
                rejoin_goal = compute_rejoin_point(state[:2], return_source_idx, next_target_idx)
            return_goal = rejoin_goal
            if return_goal is not None and np.linalg.norm(state[:2] - return_goal) < ROUTE_RETURN_THRESHOLD:
                return_to_route = False
                return_source_idx = None
                rejoin_goal = None
                target_idx += 1
        elif target_idx < len(TARGETS) and observed[target_idx] and not return_to_route:
            target_idx += 1

        hold_for_window = False
        if target_idx < len(TARGETS) and not observed[target_idx] and not return_to_route:
            pose_dist_now = np.linalg.norm(state[:2] - OBS_POSES[target_idx])
            _, _, target_dist_now = target_geometry(state, TARGETS[target_idx])
            hold_for_window = (
                pose_dist_now <= POSE_HOLD_RADIUS
                and target_dist_now <= OBSERVATION_RANGE
                and current_time <= WINDOWS[target_idx, 1]
            )

        if hold_for_window:
            theta_goal, _, _ = target_geometry(state, TARGETS[target_idx])
            theta_c_des = wrap_to_pi(theta_goal - state[2])
            u_c = np.clip(wrap_to_pi(theta_c_des - state[3]) / DT, -UC_MAX, UC_MAX)
            u = np.array([0.0, 0.0, u_c])
            had_safe = True
        elif target_idx >= len(TARGETS):
            u = final_transit_control(state)
            had_safe = True
        else:
            u, had_safe = score_controls(state, u_prev, target_idx, observed, return_to_route, current_time, return_goal)
        if np.all(observed):
            u[2] = np.clip(wrap_to_pi(-state[3]) / DT, -UC_MAX, UC_MAX)
        state = step_dynamics(state, u)
        if target_idx >= len(TARGETS) and state[1] >= FINAL_Y - FINAL_STOP_RADIUS and np.allclose(u[:2], 0.0):
            final_dwell_steps += 1
        else:
            final_dwell_steps = 0
        pd_step = update_observation(state, target_idx, observed_time, current_time)

        if target_idx < len(TARGETS):
            _, beta, dist_target = target_geometry(state, TARGETS[target_idx])
            pose_dist = np.linalg.norm(state[:2] - OBS_POSES[target_idx])
            (
                pd_val,
                p_range,
                p_angle,
                p_fov,
                p_range_gate,
                lambda_spatial,
                lambda_temporal,
                lambda_total,
            ) = detection_terms_to_target(state, TARGETS[target_idx], target_idx, current_time)
        else:
            beta = 0.0
            dist_target = 0.0
            pose_dist = np.linalg.norm(state[:2] - np.array([ROUTE_X, FINAL_Y]))
            pd_val, p_range, p_angle, p_fov, p_range_gate = 0.0, 0.0, 0.0, 0.0, 0.0
            lambda_spatial, lambda_temporal, lambda_total = 0.0, 0.0, 0.0
        h_min, safety_dist = nearest_obstacle_data(state)

        states.append(state.copy())
        controls.append(u)
        times.append(current_time)
        target_hist.append(min(target_idx, len(TARGETS) - 1))
        phase_hist.append(mission_phase(target_idx, observed, return_to_route))
        obs_frac_hist.append(observed_time / REQUIRED_OBSERVATION_TIME)
        beta_hist.append(beta)
        distance_hist.append(dist_target)
        pose_distance_hist.append(pose_dist)
        route_error_hist.append(abs(state[0]))
        h_hist.append(h_min)
        safety_dist_hist.append(safety_dist)
        safe_hist.append(had_safe)
        observing_hist.append(pd_step > 0.0)
        holding_hist.append(hold_for_window)
        pd_hist.append(pd_val)
        range_term_hist.append(p_range)
        angle_term_hist.append(p_angle)
        fov_gate_hist.append(p_fov)
        range_gate_hist.append(p_range_gate)
        lambda_spatial_hist.append(lambda_spatial)
        lambda_temporal_hist.append(lambda_temporal)
        lambda_total_hist.append(lambda_total)
        u_prev = u

    return np.array(states), {
        "controls": np.array(controls),
        "times": np.array(times),
        "target": np.array(target_hist, dtype=int),
        "phase": np.array(phase_hist, dtype=object),
        "coverage": np.array(obs_frac_hist),
        "observed_time": observed_time,
        "observed": observed,
        "missed": missed,
        "arrival_delay": arrival_delay,
        "beta": np.array(beta_hist),
        "target_distance": np.array(distance_hist),
        "pose_distance": np.array(pose_distance_hist),
        "route_error": np.array(route_error_hist),
        "h_min": np.array(h_hist),
        "dist_to_safety": np.array(safety_dist_hist),
        "safe_candidate": np.array(safe_hist, dtype=bool),
        "observing": np.array(observing_hist, dtype=bool),
        "holding": np.array(holding_hist, dtype=bool),
        "pd": np.array(pd_hist),
        "range_term": np.array(range_term_hist),
        "angle_term": np.array(angle_term_hist),
        "fov_gate": np.array(fov_gate_hist),
        "range_gate": np.array(range_gate_hist),
        "lambda_spatial": np.array(lambda_spatial_hist),
        "lambda_temporal": np.array(lambda_temporal_hist),
        "lambda_total": np.array(lambda_total_hist),
    }


def target_color(i, metrics, frame):
    idx = min(frame, len(metrics["coverage"]) - 1)
    active = metrics["target"][min(max(frame - 1, 0), len(metrics["target"]) - 1)] if len(metrics["target"]) else 0
    if metrics["missed"][i]:
        return "gray"
    if metrics["coverage"][idx, i] >= 1.0:
        return "limegreen"
    if i == active:
        return "red"
    return "yellow"


def target_status(i, metrics, frame):
    idx = min(frame, len(metrics["coverage"]) - 1)
    active = metrics["target"][min(max(frame - 1, 0), len(metrics["target"]) - 1)] if len(metrics["target"]) else 0
    if metrics["missed"][i]:
        return "MISSED"
    if metrics["coverage"][idx, i] >= 1.0:
        return "OBSERVED"
    if i == active:
        return "ACTIVE"
    return "FUTURE"


def schedule_box_text(metrics, frame):
    idx = min(max(frame - 1, 0), len(metrics["target"]) - 1)
    t = metrics["times"][idx] if len(metrics["times"]) else START_CLOCK
    active = metrics["target"][idx] if len(metrics["target"]) else 0
    dist = metrics["target_distance"][idx] if len(metrics["target_distance"]) else 0.0
    obs = metrics["coverage"][min(frame, len(metrics["coverage"]) - 1), active]
    lines = ["Mission schedule:"]
    for i, (start, end) in enumerate(WINDOWS):
        lines.append(f"T{i} {seconds_to_hms(start)[:5]}-{seconds_to_hms(end)[:5]} {target_status(i, metrics, frame)}")
    lines += [
        "",
        f"Current time: {seconds_to_hms(t)}",
        f"Active target: T{active}",
        f"Observation timer: {obs * REQUIRED_OBSERVATION_TIME:.0f}/{REQUIRED_OBSERVATION_TIME:.0f} s",
        f"Distance to target: {dist:.0f} m",
    ]
    return "\n".join(lines)


def draw_usv(ax, state):
    size = 90.0
    body = np.array([[size, 0.0], [-0.55 * size, 0.45 * size], [-0.55 * size, -0.45 * size]])
    rot = np.array([[np.cos(state[2]), -np.sin(state[2])], [np.sin(state[2]), np.cos(state[2])]])
    verts = body @ rot.T + state[:2]
    return ax.add_patch(Polygon(verts, closed=True, fc="white", ec="black", lw=0.8, zorder=8))


def draw_camera(ax, state):
    theta_deg = np.rad2deg(state[2] + state[3])
    theta = state[2] + state[3]
    wedge = Wedge(
        state[:2],
        CAMERA_RANGE,
        theta_deg - FOV_DEG / 2.0,
        theta_deg + FOV_DEG / 2.0,
        fc="cyan",
        ec="cyan",
        alpha=0.18,
        lw=1.1,
        zorder=4,
    )
    ax.add_patch(wedge)
    artists = [wedge]
    for ang, style, lw in [(theta, "-", 1.6), (theta - FOV_RAD / 2.0, "--", 1.0), (theta + FOV_RAD / 2.0, "--", 1.0)]:
        end = state[:2] + CAMERA_RANGE * np.array([np.cos(ang), np.sin(ang)])
        line, = ax.plot([state[0], end[0]], [state[1], end[1]], color="#007c91", ls=style, lw=lw, zorder=5)
        artists.append(line)
    return artists


def setup_map(ax, states, metrics, frame=None, show_target_fovs=False):
    ocean = "#bfe9ff"
    ax.set_facecolor(ocean)
    ax.figure.set_facecolor(ocean)
    frame = len(states) - 1 if frame is None else frame
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
    ax.scatter(HISTORICAL_POINTS[:, 0], HISTORICAL_POINTS[:, 1], s=4, c="white", alpha=0.22, linewidths=0, zorder=1)
    ax.scatter(
        SYNTHETIC_ARRIVAL_SOURCES[:, 0],
        SYNTHETIC_ARRIVAL_SOURCES[:, 1],
        s=55,
        c="black",
        marker="x",
        linewidths=1.5,
        zorder=8,
    )
    ax.plot([ROUTE_X, ROUTE_X], [ROUTE_Y_START, ROUTE_Y_END], color="#1b6ca8", lw=2.0, ls="--")
    ax.add_patch(plt.Rectangle((-X_LIMIT, Y_LOW), 2 * X_LIMIT, Y_HIGH - Y_LOW, fc="none", ec="deepskyblue", ls="--"))
    for i, obs in enumerate(OBSTACLES):
        safe_radius = obs["radius"] + SAFETY_MARGIN
        ax.add_patch(Circle(obs["center"], obs["radius"], fc="none", ec="red", lw=2.0))
        ax.add_patch(Circle(obs["center"], safe_radius, fc="none", ec="red", ls="--", lw=1.2))
        ax.text(obs["center"][0], obs["center"][1], f"O{i}", ha="center", va="center", fontsize=8)
    for i, (target, pose) in enumerate(zip(TARGETS, OBS_POSES)):
        color = target_color(i, metrics, frame or len(states) - 1)
        ax.scatter(target[0], target[1], s=180, marker="*", c=color, ec="black", zorder=9)
        ax.scatter(pose[0], pose[1], s=95, marker="s", c="white", ec=color, lw=2.5, zorder=9)
        ax.add_patch(Circle(target, OBS_RADII[i], fc="none", ec="black", ls=":", lw=1.0, alpha=0.55))
        ax.plot([pose[0], target[0]], [pose[1], target[1]], color=color, lw=1.0, alpha=0.75)
        ax.text(target[0] + 70, target[1] + 70, f"T{i}", weight="bold")
    ax.plot(states[:, 0], states[:, 1], color="#004f80", lw=2.0, zorder=6)
    hold_count = min(len(metrics.get("holding", [])), len(states) - 1)
    if hold_count:
        hold_mask = metrics["holding"][:hold_count]
        hold_states = states[1 : hold_count + 1][hold_mask]
        if len(hold_states):
            ax.scatter(hold_states[:, 0], hold_states[:, 1], s=9, c="purple", alpha=0.75, zorder=7)
    ax.text(
        0.02,
        0.98,
        schedule_box_text(metrics, frame),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "boxstyle": "round,pad=0.3"},
        zorder=12,
    )
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("GP-Guided MPC-CBF Active Sensing for Maritime Target Detection")
    legend_items = [
        mlines.Line2D([], [], color="#1b6ca8", ls="--", lw=2, label="nominal route"),
        mlines.Line2D([], [], color="#004f80", lw=2, label="USV trajectory"),
        mpatches.Patch(fc="cyan", ec="cyan", alpha=0.18, label="camera FOV"),
        mlines.Line2D([], [], color="#007c91", lw=1.6, label="FOV center ray"),
        mlines.Line2D([], [], marker="*", color="w", markerfacecolor="limegreen", markeredgecolor="black", markersize=12, label="GP-selected target"),
        mlines.Line2D([], [], marker="s", color="w", markerfacecolor="white", markeredgecolor="black", markersize=8, label="Optimized observation pose"),
        mlines.Line2D([], [], marker="o", color="red", markerfacecolor="none", markeredgecolor="red", markersize=10, markeredgewidth=2, ls="None", label="obstacle"),
        mlines.Line2D([], [], color="red", ls="--", lw=1.2, label="safety radius"),
        mlines.Line2D([], [], color="deepskyblue", ls="--", lw=1.6, label="boundary margin"),
        mlines.Line2D([], [], marker="x", color="black", ls="None", markersize=7, label="Synthetic arrival source"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=7)
    if show_target_fovs and len(states):
        for target_idx in range(len(TARGETS)):
            mask = (metrics["target"] == target_idx) & metrics["observing"]
            candidate_indices = np.where(mask)[0]
            if len(candidate_indices) == 0:
                mask = (metrics["target"] == target_idx) & metrics["holding"]
                candidate_indices = np.where(mask)[0]
            if len(candidate_indices) == 0:
                continue
            local_pd = metrics["pd"][candidate_indices] if len(metrics.get("pd", [])) else np.zeros(len(candidate_indices))
            best_metric_idx = candidate_indices[int(np.argmax(local_pd))]
            state_idx = min(best_metric_idx + 1, len(states) - 1)
            draw_camera(ax, states[state_idx])
            draw_usv(ax, states[state_idx])
    return lambda_im


def status_text(metrics, frame):
    idx = min(max(frame - 1, 0), len(metrics["target"]) - 1)
    t = metrics["times"][idx] if len(metrics["times"]) else START_CLOCK
    target_idx = metrics["target"][idx] if len(metrics["target"]) else 0
    phase = metrics["phase"][idx] if len(metrics["phase"]) else "start"
    beta = np.rad2deg(metrics["beta"][idx]) if len(metrics["beta"]) else 0.0
    dist = metrics["target_distance"][idx] if len(metrics["target_distance"]) else 0.0
    obs = metrics["coverage"][min(frame, len(metrics["coverage"]) - 1)]
    return (
        f"Time: {seconds_to_hms(t)}\n"
        f"Active: T{target_idx} ({phase})\n"
        f"Expected obs: {np.array2string(obs, precision=2, suppress_small=True)}\n"
        f"Beta: {beta:+.1f} deg\n"
        f"Target dist: {dist:.0f} m"
    )


def save_animation(states, metrics):
    fig, ax = plt.subplots(figsize=(8, 8))

    frame_step = max(1, len(states) // 360)
    frames = list(range(0, len(states), frame_step))
    if frames[-1] != len(states) - 1:
        frames.append(len(states) - 1)

    def update(frame):
        ax.clear()
        setup_map(ax, states[: frame + 1], metrics, frame=frame)
        state = states[frame]
        if frame > 0 and len(metrics["target"]):
            target = TARGETS[metrics["target"][min(frame - 1, len(metrics["target"]) - 1)]]
            ax.plot([state[0], target[0]], [state[1], target[1]], color="red", lw=1.0, zorder=6)
        draw_camera(ax, state)
        draw_usv(ax, state)
        return []

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / ANIMATION_FPS, blit=False)
    writer = animation.FFMpegWriter(fps=ANIMATION_FPS, bitrate=2200)
    ani.save(ANIMATION_FILENAME, writer=writer, dpi=ANIMATION_DPI)
    plt.close(fig)


def save_map(states, metrics):
    fig, ax = plt.subplots(figsize=(9, 9))
    lambda_im = setup_map(ax, states, metrics, show_target_fovs=True)
    cbar = fig.colorbar(lambda_im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized GP target-arrival likelihood")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "trajectory_map.png", dpi=180)
    plt.close(fig)


def save_series(filename, y, ylabel, title, x=None, xlabel="step"):
    fig, ax = plt.subplots(figsize=(9, 4))
    x_values = np.arange(len(y)) if x is None else x
    ax.plot(x_values, y, lw=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=170)
    plt.close(fig)


def save_observation_timer(metrics):
    fig, ax = plt.subplots(figsize=(9, 4))
    if len(metrics["times"]):
        x = np.concatenate([[0.0], (metrics["times"] - START_CLOCK) / 60.0])
        xlabel = "minutes after 10:00"
    else:
        x = np.arange(len(metrics["coverage"]))
        xlabel = "step"
    for i in range(len(TARGETS)):
        ax.plot(x, metrics["coverage"][:, i], lw=1.6, label=f"T{i}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("expected observation fraction")
    ax.set_title("Expected observation progress from detection probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "observation_timer.png", dpi=170)
    plt.close(fig)


def save_schedule_status(metrics):
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [f"T{i}" for i in range(len(TARGETS))]
    for i, (start, end) in enumerate(WINDOWS):
        ax.barh(i, (end - start) / 60.0, left=(start - START_CLOCK) / 60.0, color="lightgray", edgecolor="black")
        mask = (metrics["target"] == i) & metrics["observing"]
        if np.any(mask):
            times_min = (metrics["times"][mask] - START_CLOCK) / 60.0
            pd_minutes = np.sum(metrics["pd"][mask]) * DT / 60.0
            ax.barh(i, pd_minutes, left=times_min[0], color="green", alpha=0.75)
        arrival_time = start + metrics["arrival_delay"][i]
        if metrics["arrival_delay"][i] > 0:
            ax.axvline((arrival_time - START_CLOCK) / 60.0, color="red", lw=1.2)
    ax.set_yticks(range(len(TARGETS)), labels)
    ax.set_xlabel("minutes after 10:00")
    ax.set_title("Schedule status: expected observation time from P_d")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "schedule_status.png", dpi=170)
    plt.close(fig)


def save_mission_timeline(metrics):
    fig, ax = plt.subplots(figsize=(10, 3))
    if len(metrics["target"]):
        ax.step((metrics["times"] - START_CLOCK) / 60.0, metrics["target"], where="post")
    ax.set_yticks(range(len(TARGETS)), [f"T{i}" for i in range(len(TARGETS))])
    ax.set_xlabel("minutes after 10:00")
    ax.set_ylabel("active target")
    ax.set_title("Mission timeline")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mission_timeline.png", dpi=170)
    plt.close(fig)


def save_gp_lambda_map():
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(
        LAMBDA_BAR_MAP,
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.9,
    )
    ax.scatter(HISTORICAL_POINTS[:, 0], HISTORICAL_POINTS[:, 1], s=5, c="white", alpha=0.35, linewidths=0)
    ax.scatter(
        SYNTHETIC_ARRIVAL_SOURCES[:, 0],
        SYNTHETIC_ARRIVAL_SOURCES[:, 1],
        s=65,
        c="black",
        marker="x",
        linewidths=1.8,
        label="X : Synthetic arrival source",
        zorder=5,
    )
    ax.scatter(TARGETS[:, 0], TARGETS[:, 1], s=170, marker="*", c="red", ec="black", label="* : GP-selected target", zorder=5)
    ax.scatter(OBS_POSES[:, 0], OBS_POSES[:, 1], s=90, marker="s", c="white", ec="black", label="square : Optimized observation pose", zorder=5)
    for i, target in enumerate(TARGETS):
        ax.text(target[0] + 80, target[1] + 80, f"T{i}", color="white", weight="bold")
    for obs in OBSTACLES:
        ax.add_patch(Circle(obs["center"], obs["radius"], fc="none", ec="red", lw=1.5))
    ax.plot([ROUTE_X, ROUTE_X], [ROUTE_Y_START, ROUTE_Y_END], color="white", lw=2.0, ls="--")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("GP target selection and optimized observation poses")
    ax.legend(loc="lower right", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized GP target-arrival likelihood")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gp_lambda_map.png", dpi=180)
    plt.close(fig)


def print_selected_target_summary():
    print("Synthetic arrival sources:")
    print(SYNTHETIC_ARRIVAL_SOURCES)
    print("Automatically selected TARGETS:")
    print(TARGETS)
    print("TARGET_TYPES:")
    print(TARGET_TYPES)
    print("Spatial lambda at selected targets:")
    for i, target in enumerate(TARGETS):
        print(f"T{i}: {lambda_bar_at_position(target):.3f}")


def main():
    np.random.default_rng(SEED)
    print_selected_target_summary()
    states, metrics = simulate()
    save_gp_lambda_map()
    save_map(states, metrics)
    save_schedule_status(metrics)
    save_observation_timer(metrics)
    time_min = (metrics["times"] - START_CLOCK) / 60.0 if len(metrics["times"]) else None
    time_label = "minutes after 10:00" if time_min is not None else "step"
    save_series("cbf_h.png", metrics["h_min"], "min h", "Minimum CBF h", time_min, time_label)
    save_series("detection_probability.png", metrics["pd"], "P_d", "Target detection probability", time_min, time_label)
    if GENERATE_MP4:
        save_animation(states, metrics)

    print(f"Saved core PNG outputs in {PLOT_DIR}/")
    if GENERATE_MP4:
        print(f"Saved {ANIMATION_FILENAME}")
    else:
        print("MP4 generation disabled.")
    print(f"Final observed times: {metrics['observed_time']}")
    print(f"Missed targets: {metrics['missed']}")
    print(f"Final state: {states[-1]}")
    print(f"Minimum CBF h: {np.min(metrics['h_min']):.3f}")
    print(f"Steps with no safe sampled candidate: {np.sum(~metrics['safe_candidate'])}")


if __name__ == "__main__":
    main()
