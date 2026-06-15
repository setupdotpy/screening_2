import csv
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from pathlib import Path

import usv_active_sensing_mpc_cbf_sim_research_model as base


base.GENERATE_MP4 = False
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

METHODS = {
    "Ours": {"color": "#004f80", "ls": "-", "lw": 2.4},
    "B2 Greedy nearest": {"color": "#f28e2b", "ls": "--", "lw": 2.2},
    "B3 No-CBF": {"color": "#7b3294", "ls": ":", "lw": 2.7},
}

B2_POSE_HOLD_RADIUS = 60.0
GENERATE_COMPARISON_MP4 = True
COMPARISON_MP4_FILENAME = PLOT_DIR / "comparison_test.mp4"
COMPARISON_ANIMATION_STRIDE = 20
COMPARISON_ANIMATION_FPS = 20

EXTRA_COMPARISON_OBSTACLES = [
    {"center": np.array([-260.0, 1450.0]), "radius": 100.0},
    {"center": np.array([1600.0, 6920.0]), "radius": 180.0},
    {"center": np.array([2200.0, 7110.0]), "radius": 180.0},
    {"center": np.array([2850.0, 7360.0]), "radius": 180.0},
    {"center": np.array([580.0, 9260.0]), "radius": 100.0},
]

base.OBSTACLES = base.OBSTACLES + EXTRA_COMPARISON_OBSTACLES
base.OBS_POSES, base.REJECTED_OBS_POSES, base.OBS_RADII = base.observation_poses(base.TARGETS)


def rollout_candidates_no_cbf(state, controls):
    n = len(controls)
    pred = np.repeat(state[None, :], n, axis=0)
    safe = np.ones(n, dtype=bool)
    history = []
    v = controls[:, 0]
    omega = controls[:, 1]
    u_c = controls[:, 2]

    for _ in range(base.HORIZON):
        pred[:, 0] += base.DT * v * np.cos(pred[:, 2])
        pred[:, 1] += base.DT * v * np.sin(pred[:, 2])
        pred[:, 2] = base.wrap_to_pi(pred[:, 2] + base.DT * omega)
        pred[:, 3] = base.wrap_to_pi(pred[:, 3] + base.DT * u_c)
        history.append(pred.copy())

        safe &= (pred[:, 0] >= -base.X_LIMIT) & (pred[:, 0] <= base.X_LIMIT)
        safe &= (pred[:, 1] >= base.Y_LOW) & (pred[:, 1] <= base.Y_HIGH)

    return safe, pred, np.stack(history, axis=0)


def greedy_nearest_observation_poses():
    poses = []
    alphas = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    for target in base.TARGETS:
        route_point = np.array([base.ROUTE_X, target[1]])
        valid = []
        for alpha in alphas:
            pose = target + base.R_BEST * np.array([np.cos(alpha), np.sin(alpha)])
            if not base.point_is_safe(pose):
                continue
            valid.append((np.linalg.norm(pose - route_point), pose))
        if valid:
            poses.append(min(valid, key=lambda item: item[0])[1])
        else:
            print(f"Warning: B2 found no safe pose near target {target}; using optimized base pose.")
            poses.append(base.OBS_POSES[len(poses)])
    return np.array(poses)


def select_safe_b2_control(state, desired_control):
    safe, _, _ = base.rollout_candidates(state, base.CONTROL_CANDIDATES)
    if not np.any(safe):
        return base.fallback_control(state), False

    controls = base.CONTROL_CANDIDATES[safe]
    scale = np.array([1.0, 8.0, 1.5])
    distances = np.linalg.norm((controls - desired_control[None, :]) * scale[None, :], axis=1)
    return controls[int(np.argmin(distances))].copy(), True


def b2_guidance_point(state, goal_pose):
    blocking = []
    p0 = state[:2]
    for obs in base.OBSTACLES:
        radius = obs["radius"] + base.SAFETY_MARGIN + 120.0
        if base.segment_intersects_circle(p0, goal_pose, obs["center"], radius):
            blocking.append((np.linalg.norm(p0 - obs["center"]), obs, radius))
    if not blocking:
        return goal_pose

    _, obs, radius = min(blocking, key=lambda item: item[0])
    path = goal_pose - p0
    path_norm = np.linalg.norm(path)
    if path_norm < 1e-6:
        return goal_pose
    unit = path / path_norm
    perp = np.array([-unit[1], unit[0]])
    candidates = []
    for sign in [-1.0, 1.0]:
        detour = obs["center"] + sign * perp * (radius + 260.0)
        detour[0] = np.clip(detour[0], -base.X_LIMIT, base.X_LIMIT)
        detour[1] = np.clip(detour[1], base.Y_LOW, base.Y_HIGH)
        if not base.point_is_safe(detour, extra_margin=20.0):
            continue
        blocked_to_detour = base.segment_intersects_circle(p0, detour, obs["center"], radius)
        cost = np.linalg.norm(p0 - detour) + np.linalg.norm(detour - goal_pose)
        if blocked_to_detour:
            cost += 5000.0
        candidates.append((cost, detour))
    if not candidates:
        return goal_pose
    return min(candidates, key=lambda item: item[0])[1]


def b2_waypoint_control(state, goal_pose, target_idx, current_time):
    del current_time
    guidance = b2_guidance_point(state, goal_pose)
    to_goal = guidance - state[:2]
    distance_to_pose = float(np.linalg.norm(to_goal))
    desired_heading = math.atan2(to_goal[1], to_goal[0]) if distance_to_pose > 1e-6 else state[2]
    heading_error = base.wrap_to_pi(desired_heading - state[2])

    omega = np.clip(0.8 * heading_error, -base.OMEGA_MAX, base.OMEGA_MAX)
    speed = base.CRUISE_SPEED
    if distance_to_pose < 300.0:
        speed = min(speed, 1.0)
    if distance_to_pose < 80.0:
        speed = min(speed, 0.4)
    if B2_POSE_HOLD_RADIUS < distance_to_pose < 80.0:
        speed = 1.0
    if abs(heading_error) > np.deg2rad(70.0):
        speed = 0.0
    elif abs(heading_error) > np.deg2rad(35.0):
        speed = min(speed, 1.0)

    if target_idx < len(base.TARGETS):
        theta_goal, _, _ = base.target_geometry(state, base.TARGETS[target_idx])
        theta_c_des = base.wrap_to_pi(theta_goal - state[2])
        u_c = np.clip(base.wrap_to_pi(theta_c_des - state[3]) / base.DT, -base.UC_MAX, base.UC_MAX)
    else:
        u_c = np.clip(base.wrap_to_pi(-state[3]) / base.DT, -base.UC_MAX, base.UC_MAX)

    desired = np.array([speed, omega, u_c])
    return select_safe_b2_control(state, desired)


def score_controls_variant(
    state,
    u_prev,
    target_idx,
    observed,
    return_to_route,
    current_time,
    obs_poses,
    use_cbf=True,
    return_goal=None,
):
    controls = base.CONTROL_CANDIDATES
    rollout_fn = base.rollout_candidates if use_cbf else rollout_candidates_no_cbf
    safe, final_states, rollout_states = rollout_fn(state, controls)
    if not np.any(safe):
        return base.fallback_control(state), False

    camera_target_idx = target_idx
    if target_idx >= len(base.TARGETS):
        goal_pose = np.array([base.ROUTE_X, base.FINAL_Y])
        target = goal_pose
        target_type = "route_visible"
    else:
        goal_pose = obs_poses[target_idx]
        target = base.TARGETS[target_idx]
        target_type = base.TARGET_TYPES[target_idx]
        if not return_to_route:
            pose_dist_now = np.linalg.norm(state[:2] - goal_pose)
            target_dist_now = np.linalg.norm(state[:2] - target)
            if pose_dist_now < 120.0 and target_dist_now > base.OBSERVATION_RANGE - base.OBSERVATION_GATE_BUFFER:
                direction = goal_pose - target
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    goal_pose = target + direction / norm * (base.OBSERVATION_RANGE - base.OBSERVATION_GATE_BUFFER)

    if return_to_route:
        goal_pose = return_goal if return_goal is not None else np.array([base.ROUTE_X, min(base.Y_HIGH, state[1] + 3000.0)])
        if target_idx + 1 < len(base.TARGETS):
            camera_target_idx = target_idx + 1
            target = base.TARGETS[camera_target_idx]
            target_type = base.TARGET_TYPES[camera_target_idx]

    d_now = np.linalg.norm(state[:2] - goal_pose)
    d_pred = np.linalg.norm(final_states[:, :2] - goal_pose[None, :], axis=1)
    j_progress = (d_now - d_pred) / base.PROGRESS_SCALE

    desired_heading = np.arctan2(goal_pose[1] - final_states[:, 1], goal_pose[0] - final_states[:, 0])
    heading_error = base.wrap_to_pi(desired_heading - final_states[:, 2])
    j_return_heading = np.exp(-(heading_error**2) / (2.0 * 0.45**2))

    if target_idx < len(base.TARGETS) and current_time < base.WINDOWS[target_idx, 0] and d_now <= base.POSE_HOLD_RADIUS:
        j_progress = -d_pred / base.PROGRESS_SCALE

    if target_idx < len(base.TARGETS):
        time_to_window = max(base.WINDOWS[target_idx, 0] - current_time, 1.0)
    else:
        time_to_window = 1.0
    urgency = float(np.clip(base.URGENCY_SCALE / max(time_to_window, 1.0), 0.25, 4.0))

    rollout_flat = rollout_states.reshape(-1, 4)
    _, beta_roll, _ = base.target_geometry_many(rollout_flat, target)
    beta_roll = beta_roll.reshape(base.HORIZON, -1)
    j_cam = np.mean(np.exp(-(beta_roll**2) / (2.0 * base.SIGMA_ALIGN**2)), axis=0)
    p_detect = base.detection_prob_many(final_states, target, camera_target_idx, current_time)

    j_route = -np.abs(final_states[:, 0] - base.ROUTE_X)
    if return_to_route:
        dist_to_rejoin = np.linalg.norm(state[:2] - goal_pose)
        w_route = base.W_ROUTE_RETURN if dist_to_rejoin < 500.0 else 0.2
    else:
        w_route = base.W_ROUTE_TARGET.get(target_type, 0.3)

    if target_idx >= len(base.TARGETS):
        j_cam *= 0.0
        p_detect *= 0.0

    effort = base.LAMBDA_U * (controls[:, 0] ** 2 + 0.2 * controls[:, 1] ** 2 + 0.1 * controls[:, 2] ** 2)
    smooth = base.LAMBDA_S * np.sum((controls - u_prev[None, :]) ** 2, axis=1)
    progress_weight = base.W_RETURN_PROGRESS if return_to_route else base.W_PROGRESS * urgency
    scores = progress_weight * j_progress + base.W_CAM * j_cam + base.W_DETECT * p_detect + w_route * j_route / 1000.0
    if return_to_route:
        scores += base.W_RETURN_HEADING * j_return_heading
        smooth *= 0.2
    scores -= effort + smooth

    if target_idx < len(base.TARGETS) and not return_to_route and d_now < base.SLOWDOWN_RADIUS:
        scores[controls[:, 0] > base.APPROACH_SPEED_LIMIT] = -np.inf
    scores[~safe] = -np.inf

    if not np.isfinite(np.max(scores)):
        return base.fallback_control(state), False
    best = controls[int(np.argmax(scores))].copy()
    if (
        not return_to_route
        and target_idx < len(base.TARGETS)
        and base.TARGET_TYPES[target_idx] == "off_route"
        and d_now > 1500.0
        and abs(best[0]) < 1e-9
    ):
        desired_now = math.atan2(goal_pose[1] - state[1], goal_pose[0] - state[0])
        heading_error_now = base.wrap_to_pi(desired_now - state[2])
        if abs(heading_error_now) > 0.35:
            best[1] = np.clip(0.6 * heading_error_now, -base.OMEGA_MAX, base.OMEGA_MAX)
    return best, True


def update_observation_variant(state, target_idx, observed_time, current_time):
    if target_idx >= len(base.TARGETS):
        return 0.0
    start, end = base.WINDOWS[target_idx]
    if not (start <= current_time <= end):
        return 0.0
    pd = base.detection_prob_to_target(state, base.TARGETS[target_idx], target_idx, current_time)
    observed_time[target_idx] = min(base.REQUIRED_OBSERVATION_TIME, observed_time[target_idx] + base.DT * pd)
    return pd


def simulate_variant(method_name, obs_poses, use_cbf=True, use_rejoin=True, controller_mode="mpc"):
    state = base.START_STATE.copy()
    u_prev = np.zeros(3)
    target_idx = 0
    return_to_route = False
    return_source_idx = None
    rejoin_goal = None
    observed_time = np.zeros(len(base.TARGETS))
    observed = np.zeros(len(base.TARGETS), dtype=bool)
    missed = np.zeros(len(base.TARGETS), dtype=bool)

    states = [state.copy()]
    controls = []
    times = []
    target_hist = []
    phase_hist = []
    obs_frac_hist = [observed_time / base.REQUIRED_OBSERVATION_TIME]
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
    arrival_delay = np.zeros(len(base.TARGETS))
    arrival_recorded = np.zeros(len(base.TARGETS), dtype=bool)
    final_dwell_steps = 0

    for step in range(base.STEPS):
        current_time = base.START_CLOCK + step * base.DT
        if (
            target_idx >= len(base.TARGETS)
            and state[1] >= base.FINAL_Y - base.FINAL_STOP_RADIUS
            and abs(state[3]) < 1e-3
            and final_dwell_steps >= base.FINAL_DWELL_STEPS
        ):
            break

        if target_idx < len(base.TARGETS):
            hold_radius = B2_POSE_HOLD_RADIUS if controller_mode == "b2_waypoint" else base.POSE_HOLD_RADIUS
            pose_dist = np.linalg.norm(state[:2] - obs_poses[target_idx])
            if pose_dist <= hold_radius and not arrival_recorded[target_idx]:
                arrival_delay[target_idx] = current_time - base.WINDOWS[target_idx, 0]
                arrival_recorded[target_idx] = True
            if observed_time[target_idx] >= base.REQUIRED_OBSERVATION_TIME:
                observed[target_idx] = True
                if use_rejoin and base.TARGET_TYPES[target_idx] == "off_route" and not return_to_route:
                    return_to_route = True
                    return_source_idx = target_idx
                    next_target_idx = return_source_idx + 1 if return_source_idx + 1 < len(base.TARGETS) else None
                    rejoin_goal = base.compute_rejoin_point(state[:2], return_source_idx, next_target_idx)
            if current_time > base.WINDOWS[target_idx, 1] and not observed[target_idx]:
                missed[target_idx] = True
                observed[target_idx] = True
                if use_rejoin and base.TARGET_TYPES[target_idx] == "off_route":
                    return_to_route = True
                    return_source_idx = target_idx
                    next_target_idx = return_source_idx + 1 if return_source_idx + 1 < len(base.TARGETS) else None
                    rejoin_goal = base.compute_rejoin_point(state[:2], return_source_idx, next_target_idx)

        return_goal = None
        if return_to_route:
            if rejoin_goal is None and return_source_idx is not None:
                next_target_idx = return_source_idx + 1 if return_source_idx + 1 < len(base.TARGETS) else None
                rejoin_goal = base.compute_rejoin_point(state[:2], return_source_idx, next_target_idx)
            return_goal = rejoin_goal
            if return_goal is not None and np.linalg.norm(state[:2] - return_goal) < base.ROUTE_RETURN_THRESHOLD:
                return_to_route = False
                return_source_idx = None
                rejoin_goal = None
                target_idx += 1
        elif target_idx < len(base.TARGETS) and observed[target_idx] and not return_to_route:
            target_idx += 1

        hold_for_window = False
        if target_idx < len(base.TARGETS) and not observed[target_idx] and not return_to_route:
            hold_radius = B2_POSE_HOLD_RADIUS if controller_mode == "b2_waypoint" else base.POSE_HOLD_RADIUS
            pose_dist_now = np.linalg.norm(state[:2] - obs_poses[target_idx])
            _, _, target_dist_now = base.target_geometry(state, base.TARGETS[target_idx])
            hold_for_window = pose_dist_now <= hold_radius and current_time <= base.WINDOWS[target_idx, 1]
            if controller_mode != "b2_waypoint":
                hold_for_window = hold_for_window and target_dist_now <= base.OBSERVATION_RANGE

        if hold_for_window:
            theta_goal, _, _ = base.target_geometry(state, base.TARGETS[target_idx])
            theta_c_des = base.wrap_to_pi(theta_goal - state[2])
            u_c = np.clip(base.wrap_to_pi(theta_c_des - state[3]) / base.DT, -base.UC_MAX, base.UC_MAX)
            u = np.array([0.0, 0.0, u_c])
            had_safe = True
        elif controller_mode == "b2_waypoint" and target_idx >= len(base.TARGETS):
            u, had_safe = b2_waypoint_control(state, np.array([base.ROUTE_X, base.FINAL_Y]), target_idx, current_time)
            if np.linalg.norm(state[:2] - np.array([base.ROUTE_X, base.FINAL_Y])) <= base.FINAL_STOP_RADIUS:
                u = base.final_transit_control(state)
                had_safe = True
        elif target_idx >= len(base.TARGETS) and use_cbf:
            u, had_safe = score_controls_variant(
                state,
                u_prev,
                target_idx,
                observed,
                False,
                current_time,
                obs_poses,
                use_cbf=True,
                return_goal=None,
            )
            if np.linalg.norm(state[:2] - np.array([base.ROUTE_X, base.FINAL_Y])) <= base.FINAL_STOP_RADIUS:
                u = base.final_transit_control(state)
                had_safe = True
        elif target_idx >= len(base.TARGETS):
            u = base.final_transit_control(state)
            had_safe = True
        elif controller_mode == "b2_waypoint":
            u, had_safe = b2_waypoint_control(state, obs_poses[target_idx], target_idx, current_time)
        else:
            u, had_safe = score_controls_variant(
                state,
                u_prev,
                target_idx,
                observed,
                return_to_route,
                current_time,
                obs_poses,
                use_cbf=use_cbf,
                return_goal=return_goal,
            )
        if np.all(observed):
            u[2] = np.clip(base.wrap_to_pi(-state[3]) / base.DT, -base.UC_MAX, base.UC_MAX)

        state = base.step_dynamics(state, u)
        if target_idx >= len(base.TARGETS) and state[1] >= base.FINAL_Y - base.FINAL_STOP_RADIUS and np.allclose(u[:2], 0.0):
            final_dwell_steps += 1
        else:
            final_dwell_steps = 0
        pd_step = update_observation_variant(state, target_idx, observed_time, current_time)

        if target_idx < len(base.TARGETS):
            _, beta, dist_target = base.target_geometry(state, base.TARGETS[target_idx])
            pose_dist = np.linalg.norm(state[:2] - obs_poses[target_idx])
            pd_val = base.detection_prob_to_target(state, base.TARGETS[target_idx], target_idx, current_time)
        else:
            beta = 0.0
            dist_target = 0.0
            pose_dist = np.linalg.norm(state[:2] - np.array([base.ROUTE_X, base.FINAL_Y]))
            pd_val = 0.0
        h_min, safety_dist = base.nearest_obstacle_data(state)

        states.append(state.copy())
        controls.append(u)
        times.append(current_time)
        target_hist.append(min(target_idx, len(base.TARGETS) - 1))
        phase_hist.append(base.mission_phase(target_idx, observed, return_to_route))
        obs_frac_hist.append(observed_time / base.REQUIRED_OBSERVATION_TIME)
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
        u_prev = u

    metrics = {
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
        "obs_poses": obs_poses,
    }
    return np.array(states), metrics


def simulate_method(method_name):
    if method_name == "Ours":
        states, metrics = base.simulate()
        metrics["obs_poses"] = base.OBS_POSES
        return states, metrics
    if method_name == "B2 Greedy nearest":
        return simulate_variant(
            method_name,
            greedy_nearest_observation_poses(),
            use_cbf=True,
            use_rejoin=False,
            controller_mode="b2_waypoint",
        )
    if method_name == "B3 No-CBF":
        return simulate_variant(method_name, base.OBS_POSES.copy(), use_cbf=False, use_rejoin=True)
    raise ValueError(f"Unknown method {method_name}")


def compute_metrics(states, metrics):
    pd = np.asarray(metrics["pd"], dtype=float)
    h = np.asarray(metrics["h_min"], dtype=float)
    route = np.asarray(metrics["route_error"], dtype=float)
    diffs = np.diff(states[:, :2], axis=0)
    final_time = metrics["times"][-1] if len(metrics["times"]) else base.START_CLOCK
    return {
        "observed_targets": int(np.sum(metrics["observed_time"] >= base.REQUIRED_OBSERVATION_TIME)),
        "missed_targets": int(np.sum(metrics["missed"])),
        "total_pd_integral": float(np.sum(pd) * base.DT),
        "mean_pd": float(np.mean(pd)) if len(pd) else 0.0,
        "path_length": float(np.sum(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0,
        "mean_route_error": float(np.mean(route)) if len(route) else 0.0,
        "max_route_error": float(np.max(route)) if len(route) else 0.0,
        "min_cbf_h": float(np.min(h)) if len(h) else float("nan"),
        "safety_violations": int(np.sum(h < 0.0)) if len(h) else 0,
        "final_time": base.seconds_to_hms(final_time),
    }


def save_comparison_csv(results):
    rows = []
    for name, result in results.items():
        row = {"method": name}
        row.update(result["summary"])
        rows.append(row)
    fields = ["method"] + list(next(iter(results.values()))["summary"].keys())
    with open(PLOT_DIR / "comparison_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_table(results):
    columns = [
        "method",
        "observed",
        "missed",
        "Pd integral",
        "mean Pd",
        "path m",
        "mean |x|",
        "max |x|",
        "min h",
        "violations",
        "final time",
    ]
    rows = []
    for name, result in results.items():
        s = result["summary"]
        rows.append(
            [
                name,
                s["observed_targets"],
                s["missed_targets"],
                f"{s['total_pd_integral']:.1f}",
                f"{s['mean_pd']:.3f}",
                f"{s['path_length']:.0f}",
                f"{s['mean_route_error']:.0f}",
                f"{s['max_route_error']:.0f}",
                f"{s['min_cbf_h']:.0f}",
                s["safety_violations"],
                s["final_time"],
            ]
        )
    fig, ax = plt.subplots(figsize=(13.5, 2.25))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold")
        elif row == 1:
            cell.set_facecolor("#eef6ff")
        elif row == 2:
            cell.set_facecolor("#fff7ed")
        elif row == 3:
            cell.set_facecolor("#f5f3ff")
    ax.set_title("Baseline comparison metrics", pad=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "comparison_metrics.png", dpi=180)
    plt.close(fig)


def draw_common_map(ax):
    ax.set_facecolor("#bfe9ff")
    ax.imshow(
        base.LAMBDA_BAR_MAP,
        extent=[base.X_MIN, base.X_MAX, base.Y_MIN, base.Y_MAX],
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.42,
        zorder=0,
    )
    ax.plot([base.ROUTE_X, base.ROUTE_X], [base.ROUTE_Y_START, base.ROUTE_Y_END], color="#1b6ca8", lw=2.0, ls="--")
    ax.add_patch(
        plt.Rectangle(
            (-base.X_LIMIT, base.Y_LOW),
            2 * base.X_LIMIT,
            base.Y_HIGH - base.Y_LOW,
            fc="none",
            ec="deepskyblue",
            ls="--",
            lw=1.4,
        )
    )
    ax.scatter(
        base.SYNTHETIC_ARRIVAL_SOURCES[:, 0],
        base.SYNTHETIC_ARRIVAL_SOURCES[:, 1],
        marker="x",
        s=55,
        c="black",
        linewidths=1.5,
        zorder=7,
    )
    for obs in base.OBSTACLES:
        ax.add_patch(Circle(obs["center"], obs["radius"], fc="none", ec="red", lw=1.8, zorder=4))
        ax.add_patch(Circle(obs["center"], obs["radius"] + base.SAFETY_MARGIN, fc="none", ec="red", ls="--", lw=1.0, zorder=4))
    for i, target in enumerate(base.TARGETS):
        ax.scatter(target[0], target[1], s=170, marker="*", c="yellow", ec="black", zorder=9)
        ax.text(target[0] + 80, target[1] + 80, f"T{i}", weight="bold", fontsize=9)


def save_comparison_trajectories(results):
    fig, ax = plt.subplots(figsize=(9, 9.5))
    draw_common_map(ax)
    for name, result in results.items():
        poses = result["metrics"]["obs_poses"]
        style = METHODS[name]
        ax.scatter(poses[:, 0], poses[:, 1], s=55, marker="s", facecolors="none", edgecolors=style["color"], lw=1.5, zorder=8)
        states = result["states"]
        ax.plot(states[:, 0], states[:, 1], color=style["color"], ls=style["ls"], lw=style["lw"], label=name, zorder=6)

    handles = [
        mlines.Line2D([], [], color="#1b6ca8", ls="--", lw=2, label="nominal route"),
        mlines.Line2D([], [], color=METHODS["Ours"]["color"], lw=2.4, label="Ours"),
        mlines.Line2D([], [], color=METHODS["B2 Greedy nearest"]["color"], ls="--", lw=2.2, label="B2 Greedy nearest"),
        mlines.Line2D([], [], color=METHODS["B3 No-CBF"]["color"], ls=":", lw=2.7, label="B3 No-CBF"),
        mlines.Line2D([], [], marker="*", color="black", markerfacecolor="yellow", markersize=12, ls="", label="GP-selected target"),
        mlines.Line2D([], [], marker="s", color="black", markerfacecolor="none", markersize=8, ls="", label="observation pose"),
        mlines.Line2D([], [], marker="x", color="black", markersize=8, ls="", label="Synthetic arrival source"),
        mpatches.Circle((0, 0), radius=5, fill=False, ec="red", label="obstacle"),
        mlines.Line2D([], [], color="red", ls="--", lw=1.2, label="safety radius"),
        mlines.Line2D([], [], color="deepskyblue", ls="--", lw=1.2, label="boundary margin"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.92)
    ax.set_title("Trajectory comparison with GP-guided MPC-CBF active sensing")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(base.X_MIN, base.X_MAX)
    ax.set_ylim(base.Y_MIN, base.Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "comparison_trajectories.png", dpi=220)
    plt.close(fig)


def elapsed_minutes(metrics):
    if len(metrics["times"]) == 0:
        return np.array([])
    return (metrics["times"] - base.START_CLOCK) / 60.0


def save_comparison_timeseries(results):
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, result in results.items():
        t = elapsed_minutes(result["metrics"])
        ax.plot(t, result["metrics"]["pd"], color=METHODS[name]["color"], ls=METHODS[name]["ls"], lw=2, label=name)
    ax.set_title("Detection probability over time")
    ax.set_xlabel("time after 10:00 [min]")
    ax.set_ylabel("P_d")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "comparison_detection_probability.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        for name, result in results.items():
            metrics = result["metrics"]
            t = np.arange(len(metrics["coverage"])) * base.DT / 60.0
            ax.plot(t, metrics["coverage"][:, i], color=METHODS[name]["color"], ls=METHODS[name]["ls"], lw=2, label=name)
        ax.axhline(1.0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(f"T{i} observation fraction")
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("time after 10:00 [min]")
    axes[1, 1].set_xlabel("time after 10:00 [min]")
    axes[0, 0].set_ylabel("fraction")
    axes[1, 0].set_ylabel("fraction")
    axes[0, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Observation progress from detection probability")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "comparison_observation_progress.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, result in results.items():
        t = elapsed_minutes(result["metrics"])
        ax.plot(t, result["metrics"]["h_min"], color=METHODS[name]["color"], ls=METHODS[name]["ls"], lw=2, label=name)
    ax.axhline(0.0, color="red", lw=1.0, alpha=0.75)
    ax.set_title("Minimum CBF h over time")
    ax.set_xlabel("time after 10:00 [min]")
    ax.set_ylabel("min h")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "comparison_cbf_h.png", dpi=180)
    plt.close(fig)


def save_comparison_animation(results):
    max_len = max(len(result["states"]) for result in results.values())
    frames = range(0, max_len, COMPARISON_ANIMATION_STRIDE)

    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    draw_common_map(ax)
    ax.set_title("Comparison test: GP-guided MPC-CBF vs baselines")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(base.X_MIN, base.X_MAX)
    ax.set_ylim(base.Y_MIN, base.Y_MAX)
    ax.set_aspect("equal", adjustable="box")

    lines = {}
    points = {}
    fovs = {}
    fov_rays = {}
    for name, result in results.items():
        style = METHODS[name]
        line, = ax.plot([], [], color=style["color"], ls=style["ls"], lw=style["lw"], label=name, zorder=8)
        point, = ax.plot([], [], marker="o", ms=5, color=style["color"], ls="", zorder=9)
        fov = Wedge((0.0, 0.0), base.CAMERA_RANGE, 0.0, 0.0, fc=style["color"], ec=style["color"], alpha=0.12, lw=1.0, zorder=5)
        ray, = ax.plot([], [], color=style["color"], lw=1.2, alpha=0.9, zorder=7)
        ax.add_patch(fov)
        lines[name] = line
        points[name] = point
        fovs[name] = fov
        fov_rays[name] = ray

    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.4", alpha=0.85),
        zorder=10,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    def update(frame_idx):
        status = []
        for name, result in results.items():
            states = result["states"]
            idx = min(frame_idx, len(states) - 1)
            state = states[idx]
            lines[name].set_data(states[: idx + 1, 0], states[: idx + 1, 1])
            points[name].set_data([state[0]], [state[1]])
            theta_view = state[2] + state[3]
            theta_deg = np.rad2deg(theta_view)
            fovs[name].center = (state[0], state[1])
            fovs[name].set_radius(base.CAMERA_RANGE)
            fovs[name].theta1 = theta_deg - base.FOV_DEG / 2.0
            fovs[name].theta2 = theta_deg + base.FOV_DEG / 2.0
            ray_end = state[:2] + base.CAMERA_RANGE * np.array([np.cos(theta_view), np.sin(theta_view)])
            fov_rays[name].set_data([state[0], ray_end[0]], [state[1], ray_end[1]])
            metrics = result["metrics"]
            if len(metrics["target"]):
                midx = min(max(idx - 1, 0), len(metrics["target"]) - 1)
                active = int(metrics["target"][midx])
                obs = int(np.sum(metrics["observed_time"] >= base.REQUIRED_OBSERVATION_TIME))
                missed = int(np.sum(metrics["missed"]))
                status.append(f"{name}: T{active}, obs {obs}, missed {missed}")

        t_seconds = base.START_CLOCK + frame_idx * base.DT
        text.set_text("Time: " + base.seconds_to_hms(t_seconds) + "\n" + "\n".join(status))
        return list(lines.values()) + list(points.values()) + list(fovs.values()) + list(fov_rays.values()) + [text]

    ani = animation.FuncAnimation(fig, update, frames=list(frames), interval=1000 / COMPARISON_ANIMATION_FPS, blit=True)
    writer = animation.FFMpegWriter(fps=COMPARISON_ANIMATION_FPS, bitrate=1800)
    ani.save(COMPARISON_MP4_FILENAME, writer=writer, dpi=150)
    plt.close(fig)


def save_outputs(results):
    save_comparison_csv(results)
    save_comparison_table(results)
    save_comparison_trajectories(results)
    save_comparison_timeseries(results)
    if GENERATE_COMPARISON_MP4:
        save_comparison_animation(results)


def print_summary(results):
    headers = ["method", "observed", "missed", "Pd_int", "path_m", "min_h", "viol", "final_time"]
    rows = []
    for name, result in results.items():
        s = result["summary"]
        rows.append(
            [
                name,
                str(s["observed_targets"]),
                str(s["missed_targets"]),
                f"{s['total_pd_integral']:.1f}",
                f"{s['path_length']:.0f}",
                f"{s['min_cbf_h']:.0f}",
                str(s["safety_violations"]),
                s["final_time"],
            ]
        )
    widths = [max(len(row[i]) for row in rows + [headers]) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def print_b2_details(results):
    metrics = results["B2 Greedy nearest"]["metrics"]
    print("\nB2 arrival delay by target [s]:")
    for i, value in enumerate(metrics["arrival_delay"]):
        print(f"  T{i}: {value:.1f}")
    print("B2 observed_time by target [s]:")
    for i, value in enumerate(metrics["observed_time"]):
        print(f"  T{i}: {value:.1f}")
    print(f"B2 missed flags: {metrics['missed'].tolist()}")


def main():
    results = {}
    for method_name in METHODS:
        print(f"Simulating {method_name}...")
        states, metrics = simulate_method(method_name)
        results[method_name] = {
            "states": states,
            "metrics": metrics,
            "summary": compute_metrics(states, metrics),
        }
    save_outputs(results)
    print_summary(results)
    print_b2_details(results)
    print(f"Saved comparison CSV, PNG, and MP4 outputs in {PLOT_DIR}/")


if __name__ == "__main__":
    main()
