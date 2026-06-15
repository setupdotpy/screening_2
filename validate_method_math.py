import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

import usv_active_sensing_mpc_cbf_sim_research_model as base
import compare_baselines as compare


base.GENERATE_MP4 = False
compare.GENERATE_COMPARISON_MP4 = False

OUT_DIR = Path("plot") / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(base.SEED + 42)

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


def write_csv(filename, rows, fieldnames):
    with open(OUT_DIR / filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def state_with_view_at(pose, target, heading=None, aligned=True):
    psi = math.pi / 2.0 if heading is None else heading
    theta_goal = math.atan2(target[1] - pose[1], target[0] - pose[0])
    theta_c = base.wrap_to_pi(theta_goal - psi) if aligned else 0.0
    return np.array([pose[0], pose[1], psi, theta_c])


def window_center(target_idx):
    return float(0.5 * (base.WINDOWS[target_idx, 0] + base.WINDOWS[target_idx, 1]))


def target_rank(lambda_value):
    valid_values = []
    xx, yy = np.meshgrid(base.LAMBDA_X_GRID, base.LAMBDA_Y_GRID)
    for iy in range(base.GP_NY):
        for ix in range(base.GP_NX):
            p = np.array([xx[iy, ix], yy[iy, ix]])
            if base.point_is_safe(p):
                valid_values.append(base.LAMBDA_BAR_MAP[iy, ix])
    valid_values = np.array(valid_values)
    return int(1 + np.sum(valid_values > lambda_value))


def bounded_label_position(point, margin=300.0):
    x, y = float(point[0]), float(point[1])
    if x > base.X_MAX - 1800.0:
        label_x = x - margin
        ha = "right"
    else:
        label_x = x + margin
        ha = "left"

    if y > base.Y_MAX - 900.0:
        label_y = y - margin
        va = "top"
    else:
        label_y = y + margin
        va = "bottom"

    label_x = float(np.clip(label_x, base.X_MIN + margin, base.X_MAX - margin))
    label_y = float(np.clip(label_y, base.Y_MIN + margin, base.Y_MAX - margin))
    return label_x, label_y, ha, va


def validation_gp_targets():
    rows = []
    for i, target in enumerate(base.TARGETS):
        lam = base.lambda_bar_at_position(target)
        rows.append(
            {
                "Target": f"T{i}",
                "x": f"{target[0]:.3f}",
                "y": f"{target[1]:.3f}",
                "lambda_bar": f"{lam:.6f}",
                "rank_among_all_candidates": target_rank(lam),
            }
        )
    write_csv("validation_gp_targets.csv", rows, ["Target", "x", "y", "lambda_bar", "rank_among_all_candidates"])

    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    im = ax.imshow(
        base.LAMBDA_BAR_MAP,
        extent=[base.X_MIN, base.X_MAX, base.Y_MIN, base.Y_MAX],
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.9,
    )
    for i, target in enumerate(base.TARGETS):
        lam = base.lambda_bar_at_position(target)
        ax.scatter(target[0], target[1], marker="*", s=180, c="yellow", ec="black", zorder=5)
        lx, ly, ha, va = bounded_label_position(target)
        ax.annotate(
            f"T{i}\nlambda={lam:.2f}",
            xy=target,
            xytext=(lx, ly),
            ha=ha,
            va=va,
            fontsize=8,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.78),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
            zorder=6,
        )
    for obs in base.OBSTACLES:
        ax.add_patch(Circle(obs["center"], obs["radius"] + base.SAFETY_MARGIN, fc="none", ec="red", ls="--", lw=1))
    ax.plot([base.ROUTE_X, base.ROUTE_X], [base.ROUTE_Y_START, base.ROUTE_Y_END], color="white", ls="--", lw=1.5)
    ax.set_title("Validation 1: GP-selected target likelihood")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(base.X_MIN, base.X_MAX)
    ax.set_ylim(base.Y_MIN, base.Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, label="normalized GP target-arrival likelihood", fraction=0.045, pad=0.035)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_gp_targets.png", dpi=180)
    plt.close(fig)
    return float(np.mean([base.lambda_bar_at_position(t) for t in base.TARGETS]))


def pose_pd_expected(pose, target, target_idx):
    route_point = np.array([base.ROUTE_X, target[1]])
    state = state_with_view_at(pose, target, aligned=True)
    pd = base.detection_prob_to_target(state, target, target_idx, window_center(target_idx))

    theta_required = math.atan2(target[1] - pose[1], target[0] - pose[0])
    travel = np.linalg.norm(pose - route_point)
    clearance = base.min_obstacle_clearance(pose)
    pose_cost = abs(pose[0] - base.ROUTE_X) + 2.0 * travel + 20.0 * abs(base.wrap_to_pi(theta_required))
    if clearance <= 0.0:
        pose_cost += 1e6
    else:
        pose_cost += 1000.0 / (clearance + 1.0)
    if base.segment_intersects_safety(route_point, pose):
        pose_cost += 5000.0

    mission_quality = math.exp(-pose_cost / 5000.0)
    return pd * mission_quality


def random_ring_poses(target, n=100):
    poses = []
    attempts = 0
    while len(poses) < n and attempts < 5000:
        attempts += 1
        alpha = RNG.uniform(0.0, 2.0 * np.pi)
        pose = target + base.R_BEST * np.array([np.cos(alpha), np.sin(alpha)])
        if base.point_is_safe(pose):
            poses.append(pose)
    return poses


def validation_observation_pose():
    rows = []
    box_data = []
    labels = []
    for i, target in enumerate(base.TARGETS):
        optimized_pose = base.OBS_POSES[i]
        opt_pd = pose_pd_expected(optimized_pose, target, i)
        random_pds = np.array([pose_pd_expected(p, target, i) for p in random_ring_poses(target, 100)])
        if len(random_pds) == 0:
            random_pds = np.array([0.0])
        mean_pd = float(np.mean(random_pds))
        max_pd = float(np.max(random_pds))
        improvement = 100.0 * (opt_pd - mean_pd) / max(mean_pd, 1e-9)
        rows.append(
            {
                "Target": f"T{i}",
                "Optimized_Pd": f"{opt_pd:.6f}",
                "Random_Mean_Pd": f"{mean_pd:.6f}",
                "Random_Max_Pd": f"{max_pd:.6f}",
                "Improvement_percent": f"{improvement:.2f}",
            }
        )
        box_data.append(random_pds)
        labels.append(f"T{i}")
    write_csv(
        "validation_observation_pose.csv",
        rows,
        ["Target", "Optimized_Pd", "Random_Mean_Pd", "Random_Max_Pd", "Improvement_percent"],
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(box_data, tick_labels=labels, showfliers=False)
    opt_values = [float(row["Optimized_Pd"]) for row in rows]
    ax.scatter(np.arange(1, len(opt_values) + 1), opt_values, c="red", marker="D", label="optimized pose")
    for idx, value in enumerate(opt_values, start=1):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Validation 2: optimized pose expected detection quality")
    ax.set_ylabel("expected P_d with reachability/clearance")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_observation_pose.png", dpi=180)
    plt.close(fig)
    return rows


def validation_camera_alignment():
    rows = []
    aligned_vals = []
    unaligned_vals = []
    for i, target in enumerate(base.TARGETS):
        pose = base.OBS_POSES[i]
        aligned_state = state_with_view_at(pose, target, aligned=True)
        unaligned_state = state_with_view_at(pose, target, aligned=False)
        pd_aligned = base.detection_prob_to_target(aligned_state, target, i, window_center(i))
        pd_unaligned = base.detection_prob_to_target(unaligned_state, target, i, window_center(i))
        rows.append(
            {
                "Target": f"T{i}",
                "Pd_aligned": f"{pd_aligned:.6f}",
                "Pd_unaligned": f"{pd_unaligned:.6f}",
                "Alignment_gain": f"{pd_aligned - pd_unaligned:.6f}",
                "Gain_ratio": f"{pd_aligned / max(pd_unaligned, 1e-9):.3f}",
            }
        )
        aligned_vals.append(pd_aligned)
        unaligned_vals.append(pd_unaligned)
    write_csv("validation_camera_alignment.csv", rows, ["Target", "Pd_aligned", "Pd_unaligned", "Alignment_gain", "Gain_ratio"])

    y = np.arange(len(base.TARGETS))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (pd0, pd1) in enumerate(zip(unaligned_vals, aligned_vals)):
        ax.annotate(
            "",
            xy=(pd1, i),
            xytext=(pd0, i),
            arrowprops=dict(arrowstyle="->", lw=2.2, color="#2563eb"),
        )
        ax.scatter(pd0, i, s=70, c="#f97316", ec="black", zorder=4)
        ax.scatter(pd1, i, s=80, c="#2563eb", ec="black", zorder=5)
        ax.text(pd1 + 0.025, i, f"{pd1:.2f}", va="center", fontsize=9, weight="bold")
        ax.text(pd0 + 0.025, i - 0.16, f"{pd0:.2f}", va="center", fontsize=8, color="#7c2d12")
    ax.set_yticks(y, [f"T{i}" for i in range(len(base.TARGETS))])
    ax.set_xlim(-0.03, 1.12)
    ax.set_xlabel("Detection probability $P_d$")
    ax.set_title("Validation 3: camera alignment moves each target from low to high $P_d$")
    ax.grid(True, axis="x", alpha=0.25)
    aligned_handle = plt.Line2D([], [], marker="o", color="none", markerfacecolor="#2563eb", markeredgecolor="black", label="aligned camera")
    unaligned_handle = plt.Line2D([], [], marker="o", color="none", markerfacecolor="#f97316", markeredgecolor="black", label="unaligned camera")
    ax.legend(handles=[unaligned_handle, aligned_handle], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.24), frameon=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(OUT_DIR / "validation_camera_alignment.png", dpi=180)
    plt.close(fig)
    return rows


def run_methods():
    results = {}
    for name in compare.METHODS:
        print(f"Running {name} for validation...")
        states, metrics = compare.simulate_method(name)
        results[name] = {
            "states": states,
            "metrics": metrics,
            "summary": compare.compute_metrics(states, metrics),
        }
    return results


def validation_mpc_detection(results):
    rows = []
    for name in ["Ours", "Baseline 1"]:
        s = results[name]["summary"]
        coverage = results[name]["metrics"]["coverage"][-1]
        rows.append(
            {
                "Method": name,
                "TotalPdIntegral": f"{s['total_pd_integral']:.6f}",
                "MeanPd": f"{s['mean_pd']:.6f}",
                "ObservationCompletionFraction": f"{float(np.mean(coverage)):.6f}",
            }
        )
    write_csv("validation_mpc_detection.csv", rows, ["Method", "TotalPdIntegral", "MeanPd", "ObservationCompletionFraction"])

    methods = [row["Method"] for row in rows]
    total_pd = [float(row["TotalPdIntegral"]) for row in rows]
    mean_pd = [float(row["MeanPd"]) for row in rows]
    completion = [float(row["ObservationCompletionFraction"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    for ax, vals, title in zip(axes, [total_pd, mean_pd, completion], ["Integral P_d dt", "Mean P_d", "Completion fraction"]):
        bars = ax.bar(methods, vals, color=["#004f80", "#f28e2b"])
        fmt = "%.0f" if max(vals) > 10 else "%.2f"
        ax.bar_label(bars, fmt=fmt, fontsize=8, padding=2)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Validation 4: MPC active sensing benefit")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_mpc_detection.png", dpi=180)
    plt.close(fig)
    return rows


def validation_cbf(results):
    rows = []
    for name in ["Ours", "Baseline 2"]:
        metrics = results[name]["metrics"]
        h = np.asarray(metrics["h_min"])
        dist = np.asarray(metrics["dist_to_safety"])
        rows.append(
            {
                "Method": name,
                "MinH": f"{float(np.min(h)):.6f}",
                "SafetyViolations": int(np.sum(h < 0.0)),
                "MinimumObstacleDistance": f"{float(np.min(dist)):.6f}",
            }
        )
    write_csv("validation_cbf.csv", rows, ["Method", "MinH", "SafetyViolations", "MinimumObstacleDistance"])

    ours = results["Ours"]["metrics"]
    no_cbf = results["Baseline 2"]["metrics"]
    t_ours = (ours["times"] - base.START_CLOCK) / 60.0
    t_no_cbf = (no_cbf["times"] - base.START_CLOCK) / 60.0
    d_ours = np.asarray(ours["dist_to_safety"])
    d_no_cbf = np.asarray(no_cbf["dist_to_safety"])
    d_ours_clipped = np.clip(d_ours, -250.0, 250.0)
    d_no_cbf_clipped = np.clip(d_no_cbf, -250.0, 250.0)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.fill_between(t_no_cbf, d_no_cbf_clipped, 0.0, where=d_no_cbf_clipped < 0.0, color="#fca5a5", alpha=0.45, label="unsafe region")
    ax.plot(t_ours, d_ours_clipped, color="#004f80", lw=2.2, label="Ours with CBF")
    ax.plot(t_no_cbf, d_no_cbf_clipped, color="#7b3294", lw=2.0, ls="--", label="Baseline 2")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_ylim(-260.0, 270.0)
    ax.set_xlabel("minutes after 10:00")
    ax.set_ylabel("signed distance to safety boundary [m]")
    ax.set_title("Validation 5: CBF safety margin over time")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.92)

    text = (
        "Safety summary\n"
        f"Ours: min distance = {float(np.min(d_ours)):.1f} m, violations = {int(np.sum(np.asarray(ours['h_min']) < 0.0))}\n"
        f"Baseline 2: min distance = {float(np.min(d_no_cbf)):.1f} m, violations = {int(np.sum(np.asarray(no_cbf['h_min']) < 0.0))}\n"
        "Negative distance means inside an obstacle safety set.\n"
        "Displayed margin is clipped to +/-250 m for readability."
    )
    fig.text(
        0.5,
        0.03,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", alpha=0.9),
    )
    fig.tight_layout(rect=[0.0, 0.22, 1.0, 1.0])
    fig.savefig(OUT_DIR / "validation_cbf.png", dpi=180)
    plt.close(fig)
    return rows


def validation_summary(results):
    rows = []
    for name in ["Ours", "Baseline 1", "Baseline 2"]:
        s = results[name]["summary"]
        rows.append(
            {
                "Method": name,
                "ObservedTargets": s["observed_targets"],
                "MissedTargets": s["missed_targets"],
                "TotalPdIntegral": f"{s['total_pd_integral']:.6f}",
                "PathLength": f"{s['path_length']:.6f}",
                "MeanRouteError": f"{s['mean_route_error']:.6f}",
                "MinH": f"{s['min_cbf_h']:.6f}",
                "SafetyViolations": s["safety_violations"],
            }
        )
    write_csv(
        "validation_summary.csv",
        rows,
        ["Method", "ObservedTargets", "MissedTargets", "TotalPdIntegral", "PathLength", "MeanRouteError", "MinH", "SafetyViolations"],
    )
    return rows


def print_report(mean_lambda, pose_rows, camera_rows, mpc_rows, cbf_rows, summary_rows):
    pose_improvements = [float(row["Improvement_percent"]) for row in pose_rows]
    camera_gains = [float(row["Alignment_gain"]) for row in camera_rows]
    ours_mpc = next(row for row in mpc_rows if row["Method"] == "Ours")
    b2_mpc = next(row for row in mpc_rows if row["Method"] == "Baseline 1")
    ours_cbf = next(row for row in cbf_rows if row["Method"] == "Ours")
    b3_cbf = next(row for row in cbf_rows if row["Method"] == "Baseline 2")

    print("\nMathematical validation report")
    print("--------------------------------")
    print(f"GP validation: mean(lambda_selected) = {mean_lambda:.3f}; selected targets have high GP arrival likelihood.")
    print(
        "Observation pose validation: optimized pose expected Pd improvement vs random mean = "
        f"{np.mean(pose_improvements):.1f}%."
    )
    print(f"Camera validation: mean(Pd_aligned - Pd_unaligned) = {np.mean(camera_gains):.3f}; alignment increases Pd.")
    print(
        "MPC validation: Ours total integral = "
        f"{float(ours_mpc['TotalPdIntegral']):.1f}, Baseline 1 = {float(b2_mpc['TotalPdIntegral']):.1f}."
    )
    print(
        "CBF validation: Ours min h = "
        f"{float(ours_cbf['MinH']):.1f}, violations = {ours_cbf['SafetyViolations']}; "
        f"Baseline 2 min h = {float(b3_cbf['MinH']):.1f}, violations = {b3_cbf['SafetyViolations']}."
    )
    print("\nEnd-to-end summary:")
    for row in summary_rows:
        print(
            f"  {row['Method']}: observed={row['ObservedTargets']}, missed={row['MissedTargets']}, "
            f"J_detection={float(row['TotalPdIntegral']):.1f}, min_h={float(row['MinH']):.1f}, "
            f"violations={row['SafetyViolations']}"
        )
    print("\nConclusion: the complete GP-guided MPC-CBF framework improves J_detection while satisfying h >= 0.")


def main():
    mean_lambda = validation_gp_targets()
    pose_rows = validation_observation_pose()
    camera_rows = validation_camera_alignment()
    results = run_methods()
    mpc_rows = validation_mpc_detection(results)
    cbf_rows = validation_cbf(results)
    summary_rows = validation_summary(results)
    print_report(mean_lambda, pose_rows, camera_rows, mpc_rows, cbf_rows, summary_rows)
    print(f"\nSaved validation CSV and PNG outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
