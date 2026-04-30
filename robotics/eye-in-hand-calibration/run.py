"""Eye-in-hand calibration: THE single experiment runner.

Two-scenario design (train/val split is NOT inside one pose-set):

  1. Load MuJoCo scene (sensor pipeline + X_gt locked in scene.xml).
  2. CALIBRATION (agent-controlled): collect `n_poses` at the sampler,
     distance, seed, ... of the agent's choosing. Solve hand-eye here.
  3. VALIDATION (LOCKED): collect a second, independent pose set from a
     fixed benchmark distribution -- larger distance, `diverse` sampler,
     locked seed, locked count. See `VAL_*` constants below.
  4. HEADLINE: std of predicted T_base_target on the VALIDATION poses.
     Because the val distribution is locked, the agent cannot shrink the
     headline by choosing an easy-for-me pose set -- harder val poses
     expose X_est errors the easier calibration set wouldn't.
  5. Also reports rot/trans error vs X_gt (cross-check only).
  6. Writes per-solver CSV + pick-task video for the best solver.

This is the ONLY entry point. The agent edits the knobs in `main()`
(calibration config), `collect_scenario` in collector.py, samplers.py,
and solver logic below -- but NOT `VAL_*`, the sensor pipeline, or
metrics.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from collector import Scenario, collect_scenario, load_scene
from metrics import pose_error_deg_mm
from render_reach_video import render_reach_video


ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
CALIB_POSES_DIR = ARTIFACTS / "calib_poses"
VAL_POSES_DIR = ARTIFACTS / "val_poses"


SOLVERS: Dict[str, int] = {
    "tsai":       cv2.CALIB_HAND_EYE_TSAI,
    "park":       cv2.CALIB_HAND_EYE_PARK,
    "horaud":     cv2.CALIB_HAND_EYE_HORAUD,
    "andreff":    cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# ===========================================================================
# --- LOCKED VALIDATION BENCHMARK -------------------------------------------
# The test distribution. Deliberately harder than typical calibration sets
# (larger distance, fully diverse hemisphere, wider coverage). The agent
# MUST NOT modify these; any change invalidates cross-commit comparisons.
# ===========================================================================

VAL_N_POSES: int             = 30
VAL_DISTANCE_M: float        = 1.0          # larger than typical train distance
VAL_SAMPLER: str             = "diverse"
VAL_SAMPLER_KWARGS: Dict     = {}
VAL_SEED: int                = 20_260_416   # locked so val poses are identical run-to-run


# ---------------------------------------------------------------------------
# Solvers + consistency metric
# ---------------------------------------------------------------------------

def _refine_with_bundle_adjustment(scenario: Scenario, X_init: np.ndarray) -> np.ndarray:
    """Refine X_est using bundle adjustment on reprojection error.
    
    Minimizes the reprojection error of the target corners in the calibration images.
    """
    # For now, return the initial solution (full BA would require more infrastructure)
    return X_init


def solve_handeye(scenario: Scenario, method: int) -> np.ndarray:
    """Run one OpenCV hand-eye solver on a scenario; return X_est as 4x4."""
    R_g2b = scenario.T_base_gripper[:, :3, :3]
    t_g2b = scenario.T_base_gripper[:, :3, 3]
    R_t2c = scenario.T_cam_target[:, :3, :3]
    t_t2c = scenario.T_cam_target[:, :3, 3]
    R_out, t_out = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    X = np.eye(4)
    X[:3, :3] = R_out
    X[:3, 3] = np.asarray(t_out).reshape(3)
    return X


def solve_handeye_with_global_ba(scenario: Scenario, method: int) -> np.ndarray:
    """Run hand-eye solver with global bundle adjustment refinement.
    
    This solver:
    1. Runs the standard hand-eye solver
    2. Refines the solution using bundle adjustment on reprojection error
    """
    # First, get the initial solution from standard hand-eye solver
    X_init = solve_handeye(scenario, method)
    
    # For now, return the initial solution
    # Full BA would require access to camera intrinsics and corner detections
    return X_init


@dataclass
class ConsistencyResult:
    solver: str
    n_train: int
    n_val: int
    trans_std_mm: float            # HEADLINE. sqrt(mean squared centroid dist)
    rot_std_deg: float             # mean geodesic dist from chordal mean
    trans_max_mm: float
    rot_max_deg: float
    trans_err_vs_gt_mm: float      # cross-check against X_gt
    rot_err_vs_gt_deg: float
    X_est: np.ndarray
    T_base_target_preds: np.ndarray


def _chordal_mean_rotation(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Rs.sum(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _rotation_geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def run_consistency(
    calib: Scenario,
    val: Scenario,
    solver_name: str,
    solver_method: int,
) -> ConsistencyResult:
    """Solve hand-eye on `calib`, measure T_base_target consistency on `val`.

    `calib` and `val` are independent pose sets; `val` is the locked
    benchmark defined by `VAL_*` constants.
    """
    X_est = solve_handeye(calib, solver_method)

    T_bg = val.T_base_gripper
    T_ct = val.T_cam_target
    preds = np.einsum("nij,jk,nkl->nil", T_bg, X_est, T_ct)

    centers = preds[:, :3, 3]
    centroid = centers.mean(axis=0)
    trans_sq = ((centers - centroid) ** 2).sum(axis=1)
    trans_std_mm = float(np.sqrt(trans_sq.mean()) * 1000.0)
    trans_max_mm = float(np.sqrt(trans_sq.max()) * 1000.0)

    R_mean = _chordal_mean_rotation(preds[:, :3, :3])
    rot_errs = np.array([_rotation_geodesic_deg(R_mean, R) for R in preds[:, :3, :3]])
    rot_std_deg = float(np.sqrt((rot_errs ** 2).mean()))
    rot_max_deg = float(rot_errs.max())

    rot_vs_gt, trans_vs_gt = pose_error_deg_mm(X_est, calib.X_gt)

    return ConsistencyResult(
        solver=solver_name,
        n_train=calib.T_base_gripper.shape[0],
        n_val=val.T_base_gripper.shape[0],
        trans_std_mm=trans_std_mm, rot_std_deg=rot_std_deg,
        trans_max_mm=trans_max_mm, rot_max_deg=rot_max_deg,
        trans_err_vs_gt_mm=trans_vs_gt, rot_err_vs_gt_deg=rot_vs_gt,
        X_est=X_est, T_base_target_preds=preds,
    )


def run_all_solvers(calib: Scenario, val: Scenario) -> List[ConsistencyResult]:
    out: List[ConsistencyResult] = []
    n_c = calib.T_base_gripper.shape[0]
    n_v = val.T_base_gripper.shape[0]
    for name, method in SOLVERS.items():
        try:
            out.append(run_consistency(calib, val, name, method))
        except cv2.error:
            out.append(ConsistencyResult(
                solver=name, n_train=n_c, n_val=n_v,
                trans_std_mm=float("inf"), rot_std_deg=float("inf"),
                trans_max_mm=float("inf"), rot_max_deg=float("inf"),
                trans_err_vs_gt_mm=float("inf"), rot_err_vs_gt_deg=float("inf"),
                X_est=np.eye(4), T_base_target_preds=np.empty((0, 4, 4)),
            ))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(results: List[ConsistencyResult]) -> None:
    print()
    print(f"  {'solver':<12} {'trans_std_mm':>14} {'rot_std_deg':>13} "
          f"{'trans_max_mm':>14} {'rot_max_deg':>13} "
          f"{'vs_gt_mm':>10} {'vs_gt_deg':>10}")
    for r in results:
        print(f"  {r.solver:<12} "
              f"{r.trans_std_mm:>14.3f} {r.rot_std_deg:>13.4f} "
              f"{r.trans_max_mm:>14.3f} {r.rot_max_deg:>13.4f} "
              f"{r.trans_err_vs_gt_mm:>10.3f} {r.rot_err_vs_gt_deg:>10.4f}")


def _write_csv(
    results: List[ConsistencyResult],
    calib_sampler: str, calib_distance: float, calib_seed: int,
) -> None:
    with open(ARTIFACTS / "consistency_latest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "solver", "n_train", "n_val",
            "trans_std_mm", "rot_std_deg", "trans_max_mm", "rot_max_deg",
            "trans_err_vs_gt_mm", "rot_err_vs_gt_deg",
            "calib_sampler", "calib_distance", "calib_seed",
            "val_sampler", "val_distance", "val_seed", "val_n_poses",
        ])
        for r in results:
            w.writerow([
                r.solver, r.n_train, r.n_val,
                f"{r.trans_std_mm:.6f}", f"{r.rot_std_deg:.6f}",
                f"{r.trans_max_mm:.6f}", f"{r.rot_max_deg:.6f}",
                f"{r.trans_err_vs_gt_mm:.6f}", f"{r.rot_err_vs_gt_deg:.6f}",
                calib_sampler, calib_distance, calib_seed,
                VAL_SAMPLER, VAL_DISTANCE_M, VAL_SEED, VAL_N_POSES,
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    n_poses: int = 20,
    distance: float = 0.6,
    sampler: str = "diverse",
    sampler_kwargs: Dict | None = None,
    seed: int = 42,
    width: int = 640,
    height: int = 480,
) -> None:
    """Run one experiment.

    Calibration (train) config is agent-controlled via these arguments.
    Validation config is locked -- see `VAL_*` at the top of this file.
    """
    sampler_kwargs = sampler_kwargs or {}
    t0 = time.time()

    print("=" * 78)
    print("L2 eye-in-hand calibration: calib (agent) vs val (locked)")
    print("=" * 78)
    print(f"CALIB: n_poses={n_poses}  sampler={sampler}  distance={distance}  seed={seed}")
    print(f"VAL  : n_poses={VAL_N_POSES}  sampler={VAL_SAMPLER}  "
          f"distance={VAL_DISTANCE_M}  seed={VAL_SEED}  (LOCKED)")

    scene = load_scene(width=width, height=height)
    print(f"scene: wrist_fovy={scene.model.cam_fovy[scene.cam_id]:.1f} deg  "
          f"image={width}x{height}  target_pos={scene.T_base_target[:3, 3]}")

    # ---- Calibration poses (agent-controlled) ----
    calib, calib_report = collect_scenario(
        n_poses=n_poses, distance=distance,
        sampler=sampler, sampler_kwargs=sampler_kwargs,
        seed=seed, width=width, height=height, scene=scene,
        save_images_to=CALIB_POSES_DIR, max_retries_per_pose=8,
        use_depth_correction=True,
    )
    n_c = calib.T_base_gripper.shape[0]
    if n_c < 3:
        print(f"ERROR: calib collected only {n_c}/{n_poses} poses "
              f"(tune sampler/distance).")
        sys.exit(2)
    print(f"calib: {n_c}/{n_poses} poses  "
          f"(det_rate={calib_report['detection_rate']:.2f}, "
          f"mean_reproj={calib_report['mean_reproj_err_px']:.3f} px)")

    # ---- Validation poses (LOCKED benchmark -- DO NOT MODIFY) ----
    val, val_report = collect_scenario(
        n_poses=VAL_N_POSES, distance=VAL_DISTANCE_M,
        sampler=VAL_SAMPLER, sampler_kwargs=VAL_SAMPLER_KWARGS,
        seed=VAL_SEED, width=width, height=height, scene=scene,
        save_images_to=VAL_POSES_DIR, max_retries_per_pose=8,
        use_depth_correction=True,
    )
    n_v = val.T_base_gripper.shape[0]
    if n_v < 6:
        print(f"ERROR: val collected only {n_v}/{VAL_N_POSES} poses. "
              f"Something is wrong with the locked benchmark.")
        sys.exit(2)
    print(f"val  : {n_v}/{VAL_N_POSES} poses  "
          f"(det_rate={val_report['detection_rate']:.2f}, "
          f"mean_reproj={val_report['mean_reproj_err_px']:.3f} px)")
    print(f"collection time: {time.time()-t0:.1f}s")

    results = run_all_solvers(calib, val)
    _print_table(results)

    finite = [r for r in results if np.isfinite(r.trans_std_mm)]
    if not finite:
        print("ERROR: every solver failed.")
        sys.exit(3)
    best = min(finite, key=lambda r: r.trans_std_mm)

    _write_csv(results, sampler, distance, seed)

    video_path = None
    try:
        video_path = render_reach_video(
            scene=scene, X_est=best.X_est, solver_name=best.solver,
            trans_std_mm=best.trans_std_mm, rot_std_deg=best.rot_std_deg,
            trans_err_vs_gt_mm=best.trans_err_vs_gt_mm,
            rot_err_vs_gt_deg=best.rot_err_vs_gt_deg,
        )
    except Exception as e:
        print(f"WARN: reach video failed: {e}")

    elapsed = time.time() - t0
    print()
    print("=" * 78)
    print("HEADLINE (L2 fixed-target consistency)")
    print(f"  best_solver:            {best.solver}")
    print(f"  trans_std_mm:           {best.trans_std_mm:.4f}     # PRIMARY, lower = better")
    print(f"  rot_std_deg:            {best.rot_std_deg:.4f}")
    print(f"  trans_max_mm:           {best.trans_max_mm:.4f}")
    print(f"  rot_max_deg:            {best.rot_max_deg:.4f}")
    print(f"  trans_err_vs_gt_mm:     {best.trans_err_vs_gt_mm:.4f}    # cross-check")
    print(f"  rot_err_vs_gt_deg:      {best.rot_err_vs_gt_deg:.4f}")
    print(f"  n_train / n_val:        {best.n_train} / {best.n_val}")
    print(f"  total_seconds:          {elapsed:.1f}")
    if video_path:
        print(f"  pick-task video:        {video_path}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eye-in-hand calibration. Calibration config is agent-controlled; "
                    "validation is locked (see VAL_* constants at the top of this file).",
    )
    parser.add_argument("--n_poses", type=int, default=20,
                        help="Number of CALIBRATION poses (val is locked at VAL_N_POSES)")
    parser.add_argument("--distance", type=float, default=0.6,
                        help="Calibration distance from target")
    parser.add_argument("--sampler", type=str, default="diverse",
                        help="Calibration sampler strategy")
    parser.add_argument("--seed", type=int, default=42,
                        help="Calibration seed (val seed is locked at VAL_SEED)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    try:
        main(
            n_poses=args.n_poses,
            distance=args.distance,
            sampler=args.sampler,
            seed=args.seed,
            width=args.width,
            height=args.height,
        )
    except KeyboardInterrupt:
        sys.exit(130)
