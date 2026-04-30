"""Automatic PICK-TASK video render.

Replaces the old "reach 30 cm in front of the marker" animation with a test
that makes the calibration error immediately visible and practically meaningful:

  A 2 cm cube is placed at a known offset from the ArUco marker. The agent
  must grasp the cube using X_est. Any hand-eye calibration error propagates
  directly into the grasp, and the gripper lands somewhere other than the
  real cube. The video shows the gripper landing on the CUBE (HIT), grazing
  it (PARTIAL), or missing it (MISS).

Pipeline (matches what a real robot would do):

  1. Gripper starts at some visible pose, T_start.
  2. Wrist camera detects the marker (analytical, assumed perfect in sim).
     This gives T_cam_target = inv(T_start @ X_gt) @ T_base_target_true.
  3. Agent estimates the marker in base frame USING X_est:
        T_base_target_est = T_start @ X_est @ T_cam_target
     When X_est = X_gt this equals T_base_target_true. Otherwise there's a
     conjugated hand-eye error.
  4. The cube's pose relative to the marker is known by design:
        T_base_cube_est = T_base_target_est @ T_target_cube_local
  5. The commanded grasp pose is: gripper origin 11 cm above the estimated
     cube centre, identity rotation, so the two prongs (at +-3 cm lateral,
     extending z from -7.5 to -14.5 cm in gripper frame) straddle the 2 cm
     cube cleanly when the estimate is right.
  6. Truth: real cube is at T_base_cube_true = T_base_target_true @
     T_target_cube_local. Prong midpoint after the move = T_base_cube_est.
     Miss = |T_base_cube_true - T_base_cube_est|.

Video output: ONE iso panel (no misleading side-by-side, no occluding inset),
with the real cube rendered in solid green and the agent's estimate rendered
as a translucent red ghost. You see the gripper fly to the ghost. If the
calibration is good, the ghost is on top of the green cube, so the robot
grasps successfully. If the calibration is bad, the ghost is offset, the
gripper follows the ghost, and closes on empty air.

Only dependency beyond the existing stack is `cv2.VideoWriter` with `mp4v`,
which ships with opencv-python.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mujoco
import numpy as np

from collector import SceneInfo, _inv, _make_T, _quat_wxyz_from_mat


HERE = Path(__file__).parent
VIDEO_DIR = HERE / "video"

# Physical constants for the pick task.
CUBE_SIDE_M = 0.025                 # matches target_cube_geom size in scene.xml (full side)
PRONG_GAP_LATERAL_M = 0.060         # distance between prong centres in x
GRIPPER_TO_GRASP_CENTRE_Z_M = 0.110 # prong midpoint at gripper + (0, 0, -0.11)
HIT_MM = 5.0                        # <= this = HIT (cube cleanly between prongs)
PARTIAL_MM = 15.0                   # <= this = PARTIAL; else MISS
MARKER_SIDE_M = 0.10                # not used here but documented for reference


# --------------------------------------------------------------------------- #
# Commit hashing for video filenames                                          #
# --------------------------------------------------------------------------- #
def _short_commit() -> str:
    """Short commit hash from the eye_in_hand subrepo, `dirty` suffix if uncommitted."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short=9", "HEAD"],
            cwd=HERE, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=HERE, stderr=subprocess.DEVNULL,
        )
        if dirty != 0:
            h = f"{h}-dirty"
        return h
    except Exception:
        return "nogit"


# --------------------------------------------------------------------------- #
# Small pose-math helpers                                                     #
# --------------------------------------------------------------------------- #
def _cam_pose_lookat(cam_pos: np.ndarray, target_pos: np.ndarray,
                     world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """OpenCV-convention camera pose (+z toward target, +y world-down) at cam_pos."""
    forward = target_pos - cam_pos
    forward /= np.linalg.norm(forward) + 1e-12
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([1.0, 0.0, 0.0])
    down = -world_up
    right = np.cross(down, forward)
    right /= np.linalg.norm(right) + 1e-12
    down_adj = np.cross(forward, right)
    down_adj /= np.linalg.norm(down_adj) + 1e-12
    R_cam_cv = np.column_stack([right, down_adj, forward])
    return _make_T(R_cam_cv, cam_pos)


def _quat_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def _mat_from_quat(q: np.ndarray) -> np.ndarray:
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta_0 = np.arccos(max(min(d, 1.0), -1.0))
    sin0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.cos(theta) - d * np.sin(theta) / sin0
    s1 = np.sin(theta) / sin0
    return s0 * q0 + s1 * q1


def _interp_pose(T0: np.ndarray, T1: np.ndarray, t: float) -> np.ndarray:
    q0 = _quat_from_mat(T0[:3, :3])
    q1 = _quat_from_mat(T1[:3, :3])
    q = _slerp_quat(q0, q1, t)
    R = _mat_from_quat(q)
    p = (1 - t) * T0[:3, 3] + t * T1[:3, 3]
    return _make_T(R, p)


def _ease(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * t)


# --------------------------------------------------------------------------- #
# Pick-task planner                                                           #
# --------------------------------------------------------------------------- #
def sample_cube_local_offset(rng: np.random.Generator) -> np.ndarray:
    """Random cube offset in the marker's LOCAL frame.

    Sampled so the cube is always IN FRONT of the marker (local +z side, which
    is the side the wrist camera sees), at a plausible distance for a grasp:

      local x  -- world "up/down" (marker's local +x is world -z).
                  U(-0.08, +0.08) m keeps the cube near marker height.
      local y  -- world "side-to-side" (marker +y == world +y).
                  U(-0.18, +0.18) m.
      local z  -- world "in front of / behind marker" (marker +z == world +x).
                  U( 0.06, +0.22) m, strictly positive so the cube is always
                  rendered in front of the marker (not occluded by it).

    Each run therefore exercises hand-eye error in a different direction,
    without ever hiding the cube behind the marker face.
    """
    return np.array([
        float(rng.uniform(-0.08, 0.08)),
        float(rng.uniform(-0.18, 0.18)),
        float(rng.uniform( 0.06, 0.22)),
    ])


def plan_pick(
    scene: SceneInfo,
    X_est: np.ndarray,
    T_start_gripper: np.ndarray,
    cube_local_offset: np.ndarray,
) -> dict:
    """Compute all the poses needed to render the pick-task animation.

    Returns a dict with:
        T_base_cube_true      -- the real cube pose (ground truth, identity rot)
        T_base_cube_est       -- where the agent *thinks* the cube is
        T_base_cube_truth_at_start  -- equals T_base_cube_true (for clarity)
        T_grasp_est           -- commanded gripper pose at the end of the move
        T_grasp_truth         -- what the grasp pose would be if X_est == X_gt
        T_start_gripper       -- echoed for convenience
        T_start_cam_true      -- where the wrist cam actually is at start
        T_cam_target_obs      -- the "detection" (inv(T_start_cam_true) @ T_target)
    """
    T_base_target_true = scene.T_base_target
    # Cube pose in marker local frame (identity rotation, just offset).
    T_target_cube_local = _make_T(np.eye(3), cube_local_offset)
    T_base_cube_true = T_base_target_true @ T_target_cube_local

    # Step 2: simulate a perfect detection from the start pose.
    T_start_cam_true = T_start_gripper @ scene.X_gt
    T_cam_target_obs = _inv(T_start_cam_true) @ T_base_target_true

    # Step 3-4: the agent's computation using X_est.
    T_base_target_est = T_start_gripper @ X_est @ T_cam_target_obs
    T_base_cube_est = T_base_target_est @ T_target_cube_local

    # Step 5: commanded grasp pose. Keep gripper upright (identity rot),
    # translate so the prong midpoint lands on the estimated cube centre.
    grasp_pos_est = T_base_cube_est[:3, 3] + np.array(
        [0.0, 0.0, GRIPPER_TO_GRASP_CENTRE_Z_M])
    T_grasp_est = _make_T(np.eye(3), grasp_pos_est)

    # Same computation with X_gt -- what the grasp WOULD be under perfect calib.
    grasp_pos_truth = T_base_cube_true[:3, 3] + np.array(
        [0.0, 0.0, GRIPPER_TO_GRASP_CENTRE_Z_M])
    T_grasp_truth = _make_T(np.eye(3), grasp_pos_truth)

    return dict(
        T_base_cube_true=T_base_cube_true,
        T_base_cube_est=T_base_cube_est,
        T_grasp_est=T_grasp_est,
        T_grasp_truth=T_grasp_truth,
        T_start_gripper=T_start_gripper,
        T_start_cam_true=T_start_cam_true,
        T_cam_target_obs=T_cam_target_obs,
    )


def _start_gripper_pose(target_pos: np.ndarray,
                        cube_pos: np.ndarray,
                        X_gt: np.ndarray) -> np.ndarray:
    """Start pose: wrist cam aimed at the marker from a sensible stand-off.

    The cam is placed ~60 cm along the marker-normal direction + offset,
    pitched slightly over the scene. Gripper = cam @ inv(X_gt).
    """
    # Use a fixed, scene-independent start viewpoint: 60 cm in +x from
    # marker, offset +25 cm in -y (toward the observer cam), elevated 20 cm.
    start_cam = target_pos + np.array([0.60, -0.25, 0.20])
    T_start_cam_cv = _cam_pose_lookat(start_cam, target_pos)
    return T_start_cam_cv @ _inv(X_gt)


# --------------------------------------------------------------------------- #
# Rendering helpers                                                           #
# --------------------------------------------------------------------------- #
def _set_pose(data, qadr: int, T: np.ndarray) -> None:
    data.qpos[qadr:qadr + 3] = T[:3, 3]
    data.qpos[qadr + 3:qadr + 7] = _quat_wxyz_from_mat(T[:3, :3])


def _render_scene(
    scene: SceneInfo,
    renderer: mujoco.Renderer,
    opt: mujoco.MjvOption,
    T_base_gripper: np.ndarray,
    T_cube_true: np.ndarray,
    T_cube_est: np.ndarray,
    cam_name: str,
    qadr_gripper: int,
    qadr_cube_true: int,
    qadr_cube_est: int,
) -> np.ndarray:
    data = scene.data
    _set_pose(data, qadr_gripper, T_base_gripper)
    _set_pose(data, qadr_cube_true, T_cube_true)
    _set_pose(data, qadr_cube_est, T_cube_est)
    mujoco.mj_forward(scene.model, data)
    renderer.update_scene(data, camera=cam_name, scene_option=opt)
    img = renderer.render()
    return cv2.convertScaleAbs(img, alpha=1.18, beta=10)


def _project_point(
    p_world: np.ndarray,
    scene: SceneInfo,
    renderer: mujoco.Renderer,
    cam_name: str,
) -> Optional[np.ndarray]:
    """Project a world point to pixel coords in the given fixed camera."""
    model = scene.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        return None
    cam_pos = np.array(model.cam_pos0[cam_id])
    cam_mat = np.array(model.cam_mat0[cam_id]).reshape(3, 3)
    p_cam = cam_mat.T @ (p_world - cam_pos)
    if p_cam[2] >= 0:
        return None
    h, w = renderer.height, renderer.width
    fovy = float(model.cam_fovy[cam_id])
    f = 0.5 * h / np.tan(np.deg2rad(fovy) / 2.0)
    x = w / 2.0 + f * (p_cam[0] / -p_cam[2])
    y = h / 2.0 - f * (p_cam[1] / -p_cam[2])
    return np.array([x, y], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Annotation helpers                                                          #
# --------------------------------------------------------------------------- #
def _annotate_top_bottom(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    band: int = 52,
) -> None:
    """Black top + bottom bars with title / subtitle text."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, band), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, h - band), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, title, (20, int(band * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    # Subtitle may be long: wrap to ~110 chars of printable, draw 2 lines.
    max_line_chars = int(w / 10)
    line1 = subtitle[:max_line_chars]
    line2 = subtitle[max_line_chars:max_line_chars * 2]
    cv2.putText(frame, line1, (20, h - band + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (215, 215, 215), 1, cv2.LINE_AA)
    if line2:
        cv2.putText(frame, line2, (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (215, 215, 215),
                    1, cv2.LINE_AA)


def _verdict_label(miss_mm: float) -> Tuple[str, Tuple[int, int, int]]:
    if miss_mm <= HIT_MM:
        return "HIT", (40, 220, 40)
    if miss_mm <= PARTIAL_MM:
        return "PARTIAL", (40, 200, 240)
    return "MISS", (40, 60, 240)


def _draw_verdict_badge(
    frame: np.ndarray,
    miss_mm: float,
    err_xyz_mm: np.ndarray,
) -> None:
    """Big centred HIT / PARTIAL / MISS banner with numeric breakdown."""
    label, color = _verdict_label(miss_mm)
    text1 = f"{label}  ({miss_mm:.1f} mm)"
    text2 = (f"err  dx={err_xyz_mm[0]:+.1f}  dy={err_xyz_mm[1]:+.1f}  "
             f"dz={err_xyz_mm[2]:+.1f}  mm")
    (tw1, th1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
    (tw2, th2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    h, w = frame.shape[:2]
    pad = 18
    box_w = max(tw1, tw2) + 2 * pad
    box_h = th1 + th2 + 3 * pad
    x0 = (w - box_w) // 2
    y0 = (h - box_h) // 2 - 30
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), color, 3)
    cv2.putText(frame, text1,
                (x0 + (box_w - tw1) // 2, y0 + pad + th1),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
    cv2.putText(frame, text2,
                (x0 + (box_w - tw2) // 2, y0 + pad + th1 + pad + th2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (230, 230, 230), 2, cv2.LINE_AA)


def _draw_error_arrow_overlay(
    frame: np.ndarray,
    scene: SceneInfo,
    renderer: mujoco.Renderer,
    cam_name: str,
    p_true: np.ndarray,
    p_est: np.ndarray,
) -> None:
    """Draw a 2D arrow from the estimated cube position to the true one."""
    pt_true = _project_point(p_true, scene, renderer, cam_name)
    pt_est = _project_point(p_est, scene, renderer, cam_name)
    if pt_true is None or pt_est is None:
        return
    p_true_px = tuple(pt_true.astype(int))
    p_est_px = tuple(pt_est.astype(int))
    # Shadow + coloured arrow for visibility.
    cv2.arrowedLine(frame, p_est_px, p_true_px, (0, 0, 0), 6,
                    cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(frame, p_est_px, p_true_px, (40, 220, 255), 3,
                    cv2.LINE_AA, tipLength=0.25)


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def render_pick_video(
    scene: SceneInfo,
    X_est: np.ndarray,
    solver_name: str,
    trans_std_mm: float,
    rot_std_deg: float,
    trans_err_vs_gt_mm: float,
    rot_err_vs_gt_deg: float,
    out_path: Optional[Path] = None,
    duration_s: float = 4.5,
    fps: int = 30,
    width: int = 960,
    height: int = 720,
    cam_name: str = "reach_cam",
    seed: int = 0,
) -> Path:
    """Render a single-panel pick-task video and return the mp4 path.

    Deterministic given `seed` (controls the random cube offset). Everything
    else flows from X_est.
    """
    VIDEO_DIR.mkdir(exist_ok=True)
    if out_path is None:
        out_path = VIDEO_DIR / f"{_short_commit()}_{solver_name}.mp4"

    rng = np.random.default_rng(seed)
    cube_local_offset = sample_cube_local_offset(rng)

    # qpos layout from scene.xml: gripper @ 0, target_cube @ 7, ghost_cube @ 14.
    model = scene.model
    qadr_gripper = int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_free")])
    qadr_cube_true = int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_cube_free")])
    qadr_cube_est = int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ghost_cube_free")])

    target_pos = scene.T_base_target[:3, 3]
    T_start = _start_gripper_pose(target_pos, None, scene.X_gt)
    plan = plan_pick(scene, X_est, T_start, cube_local_offset)

    T_end = plan["T_grasp_est"]
    T_cube_true = plan["T_base_cube_true"]
    T_cube_est  = plan["T_base_cube_est"]

    # The resulting miss: prong midpoint at T_end @ (0, 0, -0.11) == T_base_cube_est.
    err_xyz_m = T_cube_est[:3, 3] - T_cube_true[:3, 3]
    miss_mm = float(np.linalg.norm(err_xyz_m) * 1000.0)
    err_xyz_mm = err_xyz_m * 1000.0

    renderer = mujoco.Renderer(model, height=height, width=width)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.sitegroup[:] = 0
    opt.frame = int(mujoco.mjtFrame.mjFRAME_BODY)

    n_approach = max(1, int(duration_s * fps * 0.65))
    n_hold = max(1, int(duration_s * fps * 0.35))
    n_frames = n_approach + n_hold

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"failed to open VideoWriter at {out_path}")

    title = (f"PICK TASK -- X_est ({solver_name}) used to locate the 2 cm cube "
             f"via marker detection")
    subtitle = (
        f"Real cube = green.  Agent's ESTIMATE of the cube (X_est * marker "
        f"detection) = red ghost.  Gripper flies to the ghost.  "
        f"Hand-eye error propagates 1:1 into grasp miss.  "
        f"trans_std={trans_std_mm:.2f} mm  rot_std={rot_std_deg:.2f} deg  "
        f"vs_gt={trans_err_vs_gt_mm:.2f} mm / {rot_err_vs_gt_deg:.2f} deg  "
        f"[HIT<={HIT_MM:.0f}mm  PARTIAL<={PARTIAL_MM:.0f}mm]"
    )

    try:
        for i in range(n_frames):
            if i < n_approach:
                t = _ease(i / max(1, n_approach - 1))
            else:
                t = 1.0
            is_final = (i == n_frames - 1)

            T_bg = _interp_pose(T_start, T_end, t)
            img = _render_scene(
                scene, renderer, opt,
                T_base_gripper=T_bg,
                T_cube_true=T_cube_true,
                T_cube_est=T_cube_est,
                cam_name=cam_name,
                qadr_gripper=qadr_gripper,
                qadr_cube_true=qadr_cube_true,
                qadr_cube_est=qadr_cube_est,
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Always draw the error arrow between ghost and true cube so
            # the viewer can eyeball the direction of the calibration error
            # throughout the reach.
            _draw_error_arrow_overlay(
                img, scene, renderer, cam_name,
                p_true=T_cube_true[:3, 3],
                p_est=T_cube_est[:3, 3],
            )

            # Persistent legend top-right.
            legend_y = 78
            cv2.putText(img, "GREEN = real cube  |  RED GHOST = agent's estimate",
                        (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 220, 220), 1, cv2.LINE_AA)

            if is_final:
                _draw_verdict_badge(img, miss_mm, err_xyz_mm)

            _annotate_top_bottom(img, title, subtitle)
            writer.write(img)
    finally:
        writer.release()
        renderer.close()

    return out_path


# Backward-compatible alias so existing callers keep working.
render_reach_video = render_pick_video
