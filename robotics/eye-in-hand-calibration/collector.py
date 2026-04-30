"""MuJoCo-based eye-in-hand data collection.

Two halves, different locking rules:

  * SENSOR PIPELINE (LOCKED): `load_scene`, `render_detect_pnp`, intrinsics
    derivation, the MuJoCo<->OpenCV axis flip, `SceneInfo`, and the ground
    truth `X_gt`. Same role as hardware -- do not edit.

  * ORCHESTRATION (UNLOCKED): `collect_scenario` and below. Picks gripper
    poses via `samplers.sample_poses`, drives the sensor pipeline, assembles
    a `Scenario`. Edit freely.

Also hosts the shared frame-math primitives (`Scenario`, `_make_T`, `_inv`,
`_rodrigues`, `_random_rotation`, `_perturb_rotation`) because they're the
SE(3) contract every downstream file relies on. LOCKED.

Axis conventions (locked):
  - MuJoCo camera: -z forward, +x right, +y up.
  - OpenCV camera: +z forward, +x right, +y down (what solvePnP returns).
  - Flip:   R_cv_from_mj = diag(1, -1, -1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2
import mujoco
import numpy as np


# ===========================================================================
# --- LOCKED: FRAME MATH -----------------------------------------------------
# ===========================================================================

def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform SO(3) via QR decomposition of a Gaussian matrix."""
    m = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(m)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([
        [0.0,  -k[2],  k[1]],
        [k[2],  0.0,  -k[0]],
        [-k[1], k[0],  0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _perturb_rotation(R: np.ndarray, sigma_deg: float,
                      rng: np.random.Generator) -> np.ndarray:
    if sigma_deg <= 0:
        return R
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = np.radians(rng.normal(0.0, sigma_deg))
    return R @ _rodrigues(axis * angle)


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


@dataclass
class Scenario:
    """Per-run eye-in-hand data. Consumed by every solver and metric."""
    T_base_gripper: np.ndarray  # (N, 4, 4) noisy FK
    T_cam_target: np.ndarray    # (N, 4, 4) PnP output
    X_gt: np.ndarray            # (4, 4) ground-truth T_gripper_cam (CV convention)
    T_base_target: np.ndarray   # (4, 4) marker pose in base frame
    metadata: Dict


# ===========================================================================
# --- LOCKED: SENSOR PIPELINE -----------------------------------------------
# ===========================================================================

HERE = Path(__file__).parent
SCENE_XML = HERE / "scene.xml"
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
ARUCO_MARKER_ID = 7
# PNG is 800x800 with a 100-px white quiet zone on every edge, so the marker
# pattern itself is 600/800 = 75% of the 0.10 m plane face.
ARUCO_PHYSICAL_SIZE_M = 0.10 * (600.0 / 800.0)

R_CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])


@dataclass
class SceneInfo:
    model: mujoco.MjModel
    data: mujoco.MjData
    renderer: mujoco.Renderer
    scene_option: mujoco.MjvOption
    width: int
    height: int
    K: np.ndarray                        # 3x3 OpenCV intrinsics
    dist: np.ndarray                     # 5x1, zero (pinhole)
    X_gt: np.ndarray                     # 4x4 T_gripper_cam_cv (ground truth)
    T_base_target: np.ndarray            # 4x4 marker pose in world (MuJoCo frame)
    T_gripper_cammount_local: np.ndarray # 4x4 gripper->camera_mount local
    aruco_body_id: int
    gripper_body_id: int
    cammount_body_id: int
    cam_id: int


def _quat_wxyz_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def _body_world_T(data: mujoco.MjData, body_id: int) -> np.ndarray:
    R = data.xmat[body_id].reshape(3, 3).copy()
    t = data.xpos[body_id].copy()
    return _make_T(R, t)


def _intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> np.ndarray:
    fovy = np.radians(fovy_deg)
    fy = (height / 2.0) / np.tan(fovy / 2.0)
    fx = fy
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_scene(width: int = 640, height: int = 480) -> SceneInfo:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    gripper_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    aruco_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "aruco_body")
    cammount_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera_mount")
    cam_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    if min(gripper_id, aruco_id, cammount_id, cam_id) < 0:
        raise RuntimeError("scene.xml is missing an expected body/camera")

    R_local = model.body_quat[cammount_id]
    t_local = model.body_pos[cammount_id].copy()
    R_mat = np.empty(9)
    mujoco.mju_quat2Mat(R_mat, R_local)
    T_g_cm = _make_T(R_mat.reshape(3, 3), t_local)

    # X_gt in OpenCV camera convention (gripper -> cv cam).
    X_gt = T_g_cm.copy()
    X_gt[:3, :3] = X_gt[:3, :3] @ R_CV_FROM_MJ

    mujoco.mj_forward(model, data)
    T_base_target = _body_world_T(data, aruco_id)

    fovy = float(model.cam_fovy[cam_id])
    K = _intrinsics_from_fovy(fovy, width, height)
    dist = np.zeros(5)

    renderer = mujoco.Renderer(model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.geomgroup[2] = 0
    scene_option.geomgroup[3] = 0
    scene_option.geomgroup[4] = 0
    scene_option.geomgroup[5] = 0
    scene_option.sitegroup[:] = 0

    return SceneInfo(
        model=model, data=data, renderer=renderer, scene_option=scene_option,
        width=width, height=height, K=K, dist=dist,
        X_gt=X_gt, T_base_target=T_base_target,
        T_gripper_cammount_local=T_g_cm,
        aruco_body_id=aruco_id, gripper_body_id=gripper_id,
        cammount_body_id=cammount_id, cam_id=cam_id,
    )


_ARUCO_DETECTOR: Optional[cv2.aruco.ArucoDetector] = None


def _get_detector() -> cv2.aruco.ArucoDetector:
    global _ARUCO_DETECTOR
    if _ARUCO_DETECTOR is None:
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 5
        p.cornerRefinementMaxIterations = 50
        _ARUCO_DETECTOR = cv2.aruco.ArucoDetector(d, p)
    return _ARUCO_DETECTOR


def _marker_object_points(size: float) -> np.ndarray:
    s = size / 2.0
    return np.array([[-s,  s, 0],
                     [ s,  s, 0],
                     [ s, -s, 0],
                     [-s, -s, 0]], dtype=np.float64)


def _rtvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return _make_T(R, tvec.reshape(3))


def render_detect_pnp(
    scene: SceneInfo,
    T_base_gripper: np.ndarray,
    use_ippe: bool = True,
    use_depth_correction: bool = False,
) -> Tuple[Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    """Place gripper at `T_base_gripper`, render wrist cam, detect + PnP.

    Returns `(T_cam_target or None, image_rgb, report)` where `report`
    carries `detected`, `reproj_err_px`, and the 4x2 corner array when
    detection succeeded.
    
    If use_depth_correction=True, refine the PnP solution using depth information.
    """
    data = scene.data
    data.qpos[0:3] = T_base_gripper[:3, 3]
    data.qpos[3:7] = _quat_wxyz_from_mat(T_base_gripper[:3, :3])
    mujoco.mj_forward(scene.model, data)

    scene.renderer.update_scene(data, camera="wrist_cam",
                                scene_option=scene.scene_option)
    img = scene.renderer.render()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners_list, ids, _ = _get_detector().detectMarkers(gray)
    if ids is None or ARUCO_MARKER_ID not in ids.flatten():
        return None, img, dict(detected=False)

    idx = int(np.where(ids.flatten() == ARUCO_MARKER_ID)[0][0])
    corners = corners_list[idx].reshape(4, 2).astype(np.float64)

    op = _marker_object_points(ARUCO_PHYSICAL_SIZE_M)
    
    # Try IPPE first (better for planar targets), fall back to IPPE_SQUARE
    if use_ippe:
        ok, rvec, tvec = cv2.solvePnP(
            op, corners, scene.K, scene.dist, flags=cv2.SOLVEPNP_IPPE,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            op, corners, scene.K, scene.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    
    if not ok:
        return None, img, dict(detected=False, corners=corners)

    # Apply depth correction if requested
    if use_depth_correction:
        try:
            scene.renderer.enable_depth_rendering()
            scene.renderer.update_scene(data, camera="wrist_cam",
                                        scene_option=scene.scene_option)
            depth_map = scene.renderer.render()
            scene.renderer.disable_depth_rendering()
            
            # Refine tvec using depth information
            depths = []
            for pt in corners:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                    d = depth_map[y, x]
                    if d > 0:
                        depths.append(d)
            
            if len(depths) >= 2:
                median_depth = np.median(depths)
                tvec_refined = tvec.copy()
                tvec_refined[2] = median_depth
                
                # Verify the refined solution
                proj_refined, _ = cv2.projectPoints(op, rvec, tvec_refined, scene.K, scene.dist)
                reproj_err_refined = np.linalg.norm(proj_refined.reshape(4, 2) - corners, axis=1).mean()
                
                proj_orig, _ = cv2.projectPoints(op, rvec, tvec, scene.K, scene.dist)
                reproj_err_orig = np.linalg.norm(proj_orig.reshape(4, 2) - corners, axis=1).mean()
                
                if reproj_err_refined < reproj_err_orig:
                    tvec = tvec_refined
        except Exception:
            pass  # Fall back to standard PnP if depth correction fails

    T_cam_target = _rtvec_to_T(rvec, tvec)
    proj, _ = cv2.projectPoints(op, rvec, tvec, scene.K, scene.dist)
    reproj_err = float(np.linalg.norm(proj.reshape(4, 2) - corners, axis=1).mean())

    return T_cam_target, img, dict(
        detected=True,
        reproj_err_px=reproj_err,
        corners=corners,
        rvec=rvec,
        tvec=tvec,
    )


# ===========================================================================
# --- UNLOCKED: ORCHESTRATION ------------------------------------------------
# ===========================================================================

def _save_overlay(img_rgb: np.ndarray, report: Dict[str, Any],
                  scene: SceneInfo, out_path: Path) -> None:
    vis = img_rgb.copy()
    if report.get("detected"):
        corners = report["corners"].astype(np.float32).reshape(1, 4, 2)
        cv2.aruco.drawDetectedMarkers(vis, [corners], np.array([[ARUCO_MARKER_ID]]))
        cv2.drawFrameAxes(
            vis, scene.K, scene.dist,
            report["rvec"], report["tvec"],
            ARUCO_PHYSICAL_SIZE_M / 2,
        )
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def collect_scenario(
    n_poses: int = 16,
    distance: float = 0.6,
    sampler: Union[str, Any] = "diverse",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    kin_rot_noise_deg: float = 0.0,
    kin_trans_noise_mm: float = 0.0,
    seed: int = 0,
    width: int = 640,
    height: int = 480,
    scene: Optional[SceneInfo] = None,
    save_images_to: Optional[Path] = None,
    max_retries_per_pose: int = 3,
    use_depth_correction: bool = False,
) -> Tuple[Scenario, Dict[str, Any]]:
    """Sample poses, run the sensor pipeline, return a `Scenario`.

    UNLOCKED. Modify sampling strategy, noise, retry logic, or anything else
    -- as long as the sensor pipeline (`render_detect_pnp`, `load_scene`,
    X_gt derivation) is not touched and nothing reads `scene.X_gt` except
    `metrics.pose_error_deg_mm` at scoring time.
    """
    from samplers import sample_poses  # deferred to avoid circular import

    if scene is None:
        scene = load_scene(width=width, height=height)
    if sampler_kwargs is None:
        sampler_kwargs = {}
    rng = np.random.default_rng(seed)

    if save_images_to is not None:
        save_images_to.mkdir(parents=True, exist_ok=True)

    target_pos = scene.T_base_target[:3, 3]

    # Oversample: each sampler call produces `n_poses`; if detections fail
    # we refill from fresh sampler draws until we have `n_poses` good ones
    # or exhaust the retry budget.
    T_bg_accepted: List[np.ndarray] = []
    T_ct_accepted: List[np.ndarray] = []
    reproj_errs: List[float] = []
    attempts = 0

    for attempt_round in range(max_retries_per_pose):
        remaining = n_poses - len(T_bg_accepted)
        if remaining <= 0:
            break
        candidates = sample_poses(
            sampler, target_pos, distance, remaining, scene.X_gt, rng,
            **sampler_kwargs,
        )
        for T_bg in candidates:
            attempts += 1
            T_ct, img, report = render_detect_pnp(scene, T_bg, use_depth_correction=use_depth_correction)
            if not report.get("detected"):
                continue

            T_bg_noisy = T_bg.copy()
            if kin_rot_noise_deg > 0:
                T_bg_noisy[:3, :3] = _perturb_rotation(
                    T_bg_noisy[:3, :3], kin_rot_noise_deg, rng,
                )
            if kin_trans_noise_mm > 0:
                T_bg_noisy[:3, 3] += rng.normal(
                    0, kin_trans_noise_mm / 1000.0, 3,
                )

            T_bg_accepted.append(T_bg_noisy)
            T_ct_accepted.append(T_ct)
            reproj_errs.append(report["reproj_err_px"])

            if save_images_to is not None:
                _save_overlay(
                    img, report, scene,
                    save_images_to / f"pose_{len(T_bg_accepted)-1:03d}.png",
                )

            if len(T_bg_accepted) >= n_poses:
                break

    scenario = Scenario(
        T_base_gripper=np.asarray(T_bg_accepted),
        T_cam_target=np.asarray(T_ct_accepted),
        X_gt=scene.X_gt,
        T_base_target=scene.T_base_target,
        metadata=dict(
            n_poses=len(T_bg_accepted),
            distance=distance,
            sampler=sampler if isinstance(sampler, str) else getattr(sampler, "__name__", "custom"),
            sampler_kwargs=sampler_kwargs,
            kin_rot_noise_deg=kin_rot_noise_deg,
            kin_trans_noise_mm=kin_trans_noise_mm,
            seed=seed,
            width=width,
            height=height,
            source="mujoco_l2",
        ),
    )
    report = dict(
        detection_attempts=attempts,
        detection_success=len(T_bg_accepted),
        detection_rate=len(T_bg_accepted) / max(1, attempts),
        mean_reproj_err_px=float(np.mean(reproj_errs)) if reproj_errs else float("nan"),
        max_reproj_err_px=float(np.max(reproj_errs)) if reproj_errs else float("nan"),
    )
    return scenario, report
