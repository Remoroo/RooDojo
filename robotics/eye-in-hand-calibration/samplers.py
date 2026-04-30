"""Pose-sampling strategies -- the research playground.

NOT LOCKED. Add, edit, delete, combine sampler modes as you like. A sampler
is anything registered in `SAMPLERS` that takes a fixed signature and returns
a list of gripper world poses.

A sampler's one job: choose `n_poses` gripper poses `T_base_gripper` such
that, ideally, when the camera rigidly attached to the gripper (offset by
`X_gt`) is rendered, it can see the marker at `T_base_target`. Whether that
is achieved via random hemisphere sampling, a smooth arc, a structured grid,
or something weirder, is entirely up to you.

Rules of the game:
  - Samplers return `List[np.ndarray]`, each a 4x4 `T_base_gripper`. Do not
    return image data, noise, or PnP outputs -- those belong to the sensor
    pipeline.
  - Samplers can use `X_gt` to compute the resulting camera pose and check
    visibility; that's NOT considered cheating because the camera-on-gripper
    offset is a real thing you know on a real robot too (from the CAD).
  - What WOULD be cheating: using `X_gt` to bias noise, to pre-correct the
    gripper pose, or anything else that lets `X_gt` leak into the solve.
  - Samplers must be deterministic given `rng`.

To add a sampler:
  1. Write `def my_sampler(target_pos, distance, n_poses, X_gt, rng, **kw):`
  2. Return a list of `T_base_gripper` (4x4 matrices).
  3. Add `"my_sampler": my_sampler` to `SAMPLERS`.
  4. Optionally add a sweep entry in `config.py`.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from collector import _inv, _make_T, _random_rotation, _rodrigues


# ---------------------------------------------------------------------------
# Internal helpers (not locked -- duplicate/extend as you please)
# ---------------------------------------------------------------------------

def _lookat_camera_R_mj(
    cam_pos: np.ndarray,
    target_pos: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Return a MuJoCo-convention camera rotation such that -z points at target.

    MuJoCo camera: x_cam = right, y_cam = up, z_cam = back (away from scene).
    """
    forward = target_pos - cam_pos
    n = np.linalg.norm(forward) + 1e-12
    forward = forward / n
    if abs(np.dot(forward, world_up)) > 0.98:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12
    return np.column_stack([right, up, -forward])


def _apply_roll(R_cam_world: np.ndarray, roll_rad: float) -> np.ndarray:
    """Rotate the camera about its own optical axis (cam -z)."""
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    R_roll = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return R_cam_world @ R_roll


def _gripper_from_cam_world(
    T_base_cam_mj: np.ndarray,
    X_gt: np.ndarray,
) -> np.ndarray:
    """Invert the camera-in-gripper offset to get gripper pose from camera pose.

    `X_gt` is in OpenCV convention; the MuJoCo camera sits at the same origin
    as OpenCV's, so we can use the positional part of X_gt directly via the
    MuJoCo->OpenCV axis flip encoded in X_gt's rotation. In practice the
    translation part of X_gt is what matters for reversing the offset.
    """
    # T_gripper_cam_mj = X_gt @ diag(1,-1,-1)^{-1}_4x4.
    # But that diag flip is its own inverse, so T_gripper_cam_mj has the same
    # translation as X_gt and rotation = X_gt.R @ diag(1,-1,-1).
    R_gc_mj = X_gt[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    t_gc = X_gt[:3, 3]
    T_gripper_cam_mj = _make_T(R_gc_mj, t_gc)
    return T_base_cam_mj @ _inv(T_gripper_cam_mj)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def sample_diverse(
    target_pos: np.ndarray,
    distance: float,
    n_poses: int,
    X_gt: np.ndarray,
    rng: np.random.Generator,
    roll_deg_range: float = 45.0,
) -> List[np.ndarray]:
    """Uniform hemisphere of camera positions + random roll.

    Canonical "well-conditioned AX=XB" sampler.
    """
    poses = []
    for _ in range(n_poses):
        d = rng.standard_normal(3)
        d[0] = abs(d[0])  # +x hemisphere (marker face)
        d /= np.linalg.norm(d) + 1e-12
        cam_pos = target_pos + distance * d
        R = _lookat_camera_R_mj(cam_pos, target_pos)
        R = _apply_roll(R, np.radians(rng.uniform(-roll_deg_range, roll_deg_range)))
        poses.append(_gripper_from_cam_world(_make_T(R, cam_pos), X_gt))
    return poses


def sample_planar(
    target_pos: np.ndarray,
    distance: float,
    n_poses: int,
    X_gt: np.ndarray,
    rng: np.random.Generator,
    roll_deg_range: float = 30.0,
) -> List[np.ndarray]:
    """Camera positions on the xy-plane through the marker -- 1D constrained."""
    poses = []
    for _ in range(n_poses):
        az = rng.uniform(-np.pi / 2, np.pi / 2)
        cam_pos = target_pos + distance * np.array([np.cos(az), np.sin(az), 0.0])
        R = _lookat_camera_R_mj(cam_pos, target_pos)
        R = _apply_roll(R, np.radians(rng.uniform(-roll_deg_range, roll_deg_range)))
        poses.append(_gripper_from_cam_world(_make_T(R, cam_pos), X_gt))
    return poses


def sample_collinear(
    target_pos: np.ndarray,
    distance: float,
    n_poses: int,
    X_gt: np.ndarray,
    rng: np.random.Generator,
    roll_deg_range: float = 90.0,
) -> List[np.ndarray]:
    """All cameras on the +x axis, only roll varies -- textbook-degenerate."""
    poses = []
    cam_pos = target_pos + distance * np.array([1.0, 0.0, 0.0])
    for _ in range(n_poses):
        R = _lookat_camera_R_mj(cam_pos, target_pos)
        R = _apply_roll(R, np.radians(rng.uniform(-roll_deg_range, roll_deg_range)))
        poses.append(_gripper_from_cam_world(_make_T(R, cam_pos), X_gt))
    return poses


def sample_hemispheric_grid(
    target_pos: np.ndarray,
    distance: float,
    n_poses: int,
    X_gt: np.ndarray,
    rng: np.random.Generator,
    azimuth_range_deg: float = 120.0,
    elevation_range_deg: float = 60.0,
    roll_deg_range: float = 20.0,
) -> List[np.ndarray]:
    """Structured (azimuth, elevation) grid on the +x hemisphere.

    Approximates what many calibration protocols do in practice: walk through
    a deliberate pattern of viewpoints, not random draws.
    """
    # Choose grid dims that roughly match n_poses.
    n_az = int(np.ceil(np.sqrt(n_poses * azimuth_range_deg / max(elevation_range_deg, 1))))
    n_el = int(np.ceil(n_poses / max(n_az, 1)))
    az_grid = np.linspace(-azimuth_range_deg, azimuth_range_deg, n_az)
    el_grid = np.linspace(-elevation_range_deg, elevation_range_deg, n_el)
    points = [(a, e) for a in az_grid for e in el_grid][:n_poses]

    poses = []
    for az_deg, el_deg in points:
        az, el = np.radians(az_deg), np.radians(el_deg)
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        cam_pos = target_pos + distance * d
        R = _lookat_camera_R_mj(cam_pos, target_pos)
        R = _apply_roll(R, np.radians(rng.uniform(-roll_deg_range, roll_deg_range)))
        poses.append(_gripper_from_cam_world(_make_T(R, cam_pos), X_gt))
    return poses


def sample_trajectory_arc(
    target_pos: np.ndarray,
    distance: float,
    n_poses: int,
    X_gt: np.ndarray,
    rng: np.random.Generator,
    arc_span_deg: float = 90.0,
    translation_jitter_m: float = 0.05,
    elevation_deg: float = 0.0,
    roll_deg_range: float = 15.0,
) -> List[np.ndarray]:
    """Continuous arc sweep -- consecutive poses are small increments.

    Models the common real-world protocol: move the robot smoothly through an
    arc in front of the marker and grab a frame at each step. Inter-pose
    motion is small, which stresses AX=XB conditioning.
      - arc_span_deg:        total azimuth span of the arc
      - translation_jitter_m: per-pose Gaussian jitter on camera position
      - elevation_deg:        fixed elevation angle of the arc
    """
    az_range = np.radians(arc_span_deg)
    el = np.radians(elevation_deg)
    az_values = np.linspace(-az_range / 2, az_range / 2, n_poses)

    poses = []
    for az in az_values:
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        cam_pos = target_pos + distance * d
        cam_pos += rng.normal(0, translation_jitter_m, 3)
        R = _lookat_camera_R_mj(cam_pos, target_pos)
        R = _apply_roll(R, np.radians(rng.uniform(-roll_deg_range, roll_deg_range)))
        poses.append(_gripper_from_cam_world(_make_T(R, cam_pos), X_gt))
    return poses


def sample_jitter(
    target_pos: np.ndarray,
    distance: float,
    n_poses: int,
    X_gt: np.ndarray,
    rng: np.random.Generator,
    position_jitter_m: float = 0.02,
    rotation_jitter_deg: float = 5.0,
) -> List[np.ndarray]:
    """Tight jitter around a single base pose -- pathological for AX=XB.

    Useful for stress tests: what happens if the operator only wiggles the
    robot instead of actually moving it? (Answer: very little parallax.)
    """
    cam_base = target_pos + distance * np.array([1.0, 0.0, 0.0])
    R_base = _lookat_camera_R_mj(cam_base, target_pos)

    poses = []
    for _ in range(n_poses):
        cam_pos = cam_base + rng.normal(0, position_jitter_m, 3)
        # Tiny rotation perturbation about a random axis.
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-12
        angle = np.radians(rng.normal(0, rotation_jitter_deg))
        R = R_base @ _rodrigues(axis * angle)
        poses.append(_gripper_from_cam_world(_make_T(R, cam_pos), X_gt))
    return poses


# ---------------------------------------------------------------------------
# Registry -- add new samplers here
# ---------------------------------------------------------------------------

SamplerFn = Callable[..., List[np.ndarray]]

SAMPLERS: Dict[str, SamplerFn] = {
    "diverse":           sample_diverse,
    "planar":            sample_planar,
    "collinear":         sample_collinear,
    "hemispheric_grid":  sample_hemispheric_grid,
    "trajectory_arc":    sample_trajectory_arc,
    "jitter":            sample_jitter,
}


def sample_poses(
    mode,
    target_pos: np.ndarray,
    distance: float,
    n_poses: int,
    X_gt: np.ndarray,
    rng: np.random.Generator,
    **kwargs,
) -> List[np.ndarray]:
    """Dispatch to a named sampler or a user-supplied callable."""
    if callable(mode):
        return mode(target_pos, distance, n_poses, X_gt, rng, **kwargs)
    try:
        fn = SAMPLERS[mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown sampler '{mode}'. Registered: {list(SAMPLERS)}"
        ) from exc
    return fn(target_pos, distance, n_poses, X_gt, rng, **kwargs)
