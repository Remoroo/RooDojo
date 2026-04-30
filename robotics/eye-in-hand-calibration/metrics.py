"""Read-only. Ground-truth error computation for eye-in-hand calibration.

This file is LOCKED -- the autoresearch agent does not modify it. The two
scalars it emits, rotation error in degrees and translation error in
millimetres, are the ground truth for every experiment. The simulator owns
`X_gt`; everything else is derived here.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def pose_error_deg_mm(X_est: np.ndarray, X_gt: np.ndarray) -> Tuple[float, float]:
    """Return (rotation error in degrees, translation error in millimetres).

    Rotation error is the geodesic distance on SO(3) -- the angle of the
    relative rotation `R_est^T R_gt`. Translation error is the Euclidean
    distance between the two translation vectors, converted to mm.
    """
    R_est = X_est[:3, :3]
    R_gt = X_gt[:3, :3]
    R_rel = R_est.T @ R_gt
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    rot_err_deg = float(np.degrees(np.arccos(cos_angle)))

    t_err_mm = float(np.linalg.norm(X_est[:3, 3] - X_gt[:3, 3]) * 1000.0)
    return rot_err_deg, t_err_mm


def summarize(errors: Iterable[Tuple[float, float]]) -> Dict[str, float]:
    """Aggregate per-scenario (rot_deg, trans_mm) pairs into summary stats."""
    arr = np.asarray(list(errors), dtype=float)
    if arr.size == 0:
        return dict(
            n=0,
            median_rot_deg=float("inf"),
            median_trans_mm=float("inf"),
            p90_rot_deg=float("inf"),
            p90_trans_mm=float("inf"),
            mean_rot_deg=float("inf"),
            mean_trans_mm=float("inf"),
        )

    # Replace inf/nan failures with a very large finite value so percentiles
    # still rank them "worst" rather than propagating NaN.
    finite = np.where(np.isfinite(arr), arr, 1e9)

    return dict(
        n=int(arr.shape[0]),
        median_rot_deg=float(np.median(finite[:, 0])),
        median_trans_mm=float(np.median(finite[:, 1])),
        p90_rot_deg=float(np.percentile(finite[:, 0], 90)),
        p90_trans_mm=float(np.percentile(finite[:, 1], 90)),
        mean_rot_deg=float(np.mean(finite[:, 0])),
        mean_trans_mm=float(np.mean(finite[:, 1])),
    )
