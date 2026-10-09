"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

# vi: from fusion_lab.workspace_support import get_tracking_params
MIN_DEPTH = 1e-6


def _position_in_sensor(x: Matrix, sensor: Any) -> np.ndarray:
    """Rotate and translate the vehicle-frame track position into sensor axes."""
    transform = np.asarray(sensor.veh_to_sens, dtype=float)
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    return transform[:3, :3] @ position + transform[:3, 3]


def _is_camera(sensor: Any) -> bool:
    return getattr(sensor, "name", None) == "camera" or hasattr(sensor, "f_i")


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    position = _position_in_sensor(x, sensor)
    if not np.isfinite(position).all():
        return False
    depth, left, _ = position
    if _is_camera(sensor):
        if depth <= MIN_DEPTH:
            return False
        # Horizontal opening of the image: pixel column 0 .. width.
        lower = np.arctan((sensor.c_i - sensor.image_width) / sensor.f_i)
        upper = np.arctan(sensor.c_i / sensor.f_i)
    else:
        lower, upper = sensor.fov
    angle = np.arctan2(left, depth)
    return bool(min(lower, upper) <= angle <= max(lower, upper))


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    position = _position_in_sensor(x, sensor)
    if not np.isfinite(position).all() or position[0] <= MIN_DEPTH:
        raise ValueError(
            "Camera projection needs finite sensor coordinates and depth "
            f"> {MIN_DEPTH:g}; got sensor position={position.tolist()}"
        )
    depth, left, up = position
    u = sensor.c_i - sensor.f_i * left / depth
    v = sensor.c_j - sensor.f_j * up / depth
    return np.asmatrix([[u], [v]], dtype=float)


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    pixels = np.asarray(z, dtype=float).reshape(-1)[:2]
    return {
        "z": np.asmatrix(pixels).T,
        "R": np.asmatrix(np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2])),
        "sensor": sensor,
    }
