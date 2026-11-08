"""Attention model based on InsightFace keypoints (head pose from yaw/pitch/roll)."""

import cv2
import numpy as np


class InsightFaceAttentionModel:
    """Estimate head pose and compute attention score from 5 facial keypoints."""

    def __init__(self, yaw_threshold_deg: float = 25.0, pitch_threshold_deg: float = 20.0):
        self.yaw_threshold_deg = float(yaw_threshold_deg)
        self.pitch_threshold_deg = float(pitch_threshold_deg)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def estimate_head_pose(self, kps: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[float, float, float] | None:
        """Return (yaw, pitch, roll) in degrees, or None when pose estimation fails."""
        if kps is None:
            return None

        pts = np.asarray(kps, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 5 or pts.shape[1] < 2:
            return None

        image_points = np.array([
            pts[0],  # left eye
            pts[1],  # right eye
            pts[2],  # nose
            pts[3],  # left mouth
            pts[4],  # right mouth
        ], dtype=np.float64)

        # Rough 3D face model points for 5 landmarks.
        model_points = np.array([
            (-30.0, 35.0, -30.0),
            (30.0, 35.0, -30.0),
            (0.0, 0.0, 0.0),
            (-25.0, -35.0, -25.0),
            (25.0, -35.0, -25.0),
        ], dtype=np.float64)

        h, w = image_shape[:2]
        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rot_mtx, _ = cv2.Rodrigues(rvec)
        proj_mtx = np.hstack((rot_mtx, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mtx)

        pitch = float(euler[0, 0])
        yaw = float(euler[1, 0])
        roll = float(euler[2, 0])
        return yaw, pitch, roll

    def evaluate(self, kps: np.ndarray, image_shape: tuple[int, int, int]) -> dict | None:
        pose = self.estimate_head_pose(kps, image_shape)
        if pose is None:
            return None

        yaw, pitch, roll = pose
        yaw_norm = self._clamp(abs(yaw) / max(self.yaw_threshold_deg, 1e-6), 0.0, 1.0)
        pitch_norm = self._clamp(abs(pitch) / max(self.pitch_threshold_deg, 1e-6), 0.0, 1.0)

        attention_score = self._clamp(1.0 - (0.6 * yaw_norm + 0.4 * pitch_norm), 0.0, 1.0)
        attentive = abs(yaw) <= self.yaw_threshold_deg and abs(pitch) <= self.pitch_threshold_deg

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "attention_score": attention_score,
            "attentive": bool(attentive),
            "source": "insightface_attention",
        }
