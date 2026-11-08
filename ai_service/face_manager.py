"""Quản lý face embeddings - load từ API, so sánh khuôn mặt."""

import numpy as np
import requests
import logging
from config import API_GET_USERS_URL, FACE_MATCH_THRESHOLD

logger = logging.getLogger(__name__)


class FaceManager:
    """Quản lý danh sách embeddings và so sánh khuôn mặt."""

    def __init__(self):
        self.known_faces = []  # [{"user_id": int, "name": str, "embedding": np.array}]

    def load_from_api(self):
        """Tải danh sách embeddings từ Flask API."""
        try:
            resp = requests.get(API_GET_USERS_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            self.known_faces = []
            for user in data.get("users", []):
                embedding = np.array(user["embedding"], dtype=np.float32)
                self.known_faces.append({
                    "user_id": user["id"],
                    "name": user["name"],
                    "embedding": embedding,
                })
            logger.info("Đã load %d người từ API", len(self.known_faces))
        except requests.RequestException as e:
            logger.error("Không thể load embeddings từ API: %s", e)

    def find_match(self, embedding):
        """So sánh embedding với danh sách đã biết.

        Returns:
            dict hoặc None: {"user_id", "name", "similarity"} nếu match
        """
        if not self.known_faces:
            return None

        best_match = None
        best_sim = -1

        emb_norm = embedding / np.linalg.norm(embedding)

        for known in self.known_faces:
            known_norm = known["embedding"] / np.linalg.norm(known["embedding"])
            sim = float(np.dot(emb_norm, known_norm))

            if sim > best_sim:
                best_sim = sim
                best_match = known

        if best_sim >= FACE_MATCH_THRESHOLD and best_match is not None:
            return {
                "user_id": best_match["user_id"],
                "name": best_match["name"],
                "similarity": round(best_sim, 4),
            }

        return None

    def find_best(self, embedding):
        """Luôn trả về best match (kể cả dưới ngưỡng) để debug."""
        if not self.known_faces:
            return None

        best_match = None
        best_sim = -1.0

        emb_norm = embedding / np.linalg.norm(embedding)
        for known in self.known_faces:
            known_norm = known["embedding"] / np.linalg.norm(known["embedding"])
            sim = float(np.dot(emb_norm, known_norm))
            if sim > best_sim:
                best_sim = sim
                best_match = known

        if best_match is None:
            return None

        return {
            "user_id": best_match["user_id"],
            "name": best_match["name"],
            "similarity": float(best_sim),
        }
