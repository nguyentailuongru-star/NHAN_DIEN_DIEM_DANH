"""Custom FaceAnalysis wrapper - hỗ trợ buffalo_l trên insightface 0.2.x.

Insightface 0.2.x chỉ có model routing cho SCRFD (detection) và ArcFaceONNX (recognition).
buffalo_l gồm 5 model: det_10g, w600k_r50, 1k3d68, 2d106det, genderage.
Wrapper này load trực tiếp các model cần thiết (detection + recognition).
"""

import os
import glob
import numpy as np
import cv2
import onnxruntime
import logging

logger = logging.getLogger(__name__)


def _create_ort_session(model_path: str, ctx_id: int):
    """Tạo ONNXRuntime InferenceSession.

    - Nếu ctx_id >= 0: ưu tiên CUDAExecutionProvider nếu khả dụng.
    - Nếu CUDA không khả dụng / init lỗi: tự fallback về CPUExecutionProvider.
    """
    available = set(onnxruntime.get_available_providers())
    want_cuda = ctx_id >= 0 and "CUDAExecutionProvider" in available

    if ctx_id < 0:
        providers = ["CPUExecutionProvider"]
    elif want_cuda:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess_options = onnxruntime.SessionOptions()
    # Giảm spam warning của ORT (đặc biệt với SCRFD khi dùng input_size lớn).
    # 0=verbose, 1=info, 2=warning, 3=error, 4=fatal
    sess_options.log_severity_level = 3

    try:
        return onnxruntime.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    except Exception as e:
        # Thử fallback CPU nếu CUDA init fail
        if "CUDAExecutionProvider" in providers:
            logger.warning(
                "Không thể khởi tạo CUDAExecutionProvider cho %s (%s). Fallback về CPU.",
                model_path,
                str(e).strip(),
            )
            return onnxruntime.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        raise


class FaceDetector:
    """SCRFD face detector - load det_10g.onnx."""

    def __init__(self, model_path, ctx_id=0):
        self.session = _create_ort_session(model_path, ctx_id)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = (640, 640)
        self.fmc = 3  # feature map count
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.det_thresh = 0.5
        self.nms_thresh = 0.4

    def _distance2bbox(self, points, distance):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points, distance):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.extend([px, py])
        return np.stack(preds, axis=-1)

    def detect(self, img, input_size=None):
        if input_size is None:
            input_size = self.input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]

        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))

        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self._feat_stride_fpn):
            # det_10g.onnx thường trả output không có batch-dimension:
            # scores: (N,1), bboxes: (N,4), kps: (N,10)
            # (một số bản export khác có thể là (1,N,1)/(1,N,4)/(1,N,10)).
            scores = outputs[idx]
            bbox_preds = outputs[idx + self.fmc]
            kps_preds = outputs[idx + self.fmc * 2] if idx + self.fmc * 2 < len(outputs) else None

            scores = np.squeeze(scores)
            bbox_preds = np.squeeze(bbox_preds)
            if kps_preds is not None:
                kps_preds = np.squeeze(kps_preds)

            if scores.ndim != 1:
                scores = scores.reshape(-1)
            if bbox_preds.ndim != 2 or bbox_preds.shape[-1] != 4:
                bbox_preds = bbox_preds.reshape((-1, 4))
            bbox_preds = bbox_preds * stride

            if kps_preds is not None:
                if kps_preds.ndim != 2 or kps_preds.shape[-1] != 10:
                    kps_preds = kps_preds.reshape((-1, 10))
                kps_preds = kps_preds * stride

            height = input_size[1] // stride
            width = input_size[0] // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))

            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(scores >= self.det_thresh)[0]
            if len(pos_inds) == 0:
                continue

            bboxes = self._distance2bbox(anchor_centers[pos_inds], bbox_preds[pos_inds])
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes)

            if kps_preds is not None:
                kpss = self._distance2kps(anchor_centers[pos_inds], kps_preds[pos_inds])
                kpss_list.append(kpss)

        if len(scores_list) == 0:
            return np.empty((0, 5)), np.empty((0, 5, 2))

        scores = np.concatenate([np.asarray(s).reshape(-1) for s in scores_list], axis=0).ravel()
        bboxes = np.vstack(bboxes_list) / det_scale

        if len(kpss_list) > 0:
            kpss = np.vstack(kpss_list) / det_scale
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
        else:
            kpss = None

        # NMS
        pre_det = np.hstack([bboxes, scores[:, None]])
        keep = self._nms(pre_det, self.nms_thresh)
        det = pre_det[keep]

        if kpss is not None:
            kpss = kpss[keep]
        else:
            kpss = np.empty((det.shape[0], 5, 2))

        return det, kpss

    @staticmethod
    def _nms(dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


class FaceRecognizer:
    """ArcFace recognizer - load w600k_r50.onnx."""

    def __init__(self, model_path, ctx_id=0):
        self.session = _create_ort_session(model_path, ctx_id)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def get_embedding(self, img, kps):
        """Extract face embedding from aligned face."""
        aligned = self._align_face(img, kps)
        return self._embedding_from_aligned(aligned)

    def get_embedding_from_bbox(self, img, bbox, margin: float = 0.25):
        """Fallback: extract embedding from a bbox crop (no keypoints alignment).

        bbox: [x1, y1, x2, y2] in the coordinate system of img.
        """
        try:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in bbox]

            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            pad = float(margin) * max(bw, bh)

            x1 = int(max(0, np.floor(x1 - pad)))
            y1 = int(max(0, np.floor(y1 - pad)))
            x2 = int(min(w, np.ceil(x2 + pad)))
            y2 = int(min(h, np.ceil(y2 + pad)))

            if x2 <= x1 or y2 <= y1:
                return None

            crop = img[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                return None

            aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
            return self._embedding_from_aligned(aligned)
        except Exception:
            return None

    def _embedding_from_aligned(self, aligned_112_bgr: np.ndarray):
        blob = cv2.dnn.blobFromImage(
            aligned_112_bgr,
            1.0 / 127.5,
            (112, 112),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        output = self.session.run(None, {self.input_name: blob})[0]
        embedding = output.flatten()
        denom = np.linalg.norm(embedding)
        if denom > 0:
            embedding = embedding / denom
        return embedding

    @staticmethod
    def _align_face(img, kps):
        """Align face using 5 keypoints (ArcFace standard alignment)."""
        # ArcFace standard reference points for 112x112
        dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

        src_pts = np.float32(kps[:5])
        # Use similarity transform
        tform = cv2.estimateAffinePartial2D(src_pts, dst)[0]
        if tform is None:
            tform = cv2.getAffineTransform(src_pts[:3], dst[:3])
        aligned = cv2.warpAffine(img, tform, (112, 112), borderValue=0.0)
        return aligned


class Face:
    """Đối tượng lưu thông tin 1 khuôn mặt."""

    def __init__(self, bbox, kps, score, embedding=None):
        self.bbox = bbox
        self.kps = kps
        self.det_score = score
        self.embedding = embedding


class FaceAnalysisLocal:
    """Thay thế FaceAnalysis - load model buffalo_l trực tiếp."""

    def __init__(self, model_dir, ctx_id=0):
        det_path = os.path.join(model_dir, "det_10g.onnx")
        rec_path = os.path.join(model_dir, "w600k_r50.onnx")

        if not os.path.exists(det_path):
            raise FileNotFoundError(f"Model detection không tìm thấy: {det_path}")
        if not os.path.exists(rec_path):
            raise FileNotFoundError(f"Model recognition không tìm thấy: {rec_path}")

        logger.info("Loading detection model: %s", det_path)
        self.detector = FaceDetector(det_path, ctx_id)
        logger.info("Loading recognition model: %s", rec_path)
        self.recognizer = FaceRecognizer(rec_path, ctx_id)
        logger.info("Models loaded OK!")

    def prepare(self, ctx_id=0, det_size=(640, 640), det_thresh=0.5):
        self.detector.input_size = det_size
        self.detector.det_thresh = det_thresh

    def get(self, img):
        """Detect + recognize faces. Returns list of Face objects."""
        det, kpss = self.detector.detect(img, input_size=self.detector.input_size)
        faces = []
        for i in range(det.shape[0]):
            bbox = det[i, :4]
            score = det[i, 4]
            kps = kpss[i] if kpss is not None and i < len(kpss) else None

            embedding = None
            if kps is not None and len(kps) >= 5:
                embedding = self.recognizer.get_embedding(img, kps)

            face = Face(bbox=bbox, kps=kps, score=score, embedding=embedding)
            faces.append(face)

        return faces
