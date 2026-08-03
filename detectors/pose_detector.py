import os
import torch
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    """
    YOLOv11-Pose Tabanlı Anatomik Duruş ve Nirengi Noktası Tespiti.
    17 COCO Keypoint Noktası:
    0: Burun, 3/4: Kulaklar, 5/6: Omuzlar, 11/12: Kalçalar
    - İşçi eğilse, çömelse veya yan dönse dahi kafa ve gövde bölgesini milimetrik tespit eder.
    """
    def __init__(self, model_path="models/yolo11n-pose.pt", conf_threshold=0.30):
        if not os.path.exists(model_path):
            model_path = "yolo11n-pose.pt"
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def predict(self, frame):
        """
        Karedeki tüm kişilerin anatomi (keypoints) haritasını döndürür.
        Return: list of dict -> [{ "person_box": [x1,y1,x2,y2], "keypoints": np.array(17, 2), "head_box": [...], "torso_box": [...] }]
        """
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        pose_data = []

        if not results or len(results) == 0:
            return pose_data

        r = results[0]
        if r.boxes is None or r.keypoints is None:
            return pose_data

        boxes = r.boxes.xyxy.cpu().numpy()
        keypoints_xy = r.keypoints.xy.cpu().numpy()  # Shape: (N, 17, 2)
        keypoints_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

        for idx, box in enumerate(boxes):
            kpts = keypoints_xy[idx]  # (17, 2)
            kpt_confs = keypoints_conf[idx] if keypoints_conf is not None else np.ones(17)

            px1, py1, px2, py2 = box
            p_w = px2 - px1
            p_h = py2 - py1

            # ── 1. Kafa Bölgesi Hesabı (Anatomik) ───────────────────────────
            # Burun(0), Kulaklar(3,4), Omuzlar(5,6)
            nose = kpts[0] if kpt_confs[0] > 0.3 else None
            l_ear = kpts[3] if kpt_confs[3] > 0.3 else None
            r_ear = kpts[4] if kpt_confs[4] > 0.3 else None
            l_shoulder = kpts[5] if kpt_confs[5] > 0.3 else None
            r_shoulder = kpts[6] if kpt_confs[6] > 0.3 else None

            # Omuz genişliği
            shoulder_width = p_w * 0.4
            if l_shoulder is not None and r_shoulder is not None:
                shoulder_width = abs(l_shoulder[0] - r_shoulder[0])

            # Kafa merkezi
            head_pts = [pt for pt in [nose, l_ear, r_ear] if pt is not None]
            if len(head_pts) > 0:
                head_cx = np.mean([pt[0] for pt in head_pts])
                head_cy = np.mean([pt[1] for pt in head_pts])
            elif l_shoulder is not None and r_shoulder is not None:
                head_cx = (l_shoulder[0] + r_shoulder[0]) / 2.0
                head_cy = min(l_shoulder[1], r_shoulder[1]) - (shoulder_width * 0.6)
            else:
                head_cx = (px1 + px2) / 2.0
                head_cy = py1 + (p_h * 0.20)

            head_radius = max(20, shoulder_width * 0.7)
            head_box = [
                max(px1 - 20, head_cx - head_radius),
                max(py1 - 20, head_cy - head_radius),
                min(px2 + 20, head_cx + head_radius),
                min(py2 + 20, head_cy + head_radius * 1.2)
            ]

            # ── 2. Gövde Bölgesi Hesabı (Yelek Bölgesi) ─────────────────────
            l_hip = kpts[11] if kpt_confs[11] > 0.3 else None
            r_hip = kpts[12] if kpt_confs[12] > 0.3 else None

            if l_shoulder is not None and r_shoulder is not None:
                torso_y1 = min(l_shoulder[1], r_shoulder[1])
                torso_x1 = min(l_shoulder[0], r_shoulder[0])
                torso_x2 = max(l_shoulder[0], r_shoulder[0])
            else:
                torso_y1 = py1 + (p_h * 0.15)
                torso_x1 = px1
                torso_x2 = px2

            if l_hip is not None and r_hip is not None:
                torso_y2 = max(l_hip[1], r_hip[1])
            else:
                torso_y2 = py1 + (p_h * 0.75)

            margin_torso = max(15, p_w * 0.15)
            torso_box = [
                max(px1 - margin_torso, torso_x1 - margin_torso),
                max(py1, torso_y1 - 10),
                min(px2 + margin_torso, torso_x2 + margin_torso),
                min(py2, torso_y2 + 10)
            ]

            head_pts_valid = [c for c in [kpt_confs[0], kpt_confs[3], kpt_confs[4]] if c > 0.5]
            torso_pts_valid = [c for c in [kpt_confs[5], kpt_confs[6]] if c > 0.5]

            pose_data.append({
                "person_box": [float(px1), float(py1), float(px2), float(py2)],
                "keypoints": kpts,
                "keypoint_confs": kpt_confs,
                "has_valid_head_keypoints": len(head_pts_valid) > 0,
                "has_valid_torso_keypoints": len(torso_pts_valid) >= 2,
                "head_box": [float(x) for x in head_box],
                "torso_box": [float(x) for x in torso_box]
            })

        return pose_data
