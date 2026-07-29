import cv2
import numpy as np


def drawing_person(frame, tracked_persons):
    """Kişileri, baret/yelek durumlarını ve ihlal renklerini çerçeve üzerine çizer."""
    for person in tracked_persons:
        x1, y1, x2, y2 = person["bbox"]
        tracker_id = person["tracker_id"]
        has_helmet = person.get("has_helmet", False)
        has_vest = person.get("has_vest", False)

        # İhlal durumu varsa KIRMIZI/TURUNCU, yoksa YEŞİL
        if has_helmet and has_vest:
            color = (0, 255, 0)  # Yeşil
        else:
            color = (0, 0, 255)  # Kırmızı (İhlal/Eksik ekipman)

        # Kişi dikdörtgeni
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Metin Etiketleri
        helmet_str = "Helmet: Present" if has_helmet else "Helmet: Missing"
        vest_str = "Vest: Present" if has_vest else "Vest: Missing"

        labels = [
            f"ID: {tracker_id}",
            helmet_str,
            vest_str
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 22

        # Arka plan kutusu boyutları
        max_width = 0
        for l in labels:
            (w, h), _ = cv2.getTextSize(l, font, font_scale, thickness)
            if w > max_width:
                max_width = w

        total_height = line_height * len(labels)
        bg_y1 = max(0, y1 - total_height - 10)
        bg_y2 = bg_y1 + total_height + 8
        bg_x1 = x1
        bg_x2 = x1 + max_width + 12

        # Yazı arkasına şeffaf koyu zemin
        sub_img = frame[bg_y1:bg_y2, bg_x1:bg_x2]
        if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
            black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.3, black_rect, 0.7, 0)
            frame[bg_y1:bg_y2, bg_x1:bg_x2] = res

        # Metinleri çiz
        for idx, text in enumerate(labels):
            ty = bg_y1 + (idx + 1) * line_height - 4
            txt_color = (255, 255, 255)
            if "Missing" in text:
                txt_color = (50, 50, 255)
            elif "Present" in text:
                txt_color = (50, 255, 50)
            cv2.putText(frame, text, (bg_x1 + 6, ty), font, font_scale, txt_color, thickness)

    return frame


def draw_osd_panel(frame, detected_count, helmet_violations, vest_violations, fps):
    """Ekranın sol üst köşesine genel özet bilgi paneli (OSD) çizer."""
    panel_x1, panel_y1 = 15, 15
    panel_w, panel_h = 260, 125
    panel_x2, panel_y2 = panel_x1 + panel_w, panel_y1 + panel_h

    # Şeffaf zemin
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 255, 255), 2)

    lines = [
        f"Detected Persons: {detected_count}",
        f"Helmet Violations: {helmet_violations}",
        f"Vest Violations: {vest_violations}",
        f"FPS: {fps:.1f}"
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    start_y = panel_y1 + 25

    for i, line in enumerate(lines):
        color = (255, 255, 255)
        if "Helmet Violations" in line and helmet_violations > 0:
            color = (100, 100, 255)
        elif "Vest Violations" in line and vest_violations > 0:
            color = (100, 100, 255)

        cv2.putText(frame, line, (panel_x1 + 12, start_y + (i * 24)), font, font_scale, color, thickness)

    return frame

