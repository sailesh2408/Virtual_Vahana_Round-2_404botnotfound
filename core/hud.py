import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class ADASHUD:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.font_path = os.path.join(self.base_dir, "SF-Pro.ttf")

    def _draw_modern_text(self, frame, text, x, y, font_size, bgr_color):
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((x, y - font_size), text, font=font, fill=(bgr_color[2], bgr_color[1], bgr_color[0]))
        frame[:] = np.array(img_pil)

    def _draw_rounded_glass_panel(self, frame, x, y, w, h, radius=35, alpha=0.35, blur_kernel=(51, 51), panel_color=(255, 255, 255)):
        h_frame, w_frame = frame.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(full_mask, (radius, radius), radius, 255, -1, cv2.LINE_AA)
        cv2.circle(full_mask, (w - radius, radius), radius, 255, -1, cv2.LINE_AA)
        cv2.circle(full_mask, (radius, h - radius), radius, 255, -1, cv2.LINE_AA)
        cv2.circle(full_mask, (w - radius, h - radius), radius, 255, -1, cv2.LINE_AA)
        cv2.rectangle(full_mask, (radius, 0), (w - radius, h), 255, -1)
        cv2.rectangle(full_mask, (0, radius), (w, h - radius), 255, -1)
        x1, x2, y1, y2 = max(0, x), min(w_frame, x + w), max(0, y), min(h_frame, y + h)
        if x1 >= x2 or y1 >= y2: return
        roi = frame[y1:y2, x1:x2]
        visible_mask = full_mask[y1-y:y2-y, x1-x:x2-x]
        mask_3d = cv2.cvtColor(visible_mask, cv2.COLOR_GRAY2BGR) / 255.0
        blurred_roi = cv2.GaussianBlur(roi, blur_kernel, 0)
        glass_effect = cv2.addWeighted(blurred_roi, 0.65, np.full_like(roi, panel_color), 0.35, 0)
        frame[y1:y2, x1:x2] = (glass_effect * mask_3d + roi * (1.0 - mask_3d)).astype(np.uint8)
        cv2.drawContours(frame, [c + [x1, y1] for c in cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]], -1, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_telemetry(self, frame, x, y, speed_kmh, target_kmh, steer):
        self._draw_rounded_glass_panel(frame, x, y, 300, 150, radius=40)
        self._draw_modern_text(frame, f"{int(speed_kmh)} / {int(target_kmh)}", x + 30, y + 60, 32, (40, 40, 40))
        self._draw_modern_text(frame, "KM/H", x + 210, y + 55, 14, (60, 60, 60))
        steer_col = (0, 150, 0) if abs(steer) < 0.1 else (0, 100, 255)
        self._draw_modern_text(frame, f"STEER: {steer:.2f}", x + 30, y + 100, 18, steer_col)
        cv2.rectangle(frame, (x + 30, y + 120), (x + 270, y + 128), (200, 200, 200), -1)
        cv2.rectangle(frame, (x + 150, y + 118), (x + 150 + int(steer*120), y + 130), steer_col, -1)

    def draw_status_bar(self, frame, x, y, status_text, status_color, aeb_active=False):
        panel_tint = (0, 0, 255) if aeb_active else (255, 255, 255)
        self._draw_rounded_glass_panel(frame, x, y, 380, 80, radius=35, panel_color=panel_tint)
        self._draw_modern_text(frame, status_text, x + 40, y + 50, 22, status_color)

    def draw_confidence_meter(self, frame, x, y, trust_score):
        self._draw_rounded_glass_panel(frame, x, y, 280, 80, radius=35)
        color = (0, 150, 0) if trust_score > 80 else (0, 0, 200)
        self._draw_modern_text(frame, f"TRUST: {trust_score:.1f}%", x + 30, y + 45, 16, color)
        cv2.rectangle(frame, (x + 30, y + 58), (x + 250, y + 66), (200, 200, 200), -1) 
        cv2.rectangle(frame, (x + 30, y + 58), (x + 30 + int((trust_score/100)*220), y + 66), color, -1)

    def draw_sidebar_widgets(self, frame, frenet_img, ai_vision_img):
        self._draw_rounded_glass_panel(frame, 950, 20, 350, 350, radius=35)
        frame[50:340, 980:1270] = cv2.resize(frenet_img, (290, 290))
        self._draw_rounded_glass_panel(frame, 950, 390, 700, 550, radius=45)
        frame[440:890, 1000:1600] = cv2.resize(ai_vision_img, (600, 450))
        self._draw_modern_text(frame, "AI PERCEPTION FEED", 970, 425, 18, (255, 0, 255))

    def draw_minimap(self, frame, minimap_img, heading_deg=0, is_maximized=False):
        size = (650, 600) if is_maximized else (290, 290)
        m_img = cv2.resize(minimap_img, size)
        
        # Heading Arrow Logic
        cx, cy = size[0] // 2, size[1] // 2
        angle_rad = np.radians(heading_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        length, width = (30, 16) if is_maximized else (15, 8)
        pts = np.array([[0, -length], [-width, length], [0, length // 2], [width, length]])
        rot_pts = np.zeros_like(pts)
        rot_pts[:, 0] = pts[:, 0] * cos_a - pts[:, 1] * sin_a + cx
        rot_pts[:, 1] = pts[:, 0] * sin_a + pts[:, 1] * cos_a + cy
        cv2.fillPoly(m_img, [rot_pts.astype(np.int32)], (255, 120, 0)) # BGR: Blue Arrow
        
        # Compass Corner
        nx, ny = size[0] - 30, 30
        cv2.circle(m_img, (nx, ny), 15, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.putText(m_img, "N", (nx - 5, ny + 5), 0, 0.4, (255, 255, 255), 1)

        if is_maximized:
            self._draw_rounded_glass_panel(frame, 100, 100, 750, 750, radius=60)
            frame[200:800, 150:800] = m_img
            self._draw_modern_text(frame, "CLOSE MAP [X]", 350, 830, 24, (255, 255, 255))
        else:
            self._draw_rounded_glass_panel(frame, 1320, 20, 350, 350, radius=35)
            frame[50:340, 1350:1640] = m_img
            self._draw_modern_text(frame, "TAP TO ENLARGE", 1335, 45, 12, (200, 200, 200))