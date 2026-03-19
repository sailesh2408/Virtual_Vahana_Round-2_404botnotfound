import sys
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(__file__)

class PerceptionModule:
    def __init__(self, yolo_path='models/yolov8s.pt'):
        yolo_abs = os.path.abspath(os.path.join(current_dir, '..', yolo_path))
        print(f"Loading YOLOv8 model from {yolo_abs}...")
        self.model = YOLO(yolo_abs)
        self.target_classes = {
            0: 'pedestrian', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 13: 'bench'
        }

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        print("Loading YOLOP Segmentation Model via PyTorch Hub...")
        try:
            self.yolop = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
            self.yolop.to(self.device)
            self.yolop.eval()
            self.yolop_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.yolop_available = True
            print("SUCCESS: YOLOP Segmentation Model loaded.")
        except Exception as e:
            print(f"CRITICAL WARNING: YOLOP failed to load. Error: {e}")
            self.yolop_available = False

        self.frame_counter = 0

    def process_frame(self, frame, run_lanenet=False):
        annotated_frame = frame.copy()
        self.frame_counter += 1
        h, w = frame.shape[:2]
        
        # ---------------------------------------------------------
        # 1. YOLOv8 OBJECT DETECTION
        # ---------------------------------------------------------
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if class_id in self.target_classes and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{self.target_classes[class_id]} {conf:.2f}"
                detections.append({'class': self.target_classes[class_id], 'bbox': (x1, y1, x2, y2), 'confidence': conf})
                color = (0, 0, 255) if class_id == 0 else (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(15, y1 - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        extracted_lanes = [] 
        
        # ---------------------------------------------------------
        # 2. YOLOP PIXEL SEGMENTATION
        # ---------------------------------------------------------
        if self.yolop_available and run_lanenet:
            extracted_lanes.append("YOLOP_ACTIVE") 

            img_resized = cv2.resize(frame, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = self.yolop_transforms(img_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.yolop(img_tensor)
                if len(out) == 3:
                    _, da_seg_out, ll_seg_out = out
                else:
                    da_seg_out, ll_seg_out = out[1], out[2]
                
            # 1. DRIVABLE AREA (Green Carpet)
            da_mask = torch.argmax(da_seg_out, dim=1).squeeze().cpu().numpy()
            da_mask_resized = cv2.resize(da_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            green_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
            green_mask[da_mask_resized == 1] = [0, 200, 0] 
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, green_mask, 0.25, 0)
            
            # 2. LANE LINES (Crisp 50% Threshold)
            lane_probs = torch.nn.functional.softmax(ll_seg_out, dim=1)[0, 1, :, :].cpu().numpy()
            ll_mask = (lane_probs > 0.50).astype(np.uint8)
            
            # ==========================================
            # THE CROSSWALK ERASER
            # Uses a vertical kernel to destroy horizontal zebra stripes!
            # ==========================================
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 9))
            ll_mask = cv2.morphologyEx(ll_mask, cv2.MORPH_OPEN, kernel)
            
            ll_mask_resized = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Draw solid magenta
            annotated_frame[ll_mask_resized == 1] = [255, 0, 255]

            if self.frame_counter % 30 == 0:
                print(f"[VISION DEBUG] YOLOP Drivable Area Active. Max Lane Confidence: {np.max(lane_probs)*100:.1f}%")

        return annotated_frame, detections, extracted_lanes