# 🚗 Virtual Vahana: Advanced ADAS & Semi-Autonomous Driving Stack

<p align="center">
  <b>Team:</b> 404botnotfound &nbsp;|&nbsp; <b>Competition:</b> Student ADAS & Semi-Autonomous Driving Challenge (Round 2)
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/CARLA-Simulator-1d2021?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Architecture-Perception→Planning→Control-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge"/>
</p>

---

## 🧭 Overview

**Virtual Vahana** is a modular, high-performance **ADAS + Semi-Autonomous Driving system** built on the **CARLA Simulator**.

It implements a strict and scalable:

> 🧠 **Perception → Planning → Control pipeline**

with **sensor fusion**, **active safety**, and **V2X-assisted decision making** — bridging the gap between traditional ADAS and full autonomy.

---

## ✨ Key Features

### 👁️ Perception Stack
- ⚡ **YOLOv8** → Real-time object detection (vehicles, pedestrians, traffic signs)
- 🛣️ **YOLOP** → Lane detection + drivable area segmentation
- 🎯 High FPS inference with GPU acceleration

### 🔗 Sensor Fusion
- 📡 32-channel **LiDAR + Camera Projection**
- 🧮 Accurate 3D → 2D spatial mapping
- 🚫 Ground noise filtering using Z-axis constraints (-2.2m to 1.0m)
- 📍 Provides **true spatial ground truth**

### 🛑 Active Safety System (AEB)
- 🚨 Dynamic **Panic Bubble** based on velocity
- 👀 Blindspot detection using LiDAR
- 🧠 Vision fallback for occluded pedestrians
- ⚡ Real-time braking decisions

### 🌐 V2X Traffic Compliance
- 🚦 Vision detects signals/signs
- 📡 Ground truth from CARLA ensures:
  - Traffic light compliance
  - Speed limit adherence
- 🏙️ Simulates **smart city infrastructure**

### 🗺️ Planning & Control

#### Global Planning
- 🧭 A* pathfinding for optimal routing

#### Local Planning
- 🛣️ **Frenet Trajectory Generation**
- 🚗 Smooth lane following
- 🔄 Intelligent overtaking (same direction)

#### Control System
- 🎯 **Stanley Controller** → Lateral control
- ⚙️ **PID Controller** → Longitudinal control

---

## 🧠 System Architecture

```
Sensors (Camera + LiDAR)
↓
Perception
↓
Sensor Fusion
↓
Safety & V2X Layer
↓
Planning Layer
(Global + Local)
↓
Control
↓
Vehicle
```

---

## 🛠️ Installation & Setup

### 1️⃣ CARLA Simulator

- Use **CARLA 0.9.16**
- Download: https://github.com/carla-simulator/carla/releases

```bash
./CarlaUE4.sh -quality-level=Epic
# Windows: CarlaUE4.exe
```

### 2️⃣ Clone Repository

```bash
git clone https://github.com/sailesh2408/Virtual_Vahana_Round-2_404botnotfound.git
cd Virtual_Vahana_Round-2_404botnotfound
```

### 3️⃣ Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Start CARLA

```bash
./CarlaUE4.sh
```

### Step 2: Run Autonomous Stack

```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| 🖱️ Click (Minimap) | Set destination |
| R | Reset vehicle |
| Q | Quit simulation |

---

## 📂 Project Structure

```text
├── main.py
├── core/
│   ├── perception.py
│   ├── fusion.py
│   ├── safety.py
│   ├── global_planner.py
│   ├── local_planner.py
│   ├── control.py
│   └── hud.py
├── utils/
│   └── carla_utils.py
├── models/
│   └── yolov8s.pt
└── requirements.txt
```

---

## 📦 Requirements

```txt
ultralytics
torch
torchvision
numpy
opencv-python
pillow
simple-pid
```

---

## 📊 Highlights 
✅ True **Camera-LiDAR Fusion**  
✅ Real-time **AEB with fallback logic**  
✅ Hybrid **V2X + Vision compliance**  
✅ Smooth **Frenet trajectory planning**  
✅ Industry-grade **modular architecture**

---

## 📚 References

1. **CARLA Simulator**  
   Dosovitskiy et al., CoRL 2017  
   https://carla.org/

2. **YOLOv8 – Ultralytics**  
   https://github.com/ultralytics/ultralytics

3. **YOLOP**  
   https://github.com/hustvl/YOLOP

4. **Simple PID**  
   https://github.com/m-lundberg/simple-pid

---

## 🏁 Future Improvements

- 🔥 Multi-agent traffic prediction (Graph Neural Networks)
- 🧠 End-to-end learning integration
- 🌍 Real-world dataset adaptation (KITTI / nuScenes)
- ⚡ Performance optimization (TensorRT)

---

## 👨‍💻 Team

**404botnotfound**

> Built with precision, performance, and a bit of madness 🚀

---

## ⭐ Final Note

This project is designed to **mimic real-world ADAS systems** while maintaining research-level flexibility — making it both **competition-ready and scalable**.
