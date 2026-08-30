# DriveCV: High-Performance Autonomous Driving Perception, Tracking & ADAS Suite

A real-time dashcam computer vision suite combining **Classical Computer Vision** with **YOLOPv2 Deep Multi-Task Perception** for high-speed object tracking, kinematic state estimation, and Advanced Driver Assistance Systems (ADAS).

---

## 🌟 Key Highlights & Features

1. **Modular Package Architecture (`drivecv`)**:
   - Cleanly separated into `perception`, `tracking`, `adas`, `core`, and `ui` packages.
   - Comprehensive typing, dataclass configuration, and zero monolithic bloat.

2. **High-Speed Hybrid Multi-Object Tracking Engine**:
   - **Kinematic State Estimation**: 2D Constant Velocity Kalman Filter per tracked vehicle ($[x_c, y_c, s, r, \dot{x}_c, \dot{y}_c, \dot{s}]^T$).
   - **Inter-Frame Sparse Optical Flow**: Forward-backward Lucas-Kanade feature flow with adaptive CLAHE contrast enhancement for dark cars at night.
   - **Asynchronous Deep Neural Fusion**: Background YOLOPv2 inference worker with Hungarian bipartite association and zero-drift Kalman measurement corrections.
   - **Track Lifecycle**: Robust multi-state machine (`Tentative` $\to$ `Confirmed` $\to$ `Lost` $\to$ `Deleted`) preventing false tracks.

3. **Autonomous Driver Assistance Systems (ADAS)**:
   - **Lane Departure Warning (LDW)**: Analyzes vehicle lateral offset ($d_{lat}$) relative to host lane center and calculates Time-to-Lane-Crossing ($TLC$).
   - **Forward Collision Warning (FCW)**: Monocular ground-plane pinhole depth estimation, lead vehicle ego-corridor selection, closing speed calculation ($v_{rel}$), and Time-to-Collision ($TTC$) with multi-tier alerts (`Safe`, `Caution`, `Warning`, `Critical Brake`).

4. **Preserved Classical Lane & AR Drivable Corridor Tracker (160+ FPS)**:
   - Canny edge and Hough transform on lower road ROI with analytical vanishing point solver (guarantees boundary lines never cross).
   - Augmented reality drivable path (Electric Cyan) extending directly and safely to the lead vehicle.

5. **Panoptic HUD & Interactive Controls**:
   - Real-time telemetric HUD with FPS, object counts, LDW departure alerts, and FCW collision warnings.
   - Interactive mouse ROI target selection and keyboard controls.

---

## 📁 Package Structure

```
drive-cv/
├── weights/
│   └── YOLOPv2.onnx                # ONNX neural weights
├── 12838618_3840_2160_25fps.mp4    # Sample dashcam video
├── main.py                         # CLI application entrypoint
├── pyproject.toml                  # Package installation metadata
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── tests/                          # Automated test suite
│   ├── test_core.py                # Camera geometry & IoU tests
│   ├── test_tracking.py            # Kalman filter & track lifecycle tests
│   └── test_adas.py                # LDW & FCW safety module tests
└── drivecv/
    ├── __init__.py                 # Top-level exports
    ├── config.py                   # Dataclass configurations
    ├── types.py                    # Core data models and enums
    ├── pipeline.py                 # Master ADAS pipeline orchestrator
    ├── core/
    │   ├── geometry.py             # Pinhole camera model & 3D distance estimation
    │   └── math_utils.py           # IoU, cost matrices, smoothing helpers
    ├── perception/
    │   ├── yolopv2.py              # YOLOPv2 multi-task ONNX perception engine
    │   ├── lane_detector.py        # Classical non-crossing lane & path tracker
    │   ├── optical_flow.py         # Sparse Lucas-Kanade with CLAHE & F-B check
    │   └── async_detector.py       # Thread-safe asynchronous perception worker
    ├── tracking/
    │   ├── kalman.py               # 2D Constant Velocity Kalman Filter
    │   ├── association.py          # Bipartite Hungarian matching
    │   ├── track.py                # Track lifecycle & kinematic management
    │   └── multi_tracker.py        # Master multi-object tracker
    ├── adas/
    │   ├── ldw.py                  # Lane Departure Warning engine
    │   ├── fcw.py                  # Forward Collision Warning & TTC engine
    │   └── adas_manager.py         # Centralized ADAS supervisor
    └── ui/
        ├── hud.py                  # Telemetry HUD and alert banners
        ├── visualizer.py           # Panoptic scene renderer & AR corridor
        └── player.py               # Interactive player & event loop
```

---

## 🛠️ Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Running the Application

### Standard Interactive Run:
Auto-detects sample video and weights:
```bash
python main.py
```

### Enable Neural Segmentation Overlays:
```bash
python main.py --show-seg
```

### Record / Export Tracked Video:
```bash
python main.py --output output_adas_demo.mp4
```

### Headless Benchmark Mode:
```bash
python main.py --headless --max-frames 200
```

---

## 🐳 Docker Deployment Options

DriveCV includes dedicated Docker Compose builds for both host Linux workstations/laptops and target Nvidia Jetson edge devices.

### 1. Host Linux Machine Docker Deployment
Run on host PC/laptop with USB camera access `/dev/video0`:
```bash
docker compose up --build
```
Access the 3D Telemetry HUD at `http://localhost:5000`.

### 2. Nvidia Jetson Orin Nano Super Docker Deployment
Targeted deployment for Nvidia Jetson Orin Nano Super (ARM64 / JetPack) utilizing `nvcr.io/nvidia/pytorch:24.05-py3-igpu` with Jetson CUDA runtime and USB UVC camera access (`/dev/video0`):
```bash
docker compose -f docker-compose.jetson.yml up --build
```
Access the 3D Telemetry HUD at `http://<jetson-ip>:5000`.

### 3. Install Jetson Boot Systemd Service
To automatically start the DriveCV Jetson container on boot as a system service:
```bash
sudo bash scripts/install_jetson_service.sh
```
**Service Control Commands:**
```bash
sudo systemctl status drivecv-jetson   # Check status
sudo systemctl start drivecv-jetson    # Start service
sudo systemctl stop drivecv-jetson     # Stop service
sudo journalctl -u drivecv-jetson -f   # View live logs
```
To uninstall: `sudo bash scripts/uninstall_jetson_service.sh`

---

## 📹 Input Source Control: Live Camera vs. Demo Video

By default, DriveCV opens the **Live USB UVC Camera** (`/dev/video0` or device `0`). If a physical camera is not attached, it automatically falls back to the recorded demo video loop.

- **Web App UI Controls**: Toggle between **Live Camera** (🔴) and **Demo Video** (🎬) instantly using the top navigation bar toggle or the System Controls drawer.
- **CLI Options**:
  ```bash
  # Force Live USB Camera mode (Default)
  python main.py --source camera --camera-device /dev/video0

  # Force Demo Video mode
  python main.py --source video --video 12838618_3840_2160_25fps.mp4
  ```

---

## 🎮 Keyboard Controls

| Key | Action | Description |
| :--- | :--- | :--- |
| **<kbd>Space</kbd>** | **Play / Pause** | Toggle video playback. |
| **`t`** | **Trigger YOLO** | Force immediate asynchronous YOLOPv2 perception pass. |
| **`a`** | **Toggle Auto-Scheduler** | Toggle automated background neural cadence. |
| **`s`** or **`r`** | **Select Target ROI** | Drag mouse to draw a box around any vehicle to lock tracking ($C=1.0$). |
| **`v`** | **Cycle Vis Mode** | Cycle between `ALL` (boxes + vectors + keypoints) $\to$ `DET_ONLY` $\to$ `MINIMAL`. |
| **`c`** | **Clear Objects** | Clear all active tracked vehicles. |
| **`d`** | **Step Frame** | Step forward 1 frame when paused. |
| **`q`** or <kbd>Esc</kbd> | **Quit** | Exit the application cleanly. |

---

## 🧪 Running Unit Tests

Run the complete test suite:
```bash
python3 -m unittest discover -s tests/ -p "test_*.py"
```
