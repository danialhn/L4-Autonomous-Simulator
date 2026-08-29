
# 🚗 Level 4 Autonomous Vehicle Decision Kernel

An advanced, multi-modal Level 4 Autonomous Driving simulator built from scratch in Python. This project demonstrates high-performance sensor fusion, exact-clearance BSM (Blind Spot Monitoring), and a robust Finite State Machine (FSM) for complex highway maneuvers like timed double-overtakes.

## 🌟 Live Demo
![L4 Simulator Demo](output_l4_hq.gif)
<img width="700" height="394" alt="output_l4_hq_compressed" src="https://github.com/user-attachments/assets/9cec44d7-b09b-4653-b9fe-4b181e61c3b6" />

*(Watch the full high-quality rendering in the `.mp4` file included in this repository).*

## 🧠 Core Features

* **Zero-Delay CVW Radar Fusion:** Physical bounding-box intersection math for zero-latency clearance detection (Closing Vehicle Warning).
* **Dynamic Multi-Stage Overtaking (FSM):** Intelligent agent capable of evaluating blind spots, holding for fast-approaching rear traffic (up to 30m range), and executing seamless multi-lane overtakes.
* **Live Engineering HUD:**
    * **Telemetry Oscilloscope:** Real-time velocity and Stanley steering graphs.
    * **Sensor Matrix:** Live status of 77GHz Front/Rear Radars, 24GHz Corner Radars, and Solid-State LiDAR.
    * **Decision Kernel:** Real-time visibility into the FSM logic and FSD confidence scoring.

## 🛠️ Tech Stack
* **Python 3.10+**
* `pygame`: For rendering the 3D Neural Vector Space and BEV (Bird's Eye View) radar.
* `numpy` & `math`: For complex kinematic physics, lateral spring-damper controllers, and 3D perspective projections.
* `imageio` & `moviepy`: For rendering telemetry buffers into high-definition `.mp4` and `.gif` formats.



