

<h3 align="center">AI-Based Worker Safety Monitoring System</h3>

<p align="center">
  Computer Vision • YOLO26 • Worker Detection • Multi-Object Tracking • PPE Detection • Safety Analytics
</p>

---

## 📌 Project Overview

> The **AI-Based Worker Safety Monitoring System** is an AI-powered computer vision application designed to monitor workplace safety by detecting workers and identifying their Personal Protective Equipment (PPE) using video footage.

> The system detects workers, safety helmets, safety vests, and other supported PPE items. It tracks individuals across consecutive video frames using unique tracking IDs and analyzes whether workers are following the required safety guidelines.

> By combining YOLO26 object detection, ByteTrack multi-object tracking, and PPE association techniques, the application provides an automated approach to workplace safety monitoring.

> The system processes video footage, generates annotated visual results, and displays safety-related information and compliance statistics to help improve workplace safety awareness.

## ✨ Key Features

- Detects workers in video streams using a custom-trained YOLO26 model
- Identifies safety helmets, safety vests, and other supported PPE classes
- Tracks workers across consecutive frames using unique tracking IDs
- Maintains worker identities using multi-object tracking
- Associates detected PPE items with individual workers
- Analyzes PPE compliance for each tracked worker
- Displays detection bounding boxes, labels, and tracking IDs
- Provides live worker monitoring and safety statistics
- Supports real-time video analysis, depending on hardware and processing performance
- Helps identify potential PPE violations in industrial environments

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| YOLO26 | Custom-trained object detection |
| Ultralytics | YOLO model framework |
| ByteTrack | Multi-object tracking |
| OpenCV | Video processing and visualization |
| NumPy | Numerical computation |
| Computer Vision | Worker and PPE identification |
| Deep Learning | AI-based object detection |
| Video Analytics | Workplace safety monitoring |

---

## 🧠 How it works

1. **Video input**: The system receives video footage from a camera or a prerecorded video file.
2. **Frame processing**: OpenCV extracts and processes individual video frames for analysis.
3. **Object detection**: The custom-trained YOLO26 model detects workers, helmets, safety vests, and other supported PPE objects.
4. **Worker tracking**: ByteTrack associates detections across consecutive frames and maintains tracking IDs for individual workers.
5. **PPE association**: Detected protective equipment is associated with workers using the project's PPE association logic.
6. **Compliance analysis**: The system evaluates whether each tracked worker has the required safety equipment based on configured rules.
7. **Visualization + analytics**: Detection results, tracking IDs, PPE status, and safety statistics are displayed through the monitoring interface.

## 📦 Installation

```bash
git clone https://github.com/<your-username>/AI-Based-Worker-Safety-Monitoring-System.git
cd AI-Based-Worker-Safety-Monitoring-System
pip install -r requirements.txt
```

`requirements.txt` (typical dependencies):

```text
ultralytics
opencv-python
numpy
```

Install any additional dependencies required by the actual project implementation.

The model weights must be available at the configured model path. If the application uses a custom-trained model, ensure that the corresponding `.pt` file is provided.

## ▶️ Usage

Run the main application using the Python script provided in the project.

```bash
python <main_script>.py
```

If the application supports video input through command-line arguments:

```bash
python <main_script>.py --video path/to/input_video.mp4
```

### Processing workflow

```text
Video Input
    |
    v
Frame Processing (OpenCV)
    |
    v
YOLO26 Object Detection
    |
    v
Worker and PPE Identification
    |
    v
ByteTrack Worker Tracking
    |
    v
PPE-to-Worker Association
    |
    v
PPE Compliance Analysis
    |
    v
Safety Monitoring Dashboard
```

*Note: Replace the example execution command with the actual entry-point filename and supported arguments in your repository.*

---

## 📊 Monitoring Information

The system can provide the following safety-related information, depending on the implemented dashboard:

| Metric | Description |
|---|---|
| Worker Count | Number of workers detected in the video |
| Tracking IDs | Unique identifiers assigned to tracked workers |
| Helmet Detection | Identifies detected safety helmets |
| Safety Vest Detection | Identifies detected safety vests |
| PPE Status | Indicates the detected protective equipment associated with a worker |
| Compliance Status | Evaluates worker PPE compliance based on configured requirements |
| FPS | Indicates video processing speed, if displayed |

---

## 🦺 PPE Compliance Monitoring

Personal Protective Equipment (PPE) is essential for protecting workers from workplace hazards.

The system focuses on identifying safety equipment such as:

- Safety Helmets
- Safety Vests
- Protective Clothing
- Other PPE categories supported by the trained model

Each detected worker is evaluated against the configured PPE requirements.

Workers with all required equipment detected can be classified as compliant, while workers with missing required equipment can be flagged for further inspection.

---

## 📁 Project Structure

Example project structure:

```text
AI-Based-Worker-Safety-Monitoring-System/
│
├── models/
│   └── best.pt
│
├── videos/
│   └── input_video.mp4
│
├── outputs/
│   └── processed_video.mp4
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: Update the example filenames and folders to match the actual repository structure.*

---

## ⚙️ Configuration

The system configuration may include the following parameters, depending on the implementation:

| Parameter | Meaning |
|---|---|
| Model Weights | Path to the trained YOLO26 model |
| Confidence Threshold | Minimum confidence required for object detection |
| Image Size | Input resolution used for model inference |
| Tracking Configuration | Settings used for maintaining worker tracking IDs |
| Video Source | Input video file or supported camera source |
| PPE Classes | Supported safety equipment categories |
| Compliance Rules | Required PPE conditions for worker evaluation |

---

## ⚠️ Limitations

- Detection accuracy depends on camera angle, lighting conditions, and video quality.
- Workers or PPE items may be missed because of occlusion.
- Small or distant objects may be difficult to detect.
- Tracking IDs may change when workers disappear from the camera view.
- PPE association may become challenging in crowded environments.
- Processing speed depends on hardware, video resolution, and model configuration.
- The system is intended to assist safety monitoring and does not replace professional workplace safety supervision.

---

## 🚀 Future Work

- Automated alerts for workers missing required PPE
- Integration with CCTV and live IP camera streams
- Worker-specific safety compliance reports
- Historical PPE violation tracking
- Exporting monitoring statistics to CSV and PDF
- Additional PPE detection classes such as gloves, goggles, and safety shoes
- Improved worker tracking under occlusion
- Web-based safety monitoring dashboard
- Multi-camera workplace monitoring
- Advanced workplace safety analytics and reporting

---

## 📈 Performance Evaluation

The system can be evaluated using the following performance metrics:

- Detection Precision
- Detection Recall
- Mean Average Precision (mAP)
- Frames Per Second (FPS)
- Tracking ID consistency
- PPE compliance classification accuracy

Actual performance values should be added after testing the system with a labeled dataset.


---

## 🙏 Acknowledgements

Special thanks to the developers and contributors of the open-source computer vision frameworks and libraries used in this project, including Ultralytics YOLO, OpenCV, and ByteTrack.

---

<h3 align="center">🦺 AI-Based Worker Safety Monitoring System</h3>
<p align="center">Making Workplaces Safer Through Artificial Intelligence and Computer Vision</p>
