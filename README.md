

# Kalman Filter Person Tracking Project

This project implements object tracking using YOLOv3 for object detection and a Kalman Filter for smooth tracking. The project is designed to track a single object in real-time, with the Kalman Filter predicting the object's position even when detections are not available due to lag.

## Features
- Real-time object detection using YOLOv3
- Kalman Filter for smooth and continuous tracking of a single object
- Display of bounding boxes and labels for detected objects
- Smoother tracking between frames using Kalman Filter predictions

## Requirements

Make sure you have Python installed (preferably Python 3.9+). All required dependencies are listed in the `requirements.txt` file.

## Installation Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/ieeeswinburne/kalman-filter.git
   cd kalman-filter
   ```

2. Download YOLOv3 files:
   You need to manually download the YOLOv3 model files and COCO labels as they are not included in this repository:

   - YOLOv3 weights: [https://www.kaggle.com/datasets/shivam316/yolov3-weights](https://www.kaggle.com/datasets/shivam316/yolov3-weights)
   - YOLOv3.cfg: [https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - COCO labels: [https://github.com/pjreddie/darknet/blob/master/data/coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

   Place the downloaded files in the parent directory of your script (e.g., `camera.py`).

3. Set up a Conda virtual environment (recommended):

   If you are using Conda for environment management, create a new environment:

   ```bash
   conda create --name kalman-filter-env python=3.9
   conda activate kalman-filter-env
   ```

4. Install dependencies:

   Once the environment is activated, install the required packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the project:

   After installing all dependencies, you can run the project using:

   ```bash
   python camera.py
   ```

   This will launch the real-time object detection and tracking using your webcam.

## Troubleshooting

- Ensure that all dependencies are installed correctly by running `pip list` or `conda list` to verify the installed packages.
- If you encounter issues with missing packages not covered in `requirements.txt`, make sure to install them via `pip` or `conda`.

## Additional Information
- Make sure your webcam is accessible for the project to capture real-time video.
- The `requirements.txt` file is intended for use with `pip`. If you are using Conda for package management, you may need to install some dependencies separately using Conda.
