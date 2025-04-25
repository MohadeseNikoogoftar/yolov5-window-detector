#  YOLOv5 Window Detector

## A deep learning project to detect and count various types of windows (square, rectangular, circular, etc.) on buildings using a custom-trained YOLOv5 model.
---

## 🗂 Dataset Preparation

1. Collected images of building facades with visible windows.
2. Annotated them using [MakeSense.ai](https://www.makesense.ai/):
   - Selected the YOLO format for annotation.
   - Labeled each window manually and exported the `.txt` label files.
3. Saved all images in `images_all/` and label files in `labels_all/`.
4. Used a custom Python script to:
   - Match each image to its annotation file.
   - Split the dataset into `train/` and `val/` sets (80/20).
   - Save the split data to the folder structure:
     ```
     dataset/
       ├── images/
       │   ├── train/
       │   └── val/
       ├── labels/
       │   ├── train/
       │   └── val/
     ```
   - Generate the `data.yaml` file with:
     ```yaml
     train: dataset/images/train
     val: dataset/images/val

     nc: 1
     names: ['window']
     ```

---

##  Model Training (YOLOv5)

1. Cloned the YOLOv5 repo:
   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   pip install -r requirements.txt
   
---

## Usage Guide
1. Clone the repo and set up your Python environment:
```bash
git clone https://github.com/yourusername/yolov5-window-detector
cd yolov5-window-detector
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
2.Place test images in the test_images/ folder.
3. Run detection:
```bash
python detect_and_count.py
```
4. Check results in output_images/.
