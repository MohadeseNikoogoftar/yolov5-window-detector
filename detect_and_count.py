import torch
from pathlib import Path
import cv2
from matplotlib import pyplot as plt

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/window_detector/weights/best.pt')

model.conf = 0.3  # Confidence threshold

# Folder with your test images
test_images_folder = 'test_images'
output_folder = 'output_images'
Path(output_folder).mkdir(exist_ok=True)

# Loop through images
for image_path in Path(test_images_folder).glob('*.*'):
    img = cv2.imread(str(image_path))
    results = model(img)

    # Results in pandas format
    df = results.pandas().xyxy[0]
    count = len(df)

    # Draw green rectangles
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Put the count on the image
    cv2.putText(img, f'Windows detected: {count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save result
    save_path = Path(output_folder) / image_path.name
    cv2.imwrite(str(save_path), img)
    print(f"Saved: {save_path}")

print("All done!")
