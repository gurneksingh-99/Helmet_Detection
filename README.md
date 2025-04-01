# Helmet Detection Project Documentation

## Project Overview
The Helmet Detection project aims to detect whether a motorcycle rider is wearing a helmet and extract the bike's number plate. The detected data is stored in an Excel file for further processing or legal enforcement. This project leverages deep learning and computer vision techniques, specifically using the YOLO (You Only Look Once) object detection model.

## Objectives
- Detect motorcycle riders in an image or video.
- Determine whether the rider is wearing a helmet.
- Extract the bike’s number plate.
- Store results, including images and extracted data, in an Excel file.

## Technologies Used
- **Programming Language:** Python
- **Deep Learning Framework:** YOLO (You Only Look Once)
- **Libraries:** OpenCV, TensorFlow/PyTorch, NumPy, Pandas, Tesseract OCR
- **Data Storage:** Excel (using Pandas)

## System Architecture
1. **Input:** Image or video feed of motorcycle riders.
2. **Preprocessing:** Resize and normalize images for YOLO input.
3. **Object Detection:** YOLO model identifies motorcycles, riders, and helmets.
4. **Helmet Classification:** Checks if a detected rider is wearing a helmet.
5. **Number Plate Detection:** Extracts the license plate using OCR.
6. **Data Storage:** Saves detection results in an Excel file.
7. **Output:** Annotated image/video and stored results.

## Dataset
- **Sources:** Custom dataset collected from traffic cameras and open datasets.
- **Classes:** Rider, Helmet, Motorcycle, Number Plate.
- **Annotations:** Labeled using YOLO annotation format.

## Model Training
- **Pre-trained Model:** YOLOv5/YOLOv8 trained on COCO dataset.
- **Custom Training:** Fine-tuned using the collected dataset.
- **Loss Function:** Cross-entropy for classification, IoU (Intersection over Union) for object detection.
- **Optimization Algorithm:** Adam/SGD.

## Implementation Details
1. **Load YOLO Model:**
   ```python
   import cv2
   import torch
   import pandas as pd
   from ultralytics import YOLO

   model = YOLO('yolov5s.pt')
   ```

2. **Perform Object Detection:**
   ```python
   img = cv2.imread('test_image.jpg')
   results = model(img)
   results.show()  # Display detected objects
   ```

3. **Helmet Detection Logic:**
   - If a rider is detected but no helmet is identified, mark as "No Helmet."

4. **License Plate Extraction:**
   - Use OCR (Tesseract) to read the detected plate.
   ```python
   import pytesseract
   plate_text = pytesseract.image_to_string(plate_crop, config='--psm 8')
   ```

5. **Saving Data to Excel:**
   ```python
   data = {'Rider': ['Detected'], 'Helmet': ['No'], 'Plate': [plate_text]}
   df = pd.DataFrame(data)
   df.to_excel('detection_results.xlsx', index=False)
   ```

## Results & Performance
- **Accuracy:** Model achieves high detection accuracy with minimal false positives.
- **Processing Speed:** YOLO provides real-time detection capabilities.
- **OCR Accuracy:** Dependent on image clarity and lighting conditions.

## Challenges & Solutions
- **Blurred Number Plates:** Used image preprocessing (grayscale conversion, thresholding) to enhance OCR accuracy.
- **Low-light Conditions:** Applied data augmentation techniques during training.

## Conclusion
This project successfully detects motorcycle riders, verifies helmet usage, and extracts number plates. It provides a foundation for traffic monitoring and safety enforcement applications.

