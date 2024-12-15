import cv2
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
import os
import numpy as np 

# Load result file
result_file = './results.txt'
df = pd.read_csv(result_file, header=None, names=['video_id', 'frame_id', 'x', 'y', 'width', 'height', 'class_id', 'confidence'])

# Parameters
video_path = os.path.abspath('./data/test/09.mp4')
# video_path = './data/test/09.mp4'  # Path to the input video file
name = np.random.rand()
output_video_path = f'output_{name:.6f}.mp4'  # Path to the output video file
fps = 10  # Frames per second

label_list = ["motorbike", "DHelmet", "DNoHelmet", "P1Helmet", "P1NoHelmet", "P2Helmet", "P2NoHelmet", "P0Helmet", "P0NoHelmet"]

# Open the video
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the video
current_frame_id = 1

while True:
    print("Start ")
    ret, frame = cap.read()
    if not ret:
        print("video_path not found")
        break

    # Get bounding boxes for the current frame
    frame_results = df[df['frame_id'] == current_frame_id]

    if len(frame_results) > 0:
        boxes = []
        scores = []
        labels = []

        for _, row in frame_results.iterrows():
            x_min = row['x'] / frame_width
            y_min = row['y'] / frame_height
            x_max = (row['x'] + row['width']) / frame_width
            y_max = (row['y'] + row['height']) / frame_height

            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(row['confidence'])
            labels.append(int(row['class_id']))

        # Convert lists to numpy arrays for WBF
        boxes = [boxes]
        scores = [scores]
        labels = [labels]

        # Apply WBF with adjusted thresholds
        boxes, scores, labels = weighted_boxes_fusion(
            boxes, scores, labels,
            iou_thr=0.6,  # Increase IoU threshold to merge more boxes
            skip_box_thr=0.1  # Filter out boxes with very low scores before fusion
        )

        # Filter out boxes with low confidence scores after WBF
        min_confidence = 0.5  # Confidence threshold to keep the box
        final_boxes = []
        final_scores = []
        final_labels = []

        for i in range(len(boxes)):
            if scores[i] >= min_confidence:
                final_boxes.append(boxes[i])
                final_scores.append(scores[i])
                final_labels.append(labels[i])

        # Convert back to pixel coordinates and draw bounding boxes
        for i in range(len(final_boxes)):
            x_min, y_min, x_max, y_max = final_boxes[i]
            x_min = int(x_min * frame_width)
            y_min = int(y_min * frame_height)
            x_max = int(x_max * frame_width)
            y_max = int(y_max * frame_height)
            class_name = label_list[int(final_labels[i]) - 1]  # Mapping class ID to label name
            label = f'{class_name} ({final_scores[i]:.2f})'
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with drawn bounding boxes to the output video
    out.write(frame)
    current_frame_id += 1
    print("End ")

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()