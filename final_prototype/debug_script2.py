import cv2
import numpy as np
import predict
from ultralytics import YOLO

img = cv2.imread('test_crop.png')
model = YOLO(predict.find_best_model())
boxes, _ = predict.detect_gates('test_crop.png', model)
num_labels, labels_im = predict.trace_wires(img, boxes)

for b in boxes:
    in_region = labels_im[
        max(0, int(b['y'])-5):min(img.shape[0], int(b['y']+b['h'])+5), 
        max(0, int(b['x'])-40):int(b['x']+b['w']//2)
    ]
    u = np.unique(in_region)
    print(f"Gate {b['id']} ({b['cls']}) inputs: {u}")
