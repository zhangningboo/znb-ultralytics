import cv2
from pathlib import Path


vid_1 = cv2.VideoCapture("runs/detect/track7/car_track_test.avi")


cnt = 0

while vid_1.isOpened():
    ret, frame = vid_1.read()
    if not ret:
        break
    cv2.imwrite(f'runs/detect/track7/frames/{cnt}.jpg', frame)
    cnt += 1