from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("yolo11n.pt")

options = {
    "source": "./car_track_test.mp4",  # video source
    "conf": 0.3,  # confidence threshold
    "iou": 0.5,  # NMS IoU threshold
    "show": False,  # show results
    "save": True,  # save results to 'runs/track'
    "tracker": "botsort.yaml",  # tracker configuration file
    "device": "0",  # device to run the model on
    # "with_reid": True,  # use ReID model
}

results = model.track(**options)