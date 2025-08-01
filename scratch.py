model_path = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_training/runs/detect/train2/weights/best.pt'

video_path= r'/home/minhpn/Desktop/Green_Parking/Data/video_data/4.mp4'

from ultralytics import YOLO

model = YOLO(model_path)

results = model.track(video_path, save=True, project=r'/home/minhpn/Desktop/Green_Parking/Data/video_data/save')