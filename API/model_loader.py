from ultralytics import YOLO

def load_model(model_dir):
    model = YOLO(model_dir)
    return model