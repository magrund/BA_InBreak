from ultralytics import YOLO
import yaml

MODEL_VERSION = 'MODEL_VERSION'
DATA_PATH = 'data.yaml'
EPOCHS = "INT"
BATCH_SIZE = "INT"
FREEZE_LAYER = "INT"
IMAGE_SIZE = "INT"

def check_data_yaml(data_path):
    try:
        with open(data_path, 'r') as file:
            data = yaml.safe_load(file)
            print("Content of data.yaml:", data)
            return True
    except Exception as e:
        print(f"Error loading data.yaml: {e}")
        return False

model = YOLO(MODEL_VERSION)

def train_and_export(model, data_path, epochs, batch_size, image_size, freeze_layer, export_path='best.pt'):
    model.train(data=data_path, epochs=epochs, batch=batch_size, imgsz = image_size, freeze=freeze_layer, device=0)

    model.save(export_path)
    print(f"Model successfully exported:'{export_path}'")

if check_data_yaml(DATA_PATH):
    train_and_export(model, DATA_PATH, EPOCHS, BATCH_SIZE, IMAGE_SIZE, FREEZE_LAYER, export_path='best.pt')
else:
    print("Training and export were cancelled because the data.yaml file could not be loaded.")
