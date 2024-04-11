#from roboflow import Roboflow
#rf = Roboflow(api_key="FjNNE4nwARLc7CLcN6eJ")
#project = rf.workspace("adrin-garca-domingo").project("wheel-object-detection")
#dataset = project.version(7).download("yolov8")

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Entrena el modelo con 2 GPUs
    results = model.train(data='data.yaml', epochs=200, imgsz=1440, device='cuda', batch=4)
