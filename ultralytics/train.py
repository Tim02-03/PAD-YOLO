from ultralytics.models import YOLO
# import torch

# torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":

    model = YOLO(r".yaml" )
    results = model.train(
        name='MODEL', 
 
        # 核心配置
        data=r'DATA',

        epochs=200,
        imgsz=640,
        batch=32,         
        workers=4,
        cache=False,
        device=0,         
        amp=True,         
        project=r'PAD_YOLO',

        patience=50,      

    )