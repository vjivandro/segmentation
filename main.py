import torch
from ultralytics import YOLO

# print("CUDA available:", torch.cuda.is_available())  
# print("CUDA version:", torch.version.cuda)           
# print("GPU detected:", torch.cuda.get_device_name(0))

def main():
    # 1. CEK GPU or CPU
    if torch.cuda.is_available():
        print("CUDA tersedia. Menggunakan GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA tidak tersedia. Menggunakan CPU.")

    # 2. Model YOLO
    model = YOLO('yolo11n-seg.pt')
    model.info()

    # 3. PROSES TRAINING DATASET
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = model.train(data="bottle_segment\data.yaml", epochs=100, imgsz=640, device=device)
    results.show()

    # 4. PROSES SEGMENTASI
    predictions = model.predict(source='bottle_segment/test/images', task='segment')

    for result in predictions:
        result.show()


if __name__ == '__main__':
    main()