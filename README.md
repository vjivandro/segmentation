1. menggunakan dataset menggunakan 55 gambar botol diambil dari google image
2. proses anotasi menggunakan roboflow
3. hasil dari anotasi roboflow data.yaml ditraining menggunakan cuda dan code
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model.train(data="bottle_segment/data.yaml", epochs=100, imgsz=640, device=device
4. proses segmentasi menggunakan hasil dari training yang berada di "runs/segment/train3/weights/best.pt", dengan menggunakan code berikut:
   predictions = model.predict(source='bottle_segment/test/images', task='segment'
5. untuk melihat hasilnya dapat menjalankan main.py, namun harus menonaktifkan metode training dahulu karena code masih menjadi satu.
