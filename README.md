# do_an_tot_nghiep
# 1. Đầu tiên cần phải có Python trong máy

# 2. Cấu hình môi trường ảo
  python -m venv venv

# 3. Kích hoạt môi trường ảo

# 4. Tải các thư viện liên quan
  pip install flask matplotlib sympy spacy pyvi click pandas

# 5. Chạy file split_dataset.py để chia DATASET thành 80% train và 20% test
  python DATASET/split_dataset.py

# 6. Chạy file train_ner.py để train NER Model
  python DATASET/train_ner.py

# 7. Sau khi train model thành công thì chạy file app.py để demo project
  python app.py

# 8. Để đánh giá model chạy file evaluate_model_ver2.py
  python evaluate_model_ver2.py
