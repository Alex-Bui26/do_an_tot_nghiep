import os
import pandas as pd

def split_my_dataset(csv_path, train_ratio=0.8):
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file {csv_path}!")
        return
        
    # 1. Đọc dữ liệu gốc
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"--- Tổng số mẫu dữ liệu gốc: {total_rows} câu ---")
    
    # 2. Xáo trộn ngẫu nhiên dữ liệu (Shuffle)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 3. Tính toán vị trí cắt dữ liệu
    train_size = int(total_rows * train_ratio)
    
    # 4. Chia tách thành 2 tập con
    train_df = df_shuffled.iloc[:train_size]
    test_df = df_shuffled.iloc[train_size:]
    
    # 5. Xác định đường dẫn và lưu file
    dir_name = os.path.dirname(csv_path)
    train_file = os.path.join(dir_name, "train_data.csv")
    test_file = os.path.join(dir_name, "test_data.csv")
    
    # Lưu xuống ổ đĩa dạng UTF-8
    train_df.to_csv(train_file, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_file, index=False, encoding='utf-8-sig')
    
    print("\n--- CHIA DỮ LIỆU THÀNH CÔNG ---")
    print(f"1. File Train: '{train_file}' -> {len(train_df)} câu ({train_ratio*100:.0f}%)")
    print(f"2. File Test : '{test_file}' -> {len(test_df)} câu ({(1-train_ratio)*100:.0f}%)")

if __name__ == "__main__":
    # Đường dẫn tới file dataset của bạn
    # Nếu file nằm cùng thư mục code thì chỉ cần điền "DATASET.csv"
    target_csv = "DATASET/DATASET.csv" 
    
    if not os.path.exists(target_csv):
        target_csv = "DATASET.csv"
        
    split_my_dataset(target_csv)
