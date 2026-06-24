import os
import re
import random
import pandas as pd
import spacy
from spacy.training import Example
from spacy.matcher import Matcher

# =======================================================
# CONFIGURATION & CUSTOM TOKENIZER FOR GEOMETRIC PUNCTUATION
# =======================================================
def setup_custom_tokenizer(nlp):
    """
    Cấu hình bộ Tokenizer của spaCy tách triệt để dấu câu và dấu bằng toán học,
    loại bỏ hoàn toàn sự phụ thuộc vào các hàm làm sạch chuỗi thủ công.
    """
    # Bổ sung dấu bằng (=) vào danh sách các ký tự phân tách tiền tố/hậu tố
    infixes = nlp.Defaults.infixes + [r'(?<=[a-zA-Z0-9])(=)(?=[a-zA-Z0-9])', r'(=)']
    infix_regex = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_regex.finditer
    
    # Đảm bảo dấu chấm, dấu phẩy dính liền luôn được bẻ thành token độc lập
    suffixes = nlp.Defaults.suffixes + [r'\.', r',']
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search
    return nlp

# =======================================================
# CORE ENGINE: TOKEN-BASED AUTOMATIC ENTITY ANNOTATOR
# =======================================================
def build_perfect_training_data(csv_path, nlp):
    df = pd.read_csv(csv_path)
    training_data = []
    
    shapes_list = ["tam giác", "hình vuông", "hình chữ nhật", "hình thang", "hình thoi", "đường tròn", "hình chóp", "hình lăng trụ đứng", "hình lập phương", "đường thẳng", "mặt phẳng", "dây cung", "đường cao", "đường chéo", "tiếp tuyến"]
    
    print("--- Đang quét và gán nhãn thực thể dựa trên cấu trúc Token chuẩn hóa ---")
    for _, row in df.iterrows():
        text = str(row['Geometry_Problem']).strip()
        doc = nlp.make_doc(text)
        
        # Sử dụng hệ thống Matcher mạnh mẽ của spaCy thay cho regex thô sơ
        matcher = Matcher(nlp.vocab)
        
        # Pattern 1: Tìm VALUES (Ví dụ: AB = 5cm, MN = 4 cm, hoặc số kèm cm đứng độc lập)
        matcher.add("VALUES", [
            [{"TEXT": {"REGEX": "^[A-Z]{1,4}$"}}, {"TEXT": "="}, {"TEXT": {"REGEX": "^\d+$"}}, {"TEXT": {"REGEX": "(?i)^cm$|^độ$"}}],
            [{"TEXT": {"REGEX": "^[A-Z]{1,4}$"}}, {"TEXT": "="}, {"TEXT": {"REGEX": "^\d+(cm|độ)$"}}],
            [{"TEXT": {"REGEX": "^\d+$"}}, {"TEXT": {"REGEX": "(?i)^cm$|^độ$"}}],
            [{"TEXT": {"REGEX": "^\d+(cm|độ)$"}}]
        ])
        
        # Pattern 2: Tìm CONSTRAINTS (Mệnh đề điều kiện toán học phức tạp)
        matcher.add("CONSTRAINTS", [
            [{"TEXT": {"REGEX": "(?i)^vuông|^cân|^đều"}}, {"TEXT": {"REGEX": "(?i)^tại"}}, {"TEXT": {"REGEX": "^[A-Z]$"}}],
            [{"TEXT": {"REGEX": "(?i)^vuông"}}, {"TEXT": {"REGEX": "(?i)^góc"}}]
        ])
        
        # Áp dụng bộ lọc tìm kiếm trên Token Doc
        matches = matcher(doc)
        
        entities = []
        # Chuyển đổi kết quả Matcher sang cấu trúc Spans của spaCy
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id]
            entities.append((doc[start].idx, doc[end-1].idx + len(doc[end-1].text), label))
            
        # Quét bổ sung nhãn SHAPE dựa trên danh mục từ điển hình học
        for shape in shapes_list:
            for match in re.finditer(r'\b' + re.escape(shape) + r'\b', text, re.IGNORECASE):
                entities.append((match.start(), match.end(), "SHAPE"))
                
        # Quét bổ sung nhãn POINTS (Hỗ trợ định dạng điểm đơn, đoạn thẳng, đa giác, hình khối không gian)
        for match in re.finditer(r'\b[S]\.[A-Z]+\b|\b[A-Z]{2,4}\b|\b[A-Z]\b', text):
            entities.append((match.start(), match.end(), "POINTS"))
            
        # THUẬT TOÁN ĐỒ THỊ KHỬ TRÙNG LẶP & ĐÈ NHÃN (Entity Resolution Greedy Loop)
        # Sắp xếp theo chiều dài giảm dần (Ưu tiên các cụm từ ghép phức tạp trước)
        sorted_ents = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
        clean_entities = []
        last_end = -1
        
        for start, end, label in sorted_ents:
            # Ngăn chặn hoàn toàn hiện tượng lấn index, đảm bảo tính phân tách tuyệt đối giữa các nhãn
            if start >= last_end:
                # Kiểm tra tính hợp lệ của Span ký tự so với ranh giới Token của spaCy
                span = doc.char_span(start, end, label=label, alignment_mode="strict")
                if span is not None:
                    clean_entities.append((start, end, label))
                    last_end = end
                    
        training_data.append((text, {"entities": clean_entities}))
        
    return training_data

# =======================================================
# TRAINING CONTROLLER
# =======================================================
def train_high_precision_ner(training_data, iterations=35):
    nlp = spacy.blank("vi")
    nlp = setup_custom_tokenizer(nlp)
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
        
    ner.add_label("SHAPE")
    ner.add_label("POINTS")
    ner.add_label("VALUES")
    ner.add_label("CONSTRAINTS")
    
    # Sử dụng cấu hình drop-out động để chống Overfitting (Học vẹt)
    optimizer = nlp.begin_training()
    print(f"\n--- Khởi động chu kỳ huấn luyện độ chính xác cao ({iterations} Epochs) ---")
    
    for epoch in range(iterations):
        random.shuffle(training_data)
        losses = {}
        # Chia batch nhỏ để cập nhật trọng số tối ưu hơn
        batches = spacy.util.minibatch(training_data, size=spacy.util.compounding(4.0, 32.0, 1.001))
        
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=0.25, sgd=optimizer, losses=losses)
            
        print(f"Epoch {epoch+1:02d}/{iterations:02d} -> Mức độ lỗi (Loss NER): {losses['ner']:.6f}")
        
    return nlp

# =======================================================
# EXECUTION INTERFACE
# =======================================================
if __name__ == "__main__":
    csv_file = "DATASET/train_data.csv"
    model_dir = "my_geometry_ner_model"
    
    if not os.path.exists(csv_file):
        csv_file = "train_data.csv"
        
    if not os.path.exists(csv_file):
        print(f"Lỗi hệ thống: Không tìm thấy file dữ liệu huấn luyện!")
    else:
        # Sử dụng mô hình nền tảng tạm thời để bóc tách token
        base_nlp = spacy.blank("vi")
        base_nlp = setup_custom_tokenizer(base_nlp)
        
        train_data = build_perfect_training_data(csv_file, base_nlp)
        
        # Tiến hành tối ưu và lưu mô hình xuống đĩa
        optimized_nlp = train_high_precision_ner(train_data, iterations=30)
        optimized_nlp.to_disk(model_dir)
        print(f"\n--- ĐÃ ĐỒNG BỘ VÀ LƯU MÔ HÌNH HOÀN HẢO TẠI: '{model_dir}' ---")
        
        # =======================================================
        # KIỂM THỬ THỰC TẾ KHÔNG CẦN QUA HÀM CLEAN CHUỖI
        # =======================================================
        print("\n--- KIỂM THỬ ĐÁNH GIÁ MÔ HÌNH SAU NÂNG CẤP ---")
        test_sentence = "Cho tam giác MNPQ vuông tại M, biết MN = 5cm và đường cao MH."
        
        nlp_production = spacy.load(model_dir)
        doc = nlp_production(test_sentence)
        
        print(f"Đầu vào thực tế: '{test_sentence}'")
        print("Kết quả bóc tách thực thể:")
        for ent in doc.ents:
            print(f" -> [{ent.label_}]: {ent.text}")