import os
import re
import io
import base64
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from flask import Flask, render_template, request
import spacy
from sympy import Point, Line, symbols, solve, sqrt

app = Flask(__name__)

# Load model NLP SpaCy
MODEL_DIR = "my_geometry_ner_model"
if os.path.exists(MODEL_DIR):
    nlp = spacy.load(MODEL_DIR)
else:
    nlp = None
    print(f"WARNING: Do not find any model at '{MODEL_DIR}'. Please train before!")

def clean_text(text):
    text = str(text)
    text = re.sub(r'(?<=[a-zA-Z0-9])\.', ' .', text)
    text = re.sub(r'(?<=[a-zA-Z0-9])\,', ' ,', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# MODULE 1: AI PARSING (SPACY NER)
def parse_text_with_ai(text_input):
    if nlp is None:
        return {"SHAPE": "tam giác", "POINTS": ["A", "B", "C"], "CONSTRAINTS": ["vuông tại a"], "VALUES": ["AB = 3cm", "AC = 4cm"]}
    
    clean_sentence = clean_text(text_input)
    doc = nlp(clean_sentence)
    
    extracted_data = {"SHAPE": "", "POINTS": [], "CONSTRAINTS": [], "VALUES": []}
    for ent in doc.ents:
        if ent.label_ == "SHAPE":
            extracted_data["SHAPE"] = ent.text.lower()
        elif ent.label_ == "POINTS":
            pts = [char for char in ent.text if char.isupper()]
            extracted_data["POINTS"].extend(pts)
        elif ent.label_ == "CONSTRAINTS":
            extracted_data["CONSTRAINTS"].append(ent.text.lower())
        elif ent.label_ == "VALUES":
            extracted_data["VALUES"].append(ent.text)
            
    extracted_data["POINTS"] = list(dict.fromkeys(extracted_data["POINTS"]))
    return extracted_data

# MODULE MATHEMETICS : TRÍCH XUẤT ĐỘ DÀI THEO TÊN CẠNH THỰC TẾ
def extract_numerical_values(values_list):
    """
    Trích xuất chính xác tên cạnh và độ dài. 
    Ví dụ: 'AB = 5cm' -> {'AB': 5.0}, 'cạnh bằng 4cm' -> {'CANH_CHUNG': 4.0}
    """
    parsed_values = {}
    for val in values_list:
        match_edge = re.search(r'([A-Z]{2})\s*=\s*(\d+(?:\.\d+)?)', val, re.IGNORECASE)
        match_generic = re.search(r'(?:cạnh|bằng)\s*(\d+(?:\.\d+)?)', val, re.IGNORECASE)
        
        if match_edge:
            edge_name = match_edge.group(1).upper()
            parsed_values[edge_name] = float(match_edge.group(2))
        elif match_generic:
            parsed_values["CANH_CHUNG"] = float(match_generic.group(1))
    return parsed_values

# MODULE 2 & 3: ENGINE HÌNH HỌC & ĐỒ HỌA HOÀN CHỈNH
def generate_geometry_plot(ner_data):
    shape = ner_data.get("SHAPE", "").lower()
    constraints_list = ner_data.get("CONSTRAINTS", [])
    constraints = " ".join(constraints_list).lower()
    points = ner_data.get("POINTS", [])
    raw_values = ner_data.get("VALUES", [])
    
    # Kích hoạt bộ lọc trích xuất thông số số học thực tế
    geometry_values = extract_numerical_values(raw_values)
    
    # Khởi tạo khung vẽ Matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    has_image = False
    title_text = "Hình vẽ toán học"
    coords = {}

    def get_point_name(index, default_char):
        return points[index] if len(points) > index else default_char

    # TRƯỜNG HỢP 1: ĐOẠN THẲNG ĐỘC LẬP (LINES CASE)
    if "đoạn thẳng" in shape or "đường thẳng" in shape:
        lA = get_point_name(0, "A")
        lB = get_point_name(1, "B")
        
        line_len = 5.0 # Mặc định độ dài đoạn thẳng nếu không có thông số
        if geometry_values:
            line_len = list(geometry_values.values())[0]
            
        coords[lA] = (0.0, 0.0)
        coords[lB] = (line_len, 0.0)
        
        # Vẽ đoạn thẳng nền chính
        ax.plot([coords[lA][0], coords[lB][0]], [coords[lA][1], coords[lB][1]], color='blue', linewidth=2)
        title_text = f"Đoạn thẳng {lA}{lB} = {line_len}cm"
        has_image = True

    # TRƯỜNG HỢP 2: CÁC LOẠI TAM GIÁC (TRIANGLES CASE)
    elif "tam giác" in shape:
        lA = get_point_name(0, "A")
        lB = get_point_name(1, "B")
        lC = get_point_name(2, "C")
        
        base_len = geometry_values.get("CANH_CHUNG", 4.0)
        top_vertex, base_pt1, base_pt2 = lA, lB, lC

        # 2a. Tam giác đều đúng kích thước cạnh
        if "đều" in constraints:
            if geometry_values and base_len == 4.0:
                base_len = list(geometry_values.values())[0]
                
            match_height = re.search(r'đường\s+cao\s+([A-Z])[A-Z]', constraints)
            if match_height:
                top_vertex = match_height.group(1).upper()
                remaining = [p for p in [lA, lB, lC] if p != top_vertex]
                base_pt1 = remaining[0] if len(remaining) > 0 else "B"
                base_pt2 = remaining[1] if len(remaining) > 1 else "C"

            h = float(base_len * sqrt(3) / 2)
            coords[base_pt1] = (0.0, 0.0)
            coords[base_pt2] = (base_len, 0.0)
            coords[top_vertex] = (base_len / 2, h)
            title_text = f"Tam giác đều {lA}{lB}{lC} (Cạnh = {base_len}cm)"

        # 2b. Tam giác vuông khớp cạnh góc vuông hoành/tung độ
        elif "vuông" in constraints:
            match_vuong = re.search(r'vuông tại\s+([a-z])', constraints)
            v_vertex = match_vuong.group(1).upper() if match_vuong else lA
            other_pts = [p for p in [lA, lB, lC] if p != v_vertex]
            lA_v, lB_v, lC_v = v_vertex, other_pts[0], other_pts[1]
            
            side_vert = 3.0  
            side_horiz = 4.0 
            
            edge1, edge1_rev = f"{v_vertex}{lB_v}", f"{lB_v}{v_vertex}"
            edge2, edge2_rev = f"{v_vertex}{lC_v}", f"{lC_v}{v_vertex}"
            
            if edge1 in geometry_values: side_vert = geometry_values[edge1]
            elif edge1_rev in geometry_values: side_vert = geometry_values[edge1_rev]
            if edge2 in geometry_values: side_horiz = geometry_values[edge2]
            elif edge2_rev in geometry_values: side_horiz = geometry_values[edge2_rev]
            
            if not (edge1 in geometry_values or edge1_rev in geometry_values or edge2 in geometry_values or edge2_rev in geometry_values):
                val_list = list(geometry_values.values())
                if len(val_list) >= 2: side_vert, side_horiz = val_list[0], val_list[1]

            coords[v_vertex] = (0.0, 0.0)
            coords[lB_v] = (0.0, side_vert)
            coords[lC_v] = (side_horiz, 0.0)
            
            box_size = min(side_vert, side_horiz) * 0.1
            ax.plot([0.0, box_size, box_size, 0.0], [box_size, box_size, 0.0, 0.0], color='red', linewidth=1)
            title_text = f"Tam giác {v_vertex}{lB_v}{lC_v} vuông tại {v_vertex}"
            top_vertex, base_pt1, base_pt2 = lB_v, v_vertex, lC_v

        # 2c. Tam giác cân giải tích qua SymPy
        elif "cân" in constraints:
            match_can = re.search(r'cân tại\s+([a-z])', constraints)
            c_vertex = match_can.group(1).upper() if match_can else lA
            other_pts = [p for p in [lA, lB, lC] if p != c_vertex]
            lA_c, lB_c, lC_c = c_vertex, other_pts[0], other_pts[1]
            
            val_list = list(geometry_values.values())
            edge_bottom = val_list[1] if len(val_list) > 1 else 4.0
            edge_leg = val_list[0] if len(val_list) > 0 else 5.0
            
            coords[lB_c] = (0.0, 0.0)
            coords[lC_c] = (edge_bottom, 0.0)
            
            x_a, y_a = symbols('x_a y_a')
            P_B = Point(0, 0)
            P_A = Point(edge_bottom / 2, y_a)
            equation = P_B.distance(P_A) - edge_leg
            sol_y = solve(equation, y_a)
            real_y = float([sol for sol in sol_y if sol > 0][0])
            coords[lA_c] = (edge_bottom / 2, real_y)
            title_text = f"Tam giác {lA_c}{lB_c}{lC_c} cân tại {lA_c}"
            top_vertex, base_pt1, base_pt2 = lA_c, lB_c, lC_c
            
        # 2d. Tam giác nhọn / thường
        else:
            coords[lA] = (1.5, 3.0)
            coords[lB] = (0.0, 0.0)
            coords[lC] = (4.0, 0.0)
            title_text = f"Tam giác {lA}{lB}{lC}"

        # Vẽ khung tam giác chính
        pts_list = [coords[base_pt1], coords[base_pt2], coords[top_vertex], coords[base_pt1]]
        xs, ys = zip(*pts_list)
        ax.plot(xs, ys, color='blue', linewidth=2)

        # XỬ LÝ HẠ ĐƯỜNG CAO HÌNH CHIẾU
        if "cao" in constraints or any("đường cao" in c for c in constraints_list):
            match_height_point = re.search(r'đường\s+cao\s+[A-Z]([A-Z])', constraints)
            lH = match_height_point.group(1).upper() if match_height_point else "H"
            
            p_top = Point(coords[top_vertex])
            p_base1 = Point(coords[base_pt1])
            p_base2 = Point(coords[base_pt2])
            line_base = Line(p_base1, p_base2)
            p_foot = line_base.projection(p_top)
            
            coords[lH] = (float(p_foot.x), float(p_foot.y))
            ax.plot([coords[top_vertex][0], coords[lH][0]], [coords[top_vertex][1], coords[lH][1]], color='green', linestyle='--', linewidth=1.8)
            title_text += f" (Đường cao {top_vertex}{lH})"
        has_image = True

    # TRƯỜNG HỢP 3: HÌNH VUÔNG & HÌNH CHỮ NHẬT (QUADRILATERALS)
    elif "hình vuông" in shape or "hình chữ nhật" in shape:
        # Loại bỏ điểm O hoặc M trung điểm ra khỏi danh sách 4 đỉnh góc
        quad_points = [p for p in points if p not in ["O", "M", "I", "K"]]
        lA = quad_points[0] if len(quad_points) > 0 else "A"
        lB = quad_points[1] if len(quad_points) > 1 else "B"
        lC = quad_points[2] if len(quad_points) > 2 else "C"
        lD = quad_points[3] if len(quad_points) > 3 else "D"
        
        if "hình vuông" in shape:
            size = geometry_values.get("CANH_CHUNG", 4.0)
            if geometry_values and size == 4.0: size = list(geometry_values.values())[0]
            w, h = size, size
            title_text = f"Hình vuông {lA}{lB}{lC}{lD}"
        else:
            w, h = 5.0, 3.0
            if f"{lA}{lB}" in geometry_values: h = geometry_values[f"{lA}{lB}"]
            elif f"{lB}{lA}" in geometry_values: h = geometry_values[f"{lB}{lA}"]
            if f"{lB}{lC}" in geometry_values: w = geometry_values[f"{lB}{lC}"]
            elif f"{lC}{lB}" in geometry_values: w = geometry_values[f"{lC}{lB}"]
            
            if w == 5.0 and h == 3.0 and len(geometry_values) >= 2:
                val_lengths = list(geometry_values.values())
                w, h = val_lengths[0], val_lengths[1]
            title_text = f"Hình chữ nhật {lA}{lB}{lC}{lD}"
            
        coords[lA] = (0.0, h)
        coords[lB] = (w, h)
        coords[lC] = (w, 0.0)
        coords[lD] = (0.0, 0.0)
        
        rect_pts = [coords[lA], coords[lB], coords[lC], coords[lD]]
        polygon = patches.Polygon(rect_pts, closed=True, edgecolor='blue', facecolor='none', linewidth=2)
        ax.add_patch(polygon)
        
        # XỬ LÝ ĐƯỜNG CHÉO TỨ GIÁC
        if "chéo" in constraints or "o" in constraints or "o" in [p.lower() for p in points]:
            ax.plot([coords[lA][0], coords[lC][0]], [coords[lA][1], coords[lC][1]], color='purple', linestyle='--', linewidth=1.5)
            ax.plot([coords[lB][0], coords[lD][0]], [coords[lB][1], coords[lD][1]], color='purple', linestyle='--', linewidth=1.5)
            lO = "O"
            coords[lO] = (w / 2, h / 2)
        has_image = True

    # TRƯỜNG HỢP 4: HÌNH THOI & HÌNH THANG VUÔNG
    elif "hình thoi" in shape:
        lA, lB, lC, lD = get_point_name(0, "A"), get_point_name(1, "B"), get_point_name(2, "C"), get_point_name(3, "D")
        val_keys = list(geometry_values.keys())
        d1 = geometry_values.get(val_keys[0], 5.0) if len(val_keys) > 0 else 5.0
        d2 = geometry_values.get(val_keys[1], 3.0) if len(val_keys) > 1 else 3.0
        
        coords[lA], coords[lB] = (0.0, d2 / 2), (d1 / 2, 0.0)
        coords[lC], coords[lD] = (0.0, -d2 / 2), (-d1 / 2, 0.0)
        
        thoi_pts = [coords[lA], coords[lB], coords[lC], coords[lD]]
        polygon = patches.Polygon(thoi_pts, closed=True, edgecolor='purple', facecolor='none', linewidth=2)
        ax.add_patch(polygon)
        title_text = f"Hình thoi {lA}{lB}{lC}{lD}"
        has_image = True

    elif "hình thang" in shape and "vuông" in constraints:
        lA, lB, lC, lD = get_point_name(0, "A"), get_point_name(1, "B"), get_point_name(2, "C"), get_point_name(3, "D")
        val_keys = list(geometry_values.keys())
        base_small = geometry_values.get(val_keys[0], 3.0) if len(val_keys) > 0 else 3.0
        base_large = geometry_values.get(val_keys[1], 5.0) if len(val_keys) > 1 else 5.0
        height_th = 3.5
        
        coords[lA], coords[lB] = (0.0, height_th), (base_small, height_th)
        coords[lC], coords[lD] = (base_large, 0.0), (0.0, 0.0)
        
        thang_pts = [coords[lA], coords[lB], coords[lC], coords[lD]]
        polygon = patches.Polygon(thang_pts, closed=True, edgecolor='darkorange', facecolor='none', linewidth=2)
        ax.add_patch(polygon)
        title_text = f"Hình thang vuông {lA}{lB}{lC}{lD}"
        has_image = True

    # MODULE NÂNG CẤP: ENGINE TỰ ĐỘNG CHẤM TRUNG ĐIỂM (MIDPOINT)
    if has_image and "trung điểm" in constraints:
        # Regex tìm chuỗi mẫu: "Gọi M là trung điểm của BC" hoặc "I là trung điểm AB"
        match_midpoint = re.search(r'([a-z])\s+là\s+trung\s+điểm\s+(?:của\s+)?(?:cạnh\s+)?([a-z]{2})', constraints)
        if match_midpoint:
            lMid = match_midpoint.group(1).upper()       # Tên trung điểm (Ví dụ: M)
            target_segment = match_midpoint.group(2).upper() # Tên đoạn thẳng (Ví dụ: BC)
            
            p1 = target_segment[0]
            p2 = target_segment[1]
            
            # Nếu cả 2 đầu mút của đoạn thẳng đều đã được dựng tọa độ trước đó
            if p1 in coords and p2 in coords:
                x1, y1 = coords[p1]
                x2, y2 = coords[p2]
                
                # Áp dụng công thức nội suy trung điểm giải tích phẳng
                coords[lMid] = ((x1 + x2) / 2, (y1 + y2) / 2)
                title_text += f" (có trung điểm {lMid} của {target_segment})"

    # ĐÓNG GÓI ĐỒ HỌA & KHÓA TỶ LỆ KHUNG HÌNH KHÔNG ĐỔI
    if not has_image:
        ax.text(0.5, 0.5, "Hệ thống chưa tìm thấy cấu hình hình học phù hợp!", ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Vẽ các chấm tròn đen đại diện cho các đỉnh hình học và in nhãn chữ
        for label, pos in coords.items():
            ax.plot(pos[0], pos[1], 'ko', markersize=5)
            ax.text(pos[0] + 0.15, pos[1] + 0.15, label, fontsize=12, fontweight='bold')

        # Căn chỉnh lề biên bao quát toàn bộ tọa độ
        all_x = [p[0] for p in coords.values()]
        all_y = [p[1] for p in coords.values()]
        ax.set_xlim(min(all_x) - 1.0, max(all_x) + 1.0)
        ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
            
        # KHÓA CHẶT TỶ LỆ PIXEL 1:1 CHỐNG MÉO HÌNH KHI CO GIÃN TRÌNH DUYỆT
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title(title_text, fontsize=13, fontweight='bold', color='#1e3d59', pad=15)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return f"data:image/png;base64,{img_base64}"

# ROUTING FLASK SERVER
@app.route('/', methods=['GET', 'POST'])
def index():
    image_data = None
    ai_output = None
    user_input = ""
    
    if request.method == 'POST':
        user_input = request.form.get('geometry_text', '')
        if user_input.strip():
            ai_output = parse_text_with_ai(user_input)
            image_data = generate_geometry_plot(ai_output)
            
    return render_template('index.html', 
                           user_input=user_input, 
                           ai_output=ai_output, 
                           image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)