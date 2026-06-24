import os
import pandas as pd
import spacy
from collections import defaultdict

MODEL_DIR = "my_geometry_ner_model"
CSV_TEST_PATH = os.path.join("DATASET", "test_data.csv")


def evaluate_model():

    if not os.path.exists(MODEL_DIR):
        print("❌ Không tìm thấy model.")
        return

    if not os.path.exists(CSV_TEST_PATH):
        print("❌ Không tìm thấy test_data.csv")
        return

    print("📦 Loading model...")
    nlp = spacy.load(MODEL_DIR)

    df = pd.read_csv(CSV_TEST_PATH)

    total_samples = len(df)

    success_ner = 0
    success_shape = 0

    category_stats = defaultdict(lambda: {"total": 0, "success": 0})

    total_shapes = 0
    total_points = 0
    total_values = 0
    total_constraints = 0

    print("\n📊 Evaluating...")

    for _, row in df.iterrows():

        text = str(row["Geometry_Problem"])
        category = str(row["Category"]).strip()

        doc = nlp(text)

        ents = list(doc.ents)

        shapes = [e for e in ents if e.label_ == "SHAPE"]
        points = [e for e in ents if e.label_ == "POINTS"]
        values = [e for e in ents if e.label_ == "VALUES"]
        constraints = [e for e in ents if e.label_ == "CONSTRAINTS"]

        total_shapes += len(shapes)
        total_points += len(points)
        total_values += len(values)
        total_constraints += len(constraints)

        if len(ents) > 0:
            success_ner += 1

        if len(shapes) > 0:
            success_shape += 1

        category_stats[category]["total"] += 1

        if len(shapes) > 0:
            category_stats[category]["success"] += 1

    ner_success_rate = success_ner / total_samples * 100
    shape_success_rate = success_shape / total_samples * 100

    print("\n" + "=" * 60)
    print("NER EXTRACTION REPORT")
    print("=" * 60)

    print(f"Total test samples       : {total_samples}")
    print(f"NER extraction success   : {ner_success_rate:.2f}%")
    print(f"Shape detection success  : {shape_success_rate:.2f}%")

    print("\nEntity Statistics")
    print("-" * 60)
    print(f"SHAPE entities       : {total_shapes}")
    print(f"POINTS entities      : {total_points}")
    print(f"VALUES entities      : {total_values}")
    print(f"CONSTRAINT entities  : {total_constraints}")

    print("\nCategory Performance")
    print("-" * 60)

    for cat, stats in category_stats.items():

        rate = (
            stats["success"] /
            stats["total"] * 100
        )

        print(
            f"{cat:25s}"
            f"{stats['success']:3d}/"
            f"{stats['total']:3d}"
            f"  ({rate:.2f}%)"
        )

    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()