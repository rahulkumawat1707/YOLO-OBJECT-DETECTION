import json
import os

# -------------------------------
# STEP 1: CLASS MAP (ALL LOWERCASE)
# -------------------------------
class_map = {
    "tree": 0,
    "bonsai shrub": 1,
    "yellow flower": 2,
    "bird": 3,
    "car": 4,
    "humans": 5,
    "building": 6,
    "road": 7,
    "sky": 8,
    "divider": 9,
    "shoes": 10,
    "palm tree": 11,
    "bonsai": 12,
    "rock wall": 13,
    "leaf": 14,
    "fire hydrant": 15,
    "white flower": 16,
    "pink flower": 17,
    "pink-red flower": 18,
    "plant-pot": 19,
    "dustbin": 20,
    "bamboo": 21,
    "electric box": 22,
    "bike": 23,
    "grass": 24,
    "orange flower": 25,
    "pic-in-pic": 30,
    "ground": 34
}

# -------------------------------
# STEP 2: IMAGE SIZE (CHANGE IF NEEDED)
# -------------------------------
image_width = 1080
image_height = 1920

# -------------------------------
# STEP 3: FOLDERS
# -------------------------------
json_folder = "json_labels"
output_folder = "labels"

os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# STEP 4: CONVERSION
# -------------------------------
for file in os.listdir(json_folder):
    if file.endswith(".json"):

        json_path = os.path.join(json_folder, file)

        with open(json_path, "r") as f:
            data = json.load(f)

        txt_file = file.replace(".json", ".txt")
        txt_path = os.path.join(output_folder, txt_file)

        with open(txt_path, "w") as out:

            # Pixlab JSON is a LIST
            if not isinstance(data, list):
                print(f"❌ Skipping {file}: not list format")
                continue

            for obj in data:

                # -------------------------------
                # LABEL EXTRACTION
                # -------------------------------
                if "labels" in obj and "labelName" in obj["labels"]:
                    label = obj["labels"]["labelName"].lower().strip()
                else:
                    print("⚠️ No label found, skipping")
                    continue

                if label not in class_map:
                    print(f"⚠️ Unknown label skipped: {label}")
                    continue

                class_id = class_map[label]

                # -------------------------------
                # BOUNDING BOX EXTRACTION
                # -------------------------------
                if "rectMask" in obj:
                    rect = obj["rectMask"]

                    if all(k in rect for k in ["xMin", "yMin", "width", "height"]):
                        xmin = rect["xMin"]
                        ymin = rect["yMin"]
                        width = rect["width"]
                        height = rect["height"]
                    else:
                        print("⚠️ Invalid rectMask format")
                        continue
                else:
                    print("⚠️ No rectMask found")
                    continue

                # -------------------------------
                # CONVERT TO YOLO FORMAT
                # -------------------------------
                x_center = (xmin + width / 2) / image_width
                y_center = (ymin + height / 2) / image_height
                w = width / image_width
                h = height / image_height

                # -------------------------------
                # VALIDATION (IMPORTANT)
                # -------------------------------
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    print("⚠️ Invalid bbox skipped")
                    continue

                # -------------------------------
                # WRITE TO FILE
                # -------------------------------
                out.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

        print(f"✅ Converted: {file}")

print("🎉 ALL FILES CONVERTED SUCCESSFULLY!")