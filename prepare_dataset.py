from datasets import load_dataset
from pathlib import Path
import random

# ==============================
# SETTINGS
# ==============================

BASE_DIR = Path(r"D:\IC IDP\dataset")

TRAIN_IMAGES = 150
VAL_IMAGES = 40

# 40 REAL Tiny ImageNet classes
# Format: Tiny ImageNet ID -> Our class name

CLASS_MAP = {
    "n01443537": "goldfish",
    "n01629819": "salamander",
    "n01641577": "bullfrog",
    "n01698640": "alligator",
    "n01770393": "scorpion",
    "n01882714": "koala",
    "n01910747": "jellyfish",
    "n01983481": "lobster",
    "n02056570": "penguin",
    "n02099601": "golden_retriever",
    "n02099712": "labrador",
    "n02106662": "german_shepherd",
    "n02123045": "tabby_cat",
    "n02129165": "lion",
    "n02132136": "bear",
    "n02504458": "elephant",
    "n02509815": "panda",
    "n02769748": "backpack",
    "n02802426": "basketball",
    "n02814860": "lighthouse",
    "n02917067": "bullet_train",
    "n03085013": "computer_keyboard",
    "n03089624": "candy_store",
    "n03160309": "dam",
    "n03255030": "dumbbell",
    "n03388043": "fountain",
    "n03670208": "limousine",
    "n03706229": "compass",
    "n03796401": "moving_van",
    "n03902125": "payphone",
    "n04070727": "refrigerator",
    "n04118538": "rugby_ball",
    "n04146614": "school_bus",
    "n04285008": "sports_car",
    "n04356056": "sunglasses",
    "n04487081": "trolleybus",
    "n04540053": "volleyball",
    "n07583066": "guacamole",
    "n07873807": "pizza",
    "n07920052": "espresso",
}

print("Loading Tiny ImageNet dataset...")

dataset = load_dataset("zh-plus/tiny-imagenet")

train_data = dataset["train"]
val_data = dataset["valid"]

print("Dataset loaded.")
print("Training images:", len(train_data))
print("Validation images:", len(val_data))

# ==============================
# CREATE FOLDERS
# ==============================

for class_name in CLASS_MAP.values():

    (BASE_DIR / "train" / class_name).mkdir(
        parents=True,
        exist_ok=True
    )

    (BASE_DIR / "val" / class_name).mkdir(
        parents=True,
        exist_ok=True
    )

# ==============================
# PREPARE TRAIN DATA
# ==============================

print("\nPreparing training images...")

for label_id, class_name in CLASS_MAP.items():

    # Find label index
    label_index = train_data.features["label"].names.index(label_id)

    # Select images belonging to this class
    class_images = train_data.filter(
        lambda x: x["label"] == label_index
    )

    total = len(class_images)

    print(
        f"{class_name}: {total} available -> "
        f"{TRAIN_IMAGES} selected"
    )

    selected = list(range(total))
    random.seed(42)
    random.shuffle(selected)

    selected = selected[:TRAIN_IMAGES]

    for count, index in enumerate(selected):

        image = class_images[index]["image"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        save_path = (
            BASE_DIR
            / "train"
            / class_name
            / f"{class_name}_{count:04d}.jpg"
        )

        image.save(save_path, "JPEG")

# ==============================
# PREPARE VALIDATION DATA
# ==============================

print("\nPreparing validation images...")

for label_id, class_name in CLASS_MAP.items():

    label_index = val_data.features["label"].names.index(label_id)

    class_images = val_data.filter(
        lambda x: x["label"] == label_index
    )

    total = len(class_images)

    print(
        f"{class_name}: {total} available -> "
        f"{VAL_IMAGES} selected"
    )

    selected = list(range(total))
    random.seed(100)
    random.shuffle(selected)

    selected = selected[:VAL_IMAGES]

    for count, index in enumerate(selected):

        image = class_images[index]["image"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        save_path = (
            BASE_DIR
            / "val"
            / class_name
            / f"{class_name}_{count:04d}.jpg"
        )

        image.save(save_path, "JPEG")

print("\n===================================")
print("DATASET PREPARATION COMPLETE!")
print("===================================")

print(f"\nClasses: {len(CLASS_MAP)}")
print(f"Training images: {len(CLASS_MAP) * TRAIN_IMAGES}")
print(f"Validation images: {len(CLASS_MAP) * VAL_IMAGES}")

print("\nDataset location:")
print(BASE_DIR)

print("\nNext step: YOLO11 training")