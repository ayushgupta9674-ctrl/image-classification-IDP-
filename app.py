from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import os
import uuid
import re


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# AI MODEL
# ==========================================

MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

print("======================================")
print("Loading AI model...")
print("======================================")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.eval()

print("AI model loaded successfully!")
print("======================================")


# ==========================================
# AI PREDICTION FUNCTION
# ==========================================

def predict_image(image_path):

    print("\n======================================")
    print("Starting prediction...")
    print("Image:", image_path)
    print("======================================")

    # Open image
    image = Image.open(image_path).convert("RGB")

    print("Image loaded successfully.")


    # ======================================
    # PROMPT
    # ======================================

    prompt = """
Look carefully at this image.

Identify the main subject or object.

Give ONLY the name of the main subject.

Examples:
Shark
Dog
Car
Bus
Smartphone
Building
Mountain
Ocean
Pizza
Tree
Person
Durga
Hindu deity
Statue
House
City
People

Do not describe the image.
Do not give a sentence.
Do not explain your answer.

Give only one short identification.

If the main subject genuinely cannot be identified,
answer Unknown.
"""


    # ======================================
    # CHAT MESSAGE
    # ======================================

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]


    print("Creating AI prompt...")


    # ======================================
    # APPLY CHAT TEMPLATE
    # ======================================

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )


    print("Processing image...")


    # ======================================
    # PROCESS IMAGE
    # ======================================

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt"
    )


    # ======================================
    # AI INFERENCE
    # ======================================

    print("Running AI inference...")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )


    print("Inference completed.")


    # ======================================
    # GET GENERATED TEXT
    # ======================================

    generated = outputs[
        :,
        inputs["input_ids"].shape[-1]:
    ]


    result = processor.batch_decode(
        generated,
        skip_special_tokens=True
    )[0].strip()


    # ======================================
    # CLEAN RESULT
    # ======================================

    result = result.replace(
        "\n",
        " "
    ).strip()


    result = re.sub(
        r"[.!?,:;]+$",
        "",
        result
    ).strip()


    # Remove unnecessary phrases
    unwanted_phrases = [
        "the image shows",
        "the image contains",
        "this image shows",
        "this is",
        "it is",
        "i see",
        "the main subject is",
        "the main object is"
    ]


    lower_result = result.lower()


    for phrase in unwanted_phrases:

        if lower_result.startswith(phrase):

            result = result[len(phrase):].strip()

            result = result.lstrip(
                ":,- "
            )

            break


    # ======================================
    # FALLBACK
    # ======================================

    if not result:

        result = "Unknown"


    print("--------------------------------------")
    print("FINAL PREDICTION:", result)
    print("--------------------------------------")


    return result


# ==========================================
# SERVE UPLOADED IMAGES
# ==========================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ==========================================
# MAIN PAGE
# ==========================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    prediction = None

    image_path = None

    image_name = None

    error = None


    # ======================================
    # GET REQUEST
    # ======================================

    if request.method == "GET":

        return render_template(
            "index.html",
            prediction=prediction,
            image_path=image_path,
            image_name=image_name,
            error=error
        )


    # ======================================
    # CHECK FILE
    # ======================================

    if "image" not in request.files:

        error = "No image selected."

        return render_template(
            "index.html",
            prediction=None,
            image_path=None,
            image_name=None,
            error=error
        )


    file = request.files["image"]


    # ======================================
    # EMPTY FILE
    # ======================================

    if file.filename == "":

        error = "Please select an image."

        return render_template(
            "index.html",
            prediction=None,
            image_path=None,
            image_name=None,
            error=error
        )


    # ======================================
    # FILE EXTENSION
    # ======================================

    original_name = file.filename

    extension = os.path.splitext(
        original_name
    )[1].lower()


    allowed_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]


    if extension not in allowed_extensions:

        error = (
            "Please upload "
            "JPG, JPEG, PNG or WEBP image."
        )

        return render_template(
            "index.html",
            prediction=None,
            image_path=None,
            image_name=None,
            error=error
        )


    # ======================================
    # CREATE UNIQUE FILE NAME
    # ======================================

    filename = (
        str(uuid.uuid4())
        + extension
    )


    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    # ======================================
    # SAVE IMAGE
    # ======================================

    file.save(filepath)

    print("\nImage saved:")
    print(filepath)


    # ======================================
    # PREDICT
    # ======================================

    try:

        prediction = predict_image(
            filepath
        )


        # IMPORTANT:
        # Browser accesses the image
        # through Flask /uploads route.

        image_path = (
            "/uploads/"
            + filename
        )


        image_name = original_name


    except Exception as e:

        print("\n======================================")
        print("ERROR:")
        print(e)
        print("======================================")

        error = (
            "AI prediction failed: "
            + str(e)
        )


    # ======================================
    # SEND RESULT TO HTML
    # ======================================

    return render_template(
        "index.html",

        prediction=prediction,

        image_path=image_path,

        image_name=image_name,

        error=error
    )


# ==========================================
# START FLASK SERVER
# ==========================================

if __name__ == "__main__":

    print("\n======================================")
    print("AI IMAGE CLASSIFIER")
    print("======================================")
    print("Server starting...")
    print("Open: http://127.0.0.1:5000")
    print("======================================\n")


    app.run(
        debug=False,
        use_reloader=False,
        host="127.0.0.1",
        port=5000
    )