from flask import Flask, render_template, request
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import os
import uuid
import re

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

print("Loading AI model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.eval()

print("AI model loaded successfully!")


def predict_image(image_path):

    print("Starting prediction...")

    image = Image.open(image_path).convert("RGB")

    print("Image loaded.")

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

Do not describe the image.
Do not give a sentence.
Give only the most likely object or subject name.

If the main subject cannot be identified, answer:
Unknown
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    print("Creating prompt...")

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    print("Processing image...")

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt",
    )

    print("Running AI inference...")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    print("Inference completed.")

    generated = outputs[:, inputs["input_ids"].shape[-1]:]

    result = processor.batch_decode(
        generated,
        skip_special_tokens=True
    )[0].strip()

    result = result.replace("\n", " ").strip()

    print("Raw prediction:", result)

    if not result:
        result = "Unknown"

    # Remove unnecessary punctuation
    result = re.sub(r"[.!?,:;]+$", "", result).strip()

    # Keep prediction short
    words = result.split()

    if len(words) > 5:
        result = " ".join(words[:5])

    print("Final Prediction:", result)

    return result


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    image_path = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:

            error = "No image selected."

            return render_template(
                "index.html",
                prediction=prediction,
                image_path=image_path,
                error=error
            )

        file = request.files["image"]

        if file.filename == "":

            error = "Please select an image."

            return render_template(
                "index.html",
                prediction=prediction,
                image_path=image_path,
                error=error
            )

        extension = os.path.splitext(
            file.filename
        )[1].lower()

        allowed_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        ]

        if extension not in allowed_extensions:

            error = "Please upload JPG, JPEG, PNG or WEBP image."

            return render_template(
                "index.html",
                prediction=prediction,
                image_path=image_path,
                error=error
            )

        filename = str(uuid.uuid4()) + extension

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        print("Image saved:", filepath)

        try:

            prediction = predict_image(filepath)

            image_path = "/" + filepath.replace("\\", "/")

        except Exception as e:

            print("ERROR:", e)

            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        error=error
    )


if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False,
        host="127.0.0.1",
        port=5000
    )