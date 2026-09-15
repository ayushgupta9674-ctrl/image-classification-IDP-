from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import os

MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

IMAGE_PATH = r"D:\IC IDP\durga maa.jpg"

print("Loading model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.eval()

print("Model loaded successfully!")

# Check image
if not os.path.exists(IMAGE_PATH):
    print("ERROR: Image not found!")
    print(IMAGE_PATH)
    exit()

image = Image.open(IMAGE_PATH).convert("RGB")

prompt = """
What is the main subject or object in this image?

Identify it accurately.

Reply with ONLY a short name.

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
Statue

If you genuinely cannot identify the main subject, reply:
Unknown
"""

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

print("Creating prompt...")

text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)

print("Processing image...")

inputs = processor(
    text=text,
    images=[image],
    return_tensors="pt"
)

print("Running AI inference...")

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )

print("Inference completed.")

generated = outputs[
    :,
    inputs["input_ids"].shape[-1]:
]

result = processor.batch_decode(
    generated,
    skip_special_tokens=True
)[0].strip()

print()
print("==============================")
print("IMAGE:", IMAGE_PATH)
print("PREDICTION:", result)
print("==============================")