AI Image Classification System
An AI-powered Image Classification Web Application that automatically analyzes uploaded images and predicts what they contain.

The project combines YOLO-based image classification with SmolVLM for intelligent image understanding and provides a simple web interface for users to upload and analyze images.

Features
Upload JPG, JPEG, and PNG images
AI-powered image classification
Automatic prediction of image categories
Uses YOLO for image classification
Uses SmolVLM-500M-Instruct for image understanding
User-friendly web interface
Fast local inference
Displays prediction results with confidence
Runs completely on your local machine
No image data needs to be uploaded to an external server
Tech Stack
AI / Machine Learning
Python
PyTorch
Ultralytics YOLO
Hugging Face Transformers
SmolVLM-500M-Instruct
Backend
Flask
Frontend
HTML
CSS
JavaScript
Libraries
torch
transformers
Pillow
ultralytics
flask
Project Structure
image-classification/
│
├── app.py
├── index.html
├── requirements.txt
│
├── best.pt
│
├── static/
│   └── ...
│
├── templates/
│   └── index.html
│
├── runs/
│   └── classify/
│       └── ...
│
└── README.md
Your exact folder structure may vary depending on where you keep the trained YOLO model and frontend files.

Models Used
YOLO Image Classification
The project uses a trained YOLO classification model to identify the category of an image.

The trained model is stored as:

best.pt
The model was trained using a custom image classification dataset.

SmolVLM
The project also uses:

HuggingFaceTB/SmolVLM-500M-Instruct
SmolVLM is a lightweight Vision-Language Model capable of analyzing images and generating useful descriptions or interpretations.

⚙️ Installation
1. Clone the Repository
git clone https://github.com/YOUR-USERNAME/image-classification.git
Move into the project directory:

cd image-classification
2. Create a Virtual Environment
python -m venv venv
Activate it on Windows:

venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
If you haven't created requirements.txt, install the main dependencies:

pip install flask torch torchvision transformers pillow ultralytics
Running the Application
Start the Flask application:

python app.py
You should see something similar to:

Running on http://127.0.0.1:5000
Open your browser and visit:

http://127.0.0.1:5000
How to Use
Open the web application.
Click the Upload Image button.
Select a JPG, JPEG, or PNG image.
The application processes the image using the AI models.
The predicted class and results are displayed on the screen.
Example:

Input Image
     ↓
Image Upload
     ↓
YOLO Classification
     ↓
Prediction + Confidence
     ↓
SmolVLM Image Understanding
     ↓
Result Displayed
Workflow
              ┌─────────────────┐
              │   User Image    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Flask Backend  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Image Processing│
              └────────┬────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      ┌──────────────┐    ┌──────────────┐
      │ YOLO Model   │    │   SmolVLM    │
      │ Classification│    │ Vision Model │
      └──────┬───────┘    └──────┬───────┘
             │                   │
             └─────────┬─────────┘
                       ▼
              ┌─────────────────┐
              │ Classification  │
              │     Result      │
              └─────────────────┘
Example Use Cases
This system can be extended for:

Building classification
Forest / nature classification
Mountain detection
Sea / beach classification
Street scene classification
Vehicle-related image classification
Urban scene classification
General image categorization
Custom dataset classification
Model Training
The YOLO classification model can be trained using a dataset organized into training and validation classes.

Example:

dataset/
│
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── class_3/
│
└── val/
    ├── class_1/
    ├── class_2/
    └── class_3/
Example training command:

yolo classify train model=yolo11n-cls.pt data=dataset epochs=20 imgsz=224
After training, the best-performing model can be found in the generated runs directory.

Prediction
The trained model generates a predicted class along with its confidence score.

Example:

Prediction: Mountain
Confidence: 94.6%
The result can then be displayed through the web interface.

Future Improvements
 Add more image categories
 Improve model accuracy
 Add drag-and-drop image upload
 Add prediction history
 Add confidence graphs
 Add webcam-based classification
 Deploy the application online
 Add mobile-responsive UI
 Add multiple image classification
 Add downloadable prediction reports
Author
Ayush Gupta

B.Tech CSE Student Adamas University

Project Status
Currently under development

The core image classification and web application functionality has been implemented. Further improvements and additional classes can be added in future versions.

License
This project is created for educational and academic purposes.

You are free to modify and improve the project for learning and experimentation.
