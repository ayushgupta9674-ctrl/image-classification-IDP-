# AI Image Classification System

An AI-powered **Image Classification Web Application** that analyzes uploaded images and predicts their category using **YOLO** and **SmolVLM**.

The system provides a simple web interface where users can upload an image and receive an AI-generated classification result along with a confidence score and image understanding.

---

##  Features

*  Upload **JPG, JPEG, and PNG** images
*  AI-powered image classification
*  Custom-trained **YOLO classification model**
*  Image understanding using **SmolVLM-500M-Instruct**
*  Displays predicted class and confidence score
*  Simple and user-friendly web interface
*  Fast local inference
*  Images can be processed locally without sending them to an external server
*  Flask-based backend
*  HTML, CSS and JavaScript frontend

---

##  Tech Stack

### Artificial Intelligence / Machine Learning

* **Python**
* **PyTorch**
* **Ultralytics YOLO**
* **Hugging Face Transformers**
* **SmolVLM-500M-Instruct**

### Backend

* **Flask**

### Frontend

* **HTML5**
* **CSS3**
* **JavaScript**

### Python Libraries

```text
torch
torchvision
transformers
Pillow
ultralytics
flask
```

---

##  Models Used

### 1. YOLO Image Classification

The primary classification is performed using a custom-trained **YOLO classification model**.

The trained model is stored as:

```text
best.pt
```

The model predicts the category of the uploaded image and provides a confidence score.

Example:

```text
Prediction: Mountain
Confidence: 94.6%
```

The model can be trained on a custom dataset containing any number of image categories.

---

### 2. SmolVLM

The project also uses:

```text
HuggingFaceTB/SmolVLM-500M-Instruct
```

**SmolVLM** is a lightweight Vision-Language Model that can analyze images and provide additional visual understanding.

It can be used to generate descriptions, interpretations, or additional information about the uploaded image.

---

#  Project Structure

```text
image-classification/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── runs/
    └── classify/
        └── ...
```

> The exact folder structure may vary depending on the project configuration.

---

#  Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ayushgupta9674-ctrl/Image-classification-.git
```

Move into the project directory:

```bash
cd Image-classification-
```

---

## 2. Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv venv
```

### Windows

Activate the environment:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies manually:

```bash
pip install flask torch torchvision transformers pillow ultralytics
```

---

#  Running the Application

Start the Flask server:

```bash
python app.py
```

After successfully starting the application, you should see something similar to:

```text
Running on http://127.0.0.1:5000
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

#  How to Use

1. Open the web application.
2. Click **Upload Image**.
3. Select a JPG, JPEG, or PNG image.
4. The image is processed by the AI models.
5. YOLO predicts the image category.
6. The confidence score is calculated.
7. SmolVLM can provide additional image understanding.
8. The final result is displayed through the web interface.

---

#  System Workflow

```text
             ┌──────────────────┐
             │    User Image    │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  Flask Web App   │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Image Processing │
             └────────┬─────────┘
                      │
             ┌────────┴─────────┐
             │                  │
             ▼                  ▼
      ┌──────────────┐   ┌──────────────┐
      │ YOLO Model   │   │   SmolVLM    │
      │ Classification│   │ Vision Model │
      └──────┬───────┘   └──────┬───────┘
             │                  │
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ Prediction Result│
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  Web Interface   │
             └──────────────────┘
```

---

#  Example Prediction

For an uploaded image, the system may produce:

```text
Prediction: Mountain
Confidence: 94.6%
```

The prediction and confidence can then be displayed on the web interface.

---

#  Model Training

The YOLO classification model can be trained using a dataset organized by classes.

Example dataset structure:

```text
dataset/
│
├── train/
│   ├── building/
│   ├── forest/
│   └── mountain/
│
└── val/
    ├── building/
    ├── forest/
    └── mountain/
```

Train the YOLO classification model using:

```bash
yolo classify train model=yolo11n-cls.pt data=dataset epochs=20 imgsz=224
```

After training, the best-performing model can be found inside the generated `runs` directory.

The trained model can then be saved as:

```text
best.pt
```

and used by the Flask application for prediction.

---

#  Possible Applications

The system can be adapted for different classification tasks, including:

*  Building classification
*  Forest and nature classification
*  Mountain classification
*  Beach and sea classification
*  Vehicle-related image classification
*  Street scene classification
*  Urban scene classification
*  General image categorization
*  Custom dataset classification

---

#  Privacy

The application is designed to support **local image processing**.

Images can be processed on the user's local machine without requiring them to be uploaded to an external image-hosting service.

> Actual privacy behavior depends on how the Flask application and AI models are configured.

---

#  Future Improvements

The project can be extended with:

* [ ] Drag-and-drop image upload
* [ ] Multiple image classification
* [ ] Prediction history
* [ ] Confidence graphs
* [ ] Webcam-based classification
* [ ] Mobile-responsive interface
* [ ] More image categories
* [ ] Improved model accuracy
* [ ] Online deployment
* [ ] Downloadable prediction reports
* [ ] Real-time image classification
* [ ] Detailed AI-generated image descriptions

---

#  Project Status

** Currently Under Development**

The core image classification and web application functionality has been implemented.

Future versions can improve the user interface, model accuracy, prediction history, image understanding, and deployment capabilities.

---

#  Author

**Ayush Gupta**

B.Tech CSE Student
**Adamas University**

---

#  License

This project is created for **educational and academic purposes**.

You are free to modify and improve the project for learning, experimentation, and academic projects.
