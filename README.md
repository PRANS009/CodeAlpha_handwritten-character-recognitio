# Handwritten Alphabet Recognition Using AI

An AI/ML-based system that recognizes handwritten uppercase letters and converts them into digital text. Built with a Convolutional Neural Network (CNN) trained on the EMNIST Letters dataset and served through a Flask web application.

## ✨ Features

- **26-class recognition** — handwritten uppercase letters A-Z
- **Real-time prediction** — draw a letter and get instant AI-powered recognition
- **Top-3 predictions** — see the model's confidence for multiple possibilities
- **Beautiful web UI** — dark glassmorphism design with smooth animations
- **Undo & Clear** — full drawing control with stroke history

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **ML Framework** | PyTorch / TorchVision |
| **Model** | Convolutional Neural Network (CNN) |
| **Dataset** | EMNIST Letters |
| **Backend** | Flask (Python) |
| **Frontend** | HTML5 Canvas, CSS3, JavaScript |
| **Image Processing** | Pillow (PIL) |

## 🏗️ Architecture

```
✍️ User draws a letter
    ↓
📷 Canvas captures image (base64 PNG)
    ↓
🧹 Image preprocessing
    • Grayscale conversion & background cleaning
    • Contrast auto-normalization
    • Crop to bounding box
    • Resize to 20×20, center on 28×28
    • Normalize to [0, 1] tensor
    ↓
🧠 PyTorch CNN Model analyses (Conv2D → BatchNorm → Dense)
    ↓
✅ Top-3 predictions with confidence scores
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

- Download the EMNIST Letters dataset automatically via TorchVision
- Train a CNN with Adam optimizer and MPS/GPU acceleration
- Map the dataset's 26 classes to uppercase letters A-Z
- Save the model to `model/handwritten_model.pth`
- Generate training plots in `plots/training_history.png`

Train the model before starting the web app. The existing checkpoint, if present, may be a digit model and cannot recognize letters.

### 3. Run the Web App

```bash
python app.py
```

Open **http://127.0.0.1:5001** in your browser. Draw an uppercase letter and click **Predict**!

## 📊 Model Architecture

```
Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
Conv2D(128) → BatchNorm → ReLU → Dropout(0.25)
Flatten → Dense(256) → BatchNorm → ReLU → Dropout(0.5) → Dense(26, Logits)
```

The validation accuracy is reported at the end of training and in the training plots.

## 🔥 Applications

- 📚 Digital note-taking
- 📝 Form digitization
- 🏦 Bank cheque processing
- 📖 OCR systems
- ♿ Assistive technologies
- 🏢 Document automation

## 📁 Project Structure

```
handwritten/
├── app.py                  # Flask server & prediction API
├── train_model.py          # PyTorch CNN training pipeline
├── requirements.txt        # Python dependencies
├── README.md
├── model/
│   ├── handwritten_model.pth    # Trained PyTorch model
│   └── label_mapping.json       # Class index → uppercase letter mapping
├── plots/
│   └── training_history.png     # Accuracy & loss curves
├── templates/
│   └── index.html               # Web UI
└── static/
    ├── style.css                # Modern dark theme styles
    └── app.js                   # Canvas & prediction logic
```

## 📜 License

This project is for educational and demonstration purposes.

---

*From **handwriting → AI → digital text** 🤖✨*
