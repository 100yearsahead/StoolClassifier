# StoolClassifier
A PyTorch image classifier trained on the Blood Cell Images dataset using transfer learning with ResNet18. Classifies images into 4 medical classes (eosinophil, lymphocyte, monocyte, neutrophil). 
# 🧬 Medical Image Classifier (a.k.a. ResNet Saves Lives... Kinda)

Welcome to the world's most over-engineered blood cell sorter. This is a PyTorch-powered, ResNet-fueled, metric-spitting machine that takes one look at a cell and yells “That’s a Lymphocyte!” faster than your bio teacher ever could. 🧪

---

## 🧠 What Is This?

We built a deep learning model using **transfer learning** with ResNet18 to classify images of blood cells. We aim on you're catching eosinophils, lymphocytes, monocytes, and neutrophils.

---

## 🗂️ Dataset

We use the [Blood Cell Images dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) from Kaggle. It's a clean, well-organized dataset.
Directory structure (after running `split_dataset.py`):

```
data/
├── train/
│   ├── EOSINOPHIL/
│   ├── LYMPHOCYTE/
│   ├── MONOCYTE/
│   └── NEUTROPHIL/
├── val/
└── test/
```

Use `split_dataset.py` to auto-slice your data like a pro chef.

---

## 🚀 Quickstart (a.k.a. How to Look Smart Fast)

### 1. Install Stuff
```bash
python -m venv venv
source venv/bin/activate  # Or use your OS equivalent
pip install -r requirements.txt
```

### 2. Train the Model (Flex those GPUs 💪)
```bash
python model/train.py
```

### 3. Predict an Image (Instant Gratification)
```bash
python predict.py --image path_to_image.jpg
```

The model will politely whisper (or scream) the predicted class.

---

## 📊 What’s Under the Hood?

- 🧠 **ResNet18** (because we’re not trying to reinvent the neural wheel)
- 🔁 **Transfer Learning** (pretrained weights to the rescue do not have compute no worries)
- 🎯 **CrossEntropyLoss + Adam** optimizer
- 📈 Metrics: Accuracy, Precision, Recall, F1-score (might extend to other params!)

---

## 📁 Project Structure

```
medical-image-classifier/
├── data/                # Dataset (you provide)
├── model/
│   └── train.py         # Training script
├── predict.py           # Prediction script
├── split_dataset.py     # Splits raw data into train/val/test
├── requirements.txt     # Pip stuff
└── README.md            # This magnificent file
```

---

## ⚠️ Disclaimer (a.k.a. Please Don’t Sue Me)

This is an educational project. This model cannot diagnose, treat, or tell your grandma if her lymphocytes are acting up. For real medical advice, consult a human — preferably one with a degree.

---

## 💡 Future Enhancements

- Deploy with **Streamlit** for drag-and-drop predictions (might build the web interface from scratch insted as steamlit has its limitations)
- Add **Grad-CAM** to visualize which parts of the image the model cares about (very important more research to be done here)
- Cloud deployment with **Hugging Face Spaces** or **Gradio** (Probably Hugging Face)
- Train it on *other* datasets like stool images (yup, we’re serious) this would be the goal!

---

## ✨ Author

Made by Omari March – powered by curiosity, caffeine, and a concerning number of Stack Overflow tabs.

---

## 📜 License

Open Source

Now go forth and classify! 🔬
