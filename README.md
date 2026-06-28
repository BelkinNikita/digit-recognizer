# 4️⃣ Digit Recognizer

## 📌 Overview

This project is an educational machine learning application that recognizes handwritten digits drawn by the user.

The application is built with Python, scikit-learn, Tkinter, and Pillow. A Random Forest classifier is trained on the built-in load_digits dataset from scikit-learn, and a graphical interface allows users to draw a digit and receive an instant prediction.

---

## 🎯 Goal

The goal of this project is to:

* Learn the complete machine learning workflow
* Work with image classification
* Train and save a machine learning model
* Build a graphical user interface with Tkinter
* Process user-drawn images for prediction

---

## 📊 Dataset

The project uses the load_digits dataset from scikit-learn.

Dataset information:

* 1,797 handwritten digit images
* 10 classes (digits 0–9)
* 8×8 grayscale images
* 64 numerical features per image

---

## 🤖 Model

The project uses a Random Forest Classifier.

Reasons for choosing this model:

* Easy to understand
* Fast to train
* High accuracy on small datasets
* Requires minimal preprocessing

---

## 🚀 How to Run

### 1. Install dependencies

```bash

pip install -r requirements.txt

```

### 2. Run the program

```bash

python main.py

```

### 3. Draw a digit

* Draw a handwritten digit (0–9).
* Click Predict to classify it.
* Click Clear to erase the canvas and draw another digit.

