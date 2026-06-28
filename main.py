"""""
Digit Recognizer Project
Author: Belkin Nikita
"""
# Using a drawing library for user input
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

# Using the default load_digits dataset from scikit-learn to train a Random Forest classifier
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# ---------------- MODEL INITIALIZING ----------------
# I intended here to save dataset as isolated variables and to show its initial numbers on the screen
digits = load_digits()

fix, axes = plt.subplots(2,5, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"label: { digits.target[i]}")
    ax.axis("off")

# I used here default settings because the dataset is small and tuning of model won't affect too much
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- DRAW WIDGET ----------------
# After training the model, initialize the drawing interface
root = tk.Tk()
root.title("Digit Recognizer")

draw_canvas = tk.Canvas(root, width=280, height=280, bg="black")
draw_canvas.pack(side="left")

image = Image.new("L", (280, 280), "black")
draw = ImageDraw.Draw(image)

prediction_label = tk.Label(root, text="Prediction: ")
prediction_label.pack()

# Called whenever the user draws on the canvas. I set a thickness of the pen as 30 because it gave the best results
def paint(event):
    x,y = event.x, event.y

    draw_canvas.create_oval(
        x - 30, y - 30,
        x + 30, y + 30,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        (x - 5, y - 5, x + 5, y + 5),
        fill="white"
    )

draw_canvas.bind("<B1-Motion>", paint)

# Clear the drawing canvas
def clear_canvas():
    global image, draw

    draw_canvas.delete("all")

    image = Image.new("L", (280, 280), "black")
    draw = ImageDraw.Draw(image)

    prediction_label.config(text="Prediction: ?")

# Save the drawing as a PNG image, resize it, and pass it to the model for prediction
def predict():
    global image

    image = image.convert("L")
    image = image.resize((8, 8))
    image.save("digit.png")

    prediction_label.config(text="saved as digit.png")
    img_array = np.array(image)
    img_array = img_array.reshape(1, -1)
    prediction = model.predict(img_array)
    print(prediction)

# Connect the buttons to their corresponding functions
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

# Start the application window
plt.tight_layout()
plt.show()

predictions = model.predict(X_test)

# Display the model's accuracy and confirm that the trained model has been saved
print("Accuracy: ", accuracy_score(y_test, predictions))
with open("digit_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("model saved as digit_model.pkl")



