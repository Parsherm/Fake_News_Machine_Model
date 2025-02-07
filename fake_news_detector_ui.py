import numpy as np
import pandas as pd
import pickle
import os
import csv
import customtkinter as ctk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Initialize file paths
model_file_path = "fake_news_model.pkl"
vectorizer_file_path = "tfidf_vectorizer.pkl"

# Check if model and vectorizer exist, otherwise train them

print("Training model...")

# Read dataset
df = pd.read_csv("news.csv")

# Combine title and text for better context
df["combined_text"] = df["title"] + " " + df["text"]

# Get labels
labels = df["label"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(df["combined_text"], labels, test_size=0.2, random_state=7)

# Train TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Save the vectorizer
with open(vectorizer_file_path, "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Train model
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Evaluate model
y_pred = pac.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained with accuracy: {round(accuracy * 100, 2)}%")

# Save model
with open(model_file_path, "wb") as model_file:
    pickle.dump(pac, model_file)

# Load trained model and vectorizer
with open(model_file_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_file_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to predict news authenticity
def predict_news():
    title = title_entry.get()
    text = text_entry.get("1.0", "end-1c")  # Get text from Text widget
    combined_input = title + " " + text
    article_tfidf = vectorizer.transform([combined_input])
    prediction = model.predict(article_tfidf)[0]
    
    # Update result label
    result_text.set(f"Prediction: {'FAKE' if prediction == 'FAKE' else 'REAL'}")


# Function to reset the text in the boxes
def reset_text():
    title_entry.delete(0, "end")
    text_entry.delete("1.0", "end")
    result_text.set("")

# ----------------- CustomTkinter UI -----------------
ctk.set_appearance_mode("dark")  # Dark mode UI
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Fake News Detector")
app.geometry("600x400")

# Title input
ctk.CTkLabel(app, text="News Title:", font=("Arial", 14)).pack(pady=(10, 0))
title_entry = ctk.CTkEntry(app, width=500)
title_entry.pack(pady=5)

# Text input
ctk.CTkLabel(app, text="News Content:", font=("Arial", 14)).pack()
text_entry = ctk.CTkTextbox(app, width=500, height=100)
text_entry.pack(pady=5)

# Predict button
predict_button = ctk.CTkButton(app, text="Check", command=predict_news)
predict_button.pack(pady=10)

# Button to reset the text
reset_button = ctk.CTkButton(app, text="Reset", command=reset_text)
reset_button.pack(pady=10)

# Prediction result label
result_text = ctk.StringVar()
result_label = ctk.CTkLabel(app, textvariable=result_text, font=("Arial", 16), text_color="white")
result_label.pack(pady=10)

# Accuracy label in the bottom right corner
accuracy_label = ctk.CTkLabel(app, text=f"Model Accuracy: {round(accuracy * 100, 2)}%", 
                              font=("Arial", 12), text_color="grey")
accuracy_label.pack(anchor="se", padx=10, pady=10)


# Run the Tkinter app
app.mainloop()
