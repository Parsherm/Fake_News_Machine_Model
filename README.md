# Fake News Machine Learning Model
This project aims to detect whether a given news article is REAL or FAKE using a machine learning model. It utilizes a Passive-Aggressive Classifier and TF-IDF Vectorizer to classify articles based on their content. The project includes a graphical user interface (GUI) built with CustomTkinter, making it easy for users to interact with the model.

# Features
- Train the model: The application will train the model on a dataset (news.csv) of labeled news articles.
- Prediction Interface: Users can input a news article title and content, and the model will predict if the article is real or fake.
- Accuracy Display: The accuracy of the model is displayed in the bottom right corner.
- Result Reset: After each prediction, input fields and the result label are cleared.

# Installing
The easiest way to install the program with all dependencies is to just download "FakeNewsML.exe" from this repository. However there is another option should you wish.
1. Download all files EXCEPT "FakeNewsML.exe"
2. Run the command "pip install -r requirements.txt" to download dependencies in the terminal
3. Run the command "python fake_news_detector_ui.py" in the terminal

However, I still just recommend to download "FakeNewsML.exe" and run it as it is the easier option. If you choose to do so, from the resulting folder you will only have to run "fake_news_detector_ui.exe" to start the program.

# Using the Application
1. Enter the title and content of a news article in the respective fields.
2. Click the "Check" button to get the prediction.
3. The model will predict whether the article is REAL or FAKE and display the result.
4. The accuracy of the model is shown in the bottom right corner.
5. The input fields will clear after pressing the "Reset" button.

# Files in this Project
- **fake_news_detector_ui.py:** The main Python script with the CustomTkinter GUI.
- **news.csv:** The dataset containing news articles labeled as FAKE or REAL.
- **fake_news_model.pkl:** The trained model file (generated after training).
- **tfidf_vectorizer.pkl:** The trained TF-IDF vectorizer (generated after training).
- **requirements.txt:** A file listing all the dependencies needed to run the project

# Acknowledgements
The news.csv dataset used for this project is a commonly used dataset for fake news detection tasks. You can find similar datasets or use your own.
CustomTkinter for creating the GUI in Python.

# DISCLAIMER
The dataset used to train this model is older, and was last updated in 2019. Because of this, some predictions may be inaccurate as media can change over time.
