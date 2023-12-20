from flask import Flask, request, jsonify
import joblib
import re
from deep_translator import GoogleTranslator
from flask_cors import CORS  # Import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# Load your trained model and vectorizer
model = joblib.load('language_detection_model.pkl')
cv = joblib.load('count_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict_language():
    data = request.json
    text = data['text']
    target_language= data['target']
    # Preprocess and vectorize the text as you did in your notebook
    # For example:
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    vectorized_text = cv.transform([text])

    # Predict the language
    prediction = model.predict(vectorized_text)
    # Convert prediction to readable format if necessary
    prediction = prediction.tolist()  # Convert the numpy array to a list
    language = le.inverse_transform(prediction)[0]

    translated = GoogleTranslator(source=language.lower(), target=target_language).translate(str(data['text']))

    return jsonify({'text': translated, 'selected' : language})

if __name__ == '__main__':
    app.run(debug=True)
