import string
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__, template_folder='templatess')
CORS(app)

# Load the English, German, and French models and tokenizers
with open("final_sms_spam_detection_english.pkl", "rb") as file:
    model_english = pickle.load(file)
with open("tokenizer_english.pkl", "rb") as file:
    tokenizer_english = pickle.load(file)

with open("final_sms_spam_detection_german.pkl", "rb") as file:
    model_german = pickle.load(file)
with open("tokenizer_german.pkl", "rb") as file:
    tokenizer_german = pickle.load(file)

with open("final_sms_spam_detection_french.pkl", "rb") as file:
    model_french = pickle.load(file)
with open("tokenizer_french.pkl", "rb") as file:
    tokenizer_french = pickle.load(file)

@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/index2', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        language = request.form['language']
        user_input = request.form['user_input']

        if language == 'english':
            model = model_english
            tokenizer = tokenizer_english
        elif language == 'german':
            model = model_german
            tokenizer = tokenizer_german
        elif language == 'french':
            model = model_french
            tokenizer = tokenizer_french

        preprocessed_input = preprocess_user_input(user_input, tokenizer)
        prediction = predict_spam_or_not(preprocessed_input, model)

        return render_template('result2.html', prediction=prediction)

    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        language = request.form['language']
        user_input = request.form['user_input']

        if language == 'english':
            model = model_english
            tokenizer = tokenizer_english
        elif language == 'german':
            model = model_german
            tokenizer = tokenizer_german
        elif language == 'french':
            model = model_french
            tokenizer = tokenizer_french

        preprocessed_input = preprocess_user_input(user_input, tokenizer)
        prediction = predict_spam_or_not(preprocessed_input, model)

        return render_template('result2.html', prediction=prediction)

@app.route('/result2')
def result():
    return render_template('result2.html')

def preprocess_user_input(user_input, tokenizer):
    user_input = data_preprocessing(user_input)
    user_input_sequence = pad_sequences(tokenizer.texts_to_sequences([user_input]))
    return user_input_sequence

def predict_spam_or_not(user_input, model):
    prediction = model.predict(user_input)
    label = "Spam" if prediction[0][0] > 0.5 else "Not spam"
    return label

def data_preprocessing(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    arr = [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    s = ''
    for x in arr:
        s += x + ' '
    return s

if __name__ == '__main__':
    app.run(debug=True)