import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

df = pd.read_csv('data-en-de-fr.csv',encoding = "ISO-8859-1")
print('Shape of File:- ',df.shape)
df.head()
"""Data Preprocessing for english"""

import string
def data_preprocessing(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    arr = [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    s = ''
    for x in arr:
        s += x +' '
    return s

df['f_en']=df['text'].apply(data_preprocessing)
df.head()

# missing values
df.isnull().sum()

# check for duplicate values
df.duplicated().sum()

# remove duplicates
df = df.drop_duplicates(keep='first')

# after removing duplicates checking
df.duplicated().sum()

df.shape

"""# word count after preprocessing"""

"""# testing and training split"""

from sklearn.model_selection import train_test_split
total_vocablary_size = 800
msg_label = df['labels'].map({'ham': 0, 'spam': 1}).values
X_train, X_test, y_train, y_test = train_test_split(df['f_en'],msg_label,test_size=0.2)
print(len(X_train),len(X_test))

"""# model"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words = total_vocablary_size)
tokenizer.fit_on_texts(X_train)
X_training = pad_sequences (tokenizer.texts_to_sequences(X_train))
X_testing = pad_sequences(tokenizer.texts_to_sequences(X_test))
epoch_count = 20

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
def dense_model_english():
    dense_model_english = Sequential()
    dense_model_english.add(Embedding(total_vocablary_size, 16))
    dense_model_english.add(GlobalAveragePooling1D())
    dense_model_english.add(Dense(24, activation='relu'))
    dense_model_english.add(Dropout(0.1))
    dense_model_english.add(Dense(1, activation='sigmoid'))

    dense_model_english.summary()

    dense_model_english.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])

    dense_hist = dense_model_english.fit(X_training, y_train, epochs=epoch_count, validation_data=(X_testing, y_test))

    x = dense_model_english.evaluate(X_testing, y_test)
    dense_los = x[0]
    dense_acc = x[1]
    print(x)
    print('Loss:- ',x[0])
    print('Accuracy:- ',x[1])
    '''
    plt.plot(range(1,(epoch_count+1)),dense_hist.history['accuracy'],label='Training Accuracy')
    plt.plot(range(1,(epoch_count+1)),dense_hist.history['val_accuracy'],label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Accuracy Graph for Dense Classifier')
    plt.legend()
    plt.show()

    plt.plot(range(1,(epoch_count+1)),dense_hist.history['loss'],label='Training Loss')
    plt.plot(range(1,(epoch_count+1)),dense_hist.history['val_loss'],label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Validation Graph for Dense Classifier')
    plt.legend()
    plt.show()
    '''
    return dense_model_english, dense_hist, dense_los, dense_acc

dense_model_english, dense_hist, dense_los, dense_acc = dense_model_english()

def preprocess_user_input(user_input):

    user_input = data_preprocessing(user_input)
    user_input_sequence = pad_sequences(tokenizer.texts_to_sequences([user_input]))
    return user_input_sequence
def predict_spam_or_not(user_input, model):

    preprocessed_input = preprocess_user_input(user_input)
    prediction = model.predict(preprocessed_input)
    label = "spam" if prediction[0][0] > 0.5 else "ham"

    return label
user_input = "Alert: Beware of fraudsters sending SMS related to 5G upgrade / SIM block / suspension / Pending KYC or Document verification. DO NOT CALL BACK as they may ask you to download Apps to access your mobile phone and may get confidential information including OTP. DO NOT RESPOND to SMS from unknown numbers, asking for such information other than from our official SMS ID - ViCARE. DO NOT CLICK on unknown links received through social media or private messaging apps. Avoid falling victim to such frauds, block these numbers and report to https://cybercrime.gov.in"
prediction = predict_spam_or_not(user_input, dense_model_english)
print("Predicted label:",prediction)

def preprocess_user_input(user_input):

    user_input = data_preprocessing(user_input)
    user_input_sequence = pad_sequences(tokenizer.texts_to_sequences([user_input]))
    return user_input_sequence
def predict_spam_or_not(user_input, model):

    preprocessed_input = preprocess_user_input(user_input)
    prediction = model.predict(preprocessed_input)
    label = "spam" if prediction[0][0] > 0.5 else "ham"

    return label
user_input = "U dun say so early hor... U c already then say...!"
prediction = predict_spam_or_not(user_input, dense_model_english)
print("Predicted label:",prediction)

"""# for german"""


"""# data preprocessing"""

import string
def data_preprocessing(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    arr = [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('german')]
    s = ''
    for x in arr:
        s += x +' '
    return s

df['f_de']=df['text_de'].apply(data_preprocessing)
df.head()

# missing values
df.isnull().sum()

# check for duplicate values
df.duplicated().sum()
df = df.drop_duplicates(keep='first')

df.shape


"""train and test"""

from sklearn.model_selection import train_test_split
total_vocablary_size = 800
msg_label = df['labels'].map({'ham': 0, 'spam': 1}).values
X_train, X_test, y_train, y_test = train_test_split(df['f_de'],msg_label,test_size=0.2)
print(len(X_train),len(X_test))

"""# model for german"""
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words = total_vocablary_size)
tokenizer.fit_on_texts(X_train)
X_training = pad_sequences (tokenizer.texts_to_sequences(X_train))
X_testing = pad_sequences(tokenizer.texts_to_sequences(X_test))
epoch_count = 20


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional
def dense_model_german():
    dense_model_german = Sequential()
    dense_model_german.add(Embedding(total_vocablary_size, 16))
    dense_model_german.add(GlobalAveragePooling1D())
    dense_model_german.add(Dense(24, activation='relu'))
    dense_model_german.add(Dropout(0.1))
    dense_model_german.add(Dense(1, activation='sigmoid'))

    dense_model_german.summary()

    dense_model_german.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])

    dense_hist = dense_model_german.fit(X_training, y_train, epochs=epoch_count, validation_data=(X_testing, y_test))

    x = dense_model_german.evaluate(X_testing, y_test)
    dense_los = x[0]
    dense_acc = x[1]
    print(x)
    print('Loss:- ',x[0])
    print('Accuracy:- ',x[1])
    '''
    plt.plot(range(1,(epoch_count+1)),dense_hist.history['accuracy'],label='Training Accuracy')
    plt.plot(range(1,(epoch_count+1)),dense_hist.history['val_accuracy'],label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Accuracy Graph for Dense Classifier')
    plt.legend()
    plt.show()

    plt.plot(range(1,(epoch_count+1)),dense_hist.history['loss'],label='Training Loss')
    plt.plot(range(1,(epoch_count+1)),dense_hist.history['val_loss'],label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Validation Graph for Dense Classifier')
    plt.legend()
    plt.show()
    '''
    return dense_model_german, dense_hist, dense_los, dense_acc

dense_model_german, dense_hist, dense_los, dense_acc = dense_model_german()

def preprocess_user_input(user_input):

    user_input = data_preprocessing(user_input)
    user_input_sequence = pad_sequences(tokenizer.texts_to_sequences([user_input]))
    return user_input_sequence
def predict_spam_or_not(user_input, model):

    preprocessed_input = preprocess_user_input(user_input)
    prediction = model.predict(preprocessed_input)
    label = "spam" if prediction[0][0] > 0.5 else "ham"

    return label
user_input = "Herzlichen Glückwunsch, Sie haben einen kostenlosen Urlaub gewonnen! Sichern Sie sich jetzt Ihren Preis!"
prediction = predict_spam_or_not(user_input, dense_model_german)
print("Predicted label:",prediction)

def preprocess_user_input(user_input):

    user_input = data_preprocessing(user_input)
    user_input_sequence = pad_sequences(tokenizer.texts_to_sequences([user_input]))
    return user_input_sequence
def predict_spam_or_not(user_input, model):

    preprocessed_input = preprocess_user_input(user_input)
    prediction = model.predict(preprocessed_input)
    label = "spam" if prediction[0][0] > 0.5 else "ham"

    return label
user_input = "Du sagst es nicht so früh oder... A c sag dann schon...!"
prediction = predict_spam_or_not(user_input, dense_model_german)
print("Predicted label:",prediction)

"""# for french"""

"""Data Preprocessing"""

import string
def data_preprocessing(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    arr = [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('french')]
    s = ''
    for x in arr:
        s += x +' '
    return s

df['f_fr']=df['text_fr'].apply(data_preprocessing)
df.head()

# missing values
df.isnull().sum()

# check for duplicate values
df.duplicated().sum()

df = df.drop_duplicates(keep='first')

df.shape


"""Test-Train Split"""

from sklearn.model_selection import train_test_split
total_vocablary_size = 800
msg_label = df['labels'].map({'ham': 0, 'spam': 1}).values
X_train, X_test, y_train, y_test = train_test_split(df['f_fr'],msg_label,test_size=0.2)
print(len(X_train),len(X_test))


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words = total_vocablary_size)
tokenizer.fit_on_texts(X_train)
X_training = pad_sequences (tokenizer.texts_to_sequences(X_train))
X_testing = pad_sequences(tokenizer.texts_to_sequences(X_test))
epoch_count = 20


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
def dense_model_french():
    dense_model_french = Sequential()
    dense_model_french.add(Embedding(total_vocablary_size, 16))
    dense_model_french.add(GlobalAveragePooling1D())
    dense_model_french.add(Dense(24, activation='relu'))
    dense_model_french.add(Dropout(0.1))
    dense_model_french.add(Dense(1, activation='sigmoid'))

    dense_model_french.summary()

    dense_model_french.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])

    dense_hist = dense_model_french.fit(X_training, y_train, epochs=epoch_count, validation_data=(X_testing, y_test))

    x = dense_model_french.evaluate(X_testing, y_test)
    dense_los = x[0]
    dense_acc = x[1]
    print(x)
    print('Loss:- ',x[0])
    print('Accuracy:- ',x[1])
    '''
    plt.plot(range(1,(epoch_count+1)),dense_hist.history['accuracy'],label='Training Accuracy')
    plt.plot(range(1,(epoch_count+1)),dense_hist.history['val_accuracy'],label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Accuracy Graph for Dense Classifier')
    plt.legend()
    plt.show()

    plt.plot(range(1,(epoch_count+1)),dense_hist.history['loss'],label='Training Loss')
    plt.plot(range(1,(epoch_count+1)),dense_hist.history['val_loss'],label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Validation Graph for Dense Classifier')
    plt.legend()
    plt.show()
    '''
    return dense_model_french, dense_hist, dense_los, dense_acc

dense_model_french, dense_hist, dense_los, dense_acc = dense_model_french()

def preprocess_user_input(user_input):

    user_input = data_preprocessing(user_input)
    user_input_sequence = pad_sequences(tokenizer.texts_to_sequences([user_input]))
    return user_input_sequence
def predict_spam_or_not(user_input, model):

    preprocessed_input = preprocess_user_input(user_input)
    prediction = model.predict(preprocessed_input)
    label = "spam" if prediction[0][0] > 0.5 else "ham"

    return label
user_input = "Félicitations, vous avez gagné des vacances gratuites ! Réclamez votre prix maintenant !"
prediction = predict_spam_or_not(user_input, dense_model_french)
print("Predicted label:",prediction)

def preprocess_user_input(user_input):

    user_input = data_preprocessing(user_input)
    user_input_sequence = pad_sequences(tokenizer.texts_to_sequences([user_input]))
    return user_input_sequence
def predict_spam_or_not(user_input, model):

    preprocessed_input = preprocess_user_input(user_input)
    prediction = model.predict(preprocessed_input)
    label = "spam" if prediction[0][0] > 0.5 else "ham"

    return label
#Enter the SMS text:FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv
user_input = "Entrez le texte du SMS : FreeMsg Hé, chérie, cela fait 3 semaines maintenant et aucun mot en retour ! J'aimerais encore t'amuser un peu ? Très bien ! XxX std chgs à envoyer, 1,50 £ à rcv"
prediction = predict_spam_or_not(user_input, dense_model_french)
print("Predicted label:",prediction)

#pickles and tokenizer 
import pickle
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Define and fit the English tokenizer
tokenizer_english = Tokenizer(num_words=total_vocablary_size)
tokenizer_english.fit_on_texts(X_train)

# Define and fit the French tokenizer
tokenizer_french = Tokenizer(num_words=total_vocablary_size)
tokenizer_french.fit_on_texts(X_train)

# Define and fit the German tokenizer
tokenizer_german = Tokenizer(num_words=total_vocablary_size)
tokenizer_german.fit_on_texts(X_train)

# Save the English model and tokenizer
with open("final_sms_spam_detection_english.pkl", "wb") as file:
    pickle.dump(dense_model_english, file)
with open("tokenizer_english.pkl", "wb") as file:
    pickle.dump(tokenizer_english, file)

# Save the French model and tokenizer
with open("final_sms_spam_detection_french.pkl", "wb") as file:
    pickle.dump(dense_model_french, file)
with open("tokenizer_french.pkl", "wb") as file:
    pickle.dump(tokenizer_french, file)

# Save the German model and tokenizer
with open("final_sms_spam_detection_german.pkl", "wb") as file:
    pickle.dump(dense_model_german, file)
with open("tokenizer_german.pkl", "wb") as file:
    pickle.dump(tokenizer_german, file)