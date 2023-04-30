import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import pickle

# Load model
with open('svm_model.pkl', 'rb') as file:
    # Call load method to deserialze
    model = pickle.load(file)

# Load Vectorizer
with open('vectorizer.pkl', 'rb') as file:
    # Call load method to deserialze
    vectorizer = pickle.load(file)

def detect(text:str):
    labels = ["Hate Speech","Offensive","Neutral"]
    # convert to lower case
    text = text.lower()
    # tokenize
    text = word_tokenize(text)
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    text_final = [str(Final_words)]
    vectorized_text = vectorizer.transform(text_final)
    predictions = model.predict(vectorized_text)
    print(f"Category of given text is '{labels[predictions[0]]}' ")

while True:
    text = input("Please enter the text: ")
    detect(text)