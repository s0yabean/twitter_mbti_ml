# Logistic Regression applied on the twitter datset

#Accuracy rate was about 37%

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

# Importing the dataset
dataset = pd.read_csv("/Users/tonytonggg/Desktop/Side_Project/DataSets/mbti_1200.csv", header = None ,encoding="ISO-8859-1")
#dataset.rename(columns={0: 'Category', 1: 'Class', 2: 'Number', 3:'Tweet'}, inplace=True)

X = dataset.iloc[:, 3].values
y = dataset.iloc[:, 2].values

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems
########

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 1000
)

corpus_data_features = vectorizer.fit_transform(
    X.tolist())

corpus_data_features_nd = corpus_data_features.toarray()
#corpus_data_features_nd.shape

vocab = vectorizer.get_feature_names()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        corpus_data_features_nd[0:len(X)],
        y,
        train_size=0.85,
        random_state=0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)
diagonal = [cm[i][i] for i in range(len(cm)) ]
diagonal = sum(diagonal)
statement = "Accuracy: "
print (statement)
print (diagonal/ np.sum(cm))


# Making a classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
