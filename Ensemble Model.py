# Building our ensemble model using SVM, Random Forest, Logistic Regression and Naives Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("/Users/tonytonggg/Desktop/Side_Project/Datasets/mbti_cleaned_2.csv", encoding="ISO-8859-1")
#Cleaning some erroneous tweets
dataset.Tweet[871] = "null"
dataset.Tweet[969] = "null"
dataset.Tweet[1003] = "null"

#run this earlier, this takes ~10 mins to run
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1187):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', dataset['Tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000) #How does this affect?
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# ------- NAIVE BAYES -------- ACCURACY ~ 37%

# Fitting Naive Bayes to training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

# Applying k-Fold Cross Validation ** Did not apply to the other 3 models!
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print ("K-fold: ")
print(accuracies.mean())
#print(accuracies.std())

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)
print (cm)
diagonal = [cm[i][i] for i in range(len(cm)) ]
diagonal = sum(diagonal)
statement = "Accuracy: "
print (statement)
print (diagonal/ np.sum(cm))

# ------- KERNEL SVM -------- ACCURACY ~ 35.83%%

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred2 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print ("K-fold: ")
print(accuracies.mean())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
print (cm)
diagonal = [cm[i][i] for i in range(len(cm)) ]
diagonal = sum(diagonal)
statement = "Accuracy: "
print (statement)
print (diagonal/ np.sum(cm))

# ------- RANDOM FOREST -------- ACCURACY ~ 44.58%%

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred3 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print ("K-fold: ")
print(accuracies.mean())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)
print (cm)
diagonal = [cm[i][i] for i in range(len(cm)) ]
diagonal = sum(diagonal)
statement = "Accuracy: "
print (statement)
print (diagonal/ np.sum(cm))

# ------- LOGISTIC REGRESSION CLASSIFIER -------- ACCURACY ~ 25.83%%

# Fitting a Logistic Regression To The Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred4 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred4) # true = (65 + 24), false = (8 + 3)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print ("K-fold: ")
print(accuracies.mean())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred4)
print (cm)
diagonal = [cm[i][i] for i in range(len(cm)) ]
diagonal = sum(diagonal)
statement = "Accuracy: "
print (statement)
print (diagonal/ np.sum(cm))

#################

########################################################################
# ------- ENSEMBLE MODEL --------
# Merging the response variables from the 4 tests above into a combined dataframe
data = np.column_stack((y_pred1,y_pred2))
data = np.column_stack((data,y_pred3))
data = np.column_stack((data,y_pred4))
data = pd.DataFrame(data)

X = data.iloc[:, [0, 1, 2, 3]].values
#y = pd.DataFrame(y_test)
data = np.column_stack((data,y_test))
#y_test here comes from the initial y test above, which is used as full (training + testing ) data here

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test2 = train_test_split(X, y_test, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

########################################################################
#SVM Ensemble Model
# Fitting classifier to the Training set
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0) #what do all the different kernels do?
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred5 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2, y_pred5)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print ("K-fold: ")
print(accuracies.mean())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2, y_pred5)
print (cm)
diagonal = [cm[i][i] for i in range(len(cm)) ]
diagonal = sum(diagonal)
statement = "Accuracy: "
print (statement)
print (diagonal/ np.sum(cm))

########################################
#To directly input a string into the bag of words and get an answer. Gradient descent is the current example, replace it with input from user.
inpVec = []
inp = re.sub('[^a-zA-Z]', ' ', "Gradient descent is a first-order optimization algorithm. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or of the approximate gradient) of the function at the current point. If instead one takes steps proportional to the positive of the gradient, one approaches a local maximum of that function; the procedure is then known as gradient ascent. Gradient descent is also known as steepest descent, or the method of steepest descent. Gradient descent should not be confused with the method of steepest descent for approximating integrals.")
inp = inp.lower()
inp = inp.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in inp if not word in set(stopwords.words('english'))]
inp = ' '.join(inp)
inpVec.append(inp)

newVec = CountVectorizer(vocabulary=cv.vocabulary_)
newVec = newVec.fit_transform(inpVec).toarray()

print (classifier.predict(newVec))
