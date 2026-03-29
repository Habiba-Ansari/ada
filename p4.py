import pandas as pd

data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1','v2']]
data.columns = ['label','text']

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(len(data['text'])):
    text = re.sub('[^a-zA-Z]', ' ', data['text'][i])
    text = text.lower().split()
    text = [word for word in text if word not in stopwords.words('english')]
    text = [lemmatizer.lemmatize(word) for word in text]
    corpus.append(' '.join(text))

data['text'] = corpus

from sklearn.model_selection import train_test_split

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=123
)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

X_test_cv = cv.transform(X_test)
predictions = model.predict(X_test_cv)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
