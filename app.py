from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the model and other required variables
df = pd.read_csv('train.csv')
df = df.dropna()
msg = df.copy()
msg.reset_index(inplace=True)

ps = PorterStemmer()
corpus = []

for i in range(0, len(msg)):
    review = re.sub('[^a-zA-Z]', ' ', msg['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))
X = cv.fit_transform(corpus).toarray()
y = msg['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=0)

classifier = MultinomialNB()
classifier.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
