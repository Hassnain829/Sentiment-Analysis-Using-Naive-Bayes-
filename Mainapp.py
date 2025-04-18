from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and preprocess the labeled data
data = pd.read_csv("Data/labeled_data.csv")
X = data['Sentiment']
y = data['Rating']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def main():
    message = ''
    emoji = ''
    sentiment = ''
    probabilities = ''

    if request.method == "POST":
        input = request.form.get("input")
        input_vec = vectorizer.transform([input])
        probabilities = nb.predict_proba(input_vec)[0]
        sentiment = nb.predict(input_vec)[0]

        # Map the predicted sentiment to a message and emoji
        if sentiment == 'negative':
            message = "Why you are so Negative 😠😠"
            emoji = "😠😠"
        elif sentiment == 'neutral':
            message = "Sentence is Neutral 😇😇"
            emoji = "😇😇"
        else:
            message = "Glad to hear!!It's a Positive Sentence😀😀"
            emoji = "😀😀"

    return render_template('Home.html', message=message, emoji=emoji, sentiment=sentiment, probabilities=probabilities)


if __name__ == '__main__':
    app.run(debug=True)
