from flask import Flask, request, render_template, redirect, url_for, session
import fitz
import nltk
import re
import gensim
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.secret_key = 'supersecretkey'  # Needed for session management

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def text_preprocessing(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)  # Remove URLs
    words = word_tokenize(cleaned_text)
    sentences = sent_tokenize(cleaned_text)
    lowercase_words = [word.lower() for word in words]
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in lowercase_words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return lemmatized_words

def load_word2vec_model(sentences):
    data = []
    for i in sentences:
        temp = []
        for j in word_tokenize(i):
            temp.append(j.lower())
        data.append(temp)
    model = gensim.models.Word2Vec(data, min_count=1, vector_size=100, sg=0, window=5, workers=4)
    return model

def embed_questions(questions, model):
    embedded_questions = []
    for question in questions:
        tokens = text_preprocessing(question)
        embeddings = [model.wv[token] for token in tokens if token in model.wv]
        if embeddings:
            embedded_questions.append(np.mean(embeddings, axis=0))
        else:
            embedded_questions.append(None)
    return embedded_questions

def find_most_similar_answer(embedded_questions, sentences, model):
    answers = []
    for embedded_question in embedded_questions:
        if embedded_question is None:
            answers.append("Unable to process the question.")
        else:
            max_similarity = -1
            most_similar_sentence = None
            for sentence in sentences:
                tokens = text_preprocessing(sentence)
                embeddings = [model.wv[token] for token in tokens if token in model.wv]
                if embeddings:
                    sentence_embedding = np.mean(embeddings, axis=0)
                    similarity = np.dot(embedded_question, sentence_embedding) / (np.linalg.norm(embedded_question) * np.linalg.norm(sentence_embedding))
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_sentence = sentence
            answers.append(most_similar_sentence)
    return answers

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf' not in request.files:
        return redirect(request.url)
    
    file = request.files['pdf']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Store file path in session
        session['file_path'] = file_path

        return redirect(url_for('ask'))

    return redirect(url_for('index'))

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if request.method == 'POST':
        question = request.form['question']
        file_path = session.get('file_path')

        if not file_path:
            return redirect(url_for('index'))

        text_pdf = extract_text(file_path)
        sentences = sent_tokenize(text_pdf)
        model = load_word2vec_model(sentences)
        embedded_question = embed_questions([question], model)
        answers = find_most_similar_answer(embedded_question, sentences, model)

        return render_template('result.html', question=question, answer=answers[0])

    return render_template('ask.html')

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
