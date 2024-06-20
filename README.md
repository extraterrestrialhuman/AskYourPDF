# AskYourPDF
# (PDF Answering AI)

Project Overview:

This project implements an AI model capable of answering questions based on the content of uploaded PDF documents. It utilizes Word2Vec Embedding techniques to extract, preprocess, and analyze text data from PDF files. Users can upload a PDF document, ask questions related to its content, and receive answers based on the most relevant sentences found within the document.

Project Pipeline:

![image](https://github.com/extraterrestrialhuman/AskYourPDF/assets/113041704/03aeb9a5-e4d8-48d5-8cf9-6e76e8dc7a1c)

Deployment:

<img width="897" alt="image" src="https://github.com/extraterrestrialhuman/AskYourPDF/assets/113041704/e1ef8e0a-9f1c-4aa7-9d2a-979a68929ac1">


<img width="897" alt="image" src="https://github.com/extraterrestrialhuman/AskYourPDF/assets/113041704/b272d773-1af2-454b-a894-73e964a20e53">


<img width="897" alt="image" src="https://github.com/extraterrestrialhuman/AskYourPDF/assets/113041704/2c41fb7a-e188-4ece-a0d5-3c2d1112b53b">




Installation Instructions:

1. Clone the repository from GitHub: https://github.com/extraterrestrialhuman/AskYourPDF
2. Navigate to the project directory. (cd 'directory-address')
3. Create a virtual environment (optional but recommended):
   
    python -m venv venv
   
    venv\Scripts\activate
5. Install dependencies by running on terminal:
   
    pip install -r requirements.txt
7. Start the Flask web server on terminal:
   
    python app.py
9. Open your web browser and go to http://localhost:5000 to access the application.


Usage:

1. Upload a PDF file containing the document you want to query.
2. Ask a question related to the document's content.
3. The system will process the question and return the most relevant sentence from the PDF as the answer.


Dependencies:

Python 3.x
Flask
PyMuPDF (fitz)
NLTK (Natural Language Toolkit)
Gensim
Scipy 1.12.0


