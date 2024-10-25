streamlit run main.py
#SetUp the Environment
git add .
git commit -m "Initial commit: added Streamlit UI for AI PDF chat"
git push origin main
#AI Chat Interface in Streamlit
import streamlit as st
from PyPDF2 import PdfReader
import requests

st.title("AI Chat with PDF Knowledge Base")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    st.write("PDF uploaded successfully!")
    st.text_area("Extracted Text", text[:500], height=300)

    query = st.text_input("Ask a question based on the PDF")

    if query:
        st.write(f"Your query: {query}")
        # Send the query to the API
        response = requests.post("http://localhost:8000/query", json={"query": query, "document": text})
        st.write(f"AI Response: {response.json().get('answer')}")
  #LLM Integration(Using Hugging Face)
 from transformers import pipeline

 def get_model():
    model = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad")
    return model

 def answer_question(model, question, context):
    result = model(question=question, context=context)
    return result['answer']
#RAG Implementation
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Generate embeddings for document
def generate_embeddings(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Indexing
def build_index(documents):
    embeddings = np.vstack([generate_embeddings(doc) for doc in documents])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Search the index
def search_index(index, query):
    query_embedding = generate_embeddings(query)
    D, I = index.search(query_embedding, k=1)
    return I[0]
#API Development
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()
model = get_model()

@app.post("/query")
async def query_pdf(request: Request):
    data = await request.json()
    query = data['query']
    document = data['document']
    
    answer = answer_question(model, query, document)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
#Unit Testing 
import pytest
from main import get_model, answer_question

def test_answer_question():
    model = get_model()
    context = "The capital of France is Paris."
    question = "What is the capital of France?"
    answer = answer_question(model, question, context)
    assert answer == "Paris"

