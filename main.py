
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import logging

# Download necessary NLTK data
nltk.download("punkt")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample FAQ data
faq_data = {
    "questions": [
        "What are your working hours?",
        "How do I reset my password?",
        "Can I change my email address?",
        "Where can I track my order?",
        "How do I contact support?",
        "Do you ship internationally?",
        "What is your return policy?",
        "How do I update my billing information?"
    ],
    "answers": [
        "We are available from 9 AM to 6 PM, Monday to Saturday.",
        "You can reset your password by clicking on 'Forgot Password' at login.",
        "Yes, you can change your email address in the account settings.",
        "You can track your order through the 'Orders' section in your account.",
        "You can contact support via our help center or email support@example.com.",
        "Yes, we ship to many countries worldwide. Check our shipping policy.",
        "You can return items within 30 days of delivery. Read our return policy for more.",
        "Go to your account settings and click 'Billing' to update your information."
    ]
}

# FastAPI app
app = FastAPI()

# CORS settings to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic model for request validation
class Query(BaseModel):
    question: str

# TF-IDF setup for question matching
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_data["questions"])

@app.post("/chat/")
def get_answer(query: Query):
    user_question = query.question.strip()

    if not user_question:
        return {"answer": "Please enter a valid question."}

    user_vec = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    best_match_index = int(np.argmax(similarity))
    best_score = float(similarity[0][best_match_index])

    logger.info(f"User asked: {user_question}")
    logger.info(f"Best match score: {best_score}")

    if best_score < 0.3:
        return {"answer": "I'm not sure how to answer that yet. Please contact support@example.com for further assistance."}

    answer = faq_data["answers"][best_match_index]
    return {"answer": answer}
