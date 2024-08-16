import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Define the maximum sequence length used during training
max_sequence_length = 100  # Ensure this matches the value used during training

# Load your trained model
model = load_model('qa_model.h5')

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load preprocessed Q&A pairs
qa_pairs = pd.read_csv('qa_pairs.csv')
questions = qa_pairs['question'].values
answers = qa_pairs['answer'].values

# Tokenizer for questions used during training
tokenizer_questions = Tokenizer()
tokenizer_questions.fit_on_texts(questions)

def preprocess_text(text):
    # Tokenize and pad text to the format expected by the model
    sequences = tokenizer_questions.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

def get_answer(question):
    processed_question = preprocess_text(question)
    
    # Predict using the model
    prediction = model.predict(processed_question)
    
    # Ensure prediction is 2D
    if prediction.ndim > 2:
        prediction = np.squeeze(prediction, axis=0)
    if prediction.ndim == 1:
        prediction = np.expand_dims(prediction, axis=0)
    
    # Debugging shapes
    print("Processed Prediction shape:", prediction.shape)
    
    # Compute embeddings for all questions in the dataset
    question_embeddings = []
    for q in questions:
        q_processed = preprocess_text(q)
        q_embedding = model.predict(q_processed)
        
        # Ensure q_embedding is 2D
        if q_embedding.ndim > 2:
            q_embedding = np.squeeze(q_embedding, axis=0)
        if q_embedding.ndim == 1:
            q_embedding = np.expand_dims(q_embedding, axis=0)
        
        question_embeddings.append(np.squeeze(q_embedding))
    
    question_embeddings = np.array(question_embeddings)
    
    # Debugging shapes
    print("Question Embeddings shape:", question_embeddings.shape)
    
    # Ensure embeddings have correct shapes
    if prediction.ndim == 1:
        prediction = np.expand_dims(prediction, axis=0)
    if question_embeddings.ndim == 1:
        question_embeddings = np.expand_dims(question_embeddings, axis=0)

    # Compute cosine similarities
    similarities = cosine_similarity(prediction, question_embeddings)
    
    # Find the index of the most similar question
    most_similar_idx = np.argmax(similarities)
    
    # Check if similarity is below a threshold or if no similarity is found
    if similarities[0, most_similar_idx] < 0.1:  # You may adjust the threshold
        return "No data found."

    # Return the corresponding answer
    return answers[most_similar_idx]

# Example usage
if __name__ == "__main__":
    question = "What is a chatbot?"
    print("Question:", question)
    answer = get_answer(question)
    print("Answer:", answer)
