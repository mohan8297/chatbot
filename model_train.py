import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle

# Load preprocessed Q&A pairs
qa_pairs = pd.read_csv('qa_pairs.csv')

# Convert all entries in questions and answers to strings
qa_pairs['question'] = qa_pairs['question'].astype(str)
qa_pairs['answer'] = qa_pairs['answer'].astype(str)

questions = qa_pairs['question'].values
answers = qa_pairs['answer'].values

# Tokenize and pad sequences for questions
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(questions)
question_sequences = tokenizer.texts_to_sequences(questions)
max_question_length = 100
padded_question_sequences = pad_sequences(question_sequences, maxlen=max_question_length)

# Tokenize and pad sequences for answers
tokenizer.fit_on_texts(answers)
answer_sequences = tokenizer.texts_to_sequences(answers)
max_answer_length = 100
padded_answer_sequences = pad_sequences(answer_sequences, maxlen=max_answer_length)

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Define the model
model = Sequential([
    Input(shape=(max_question_length,)),
    Embedding(input_dim=10000, output_dim=256),  # Increased embedding dimension
    LSTM(128, return_sequences=True),
    Dense(128, activation='relu'),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')  # Output layer size matches the tokenizer's vocabulary size
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convert answers to categorical format
answer_sequences = np.expand_dims(padded_answer_sequences, -1)  # Ensure the shape matches the output

# Train the model
model.fit(padded_question_sequences, answer_sequences, epochs=10, validation_split=0.2)

# Save the model
model.save('qa_model.h5')
