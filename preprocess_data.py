import pandas as pd
import re
from nltk.tokenize import sent_tokenize

def create_qa_pairs(text):
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Handle the case where there's an odd number of sentences
    if len(sentences) % 2 != 0:
        sentences.append("")  # Add an empty string as a dummy answer
    
    # Create Q&A pairs
    questions = []
    answers = []
    for i in range(0, len(sentences) - 1, 2):
        questions.append(sentences[i])
        answers.append(sentences[i + 1])
    
    return pd.DataFrame({"question": questions, "answer": answers})

if __name__ == "__main__":
    with open('scraped_data.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    qa_pairs = create_qa_pairs(text)
    qa_pairs.to_csv('qa_pairs.csv', index=False)
