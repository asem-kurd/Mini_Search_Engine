import os
import jellyfish
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Assuming nltk stopwords are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Directory containing text files
DOCUMENTS_DIR = "documents"

# Function to preprocess documents
def preprocess(document):
    # Tokenization and normalization
    tokens = word_tokenize(document.lower())
    # Remove stop words and non-alphabetic tokens
    return [word for word in tokens if word.isalpha() and word not in stop_words]

# Function to load documents
def load_documents():
    documents = {}
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(DOCUMENTS_DIR, filename), 'r', encoding='utf-8') as file:
                documents[filename] = preprocess(file.read())
    return documents

# Search for terms using OR logic
def search_terms(query, documents):
    query_terms = set(preprocess(query))
    results = defaultdict(int)
    
    for filename, words in documents.items():
        for term in query_terms:
            if term in words:
                results[filename] += 1  # Increment relevance score

    return sorted(results.items(), key=lambda x: x[1], reverse=True)

# Search for phrases
def search_phrases(phrase, documents):
    phrase = phrase.lower()
    results = defaultdict(int)
    
    for filename, words in documents.items():
        if phrase in ' '.join(words):
            results[filename] += 1  # Increment relevance score

    return sorted(results.items(), key=lambda x: x[1], reverse=True)

# Handle character loss (fuzzy search)
def fuzzy_search(query, documents, threshold=2):
    results = defaultdict(int)
    
    for filename, words in documents.items():
        for word in words:
            if jellyfish.levenshtein_distance(query, word) <= threshold:
                results[filename] += 1  # Increment relevance score

    return sorted(results.items(), key=lambda x: x[1], reverse=True)

# Phonetics handling using Soundex
def phonetic_search(query, documents):
    query_soundex = jellyfish.soundex(query)
    results = defaultdict(int)
    
    for filename, words in documents.items():
        for word in words:
            if jellyfish.soundex(word) == query_soundex:
                results[filename] += 1  # Increment relevance score

    return sorted(results.items(), key=lambda x: x[1], reverse=True)

# Evaluation metrics
def precision(retrieved, relevant):
    return len(set(retrieved) & set(relevant)) / len(retrieved) if retrieved else 0

def recall(retrieved, relevant):
    return len(set(retrieved) & set(relevant)) / len(relevant) if relevant else 0

def accuracy(total, correct):
    return correct / total if total else 0

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

def main():
    print("Welcome to the Enhanced Mini Search Engine!")
    
    # Load documents
    documents = load_documents()
    
    # User input
    user_query = input("Enter your search term or phrase: ")

    # Determine if it's a term or phrase search
    if '"' in user_query:
        # Phrase search
        phrase = user_query.strip('"')
        results = search_phrases(phrase, documents)
    else:
        # Term search
        results = search_terms(user_query, documents)

    # Fuzzy search and phonetic search
    fuzzy_results = fuzzy_search(user_query, documents)
    phonetic_results = phonetic_search(user_query, documents)

    # Combine results
    combined_results = sorted(set(results + fuzzy_results + phonetic_results), key=lambda x: x[1], reverse=True)

    # Display results
    if combined_results:
        print("\nSearch Results:")
        for result in combined_results:
            print(f"- {result[0]}: Relevance Score: {result[1]}")
    else:
        print("\nNo matches found.")

    # Evaluation metrics (example, you can adjust the relevant set)
    relevant_set = {'document1.txt', 'document2.txt'}  # Example relevant documents
    retrieved_set = {result[0] for result in combined_results}

    p = precision(retrieved_set, relevant_set)
    r = recall(retrieved_set, relevant_set)
    a = accuracy(len(documents), len(retrieved_set))
    f1 = f1_score(p, r)

    print(f"\nEvaluation Metrics:\nPrecision: {p:.2f}\nRecall: {r:.2f}\nAccuracy: {a:.2f}\nF1 Score: {f1:.2f}")

if __name__ == "__main__":
    main()