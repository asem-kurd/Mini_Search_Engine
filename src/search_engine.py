import os
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Soundex implementation
def soundex(name):
    soundex_mapping = {
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    if not name:
        return "0000"
    first_letter = name[0].upper()
    encoded = first_letter
    for char in name[1:].lower():
        if char in soundex_mapping:
            digit = soundex_mapping[char]
            if encoded[-1] != digit:
                encoded += digit
    return (encoded + '000')[:4]

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    stop_words = set(['the', 'is', 'and', 'in', 'to', 'a', 'of', 'it', 'for'])
    return [word for word in text.split() if word not in stop_words]

# Build inverted index
def build_inverted_index(directory):
    inverted_index = defaultdict(set)
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                documents[filename] = content
                tokens = preprocess_text(content)
                for word in set(tokens):
                    inverted_index[word].add(filename)
    return inverted_index, documents

# Rank files using TF-IDF and Cosine Similarity
def rank_files(query, documents):
    filenames = list(documents.keys())
    contents = list(documents.values())

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contents)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_files = [(filenames[i], scores[i]) for i in range(len(scores)) if scores[i] > 0]
    ranked_files = sorted(ranked_files, key=lambda x: x[1], reverse=True)
    return ranked_files

# Apply AND logic
def search_and(query_tokens, inverted_index):
    result_sets = [inverted_index.get(token, set()) for token in query_tokens]
    return set.intersection(*result_sets) if result_sets else set()

# Correct query using Soundex
def correct_query(query, inverted_index):
    query_tokens = query.lower().split()
    corrected_tokens = []

    for token in query_tokens:
        if token in inverted_index:
            corrected_tokens.append(token)
        else:
            soundex_matches = soundex_search(token, inverted_index)
            if soundex_matches:
                corrected_tokens.append(list(soundex_matches)[0])
            else:
                corrected_tokens.append(token)

    corrected_query = ' '.join(corrected_tokens)
    return corrected_query if corrected_query != query else None

# Soundex search
def soundex_search(query, inverted_index):
    query_soundex = soundex(query)
    matches = {word for word in inverted_index.keys() if soundex(word) == query_soundex}
    return matches

# Calculate evaluation metrics
def calculate_metrics(retrieved_docs, relevant_docs, total_docs):
    true_positives = len(retrieved_docs & relevant_docs)
    false_positives = len(retrieved_docs - relevant_docs)
    false_negatives = len(relevant_docs - retrieved_docs)
    true_negatives = total_docs - (true_positives + false_positives + false_negatives)

    precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
    recall = true_positives / len(relevant_docs) if relevant_docs else 0
    accuracy = (true_positives + true_negatives) / total_docs if total_docs > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, accuracy, f1_score

# Main search function
# Main search function with phrase search support
def search(query, inverted_index, documents):
    # Check if query is a phrase (enclosed in quotes)
    if query.startswith('"') and query.endswith('"'):
        phrase = query[1:-1].lower()
        matching_files = set()
        for filename, content in documents.items():
            if phrase in content.lower():
                matching_files.add(filename)
        corrected_query = None  # No correction for exact phrases
        ranked_results = rank_files(phrase, {f: documents[f] for f in matching_files})
        return ranked_results, matching_files, corrected_query

    # Split query into tokens for AND logic
    query_tokens = query.lower().split()

    # Correct the query if needed
    corrected_query = correct_query(query, inverted_index)
    if corrected_query:
        print(f"\nDid you mean: {corrected_query}?")
        query_tokens = corrected_query.split()

    # Perform AND search
    matching_files = search_and(query_tokens, inverted_index)

    # Rank files based on relevance
    ranked_results = rank_files(' '.join(query_tokens), documents)
    return ranked_results, matching_files, corrected_query

# Main program
if __name__ == "__main__":
    directory = "documents/"
    os.makedirs(directory, exist_ok=True)

    inverted_index, documents = build_inverted_index(directory)

    while True:
        query = input("\nEnter your search query (or type '0' to quit): ").strip()
        if query.lower() == "0":
            print("Exiting the search engine. Goodbye!")
            break

        ranked_results, retrieved_docs, corrected_query = search(query, inverted_index, documents)
        relevant_docs = set.union(*[inverted_index.get(token, set()) for token in query.lower().split()]) if not query.startswith('"') else retrieved_docs
        total_docs = len(documents)

        if ranked_results:
            print(f"\nRanked Results for '{corrected_query or query}':")
            for idx, (filename, score) in enumerate(ranked_results, start=1):
                print(f"{idx}. {filename}: {score:.4f}")
        else:
            print(f"\nNo results found for '{query}'.")

        precision, recall, accuracy, f1_score = calculate_metrics(set(doc for doc, _ in ranked_results), relevant_docs, total_docs)

        print("\nEvaluation Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1_score:.4f}")