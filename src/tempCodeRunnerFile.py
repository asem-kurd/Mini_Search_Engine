import os
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = preprocess_text(file.read())
                for word in set(content):
                    inverted_index[word].add(filename)
    return inverted_index

# Search for a phrase
def phrase_retrieval(query, directory):
    results = []
    query = query.lower()
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read().lower()
                if query in content:
                    results.append(filename)
    return results

# Soundex search
def soundex_search(query, inverted_index):
    query_soundex = soundex(query)
    matches = {word for word in inverted_index.keys() if soundex(word) == query_soundex}
    return matches

# Correct query using Soundex
def correct_query(query, inverted_index):
    if query in inverted_index:
        return query, "Term Search"
    soundex_matches = soundex_search(query, inverted_index)
    if soundex_matches:
        corrected = list(soundex_matches)[0]  # Take the first match
        return corrected, "Soundex Search"
    return query, "Not Found"

# Main search function
def search(query, directory, inverted_index):
    if query.startswith('"') and query.endswith('"'):
        # Phrase Search
        phrase = query.strip('"')
        results = phrase_retrieval(phrase, directory)
        return results, "Phrase Search", None
    else:
        # Correct and search term
        corrected_query, search_type = correct_query(query, inverted_index)
        if search_type == "Term Search":
            results = inverted_index.get(corrected_query, set())
        elif search_type == "Soundex Search":
            results = inverted_index.get(corrected_query, set())
        else:
            results = []
        return results, search_type, corrected_query if corrected_query != query else None

# Main program
if __name__ == "__main__":
    # Path to the documents directory
    directory = "documents/"
    os.makedirs(directory, exist_ok=True)

    # Build the inverted index
    inverted_index = build_inverted_index(directory)

    while True:
        query = input("\nEnter your search query (or type '0' to quit): ").strip()
        if query.lower() == "0":
            print("Exiting the search engine. Goodbye!")
            break

        results, search_type, corrected_query = search(query, directory, inverted_index)
        if corrected_query:
            print(f"\nDid you mean: {corrected_query}?")
        
        if results:
            print(f"\nSearch Type: {search_type}")
            print(f"Results for '{corrected_query or query}':")
            for result in results:
                print(f"- {result}")
        else:
            print(f"\nNo results found for '{query}'.")








