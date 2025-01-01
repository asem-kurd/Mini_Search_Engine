import os
import jellyfish
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from math import log10, sqrt
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class MiniSearchEngine:
    """
    A mini search engine implementation that fulfills the requirements for the Information Retrieval course project.
    Includes term-based retrieval, phrase-based retrieval, character loss handling, phonetics handling, and ranking.
    """
    
    def __init__(self, folder_path: str):
        """
        Initialize the search engine with the path to the document folder.
        
        Args:
            folder_path (str): Path to the folder containing the documents to be indexed
        """
        self.folder_path = folder_path
        self.documents = {}
        self.inverted_index = defaultdict(set)
        self.doc_lengths = {}
        self.tf_matrix = defaultdict(lambda: defaultdict(float))
        
        # Text processing tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Initialize the search engine by preprocessing documents
        self._preprocess_documents()

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing, removing stopwords, and stemming."""
        tokens = word_tokenize(text.lower())
        return [
            self.stemmer.stem(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]

    def _preprocess_documents(self):
        """Preprocess all documents in the specified folder and build the necessary indices."""
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.folder_path, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    tokens = self._preprocess_text(content)
                    self.documents[filename] = tokens
                    self.doc_lengths[filename] = len(tokens)
                    
                    for token in tokens:
                        self.inverted_index[token].add(filename)
                        self.tf_matrix[filename][token] += 1

    def cosine_similarity(self, query_vector: Dict[str, float], doc_vector: Dict[str, float]) -> float:
        """Calculate the cosine similarity between the query vector and document vector."""
        dot_product = sum(query_vector[token] * doc_vector[token] for token in query_vector if token in doc_vector)
        norm_query = sqrt(sum(value ** 2 for value in query_vector.values()))
        norm_doc = sqrt(sum(value ** 2 for value in doc_vector.values()))
        
        if norm_query == 0 or norm_doc == 0:
            return 0.0
        
        return dot_product / (norm_query * norm_doc)

    def term_search(self, query: str) -> List[Tuple[str, float]]:
        """Retrieve documents containing the specified terms, with ranking based on cosine similarity."""
        tokens = self._preprocess_text(query)
        query_vector = defaultdict(float)
        
        # Calculate TF for the query
        for token in tokens:
            query_vector[token] += 1
        
        # Calculate cosine similarity for each document
        results = []
        for doc, tf in self.tf_matrix.items():
            similarity = self.cosine_similarity(query_vector, tf)
            if similarity > 0:
                results.append((doc, similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_phrase(self, phrase: str) -> List[Tuple[str, float]]:
        """Perform OR-based retrieval using positional indexes for multiple terms."""
        tokens = self._preprocess_text(phrase)
        if not tokens:
            return []
        matching_docs = set()
        
        # Finding matching documents for any token
        for token in tokens:
            if token in self.inverted_index:
                matching_docs.update(self.inverted_index[token])
        
        results = []
        scores = defaultdict(float)
        
        # Calculate scores for documents that matched any token
        for doc in matching_docs:
            for token in tokens:
                if token in self.tf_matrix[doc]:
                    scores[doc] += self.tf_matrix[doc][token]  # Use raw TF for scoring

        # Prepare results with scores for sorting
        results = [(doc, score) for doc, score in scores.items() if score > 0]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def handle_character_loss(self, query_term: str, max_distance: int = 1) -> Set[str]:
        matches = set()
        for term in self.inverted_index.keys():
            if jellyfish.distance(query_term, term) <= max_distance:
                matches.add(term)
        return matches or {query_term}

    def handle_phonetics(self, query_term: str) -> Set[str]:
        query_soundex = jellyfish.soundex(query_term)
        return {term for term in self.inverted_index.keys() if jellyfish.soundex(term) == query_soundex}

    def interactive_search(self):
        """Interactive command-line interface for the search engine."""
        print("\nWelcome to the Enhanced Mini Search Engine!")
        while True:
            query = input("\nEnter your search term or phrase (or type '0' to quit): ").strip()
            if query == '0':
                print("Thank you for using the search engine. Goodbye!")
                break
            
            start_time = time.time()
            if len(query.split()) > 1:  # Treat as phrase search if more than one word
                results = self.search_phrase(query)
            else:
                results = self.term_search(query)
            
            search_time = time.time() - start_time
            
            if results:
                print(f"\nFound {len(results)} results in {search_time:.4f} seconds:")
                for rank, (doc, score) in enumerate(results[:10], 1):  # Display top 10 results
                    print(f"{rank}. {doc} (Score: {score:.4f})")
                
                # Provide suggestions based on Soundex for phonetic issues
                phonetic_matches = self.handle_phonetics(query)
                if phonetic_matches:
                    print(f"\nDid you mean: {', '.join(phonetic_matches)}?")
            else:
                print(f"No results found (Search time: {search_time:.4f} seconds)")

def main():
    folder_path = 'documents/'  # Ensure this folder contains your text documents
    search_engine = MiniSearchEngine(folder_path)
    search_engine.interactive_search()

if __name__ == "__main__":
    main()