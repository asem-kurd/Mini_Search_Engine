import os
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from math import log10
import time

# NLP-related imports
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.metrics.distance import edit_distance
import jellyfish
import nltk

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
        self.positions_index = defaultdict(lambda: defaultdict(list))
        self.inverted_index = defaultdict(set)
        self.doc_lengths = {}
        
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
                    
                    for position, token in enumerate(tokens):
                        self.positions_index[token][filename].append(position)
                        self.inverted_index[token].add(filename)

    def _calculate_tfidf(self, token: str, scores: Dict[str, float]) -> None:
        """Calculate TF-IDF scores for a given token."""
        if token in self.inverted_index:
            idf = log10(len(self.documents) / len(self.inverted_index[token]))
            for doc in self.inverted_index[token]:
                tf = len(self.positions_index[token][doc]) / self.doc_lengths[doc]
                scores[doc] += tf * idf

    def term_search(self, query: str) -> List[Tuple[str, float]]:
        """Retrieve documents containing the specified terms, with ranking based on TF-IDF."""
        tokens = self._preprocess_text(query)
        scores = defaultdict(float)

        for token in tokens:
            matches = self.handle_character_loss(token, max_distance=2) | self.handle_phonetics(token)
            for term in matches:
                self._calculate_tfidf(term, scores)

        results = [(doc, score / self.doc_lengths[doc]) for doc, score in scores.items()]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_phrase(self, phrase: str) -> List[Tuple[str, float]]:
        """Perform OR-based retrieval using positional indexes for multiple terms."""
        # Split the phrase into individual terms
        tokens = self._preprocess_text(phrase)
        if not tokens:
            return []

        # Using a set to collect documents that match any of the tokens
        matching_docs = set()

        # Iterate through each token and find matching documents
        for token in tokens:
            if token in self.inverted_index:
                matching_docs.update(self.inverted_index[token])

        results = []
        scores = defaultdict(float)

        # Calculate scores for documents that matched any token
        for doc in matching_docs:
            for token in tokens:
                if token in self.inverted_index and doc in self.inverted_index[token]:
                    score = log10(len(self.documents) / len(self.inverted_index[token]))  # Basic score based on IDF
                    scores[doc] += score

        # Prepare results with scores for sorting
        results = [(doc, score / self.doc_lengths[doc]) for doc, score in scores.items()]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def handle_character_loss(self, query_term: str, max_distance: int = 1) -> Set[str]:
        """
        Handle missing or incorrect characters in query terms using edit distance.
        
        Args:
            query_term (str): Query term to process
            max_distance (int): Maximum allowed edit distance for matching terms
            
        Returns:
            Set[str]: Set of matching terms
        """
        matches = set()
        for term in self.inverted_index.keys():
            if edit_distance(query_term, term) <= max_distance:  # Allow for specified character differences
                matches.add(term)
        return matches or {query_term}

    def handle_phonetics(self, query_term: str) -> Set[str]:
        """Handle phonetic variations in query terms using Metaphone algorithm."""
        query_metaphone = jellyfish.metaphone(query_term)
        return {term for term in self.inverted_index.keys() if jellyfish.metaphone(term) == query_metaphone}

    def evaluate(self, queries: List[str], relevant_docs: List[Set[str]]) -> Dict[str, float]:
        """Evaluate the search engine using standard IR metrics."""
        total_metrics = defaultdict(float)

        for query, relevant in zip(queries, relevant_docs):
            results = self.term_search(query)
            retrieved_docs = {doc for doc, _ in results}

            tp = len(relevant & retrieved_docs)
            fp = len(retrieved_docs - relevant)
            fn = len(relevant - retrieved_docs)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + len(self.documents) - len(relevant) - len(retrieved_docs)) / len(self.documents)

            total_metrics['precision'] += precision
            total_metrics['recall'] += recall
            total_metrics['f1_score'] += f1
            total_metrics['accuracy'] += accuracy

        n_queries = len(queries)
        return {metric: value / n_queries for metric, value in total_metrics.items()}

    def interactive_search(self):
        """Interactive command-line interface for the search engine."""
        print("\nWelcome to the Enhanced Mini Search Engine!")
        while True:
            print("\nOptions:\n1. Term search\n2. Phrase search\n3. Exit")
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            
            if choice == '3':
                print("Thank you for using the search engine. Goodbye!")
                break
            
            if choice not in ['1', '2']:
                print("Invalid choice. Please select 1, 2, or 3.")
                continue
            
            query = input("\nEnter your search query: ").strip()
            if not query:
                print("Please provide a search query.")
                continue
            
            start_time = time.time()
            results = self.term_search(query) if choice == '1' else self.search_phrase(query)
            search_time = time.time() - start_time
            
            if results:
                print(f"\nFound {len(results)} results in {search_time:.4f} seconds:")
                for rank, (doc, score) in enumerate(results[:10], 1):
                    print(f"{rank}. {doc} (Score: {score:.4f})")
                
                for token in self._preprocess_text(query):
                    corrections = self.handle_character_loss(token, max_distance=2)
                    phonetic_matches = self.handle_phonetics(token)
                    if corrections or phonetic_matches:
                        print(f"\nSuggestions for '{token}':")
                        print("Character-based:", ", ".join(corrections))
                        print("Phonetic-based:", ", ".join(phonetic_matches))
            else:
                print(f"No results found (Search time: {search_time:.4f} seconds)")

if __name__ == "__main__":
    folder_path = 'documents/'  # Ensure this folder contains your text documents
    search_engine = MiniSearchEngine(folder_path)
    search_engine.interactive_search()
