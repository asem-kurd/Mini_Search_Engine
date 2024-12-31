import os
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
from math import log10
import time

# NLP-related imports
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.metrics.distance import edit_distance
import jellyfish

# Download required NLTK data
import nltk
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
        self.documents = {}  # Store preprocessed documents
        self.positions_index = defaultdict(lambda: defaultdict(list))  # Positional index for phrase queries
        self.inverted_index = defaultdict(set)  # Inverted index for term queries
        self.doc_lengths = {}  # Document lengths for normalization
        
        # Text processing tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Initialize the search engine by preprocessing documents
        self._preprocess_documents()

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing, removing stopwords, and stemming.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            List[str]: List of processed tokens
        """
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords, non-alphanumeric tokens, and apply stemming
        processed_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
        return processed_tokens

    def _preprocess_documents(self):
        """
        Preprocess all documents in the specified folder and build the necessary indices.
        """
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.folder_path, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    tokens = self._preprocess_text(content)
                    
                    # Store processed documents and their lengths
                    self.documents[filename] = tokens
                    self.doc_lengths[filename] = len(tokens)
                    
                    # Build positional and inverted indices
                    for position, token in enumerate(tokens):
                        self.positions_index[token][filename].append(position)
                        self.inverted_index[token].add(filename)

    def term_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Requirement 1: Term-based Retrieval
        Retrieve documents containing the specified terms, with ranking based on TF-IDF.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Tuple[str, float]]: Ranked list of (document, score) pairs
        """
        tokens = self._preprocess_text(query)
        scores = defaultdict(float)
        
        for token in tokens:
            # Get potential matches including character loss and phonetic variations
            matches = self.handle_character_loss(token)
            matches.update(self.handle_phonetics(token))
            
            # Calculate TF-IDF score for each matching term
            for term in matches:
                if term in self.inverted_index:
                    idf = log10(len(self.documents) / len(self.inverted_index[term]))
                    for doc in self.inverted_index[term]:
                        tf = len(self.positions_index[term][doc]) / self.doc_lengths[doc]
                        scores[doc] += tf * idf
        
        # Normalize scores by document length
        results = [(doc, score / self.doc_lengths[doc]) 
                  for doc, score in scores.items()]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def phrase_search(self, phrase: str) -> List[Tuple[str, float]]:
        """
        Requirement 2: Phrase-based Retrieval
        Retrieve documents containing the exact phrase, considering word positions.
        
        Args:
            phrase (str): Search phrase
            
        Returns:
            List[Tuple[str, float]]: Ranked list of (document, score) pairs
        """
        tokens = self._preprocess_text(phrase)
        if not tokens:
            return []
            
        # Find documents containing all terms
        candidate_docs = set.intersection(
            *[self.inverted_index[token] for token in tokens]
        )
        
        results = []
        for doc in candidate_docs:
            # Check for consecutive positions of terms
            positions = self.positions_index[tokens[0]][doc]
            for i in range(1, len(tokens)):
                next_positions = {pos - i for pos in self.positions_index[tokens[i]][doc]}
                positions = {pos for pos in positions if pos in next_positions}
                if not positions:
                    break
                    
            # Score documents based on phrase frequency
            if positions:
                score = len(positions) / self.doc_lengths[doc]
                results.append((doc, score))
                
        return sorted(results, key=lambda x: x[1], reverse=True)

    def handle_character_loss(self, query_term: str) -> Set[str]:
        """
        Requirement 3: Handling Character Loss
        Handle missing or incorrect characters in query terms using edit distance.
        
        Args:
            query_term (str): Query term to process
            
        Returns:
            Set[str]: Set of matching terms
        """
        matches = set()
        for term in self.inverted_index.keys():
            if edit_distance(query_term, term) <= 1:  # Allow for one character difference
                matches.add(term)
        return matches or {query_term}

    def handle_phonetics(self, query_term: str) -> Set[str]:
        """
        Requirement 4: Phonetics Handling
        Handle phonetic variations in query terms using Metaphone algorithm.
        
        Args:
            query_term (str): Query term to process
            
        Returns:
            Set[str]: Set of phonetically matching terms
        """
        matches = set()
        query_metaphone = jellyfish.metaphone(query_term)
        for term in self.inverted_index.keys():
            if jellyfish.metaphone(term) == query_metaphone:
                matches.add(term)
        return matches

    def evaluate(self, queries: List[str], relevant_docs: List[Set[str]]) -> Dict[str, float]:
        """
        Requirement 6: System Evaluation
        Evaluate the search engine using standard IR metrics.
        
        Args:
            queries (List[str]): List of test queries
            relevant_docs (List[Set[str]]): List of sets of relevant documents for each query
            
        Returns:
            Dict[str, float]: Dictionary containing average precision, recall, F1-score, and accuracy
        """
        total_metrics = defaultdict(float)
        
        for query, relevant in zip(queries, relevant_docs):
            # Get search results
            results = self.term_search(query)
            retrieved_docs = {doc for doc, _ in results}
            
            # Calculate evaluation metrics
            tp = len(relevant & retrieved_docs)
            fp = len(retrieved_docs - relevant)
            fn = len(relevant - retrieved_docs)
            tn = len(set(self.documents.keys()) - relevant - retrieved_docs)
            
            # Calculate individual metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Accumulate metrics
            total_metrics['precision'] += precision
            total_metrics['recall'] += recall
            total_metrics['f1_score'] += f1
            total_metrics['accuracy'] += accuracy
        
        # Calculate averages
        n_queries = len(queries)
        return {
            metric: value / n_queries
            for metric, value in total_metrics.items()
        }

    def interactive_search(self):
        """
        Interactive command-line interface for the search engine.
        Allows users to perform term-based or phrase-based searches and view results.
        """
        print("\nWelcome to the Enhanced Mini Search Engine!")
        while True:
            print("\nOptions:")
            print("1. Term search")
            print("2. Phrase search")
            print("3. Exit")
            
            try:
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
                    
                # Perform search based on user choice
                start_time = time.time()
                results = []
                if choice == '1':
                    results = self.term_search(query)
                elif choice == '2':
                    results = self.phrase_search(query)
                    
                # Display results and search time
                search_time = time.time() - start_time
                if results:
                    print(f"\nFound {len(results)} results in {search_time:.4f} seconds:")
                    for rank, (doc, score) in enumerate(results[:10], 1):
                        print(f"{rank}. {doc} (Score: {score:.4f})")
                        
                    # Show spelling suggestions and phonetic matches
                    for token in self._preprocess_text(query):
                        corrections = self.handle_character_loss(token)
                        phonetic_matches = self.handle_phonetics(token)
                        if corrections or phonetic_matches:
                            print(f"\nSuggestions for '{token}':")
                            print("Character-based:", ", ".join(corrections))
                            print("Phonetic-based:", ", ".join(phonetic_matches))
                else:
                    print(f"No results found (Search time: {search_time:.4f} seconds)")
                    
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Please try again.")

if __name__ == "__main__":
    folder_path = 'documents/'
    search_engine = MiniSearchEngine(folder_path)
    search_engine.interactive_search()
