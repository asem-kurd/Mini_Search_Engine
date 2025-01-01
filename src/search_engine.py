import os
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

def calculate_soundex(token: str) -> str:
    """Manually calculate the Soundex code for a given token."""
    token = token.upper()
    soundex_mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    first_letter = token[0] if token else ''
    encoded = [first_letter]

    for char in token[1:]:
        code = soundex_mapping.get(char, '')
        if code != encoded[-1]:
            encoded.append(code)

    soundex_code = ''.join(encoded).replace('H', '').replace('W', '')
    soundex_code = soundex_code[:4].ljust(4, '0')
    return soundex_code


class MiniSearchEngine:
    """
    A mini search engine implementation that fulfills the requirements with Soundex-based misspelling handling
    and phrase-based exact match retrieval.
    """

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.documents = {}
        self.inverted_index = defaultdict(set)
        self.phrase_index = defaultdict(list)
        self.doc_lengths = {}
        self.tf_matrix = defaultdict(lambda: defaultdict(float))
        self.soundex_index = defaultdict(set)

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

                    # Build inverted and phrase indices
                    for idx, token in enumerate(tokens):
                        self.inverted_index[token].add(filename)
                        self.phrase_index[filename].append(token)
                        self.tf_matrix[filename][token] += 1
                        self.soundex_index[calculate_soundex(token)].add(token)

    def _search_phrase(self, phrase: str) -> List[str]:
        """Search for documents containing the exact phrase."""
        results = []
        for doc, tokens in self.phrase_index.items():
            if ' '.join(tokens).find(phrase.lower()) != -1:
                results.append(doc)
        return results

    def _search_and_condition(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """Search for documents containing all terms (AND condition)."""
        relevant_docs = set(self.inverted_index[tokens[0]]) if tokens else set()
        for token in tokens[1:]:
            relevant_docs &= self.inverted_index[token]

        scores = [(doc, sum(self.tf_matrix[doc][token] for token in tokens if token in self.tf_matrix[doc]))
                  for doc in relevant_docs]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def handle_phonetics(self, query_term: str) -> Set[str]:
        """Handle phonetically similar terms using Soundex."""
        query_soundex = calculate_soundex(query_term)
        return self.soundex_index.get(query_soundex, {query_term})

    def search(self, query: str):
        """Process the user query and retrieve results."""
        if '"' in query:
            # Phrase search
            phrases = re.findall(r'"(.*?)"', query)
            results = set()
            for phrase in phrases:
                results.update(self._search_phrase(phrase))
            return list(results)
        else:
            # Term-based search with AND condition
            tokens = self._preprocess_text(query)
            return self._search_and_condition(tokens)

    def interactive_search(self):
        """Interactive command-line interface for the search engine."""
        print("\nWelcome to the Mini Search Engine!")
        while True:
            query = input("\nEnter your search term or phrase (or type '0' to quit): ").strip()
            if query == '0':
                print("Thank you for using the search engine. Goodbye!")
                break

            start_time = time.time()
            results = self.search(query)
            search_time = time.time() - start_time

            if results:
                print(f"\nFound {len(results)} results in {search_time:.4f} seconds:")
                for rank, doc in enumerate(results[:10], 1):
                    print(f"{rank}. {doc}")

                phonetic_matches = self.handle_phonetics(query)
                if phonetic_matches:
                    print(f"\nDid you mean: {', '.join(phonetic_matches)}?")
            else:
                print(f"No results found (Search time: {search_time:.4f} seconds)")


def main():
    folder_path = "Documents/"  # Ensure this folder contains at least 20 documents in .txt format
    search_engine = MiniSearchEngine(folder_path)
    search_engine.interactive_search()


if __name__ == "__main__":
    main()
