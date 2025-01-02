import os
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from math import log10
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from symspellpy import SymSpell, Verbosity
from difflib import SequenceMatcher

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"  # Path to the dictionary file

# Load dictionary file with error handling
try:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
except FileNotFoundError:
    print(f"Dictionary file '{dictionary_path}' not found. Spell correction will be disabled.")
    sym_spell = None

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def correct_spelling(query: str, inverted_index: Dict[str, Set[str]]) -> Tuple[str, str]:
    """
    Correct spelling errors in the query using SymSpell and document vocabulary.
    Returns the corrected query and the closest matching word from the document collection.
    """
    # Use SymSpell for initial correction
    if sym_spell is not None:
        suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
        if suggestions:
            corrected_query = suggestions[0].term
        else:
            corrected_query = query
    else:
        corrected_query = query

    # Find the closest matching word from the document vocabulary
    closest_word = None
    max_similarity = 0
    for word in inverted_index.keys():
        similarity = SequenceMatcher(None, query.lower(), word.lower()).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            closest_word = word

    # If the closest word is significantly similar, suggest it
    if closest_word and max_similarity >= 0.6:  # Adjust threshold as needed
        return corrected_query, closest_word
    else:
        return corrected_query, None

def improved_soundex(token: str) -> str:
    """
    Improved Soundex implementation that preserves the first letter and handles misspellings.
    """
    if not token:
        return "0000"

    # Convert to uppercase and filter non-alphabetic characters
    token = ''.join([char.upper() for char in token if char.isalpha()])
    if not token:
        return "0000"

    # Soundex mapping
    soundex_mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    # Keep the first letter
    soundex_code = token[0]

    # Convert remaining letters
    for char in token[1:]:
        if char in soundex_mapping:
            code = soundex_mapping[char]
            # Avoid adding the same code consecutively
            if code != soundex_code[-1]:
                soundex_code += code

    # Pad with zeros and truncate to 4 characters
    soundex_code = soundex_code.ljust(4, '0')[:4]

    return soundex_code

def fuzzy_soundex_match(query_soundex: str, soundex_index: Dict[str, Set[str]], similarity_threshold: float = 0.8) -> Set[str]:
    """Find documents with Soundex codes similar to the query's Soundex code using difflib."""
    matching_docs = set()
    for soundex_code in soundex_index:
        similarity = SequenceMatcher(None, query_soundex, soundex_code).ratio()
        if similarity >= similarity_threshold:
            matching_docs.update(soundex_index[soundex_code])
    return matching_docs

class MiniSearchEngine:
    def __init__(self):
        """
        Initialize search engine with built-in document collection.
        """
        self.documents = {
            "doc1": "ELEPHANTS ELEPHANT",
            "doc2": "Augmented Reality (AR) overlays digital information onto the real world, enhancing users' perception of their environment. AR applications range from entertainment, such as mobile games like Pokémon GO, to practical uses like navigation and interior design. AR relies on technologies like cameras, sensors, and software algorithms to detect and display virtual objects in real-time. It is also gaining traction in education and healthcare, enabling immersive learning experiences and aiding in surgical procedures.",
            "doc3": "Big Data refers to extremely large datasets that traditional methods cannot process effectively. These datasets come from various sources, such as social media, sensors, and online transactions. Big Data analytics involves storing, processing, and analyzing this data to uncover patterns and insights. Tools like Hadoop and Spark are commonly used to manage Big Data, which is vital for decision-making in fields like marketing, healthcare, and finance.",
            "doc4": "Powers Pierce Price Perez please",
            "doc5": "ASEM asEm aSem MohAmAd",
            "doc6": "Computer networks enable multiple devices to connect and share resources, such as data, applications, and hardware. They range from small-scale local area networks (LANs) to large-scale wide area networks (WANs) like the internet. Networks are essential for communication and collaboration in modern businesses, but they require robust security measures to protect against unauthorized access and data breaches.",
            "doc7": "Lion Tiger Tiger Tiger Lion tiger vs tiger lion Is the president",
            "doc8": "ASEM MOHAMMAD ZAID amz ZaId",
            "doc9": "ZaId zaId ZAID zaiD",
            "doc10": "Lion Tigers vs tiger lion Is the presidents Lion Tiger Tiger Tiger Lion ",
            "doc11": "Information retrieval is the process of finding relevant information from a large repository, such as a database, library, or the internet. It plays a critical role in search engines, enabling users to retrieve documents that match their queries. Techniques like keyword matching, natural language processing, and ranking algorithms are central to information retrieval systems. As data continues to grow exponentially, the field faces challenges in handling vast datasets efficiently while maintaining high accuracy and speed in query responses.",
            "doc12": "HEllo world",
            "doc13": "Machine learning, a subset of artificial intelligence, enables systems to learn from data and improve their performance over time without explicit programming. This technology underpins many modern applications, such as spam filtering, facial recognition, and predictive maintenance. Machine learning models can be classified into supervised, unsupervised, and reinforcement learning. Despite its advancements, challenges like data bias and interpretability remain areas of active research, as they impact the fairness and transparency of machine learning systems.",
            "doc14": "Natural Language Processing (NLP) is a field at the intersection of linguistics and artificial intelligence that allows machines to understand, interpret, and generate human language. Applications of NLP include chatbots, language translation tools, and sentiment analysis. The technology uses techniques like tokenization, stemming, and neural networks to analyze text and speech. One of its significant achievements is the development of large language models like GPT, which can generate coherent and contextually relevant text.",
            "doc15": "Processors, commonly known as central processing units (CPUs), serve as the brain of computers. They are responsible for executing instructions, performing calculations, and running programs. Modern processors consist of billions of transistors and utilize multiple cores to handle parallel processing, significantly enhancing performance. CPUs are essential not only in personal computers but also in smartphones, gaming consoles, and data centers. Advances in processor technology, such as quantum computing and AI acceleration, are pushing the boundaries of computational capabilities.",
            "doc16": "Elephant elephant in the pool",
            "doc17": "Quantum computing leverages the principles of quantum mechanics to perform computations far beyond the capabilities of classical computers. Unlike classical bits, which represent data as 0s and 1s, quantum bits (qubits) can exist in multiple states simultaneously due to superposition. This allows quantum computers to solve complex problems, such as drug discovery and cryptographic analysis, exponentially faster. However, the technology is still in its early stages, with challenges like qubit stability and error correction to overcome.",
            "doc18": "Software engineering applies engineering principles to the design, development, testing, and maintenance of software systems. It emphasizes creating reliable, scalable, and efficient solutions to meet user needs. Software engineering methodologies, like Agile and DevOps, focus on collaboration and iterative improvements, ensuring high-quality products that align with business goals.",
            "doc19": "TIGER LION lion",
            "doc20": "Web development encompasses the creation and maintenance of websites and web applications. It includes front-end development (user interface design) and back-end development (server-side logic and database integration). Technologies like HTML, CSS, JavaScript, and frameworks like React and Node.js are commonly used. Web development continues to evolve, with trends focusing on responsive design, accessibility, and performance optimization."
        }

        self.inverted_index = defaultdict(set)
        self.phrase_index = defaultdict(list)
        self.doc_lengths = {}
        self.tf_matrix = defaultdict(lambda: defaultdict(float))
        self.soundex_index = defaultdict(set)
        self.preprocessed_docs = {}  # Cache for preprocessed documents

        # Text processing tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.preserved_chars = {'+', '@', '.', '-'}  # Characters to preserve during preprocessing

        # Initialize indices
        self._preprocess_documents()

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing, while preserving specific non-alphanumeric characters."""
        if text in self.preprocessed_docs:
            return self.preprocessed_docs[text]

        tokens = word_tokenize(text.lower())
        preprocessed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if any(c.isalnum() or c in self.preserved_chars for c in token) and token not in self.stop_words
        ]
        self.preprocessed_docs[text] = preprocessed_tokens
        return preprocessed_tokens

    def _preprocess_documents(self):
        """Process the built-in document collection and build indices."""
        for doc_id, content in self.documents.items():
            # Tokenize and preprocess the document text
            tokens = self._preprocess_text(content)
            self.doc_lengths[doc_id] = len(tokens)

            # Build indices
            for position, token in enumerate(tokens):
                # Update inverted index
                self.inverted_index[token].add(doc_id)

                # Update phrase index
                self.phrase_index[doc_id].append(token)

                # Update term frequency matrix
                self.tf_matrix[doc_id][token] += 1

                # Build soundex index
                soundex = improved_soundex(token)
                self.soundex_index[soundex].add(doc_id)  # Map Soundex code to doc_id

    def wildcard_search(self, pattern: str) -> Set[str]:
        """Search for documents containing tokens matching the wildcard pattern."""
        regex = re.compile(pattern.replace('*', '.*'))
        matching_tokens = [token for token in self.inverted_index if regex.match(token)]
        matching_docs = set.union(*[self.inverted_index[token] for token in matching_tokens])
        return matching_docs

    def search(self, query: str) -> Tuple[Set[str], Dict[str, float], List[str], Set[str]]:
        """Perform search for exact phrase, AND logic, OR logic, NOT logic, or wildcard search."""
        # Handle exact phrase search (words in exact order)
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]  # Remove quotes and preserve case
            query_tokens = [query]
            matching_docs = set()
            for doc_id, content in self.documents.items():
                # Check if the exact phrase exists in the document (case-sensitive)
                if query in content:
                    matching_docs.add(doc_id)
        else:
            # Correct spelling and find the closest matching word for non-exact phrase searches
            corrected_query, closest_word = correct_spelling(query, self.inverted_index)
            if corrected_query != query:
                if closest_word:
                    print(f"Did you mean: {closest_word}?")
                else:
                    print(f"Did you mean: {corrected_query}?")
            query = corrected_query

            # Handle wildcard search
            if '*' in query:
                matching_docs = self.wildcard_search(query)
                query_tokens = [query]
            # Handle AND logic search (words in any order but all must exist)
            else:
                query_tokens = self._preprocess_text(query)
                print(f"Query Tokens: {query_tokens}")  # Debug: Print query tokens
                # Get the sets of documents for each token
                doc_sets = [self.inverted_index[token] for token in query_tokens if token in self.inverted_index]
                # Perform intersection only if there are sets to intersect
                if doc_sets:
                    matching_docs = set.intersection(*doc_sets)
                else:
                    # Fallback to Soundex-based matching if no exact matches are found
                    soundex_codes = [improved_soundex(token) for token in query_tokens]
                    print(f"Query Soundex Codes: {soundex_codes}")  # Debug: Print query Soundex codes
                    soundex_docs = set.union(*[
                        fuzzy_soundex_match(code, self.soundex_index, similarity_threshold=0.8) for code in soundex_codes
                    ])
                    print(f"Soundex Matches: {soundex_docs}")  # Debug: Print Soundex matches
                    # Filter documents to ensure they contain at least one of the query tokens
                    matching_docs = soundex_docs  # Remove strict token filtering

        # Define ground truth relevant documents based on query tokens
        relevant_docs = set()
        for token in query_tokens:
            relevant_docs.update(self.inverted_index.get(token, set()))

        relevance_scores = {
            doc_id: self.calculate_relevance_score(doc_id, query_tokens)
            for doc_id in matching_docs
        }
        return matching_docs, relevance_scores, query_tokens, relevant_docs

    def calculate_relevance_score(self, doc_id: str, query_tokens: List[str]) -> float:
        """Calculate relevance score for a document given query tokens using BM25 and Soundex similarity."""
        score = 0
        doc_text = self._preprocess_text(self.documents[doc_id])
        total_docs = len(self.documents)
        avg_doc_length = sum(self.doc_lengths.values()) / total_docs
        k1 = 1.5
        b = 0.75

        for token in query_tokens:
            term_freq = doc_text.count(token)
            if term_freq > 0:
                # BM25 scoring for exact matches
                doc_freq = len(self.inverted_index[token])
                idf = log10((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                numerator = term_freq * (k1 + 1)
                denominator = term_freq + k1 * (1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length))
                score += idf * (numerator / denominator)
            else:
                # Soundex-based scoring for approximate matches
                soundex = improved_soundex(token)
                if soundex in self.soundex_index and doc_id in self.soundex_index[soundex]:
                    # Find the most similar token in the document
                    max_similarity = max(
                        SequenceMatcher(None, token, doc_token).ratio()
                        for doc_token in doc_text
                    )
                    # Add a score proportional to the similarity and term frequency
                    matched_token = next(
                        doc_token for doc_token in doc_text
                        if SequenceMatcher(None, token, doc_token).ratio() == max_similarity
                    )
                    term_freq = doc_text.count(matched_token)
                    score += max_similarity * term_freq * 0.5  # Adjust weight as needed

        return score

    def evaluate_metrics(self, retrieved_docs: Set[str], relevant_docs: Set[str]) -> Dict[str, float]:
        """Calculate precision, recall, accuracy, and F1-score based on ground truth relevant documents."""
        # True Positives: Retrieved documents that are relevant
        true_positives = len(retrieved_docs & relevant_docs)
        
        # False Positives: Retrieved documents that are not relevant
        false_positives = len(retrieved_docs - relevant_docs)
        
        # False Negatives: Relevant documents that were not retrieved
        false_negatives = len(relevant_docs - retrieved_docs)
        
        # True Negatives: Documents that are not relevant and were not retrieved
        true_negatives = len(self.documents) - (true_positives + false_positives + false_negatives)
        
        # Precision: Proportion of retrieved documents that are relevant
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        # Recall: Proportion of relevant documents that are retrieved
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Accuracy: Proportion of correct predictions (both relevant and non-relevant)
        accuracy = (true_positives + true_negatives) / len(self.documents)
        
        # F1-Score: Harmonic mean of precision and recall
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1-Score": f1_score
        }

    def extract_key_phrase(self, doc_id: str, query_tokens: List[str]) -> str:
        """Extract the key phrase from the document that matches the query tokens."""
        content = self.documents[doc_id]
        for token in query_tokens:
            # Find the first occurrence of the token in the document
            if token.lower() in content.lower():
                return token.upper()  # Return the token in uppercase
        return ""

    def run(self):
        """Run the search engine."""
        while True:
            clear_screen()
            print("Welcome to the Mini Search Engine")
            print("Enter your search query (or 0 to exit):")
            query = input("Search: ").strip()
            if query == "0":
                print("Exiting...")
                break

            matching_docs, relevance_scores, query_tokens, relevant_docs = self.search(query)

            print("\nSearch Results:")
            if matching_docs:
                for doc_id in sorted(matching_docs, key=lambda x: relevance_scores[x], reverse=True):
                    key_phrase = self.extract_key_phrase(doc_id, query_tokens)
                    print(f"{doc_id}: {key_phrase} ==> Score: {relevance_scores[doc_id]:.2f}")
            else:
                print("No results found.")

            metrics = self.evaluate_metrics(matching_docs, relevant_docs)
            print("\nEvaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")

            input("\nPress Enter to continue...")

if __name__ == "__main__":
    engine = MiniSearchEngine()
    engine.run()