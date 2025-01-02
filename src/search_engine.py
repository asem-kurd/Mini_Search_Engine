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

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def improved_soundex(token: str) -> str:
    """
    Improved Soundex implementation that better handles misspellings and repeated letters.
    """
    if not token:
        return "0000"

    # Remove repeated characters first
    cleaned = token[0]
    for char in token[1:]:
        if char != cleaned[-1]:
            cleaned += char

    # Convert to uppercase
    token = cleaned.upper()

    # Enhanced Soundex mapping
    soundex_mapping = {
        'A': '0', 'E': '0', 'I': '0', 'O': '0', 'U': '0', 'H': '0', 'W': '0', 'Y': '0',
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    # Keep first letter
    result = token[0]

    # Convert remaining letters
    for char in token[1:]:
        if char in soundex_mapping:
            code = soundex_mapping[char]
            if code != '0' and (not result or code != result[-1]):
                result += code

    # Pad with zeros
    result = result + '0' * 4

    return result[:4]

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
            "doc10": "Lion Tigers vs tiger lion Is the presidents Lion Tiger Tiger Tiger Lion lions ",
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

        # Text processing tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

        # Initialize indices
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
                self.soundex_index[soundex].add(token)

    def search(self, query: str) -> Tuple[Set[str], Dict[str, float]]:
        """Perform search for exact phrase or AND logic."""
        exact_match = query.startswith('"') and query.endswith('"')
        if exact_match:
            query = query[1:-1].lower()
            matching_docs = {
                doc_id for doc_id, content in self.documents.items()
                if query in content.lower()
            }
            query_tokens = [query]  # Treat the whole phrase as a single "token" for scoring
        else:
            query_tokens = self._preprocess_text(query)
            matching_docs = set.intersection(*[
                self.inverted_index[token] for token in query_tokens if token in self.inverted_index
            ])

        relevance_scores = {
            doc_id: self.calculate_relevance_score(doc_id, query_tokens)
            for doc_id in matching_docs
        }
        return matching_docs, relevance_scores

    def calculate_relevance_score(self, doc_id: str, query_tokens: List[str]) -> float:
        """Calculate relevance score for a document given query tokens."""
        score = 0
        doc_text = self._preprocess_text(self.documents[doc_id])

        for token in query_tokens:
            term_freq = doc_text.count(token)
            if term_freq > 0:
                score += term_freq * (1 + log10(term_freq))
        return score

    def evaluate_metrics(self, retrieved_docs: Set[str], relevant_docs: Set[str]) -> Dict[str, float]:
        """Calculate precision, recall, accuracy, and F1-score."""
        true_positives = len(retrieved_docs & relevant_docs)
        false_positives = len(retrieved_docs - relevant_docs)
        false_negatives = len(relevant_docs - retrieved_docs)
        true_negatives = len(self.documents) - (true_positives + false_positives + false_negatives)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(self.documents)
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1-Score": f1_score
        }

    def run(self):
        """Run the search engine."""
        while True:
            clear_screen()
            print("Welcome to the Mini Search Engine")
            print("Enter your search query (or 0 to exit):")
            query = input("Query: ").strip()
            if query == "0":
                print("Exiting...")
                break

            matching_docs, relevance_scores = self.search(query)
            relevant_docs = set(self.documents.keys())  # Assume all documents are relevant for simplicity

            print("\nSearch Results:")
            for doc_id in matching_docs:
                print(f"{doc_id}: {self.documents[doc_id]} (Score: {relevance_scores[doc_id]:.2f})")

            metrics = self.evaluate_metrics(matching_docs, relevant_docs)
            print("\nEvaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")

            input("\nPress Enter to continue...")

if __name__ == "__main__":
    engine = MiniSearchEngine()
    engine.run()