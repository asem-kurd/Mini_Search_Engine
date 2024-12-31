import nltk
import numpy
import jellyfish

def test_imports():
    try:
        # Test NLTK
        tokens = nltk.word_tokenize("Testing NLTK installation")
        print("NLTK working correctly")
        
        # Test NumPy
        arr = numpy.array([1, 2, 3])
        print("NumPy working correctly")
        
        # Test Jellyfish
        distance = jellyfish.levenshtein_distance("test", "text")
        print("Jellyfish working correctly")
        
        print("\nAll dependencies are installed and working!")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_imports()