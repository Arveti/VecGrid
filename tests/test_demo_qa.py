"""
Test case demonstrating the real-world semantic Q&A use case
using a real sentence-transformer model.
"""

import numpy as np
from vecgrid import VecGrid, InProcessTransport

# Try to import sentence-transformers, skip test if not installed
try:
    from sentence_transformers import SentenceTransformer
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

def test_semantic_qa_use_case_with_real_model():
    """
    Tests a real-world semantic search use case for a Q&A system
    using a real sentence embedding model.
    """
    if not MODEL_AVAILABLE:
        print("\nSkipping Semantic Q&A test: 'sentence-transformers' not installed.")
        print("Please run 'pip install -e .[dev]' to install test dependencies.")
        return

    # Use the in-process transport for single-file testing
    InProcessTransport.reset()
    
    # --- Model and Grid Setup ---
    # Load a pre-trained model. 'all-MiniLM-L6-v2' is small and fast.
    print("Loading sentence-transformer model... (this may take a moment on first run)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dim = model.get_sentence_embedding_dimension() # Should be 384

    grid = VecGrid(node_id="qna-node-1", dim=dim)
    grid.start()

    articles = {
        "doc-1": "Getting Started with Our API: Authentication and first calls.",
        "doc-2": "How to Optimize Application Performance: A guide to profiling and speed.",
        "doc-3": "Advanced Configuration: A deep dive into clustering and replication.",
        "doc-4": "Troubleshooting Common Connection Errors and Timeouts.",
    }

    # --- Indexing Phase ---
    print("Indexing articles with real embeddings...")
    # Convert all articles into vectors and store them in VecGrid
    contents = list(articles.values())
    vectors = model.encode(contents)
    
    for i, doc_id in enumerate(articles.keys()):
        grid.put(doc_id, vectors[i], {"content": contents[i]})

    # --- Query Phase ---
    # A user asks a question in natural language.
    user_query = "how to make my program faster"
    print(f"User Query: '{user_query}'")
    query_vector = model.encode([user_query])[0]

    # Search VecGrid to find the most semantically similar articles.
    results = grid.search(query_vector, k=1)

    # --- Assertion Phase ---
    # The result should be 'doc-2', because its content is about
    # performance, even though it doesn't contain the words "program" or "faster".
    assert len(results) > 0, "Search should return at least one result."
    
    best_match = results[0]
    
    print(f"Best Match: '{best_match.metadata['content']}' (ID: {best_match.vector_id})")
    
    assert best_match.vector_id == "doc-2", f"Expected 'doc-2' but got '{best_match.vector_id}'"
    assert "performance" in best_match.metadata['content'].lower()
    
    grid.stop()

if __name__ == "__main__":
    test_semantic_qa_use_case_with_real_model()
