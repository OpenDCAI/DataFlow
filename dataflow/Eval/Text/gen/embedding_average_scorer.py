from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize


# Embedding Average Cosine Similarity Scorer
@MODEL_REGISTRY.register()
class EmbeddingAverageScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = 1
        self.data_type = "text"
        self.scorer_name = "EmbeddingAverageScorer"
        self.score_type = float

        # Initialize the GloVe embeddings
        self.embedding_model = Embedding()

    def evaluate_batch(self, eval_batch, ref_batch=None):
        eval_data = next(iter(eval_batch.values()))  # Extract generated text
        ref_data = next(iter(ref_batch.values())) if ref_batch else eval_data  # Default to eval_data if no ref provided

        # Clean up text data
        eval_data = [e.strip() for e in eval_data]
        ref_data = [r.strip() for r in ref_data]

        # Get embeddings for eval and reference
        eval_embeddings = self.embedding_model.get_vectors(eval_data)
        ref_embeddings = self.embedding_model.get_vectors(ref_data)

        # Compute cosine similarity for each pair of evaluation text and reference text
        similarities = cosine_similarity(eval_embeddings, ref_embeddings)

        # Average cosine similarity as score
        average_similarity = np.mean(similarities, axis=1)  # Ensure it gives a list of scores, one per batch

        return average_similarity.tolist()  # Return as a list of scores


import os
try:
    from gensim.models import KeyedVectors
except ImportError:
    from gensim.models import Word2Vec as KeyedVectors

import nltk
nltk.download('punkt_tab')
class Embedding(object):
    def __init__(self):
        # Assuming a method to load the pre-trained GloVe embeddings, e.g., a .bin model file
        path = get_data_dir()  # Replace this with actual data directory logic
        self.m = KeyedVectors.load(os.path.join(path, 'glove.6B.300d.model.bin'), mmap='r')
        try:
            self.unk = self.m.vectors.mean(axis=0)
        except AttributeError:
            self.unk = self.m.syn0.mean(axis=0)

    def get_vectors(self, texts):
        # Get embedding vectors for a list of texts (tokens)
        return np.array([self.get_text_embedding(text) for text in texts])

    def get_text_embedding(self, text):
        # Tokenize text and calculate average embedding
        words = word_tokenize(text)
        embeddings = [self.vec(word) for word in words]
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding / np.linalg.norm(avg_embedding)  # Normalize

    def vec(self, word):
        # Get embedding for a single word, return a default vector if not found
        try:
            return self.m[word]
        except KeyError:
            return self.unk



def get_data_dir():
    model_dir = os.path.join('.', 'ckpt', 'glove_model')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        glove = api.load("glove-wiki-gigaword-300")  
        glove.save(os.path.join(model_dir, 'glove.6B.300d.model.bin'))

    return model_dir



