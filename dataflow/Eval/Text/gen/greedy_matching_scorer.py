import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
import gensim.downloader as api
from gensim.models import KeyedVectors

# Greedy Matching Score scorer
@MODEL_REGISTRY.register()
class GreedyMatchingScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = 1
        self.data_type = "text"
        self.scorer_name = "GreedyMatchingScorer"
        self.score_type = float

        # Initialize the embedding model
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

        # Greedy matching: max similarity in both directions (ref to hyp, hyp to ref)
        dir1 = similarities.max(axis=0).mean()  # max similarity of ref to hyp
        dir2 = similarities.max(axis=1).mean()  # max similarity of hyp to ref

        # Greedy matching score is the average of both directions
        score = (dir1 + dir2) / 2

        return [score]  # Return as a list of scores for batch processing




# Embedding class for handling GloVe embeddings
class Embedding(object):
    def __init__(self):
        # Load pre-trained GloVe embeddings
        path = get_data_dir()
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



