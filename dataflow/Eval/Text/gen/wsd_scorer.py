import gensim.downloader as api
from nltk.corpus import stopwords
from nltk import download
from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
import logging



@MODEL_REGISTRY.register()
class WsdScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = 1 
        self.data_type = "text"
        self.scorer_name = "WSDScorer"
        self.score_type = float
        model = api.load('word2vec-google-news-300')
        print('api over')
        self.model = model  

    def evaluate_batch(self, eval_batch, ref_batch=None):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



        eval_data = next(iter(eval_batch.values()))  
        ref_data = next(iter(ref_batch.values())) if ref_batch else None  

        if ref_data is None:
            raise ValueError("Reference data must be provided for WSD Scorer.")

        scores = []
        for eval_text, ref_text in zip(eval_data, ref_data):

            eval_tokens = preprocess(eval_text)
            ref_tokens = preprocess(ref_text)

 
            wmd_score = self.model.wmdistance(eval_tokens, ref_tokens)
            scores.append(wmd_score)

        return scores

def preprocess(sentence):
    download('stopwords')
    stop_words = stopwords.words('english')
    return [w for w in sentence.lower().split() if w not in stop_words]
