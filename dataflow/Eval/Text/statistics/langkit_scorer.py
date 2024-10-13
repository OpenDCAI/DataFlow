from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY
import pandas as pd
from langkit import light_metrics, extract

# Langkit quality metrics
@MODEL_REGISTRY.register()
class LangkitScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.llm_schema = light_metrics.init()
        self.metrics_to_keep = args_dict.get('metrics_to_keep', {})
        self.batch_size = 1
        self.score_type = float 
        self.data_type = 'text' 
        self.score_name = 'LangkitScore' 

    def evaluate_batch(self, batch):
        input_data = next(iter(batch.values()))
        results = {}  
        for sample in input_data:
            df = pd.DataFrame({'prompt': [sample]})  
            df['response'] = '' 
            enhanced_df = extract(df, schema=self.llm_schema)
            scores_dict = enhanced_df.to_dict(orient='records')[0]

            processed_scores = {}
            for k, v in scores_dict.items():
                if k == 'prompt':
                    continue
                elif k.startswith('prompt.'):
                    new_key = k[len('prompt.'):]  
                    processed_scores[new_key] = v
                elif not (k == 'response' or k.startswith('response.')):
                    processed_scores[k] = v  

            processed_scores.pop('has_patterns', None)  

            if self.metrics_to_keep:
                processed_scores = {k: v for k, v in processed_scores.items() if self.metrics_to_keep.get(k, True)}

            for k, v in processed_scores.items():
                score_key = f"Langkit{''.join([word.capitalize() for word in k.split('_')])}Score"
                if score_key not in results:
                    results[score_key] = [] 
                results[score_key].append(v)  

        return results 
