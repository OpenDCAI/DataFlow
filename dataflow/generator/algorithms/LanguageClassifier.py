import json
import logging
from tqdm import tqdm
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from guesslang import Guess
from dataflow.utils.registry import GENERATOR_REGISTRY
# guess = Guess()
# print(guess.supported_languages)
def batch(iterable, size):
    b = []
    for item in iterable:
        b.append(item)
        if len(b) == size:
            yield b
            b = []
    if b:
        yield b

class CodeBertClassifier:
    def __init__(self, config):
        self.config = config 
        self.input_file = config.get("input_file")
        self.output_file = config.get("output_file")
        self.input_key = config.get('input_key', 'code')
        self.output_key = config.get('output_key', 'label')
        self.batch_size = 64
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'philomath-1209/programming-language-identification'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)

    def predict(self):
        all_results = []
        # true_labels = []
        with open(self.input_file, "r", encoding="utf-8") as fin, open(self.output_file, "w", encoding="utf-8") as fout:
            lines = list(fin)
            for line_batch in tqdm(batch(lines, self.batch_size), desc="Predicting"):
                samples = [json.loads(line.strip()) for line in line_batch]
                codes = [sample[self.input_key] for sample in samples]
                # true_langs = [sample["label"] for sample in samples]
                inputs = self.tokenizer(codes, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs.to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    pred_ids = torch.argmax(logits, dim=-1).tolist()
                for pred_id in pred_ids:
                    predicted_lang = self.model.config.id2label[pred_id].lower()
                    all_results.append(predicted_lang)
                # for true_lang, pred_id in zip(true_langs, pred_ids):
                #     predicted_lang = self.model.config.id2label[pred_id]
                #     output_data = {
                #         "true_lang": true_lang.lower(),
                #         "predicted_lang": predicted_lang.lower()
                #     }
                #     all_results.append(output_data['predicted_lang'])
                #     true_labels.append(true_lang)
                #     fout.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        # print(self.calculate_accuracy(true_labels, all_results))
        return all_results
    
    # def calculate_accuracy(self, true, pred):
    #     normal_dict = {
    #         'csharp': 'c#',
    #         'java': 'java',
    #         'cpp': 'c++'
    #     }
    #     total = len(true)
    #     correct = 0
    #     for i in range(len(true)):
    #         # print(normal_dict[true[i]], pred[i])
    #         if normal_dict[true[i]] == pred[i]:
    #             correct += 1
    #     accuracy = correct / total
    #     return accuracy

class GuessLangClassifier:
    def __init__(self, config):
        self.config = config
        self.input_file = config.get('input_file')
        self.output_file = config.get('output_file')
        self.input_key = config.get('input_key', 'code')
        self.output_key = config.get('output_key', 'label')
        self.classifier = Guess()

    def predict(self):
        results = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing"):
                item = json.loads(line)
                code = item[self.input_key]
                language_id = self.classifier.language_name(code)
                results.append(language_id.lower())
        return results

@GENERATOR_REGISTRY.register()
class LanguageClassifier:
    def __init__(self, config, args=None):
        self.args = args
        self.config = config
        self.input_file = config.get('input_file')
        self.output_file = config.get('output_file')
        self.input_key = config.get('input_key', 'code')
        self.output_key = config.get('output_key', 'label')
        self.codebert_classifier = CodeBertClassifier(self.config)
        self.guesslang_calssifier = GuessLangClassifier(self.config)
        # self.guesslang_classifier = 
        logging.info("Initializing LanguageClassifier...")
        if not self.input_file or not self.output_file:
            raise ValueError("Both input_file and output_file must be specified in the config.")
    
    def run(self):
        logging.info("Start running LanguageClassifier...")
        logging.info(f"Reading input file: {self.input_file}...")
        CodeBert_Predictions = self.codebert_classifier.predict()
        GuessLang_Predictions = self.guesslang_calssifier.predict()
        # print(GuessLang_Predictions)
        final_predictions = []
        label_list = self.codebert_classifier.model.config.id2label.values()
        label_list = [label.lower() for label in label_list]
        for i in range(len(CodeBert_Predictions)):
            if not GuessLang_Predictions[i] in label_list:
                final_prediction = GuessLang_Predictions[i]
            else:
                final_prediction = CodeBert_Predictions[i]
            if final_prediction == 'c' or final_prediction == 'objective-c':
                final_prediction = 'c++'
            if final_prediction == 'typescript':
                final_prediction = 'javascript'
            if final_prediction == "c++":
                final_prediction = "cpp"
            if final_prediction == "c#":
                final_prediction = "c_sharp"
            final_predictions.append({self.output_key: final_prediction})
        with open(self.output_file, 'w', encoding='utf-8') as f:
            with open(self.input_file, 'r', encoding='utf-8') as g:
                data = [json.loads(_) for _ in g]
                for item, pred in zip(data, final_predictions):
                    item[self.output_key] = pred[self.output_key]
                    json.dump(item, f)
                    f.write('\n')


        

                

        
