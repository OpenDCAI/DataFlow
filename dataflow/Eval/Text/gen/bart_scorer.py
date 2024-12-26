from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

@MODEL_REGISTRY.register()
class BARTScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        
        self.batch_size = 1
        self.data_type = "text"
        self.scorer_name = "BARTScorer"
        self.score_type = float
        self.device = args_dict.get("device", 'cuda:0')  # Default device to 'cuda:0'
        self.max_length = args_dict.get("max_length", 1024)  # Max token length for BART
        
        # Load the BART model and tokenizer
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        self.model.eval().to(self.device)
        
        # Set up loss function
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def evaluate_batch(self, eval_batch, ref_batch=None):
        """ Evaluate a batch of generated text against reference texts using BART """
        eval_data = next(iter(eval_batch.values()))  # Extract generated text
        ref_data = next(iter(ref_batch.values())) if ref_batch else None  # Extract reference text

        if ref_data is None:
            raise ValueError("Reference data must be provided for BART Scorer.")

        scores = []
        for eval_text, ref_text in zip(eval_data, ref_data):
            try:
                with torch.no_grad():
                    # Tokenize the source and target texts
                    encoded_src = self.tokenizer(
                        [eval_text],
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        [ref_text],
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    
                    # Move tensors to device
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)
                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask'].to(self.device)
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    # Get the model output and compute the loss
                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    
                    # Store the negative loss (higher loss means worse performance)
                    curr_score_list = [-x.item() for x in loss]
                    scores += curr_score_list

            except RuntimeError:
                print(f'Error processing pair: {eval_text} -> {ref_text}')
                raise

        return scores
