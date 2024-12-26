from datasets import load_from_disk, load_dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .modeling.modeling_flash_llama import LlamaForSequenceClassification
import torch
import argparse
import numpy as np

class TokenizeAndChunk:
    def __init__(self, tokenizer_name, text_field, tokens_field, tokens, model_cache_dir):
        self.tokens = tokens
        self.tokenizer_name = tokenizer_name
        self.text_field = text_field
        self.tokens_field = tokens_field

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, cache_dir = model_cache_dir)
        self.tokenizer.pad_token_id = 0

    def __getstate__(self):
        return {
            "tokenizer_name": self.tokenizer_name,
            "text_field": self.text_field,
            "tokens_field": self.tokens_field,
            "tokens": self.tokens,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def tokenize_and_chunk(self, source_tokens):
        chunks_token_ids = []
        chunks_token_counts = []

        for seq in source_tokens:
            chunks = torch.tensor(seq, dtype=torch.long).split(self.tokens)
            chunks_token_ids.append([chunk.tolist() for chunk in chunks])
            chunks_token_counts.append([len(x) for x in chunks])

        return chunks_token_ids, chunks_token_counts

    def __call__(self, example):
        if self.tokens_field in example:
            source_tokens = example[self.tokens_field]
        else:
            source_tokens = self.tokenizer(example[self.text_field], truncation=False, padding=False, add_special_tokens=False).input_ids

        chunks_token_ids, chunks_token_counts = self.tokenize_and_chunk(source_tokens)

        assert len(example[self.text_field]) == len(chunks_token_ids)
        assert len(example[self.text_field]) == len(chunks_token_counts)

        return {
            "chunks_token_ids": chunks_token_ids,
            "chunks_token_counts": chunks_token_counts,
        }


class ModelAnnotator:
    def __init__(self, model_name, labels, device_batch_size, device, model_cache_dir):

        self.model_name = model_name
        self.labels = labels
        self.device_batch_size = device_batch_size

        self.model = LlamaForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=model_cache_dir)
        self.model.config.pad_token_id = 0
        self.model.eval()

        self.device = device
        print(f"Using device {self.device}")
        self.model.to(self.device)

        self.num_labels = len(labels)
        assert self.num_labels == self.model.config.num_labels, f"Number of labels ({self.num_labels}) does not match model config ({self.model.config.num_labels})"

    def __getstate__(self):
        return {
            "model_name": self.model_name,
            "labels": self.labels,
            "device_batch_size": self.device_batch_size,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    @torch.inference_mode()
    def score_chunks(self, chunks_token_ids, chunks_token_counts):
        sorted_indices = torch.argsort(chunks_token_counts)

        scores = torch.zeros(len(chunks_token_ids), self.num_labels, dtype=torch.float32)

        for batch_indices in sorted_indices.split(self.device_batch_size):
            max_len = chunks_token_counts[batch_indices].max()

            input_ids = torch.zeros((len(batch_indices), max_len), dtype=torch.long)
            attention_mask = torch.zeros((len(batch_indices), max_len), dtype=torch.long)

            for i, j in enumerate(batch_indices):
                seq = chunks_token_ids[j]
                input_ids[i, :len(seq)] = seq
                attention_mask[i, :len(seq)] = 1

            outputs = self.model(input_ids.to(self.device), attention_mask=attention_mask.to(self.device), use_cache=False)
            scores[batch_indices] = outputs.logits.float().cpu()
        return scores

    def __call__(self, example, indices):
        num_seqs = len(indices)

        source_ids = [i for i, counts in enumerate(example["chunks_token_counts"]) for _ in range(len(counts))]
        chunks_token_ids = [torch.tensor(chunk, dtype=torch.long) for chunks in example["chunks_token_ids"] for chunk in chunks]
        flattened_chunks_token_counts = torch.tensor([chunk for chunks in example["chunks_token_counts"] for chunk in chunks], dtype=torch.long)

        flattened_scores = self.score_chunks(chunks_token_ids, flattened_chunks_token_counts)

        chunk_token_counts = example["chunks_token_counts"]
        chunk_scores = [[[] for _ in range(num_seqs)] for _ in range(self.num_labels)]

        for source_id, score in zip(source_ids, flattened_scores):
            for label in range(self.num_labels):
                chunk_scores[label][source_id].append(score[label].item())

        output = {
            "index": indices,
            "chunk_lengths": chunk_token_counts,
            "length": [sum(counts) for counts in chunk_token_counts],
        }

        for i, label in enumerate(self.labels):
            output[f"{label}_chunks"] = chunk_scores[i]
            output[f"{label}_average"] = [
                np.average(scores, weights=token_counts).item()
                for scores, token_counts in zip(chunk_scores[i], chunk_token_counts)
            ]

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)

    parser.add_argument("-F", "--data_files", type=str, nargs="+", default=[])
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("-M", "--model", type=str, required=True)
    parser.add_argument("-t", "--tokens", type=int, default=512)
    parser.add_argument("--map_batch_size", type=int, default=512)
    parser.add_argument("-b", "--device_batch_size", type=int, default=16)
    parser.add_argument("-w", "--num_workers", type=int, default=1)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--tokens_field", type=str, default="input_ids")
    parser.add_argument("--labels", type=str, nargs="+")

    args = parser.parse_args()
    print(args)

    if args.input == "json":
        dataset = load_dataset("json", data_files=args.data_files, split="train")
    else:
        dataset = load_from_disk(args.input)

    src_dataset = dataset.shard(args.shard[1], args.shard[0], contiguous=True)
    dataset = src_dataset

    print(dataset)
    print("Total number of examples:", len(dataset))
    dataset = dataset.map(
        TokenizeAndChunk(args.model, args.text_field, args.tokens_field, args.tokens),
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=args.num_workers,
        remove_columns=dataset.column_names)

    print("After tokenization: Total number of examples:", len(dataset))
    dataset = dataset.map(
        ModelAnnotator(args.model, args.labels, args.device_batch_size),
        batched=True,
        with_indices=True,
        batch_size=args.map_batch_size,
        remove_columns=dataset.column_names)

    dataset = concatenate_datasets([dataset, src_dataset], axis=1)

    print("After annotation: Total number of examples:", len(dataset))

    print(f"Saving to {args.output}")
    dataset.save_to_disk(args.output)