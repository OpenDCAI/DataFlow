"""
PDF Table QA Pipeline with Map-Reduce Support
Pipeline for extracting tables from PDF and generating Q/A pairs using Map-Reduce approach
"""
from dataflow.operators.knowledge_cleaning import (
    FileOrURLToMarkdownConverterAPI,
    PDFTableQAGenerator,
    TableChunkSplitter,
    MapReduceQAGenerator,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request


class PDFTableQA_APIPipeline():
    """
    Pipeline for generating Q/A pairs from PDF tables.
    
    Supports two modes:
    1. Direct mode: For small documents within token limits
    2. Map-Reduce mode: For large documents exceeding token limits
    
    Steps (Map-Reduce mode):
    1. Convert PDF to Markdown (tables preserved)
    2. Split document by table boundaries
    3. Generate summaries for each chunk (Map)
    4. Plan and generate Q/A pairs (Reduce)
    """
    
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="/home/wangdeng/Github/dataflow/DataFlow/dataflow/example/PDFTableQAPipeline/pdf_table_qa_test.jsonl",
            cache_path="./.cache/pdf_table_qa",
            file_name_prefix="pdf_table_qa_step",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
            api_url="https://oneapi.hkgai.net/v1/chat/completions",
            model_name="qwen3next",
            max_workers=8,
            temperature=0.3
        )

        # Step 1: PDF to Markdown conversion (preserves tables)
        self.pdf_to_markdown_step1 = FileOrURLToMarkdownConverterAPI(
            intermediate_dir="../example_data/PDFTableQAPipeline/raw/",
            mineru_backend="vlm",
        )

        # Step 2: Split by table boundaries
        self.table_chunk_splitter_step2 = TableChunkSplitter(
            max_chunk_size=20000,
            min_chunk_size=500,
            context_before=500,
            context_after=500,
        )

        # Step 3: Map-Reduce QA generation
        self.map_reduce_qa_step3 = MapReduceQAGenerator(
            llm_serving=self.llm_serving,
            lang="en",
            max_qa=50,
            max_relevant_chunks=6,
        )

        # Legacy: Direct mode generator (for small documents)
        self.table_qa_generator_direct = PDFTableQAGenerator(
            llm_serving=self.llm_serving,
            lang="en",
            max_qa=50,
            max_text_length=1000000,
        )

    def forward(self, use_map_reduce: bool = True, merge_mode: bool = True):
        """
        Run the pipeline.
        
        Args:
            use_map_reduce: If True, use Map-Reduce approach for large documents.
                           If False, use direct approach (may fail for large docs).
            merge_mode: If True, merge all documents into one before generating Q/A.
                       This is useful when multiple documents contain related data.
        """
        import os
        import pandas as pd
        
        # Step 1: Convert PDF to Markdown
        print("[Step 1] Converting PDF to Markdown...")
        step1_storage = self.storage.step()
        self.pdf_to_markdown_step1.run(
            storage=step1_storage,
            input_key="source",
            output_key="text_path",
        )
        
        if merge_mode:
            # Merge all documents into one before processing
            print("[Merge Mode] Merging all documents...")
            step2_storage = self.storage.step()
            df = step2_storage.read("dataframe")
            
            # Read all markdown files
            all_texts = []
            all_sources = []
            for idx, row in df.iterrows():
                text_path = row.get("text_path", "")
                source = row.get("source", "")
                if text_path and os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        all_texts.append(f"\n\n--- Document {idx + 1}: {os.path.basename(source)} ---\n\n{content}")
                        all_sources.append(source)
            
            # Merge all texts into one
            merged_text = "\n".join(all_texts)
            print(f"[Merge Mode] Merged {len(all_sources)} documents, total length: {len(merged_text)} chars")
            
            # Create merged dataframe
            merged_df = pd.DataFrame([{
                "source": " + ".join(all_sources),
                "text": merged_text,
            }])
            
            if use_map_reduce:
                # Split merged text into chunks
                chunks = self.table_chunk_splitter_step2.process(merged_text)
                print(f"[Merge Mode] Split into {len(chunks)} chunks")
                
                # Generate Q/A using Map-Reduce directly
                print("[Step 3] Generating Q/A pairs using Map-Reduce on merged documents...")
                csv_output_dir = "./.cache/pdf_table_qa/tables"
                result = self.map_reduce_qa_step3.process(chunks, csv_output_dir=csv_output_dir)
                
                merged_df['chunks'] = [chunks]
                merged_df['table_qa_pairs'] = [result['qa_pairs']]
                merged_df['chunk_summaries'] = [result['summaries']]
                merged_df['qa_plans'] = [result['plans']]
                merged_df['table_csv_paths'] = [result['csv_paths']]
                step2_storage.write(merged_df)
                
                print(f"[Merge Mode] Extracted {len(result['csv_paths'])} tables to CSV")
                print(f"[Merge Mode] Generated {len(result['qa_pairs'])} Q/A pairs from merged documents")
            else:
                # Direct processing on merged text
                merged_df['text_path'] = 'merged'
                step2_storage.write(merged_df)
                
                print("[Step 2] Generating Q/A pairs directly on merged documents...")
                outputs = self.table_qa_generator_direct.process_batch([merged_text])
                
                # Extract tables to CSV
                csv_output_dir = "./.cache/pdf_table_qa/tables"
                csv_paths = self.table_qa_generator_direct._extract_tables_to_csv(
                    text=merged_text,
                    output_dir=csv_output_dir,
                    base_name="merged_docs"
                )
                
                result_df = pd.DataFrame([{
                    "source": " + ".join(all_sources),
                    "text": merged_text,
                    "table_qa_pairs": outputs[0]['qa_pairs'] if outputs else [],
                    "has_table": outputs[0]['has_table'] if outputs else False,
                    "table_csv_paths": csv_paths,
                }])
                step2_storage.write(result_df)
            
            print("[Done] Merged Q/A generation complete!")
            print("Results saved to ./.cache/pdf_table_qa/")
        
        elif use_map_reduce:
            # Map-Reduce mode for large documents (process each separately)
            print("[Step 2] Splitting document by table boundaries...")
            self.table_chunk_splitter_step2.run(
                storage=self.storage.step(),
                input_key="text",
                output_key="chunks",
            )
            
            print("[Step 3] Generating Q/A pairs using Map-Reduce...")
            self.map_reduce_qa_step3.run(
                storage=self.storage.step(),
                input_key="chunks",
                output_key="table_qa_pairs",
            )
            
            print("[Done] Map-Reduce Q/A generation complete!")
            print("Results saved to ./.cache/pdf_table_qa/")
        else:
            # Direct mode for small documents
            print("[Step 2] Generating Q/A pairs (direct mode)...")
            self.table_qa_generator_direct.run(
                storage=self.storage.step(),
                input_key="text",
                input_text_path_key="text_path",
                output_key="table_qa_pairs",
                csv_output_dir="./.cache/pdf_table_qa/tables",
            )
            
            print("[Done] Direct Q/A generation complete!")


if __name__ == "__main__":
    pipeline = PDFTableQA_APIPipeline()
    # use_map_reduce=True: For large documents (recommended)
    # use_map_reduce=False: For small documents within token limits
    # merge_mode=True: Merge all documents before generating Q/A
    # merge_mode=False: Process each document separately
    pipeline.forward(use_map_reduce=True, merge_mode=True)

