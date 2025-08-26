import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

class BM25Miner():
    """
    A self-contained class for performing BM25 search.
    It builds an in-memory index from the provided documents.
    """
    def __init__(self, documents: list, doc_ids: list):
        try:
            from pyserini import analysis
            from gensim.corpora import Dictionary
            from gensim.models import LuceneBM25Model
            from gensim.similarities import SparseMatrixSimilarity
        except ImportError:
            # Follow https://github.com/facebookresearch/ReasonIR/blob/main/synthetic_data_generation/setup_java.sh
            # export JAVA_HOME=~/jdk-23.0.1
            # export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
            # export PATH=$JAVA_HOME/bin:$PATH
            raise ImportError("Please install pyserini, gensim and JDK to use BM25 miner.")
        self.documents = documents
        self.doc_ids = doc_ids
        # Use a pre-defined Lucene analyzer for consistency
        self.analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
        
        # Process documents for BM25 model
        corpus = [self.analyzer.analyze(doc) for doc in self.documents]
        self.dictionary = Dictionary(corpus)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=0.9, b=0.4)
        bm25_corpus = self.model[list(map(self.dictionary.doc2bow, corpus))]
        
        # Create a similarity index
        self.bm25_index = SparseMatrixSimilarity(
            bm25_corpus, 
            num_docs=len(corpus), 
            num_terms=len(self.dictionary),
            normalize_queries=False, 
            normalize_documents=False
        )

    def search(self, query: str, top_k: int = 1000) -> dict:
        """
        Searches the indexed documents for a given query and returns top_k results.
        """
        tokenized_query = self.analyzer.analyze(query)
        bm25_query = self.model[self.dictionary.doc2bow(tokenized_query)]
        
        # Get similarity scores
        similarities = self.bm25_index[bm25_query].tolist()
        
        # Sort and get top results
        all_scores = sorted(
            zip(self.doc_ids, similarities), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return as a dictionary of {doc_id: score}
        return dict(all_scores[:top_k])

    def select_hard_negatives(self, query: str, gold_id: int, num_neg: int = 3, hard_neg_start_index: int = 0) -> list:
        """
        Selects hard negative documents for a given query and its positive document (gold_id).
        """
        scores = self.search(query)
        hard_negatives_docs = []
        
        # Sort retrieved documents by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for doc_id, score in sorted_scores[hard_neg_start_index:]:
            # Ensure the document is not the positive example itself
            if doc_id != gold_id:
                hard_negatives_docs.append(self.documents[doc_id])
            if len(hard_negatives_docs) == num_neg:
                break
                
        return hard_negatives_docs


@OPERATOR_REGISTRY.register()
class BM25HardNeg(OperatorABC):
    '''
    BM25HardNeg operator mines hard negatives for a given query using the BM25 algorithm.
    It reads a dataframe with queries and positive documents, and appends a column with hard negatives.
    '''

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "RAREPipeline: BM25HardNeg 算子使用 BM25 算法为给定查询挖掘困难负样本。\n\n"
                "输入参数：\n"
                "- input_question_key: 包含查询的字段名。\n"
                "- input_text_key: 包含正面文档的字段名。\n"
                "- output_negatives_key: 用于存储挖掘出的困难负样本列表的字段名。\n"
                "- num_neg: 每个查询需要挖掘的困难负样本数量。\n"
            )
        elif lang == "en":
            return (
                "RAREPipeline: BM25HardNeg operator mines hard negatives for a given query using the BM25 algorithm.\n\n"
                "Input Parameters:\n"
                "- input_question_key: Field name containing the query.\n"
                "- input_text_key: Field name containing the positive document text.\n"
                "- output_negatives_key: Field name for storing the list of mined hard negatives.\n"
                "- num_neg: The number of hard negatives to mine for each query.\n"
            )
        else:
            return "RAREPipeline: BM25HardNeg operator mines hard negatives for a given query using the BM25 algorithm."

    def _validate_dataframe(self, dataframe: pd.DataFrame, input_question_key: str, input_text_key: str, output_negatives_key: str):
        required_keys = [input_question_key, input_text_key]
        
        missing = [k for k in required_keys if k not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        
        if output_negatives_key in dataframe.columns:
            raise ValueError(f"The output column '{output_negatives_key}' already exists and would be overwritten.")

    def run(
        self,
        storage: DataFlowStorage,
        input_question_key: str = "question",
        input_text_key: str = "text",
        output_negatives_key: str = "hard_negatives",
        num_neg: int = 3
    ) -> list:
        '''
        Runs the hard negative mining process.
        '''
        self.logger.info("Starting BM25 hard negative mining process.")
        
        # 1. Read data using the storage abstraction
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe, input_question_key, input_text_key, output_negatives_key)
        
        # 2. Initialize the BM25 miner with all documents from the dataframe
        documents = dataframe[input_text_key].tolist()
        doc_ids = list(range(len(documents))) # Use simple integer indices as document IDs
        
        self.logger.info(f"Building BM25 index for {len(documents)} documents...")
        bm25_miner = BM25Miner(documents=documents, doc_ids=doc_ids)
        self.logger.info("BM25 index built successfully.")

        # 3. Process each row to find hard negatives
        hard_negatives_list = []
        for index, row in dataframe.iterrows():
            query = row[input_question_key]
            # 'index' serves as the gold_id for the current positive document
            negs = bm25_miner.select_hard_negatives(query, gold_id=index, num_neg=num_neg)
            hard_negatives_list.append(negs)
        
        dataframe[output_negatives_key] = hard_negatives_list
        
        # 4. Write data back using the storage abstraction
        output_file = storage.write(dataframe)
        self.logger.info(f"Hard negative mining complete. Results saved to {output_file}")

        # 5. Return the name of the newly created column
        return [output_negatives_key]