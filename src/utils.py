from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.document_store import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.schema import Document
from sentence_transformers import CrossEncoder
import numpy as np


class DataProcessor:
    def __init__(self):
        self.document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

    def initialize_temp_store(self):
        # Let's first get some files that we want to use
        doc_dir = "data/article_txt_got"
        s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
        fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

        # Convert files to dicts
        dicts = convert_files_to_dicts(
            dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True
        )

        # Now, let's write the dicts containing documents to our DB.
        self.document_store.write_documents(dicts)

    def update_embeddings(self, retriever):
        self.document_store.update_embeddings(retriever)


class Data_Retriever:
    def __init__(self, document_store):
        self.document_store = document_store
        self.retriever = DensePassageRetriever(
            document_store=self.document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            max_seq_len_query=64,
            max_seq_len_passage=256,
            batch_size=16,
            use_gpu=True,
            embed_title=True,
            use_fast_tokenizers=True,
        )

    def batch_retrieve(self, queries, top_k, index="document"):
        query_emb = self.retriever.embed_queries(texts=queries)
        score_matrix, vector_id_matrix = self.document_store.faiss_indexes[
            index
        ].search(query_emb, top_k)

        documents = []
        for i_vector_id_matrix in vector_id_matrix:
            vector_ids_for_query = [
                str(vector_id) for vector_id in i_vector_id_matrix if vector_id != -1
            ]
            i_documents = self.document_store.get_documents_by_vector_ids(
                vector_ids_for_query, index="document"
            )
            documents.append(i_documents)
        return documents


class Reranker:
    def __init__(
        self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=256
    ):
        self.model = CrossEncoder(model_name, max_length=max_length)

    def ranker_input_data_format(self, question, passages):
        pairs = []

        for passage in passages:
            pairs.append((question, passage))

        return pairs

    def batch_ranker_input_data_format(self, queries, retrieve_documents):
        pairs = []

        for i in range(len(queries)):
            question = queries[i]
            passages = [d.text for d in retrieve_documents[i]]
            i_pairs = self.ranker_input_data_format(question, passages)
            pairs.extend(i_pairs)

        return pairs

    def get_top_k_reranker_result(
        self, retrieve_documents, top_k_ranker, shorted_index, shorted_scores
    ):
        retrieve_reranker_documents = []
        for i in range(len(shorted_index)):
            i_index = shorted_index[i]
            i_score = shorted_scores[i]
            i_retrieve_documents = retrieve_documents[i]
            i_shorted_retrieve_documents = []

            for j in range(top_k_ranker):
                ij_retrieve_documents = i_retrieve_documents[i_index[j]]
                ij_retrieve_documents.score = i_score[j]
                i_shorted_retrieve_documents.append(ij_retrieve_documents)

            retrieve_reranker_documents.append(i_shorted_retrieve_documents)

        return retrieve_reranker_documents

    def run(self, queries, retrieve_documents, top_k_ranker=3, top_k_retrieve=5):
        pre_processed_ranker_input = self.batch_ranker_input_data_format(
            queries, retrieve_documents
        )
        scores = self.model.predict(pre_processed_ranker_input, show_progress_bar=True)
        scores = scores.reshape(-1, top_k_retrieve)

        shorted_index = np.argpartition(scores, -top_k_ranker, axis=1)[:, ::-1]
        shorted_scores = np.array(
            [scores[i][shorted_index[i]] for i in range(len(shorted_index))]
        )
        retrieve_reranker_documents = self.get_top_k_reranker_result(
            retrieve_documents, top_k_ranker, shorted_index, shorted_scores
        )
        return retrieve_reranker_documents


class Data_Reader:
    def __init__(self):
        self.reader = FARMReader(
            model_name_or_path="deepset/roberta-base-squad2", use_gpu=True
        )

    def reader_preprocessing(self, queries, retrieve_documents):
        query_doc_list = []

        for i in range(len(queries)):
            query = Document(id=i, question=queries[i], text="")
            documents = retrieve_documents[i]
            query_doc_list.append({"docs": documents, "question": query})
        return query_doc_list

    def run(self, queries, reranked_documents, batch_size=8, top_k_reader=2):
        query_doc_list = self.reader_preprocessing(queries, reranked_documents)
        results = self.reader.predict_batch(
            query_doc_list, top_k=top_k_reader, batch_size=batch_size
        )
        return results
