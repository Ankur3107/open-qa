from utils import DataProcessor DataReader DataReranker DataRetriever

class Pipeline():

    def __init__(self):
        self.data_processor = DataProcessor()
        self.data_processor.initialize_temp_store()

        self.data_retriever = DataRetriever(data_processor.document_store)
        self.data_processor.update_embeddings(data_retriever.retriever)

        self.data_reranker = DataReranker()
        self.data_reader = DataReader()

    def run(queries, top_k_retrieve=5, top_k_ranker=3):
        retrieve_documents = self.data_retriever.batch_retrieve(queries, top_k_retrieve, index='document')
        reranked_documents = data_reranker.run(queries, retrieve_documents, top_k_ranker = top_k_ranker, top_k_retrieve = top_k_retrieve)
        results = self.data_reader.run(queries, reranked_documents)
        return results