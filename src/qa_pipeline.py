from utils import DataProcessor, DataReader, DataReranker, DataRetriever

class Pipeline():

    def __init__(self):
        self.data_processor = DataProcessor()
        self.data_processor.initialize_temp_store()

        self.data_retriever = DataRetriever(self.data_processor.document_store)
        self.data_processor.update_embeddings(self.data_retriever.retriever)

        self.data_reranker = DataReranker()
        self.data_reader = DataReader()

    def run(self, queries, reranker_batch_size=32, reader_batch_size=32, top_k_retrieve=5, top_k_ranker=3):
        retrieve_documents = self.data_retriever.batch_retrieve(queries, top_k_retrieve, index='document')
        reranked_documents = self.data_reranker.run(queries, retrieve_documents, top_k_ranker = top_k_ranker, top_k_retrieve = top_k_retrieve)
        results = self.data_reader.run(queries, reranked_documents,batch_size=reader_batch_size, top_k_reader=top_k_ranker)
        return results