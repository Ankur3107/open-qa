# open-qa

Open QA is a tool for question and answering over large text collections. It has retriever, reranker and reader components.

# Install Requirements

    pip install -r requirements.txt

# Run

    from qa_pipeline import Pipeline
    qa_pipeline = Pipeline()
    queries = ["who is stark?", "Who won the war?"]
    qa_pipeline.run(queries)