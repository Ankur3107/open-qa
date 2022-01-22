# Google: Large Dual Encoders Are Generalizable Retrievers (GTR)

  https://arxiv.org/pdf/2112.07899.pdf
  
  https://tfhub.dev/google/gtr/gtr-base/1
  
  Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models (https://arxiv.org/pdf/2108.08877.pdf)
  


# Condenser and coCondenser

  https://github.com/luyug/Condenser
  
  1.  Luyu/condenser: Condenser pre-trained on BookCorpus and Wikipedia
  2.  Luyu/co-condenser-wiki: coCondenser pre-trained on Wikipedia
  3.  Luyu/co-condenser-marco: coCondenser pre-trained on MS-MARCO collection

# DensePhrases

  https://github.com/princeton-nlp/DensePhrases
  
      >>> from densephrases import DensePhrases

      # Load DensePhrases for dialogue and entity linking
      >>> model = DensePhrases(
      ...     load_dir='princeton-nlp/densephrases-multi-query-kilt-multi',
      ...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump',
      ... )

      # Retrieve relevant documents for a dialogue
      >>> model.search('I love rap music.', retrieval_unit='document', top_k=5)
      ['Rapping', 'Rap metal', 'Hip hop', 'Hip hop music', 'Hip hop production']

      # Run entity linking for the target phrase denoted as [START_ENT] and [END_ENT]
      >>> model.search('[START_ENT] Security Council [END_ENT] members expressed concern on Thursday', retrieval_unit='document', top_k=1)
      ['United Nations Security Council']
      
# docTTTTTquery document expansion model
  https://github.com/castorini/docTTTTTquery
  
  https://huggingface.co/castorini/doc2query-t5-base-msmarco

# Training Dense Retrieval with Balanced Topic Aware Sampling (TAS-Balanced)
  https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval
  
  https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco
  
# Margin-MSE Trained DistilBert for Dense Passage Retrieval
  https://github.com/sebastian-hofstaetter/neural-ranking-kd
  
  https://huggingface.co/sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco
  
# Intra-Document Cascading (IDCM)
This instance can be used to re-rank a candidate set of long documents. The base BERT architecure is a 6-layer DistilBERT.
  https://github.com/sebastian-hofstaetter/intra-document-cascade
  
  https://huggingface.co/sebastian-hofstaetter/idcm-distilbert-msmarco_doc
  
