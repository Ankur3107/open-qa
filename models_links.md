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
