"""
    - Given the prior synonyms, extend the synonyms using medical embeddings
    (BioLORD and RotatE) and LLMs, and add negatives.
    - Extract positive contexts from Pubmed etc. see KRISSBERT
    - Create paraphrased positive-pairs
    - High syntactic similarity -> small Levenhstein, or eq, distance but negative to anchor
    - High static word similarity -> small word2vec, or eq, distance but negative to anchor

We expand this with negative term pairs by random sampling, followed by selecting the top-N most dissimilar terms with (sim<0.25)
* using [BioLORD 2023 M Dutch](https://huggingface.co/FremyCompany/BioLORD-2023-M-Dutch-InContext-v1)
* using [sapMedRoBERTa.nl](https://huggingface.co/UMCU/sap_umls_medroberta.nl_meantoken)
* using [RotatE graph embedding](https://arxiv.org/abs/1902.10197)

We further expand these lists using LLMs.

We now end up with.
* ontology-based same-concept positive pairs, extended with LLMs
* ontology-based related-concept positive pairs, extended with LLMs
* BioLoRD/RotatE/sap-based negatives, extended with LLMs

"""


class EmbeddingTripletExtender():
    def __init__(self,
                 triplet_location: str,
                 embedding_file: str,
                 sbert_model: str,
                 mine_negatives: False,
                 ):


    def load_triplets(self, location: str=None):
        # dictionary: set((anchor, positive, positive))
        pass

    def load_graph_embedder(self):
        pass

    def load_span_encoder(self)
        pass

    def get_positives(self, top_K=5, threshold_minimum=0.8):
        pass


    def get_negatives(self, bottom_K=5, threshold_maximum=0.1)
        # dictionary: set((anchor, positive1, positive2))
        # ->
        # dictionary: set((anchor, positive1, negative))
        # dictionary: set((anchor, positive2, negative))
        pass


class UMLSTripletExtender():
    # use MRREF to extract related concepts for terms


class LLMTripletExtender():
    # use LLMs to extract similar/dissimilar terms given a list of positives/negatives
