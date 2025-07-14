"""
    - Given the prior synonyms, extend the synonyms using medical embeddings
    (BioLORD and RotatE) and LLMs, and add negatives.
    - Extract positive contexts from Pubmed etc. see KRISSBERT
    - Create paraphrased positive-pairs
    - High syntactic similarity -> small Levenhstein, or eq, distance but negative to anchor
    - High static word similarity -> small word2vec, or eq, distance but negative to anchor

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
