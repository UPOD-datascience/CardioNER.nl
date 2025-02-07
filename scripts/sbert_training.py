'''
    Basically:
        Model 1
        * create the triplets (anchor, positive, negative)
        * create the model with triplet loss
        Model 2
        * create pairs (anchor, positive)
        * create the model with MultipleNegativesRankingLoss

    https://sbert.net/docs/package_reference/sentence_transformer/losses.html

    
    For this, negatives need to be mined, and for both positives and negatives we need to find longer contexts, e.g. as is done in KRISSBERT

'''
