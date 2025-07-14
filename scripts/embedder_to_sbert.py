# turn contrastively trained model into sbert compatible model
#
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

def wrapper(args):
    word_embedding_model = models.Transformer(args.model, args.seq_len)

    CLS_MODE = False
    MEAN_MODE = False
    MAX_MODE = False

    if args.averaging_method == 'cls':
        CLS_MODE = True
    elif args.averaging_method == 'mean':
        MEAN_MODE = True
    elif args.averaging_method == 'max':
        MAX_MODE = True

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=CLS_MODE,
        pooling_mode_mean_tokens=MEAN_MODE,
        pooling_mode_max_tokens=MAX_MODE
    )

    sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    sbert_model.save(args.save_path)

def test(model_loc, list_of_phrases):
    """
        extract embeddings and their cosine distance, present distances in ASCII table
    """
    model = SentenceTransformer(model_loc)
    embeddings = model.encode(list_of_phrases)
    distances = cosine_similarity(embeddings)

    print(tabulate(distances, headers=list_of_phrases, tablefmt='psql'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--averaging_method', choices=['cls', 'mean', 'max'], required=True)
    args = parser.parse_args()

    wrapper(args)

    test(args.save_path, ['De patient heeft gele koorts',
                          'De patient heeft koorts',
                          'De patient heeft geelzucht',
                          'De voetbreuk is geheeld',
                          'De armbreuk is volledig hersteld'
    ])
