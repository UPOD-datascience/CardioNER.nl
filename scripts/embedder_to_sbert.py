# turn contrastively trained model into sbert compatible model
#
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from torch import bfloat16

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
    if args.bf16:
        sbert_model = sbert_model.to(bfloat16)
    sbert_model.save(args.save_path)

def test(model_loc, list_of_phrases):
    """
        extract embeddings and their cosine distance, present distances in ASCII table
    """
    model = SentenceTransformer(model_loc)
    embeddings = model.encode(list_of_phrases)
    distances = cosine_similarity(embeddings)

    print(tabulate(distances, headers=list_of_phrases, tablefmt='psql', floatfmt='.4f'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seq_len', type=int, required=False)
    parser.add_argument('--save_path', type=str, required=False)
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--averaging_method', choices=['cls', 'mean', 'max'], required=False)
    parser.add_argument('--test_only', action='store_true', default=False)
    args = parser.parse_args()

    if args.test_only:
        test(args.model, ['koorts [SEP] De patient heeft gele koorts',
                          'koorts [SEP] De patient heeft koorts',
                          'verlatingsangst [SEP] De patient heeft verlatingsangst',
                          'voetbreuk [SEP] De voetbreuk is geheeld',
                          'voetbreuk [SEP] De voetbreuk is niet hersteld'
        ])
    else:
        wrapper(args)
        test(args.save_path, ['De patient heeft gele koorts',
                            'De patient heeft koorts',
                            'De patient heeft geelzucht',
                            'De voetbreuk is geheeld',
                            'De voetbreuk is niet hersteld'
        ])
