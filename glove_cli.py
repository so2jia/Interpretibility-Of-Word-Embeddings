from Eval.glove import Glove

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Glove interpretibility')
    parser.add_argument("embedding_path", type=str,
                        help="Path to the embedding file")
    parser.add_argument("semcat_dir", type=str,
                        help="Path to the SemCat Categories directory")

    parser.add_argument("--lambda_value", type=int, required=False, default=1,
                        help="Lambda value for relaxing score computation")
    parser.add_argument("--weights_dir", type=str, required=False,
                        help="The directory where weight matrices loaded from and saved to "
                             "(f.e. I.npy, wb.npy, wbs.npy)")
    parser.add_argument("--save", action="store_true", required=False,
                        help="Flag whether you want to save the matrices or not")

    args = parser.parse_args()

    embedding_path = args.embedding_path
    semcat_dir = args.semcat_dir

    lamb = args.lambda_value

    weights_dir = args.weights_dir
    save = args.save

    model = Glove(embedding_path, semcat_dir, weights_dir="out" if weights_dir is None else weights_dir,
                  save=save, load=weights_dir is not None)
    model.calculate_score(lamb)
