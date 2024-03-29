from Eval.interpretability import Glove

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Glove interpretibility')

    # Required parameters
    parser.add_argument("embedding_path", type=str,
                        help="Path to the embedding file")
    parser.add_argument("semcat_dir", type=str,
                        help="Path to the SemCat Categories directory")

    # Embedding related parameters
    parser.add_argument('-dense_file', action='store_true', help='Mark if it is a dense embedding file')
    parser.set_defaults(dense=False)
    parser.add_argument('--lines_to_read', type=int, help='number of embeddings to read', default=-1)
    parser.add_argument('--mcrae_dir', type=str, help='path to the McRae file', default=None)
    parser.add_argument('-mcrae_words_only', action='store_true')

    # Model related parameteres
    parser.add_argument("--weights_dir", type=str, required=False,
                        help="The directory where weight matrices loaded from and saved to"
                             "(f.e. I.npy, w_b.npy, w_bs.npy). Default is \"out/\", if not provided.")
    parser.add_argument("-save_weights", action="store_true", required=False,
                        help="Flag whether you want to save the matrices")
    parser.add_argument("-load_weights", action="store_true", required=False,
                        help="Flag whether you want to load the matrices")

    # Validation
    parser.add_argument("--calculate", type=str, required=False, default="score",
                        help="[score|decomp]")
    parser.add_argument("--calculation_args", type=list, nargs='*', required=False, default=[],
                        help='''\
                            Takes calculation arguments:
                            score:
                                [int ] Lamda value which > 0. Optional: Default 1.
                            decomp:
                                [str ] The word to decompose
                                [int ] The top X category. Optional: Default 20. 
                                [bool ] Save result into file. Optional: Default False.
                        ''')

    args = parser.parse_args()

    # Required parameters
    embedding_path = args.embedding_path
    semcat_dir = args.semcat_dir

    # Embedding related parameters
    dense_file = args.dense_file
    lines_to_read = args.lines_to_read
    mcrae_dir = args.mcrae_dir
    mcrae_words_only = args.mcrae_words_only

    # Model related parameters
    weights_dir = args.weights_dir
    save_weights = args.save_weights
    load_weights = args.load_weights

    # Validation
    calculate = args.calculate
    calculation_args = [] if args.calculation_args is None else args.calculation_args

    embedding_params = {
        "input_file": embedding_path,
        "dense_file": dense_file,
        "lines_to_read": lines_to_read,
        "mcrae_dir": mcrae_dir,
        "mcrae_words_only": mcrae_words_only
    }

    eval_params = {
        "weights_dir": weights_dir,
        "save_weights": save_weights,
        "load_weights": load_weights
    }
    model = Glove(embedding_params=embedding_params, semcat_dir=semcat_dir,
                  eval_params=eval_params,
                  calculation_type=calculate, calculation_args=calculation_args)
