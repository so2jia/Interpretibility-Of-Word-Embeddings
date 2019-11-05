from Eval.interpretability import Glove
from multiprocessing import freeze_support
from argparse import ArgumentParser

if __name__ == '__main__':
    freeze_support()  # for Windows support
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
    parser.add_argument("--processes", type=int, required=False, default=2,
                        help="The number of processes to initiate for calculations")
    parser.add_argument("--name", type=str, required=False, default="default",
                        help="A prefix for the output files")
    parser.add_argument("--kde_kernel", type=str, required=False, default="gaussian",
                        help="The kernel for kernel density estimation")
    parser.add_argument("--kde_bandwidth", type=float, required=False, default=0.2,
                        help="The bandwidth for kernel density estimation")
    # RNG based dropout
    parser.add_argument("-random_drop", action="store_true", required=False,
                        help="Flag whether you want to drop random words from categories")
    parser.add_argument("--random_seed", type=int, required=False, default=None,
                        help="Seed to random number generation (Default: None)")
    parser.add_argument("--percent", type=float, required=False, default=0.1,
                        help="Percentage of the words to drop out [0, 1] (Default: 0.1)")

    # Validation
    parser.add_argument("--calculate", type=str, required=False, default="score",
                        help="[score|decomp]")
    parser.add_argument("--calculation_args", nargs='*', required=False, default=[],
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
        "load_weights": load_weights,
        "processes": args.processes,
        "name": args.name,
        "kde_params": {"kde_kernel": args.kde_kernel,
                       "kde_bandwidth": args.kde_bandwidth},
        "semcat_random": {"random": args.random_drop,
                          "seed": args.random_seed,
                          "percent": args.percent}
    }
    model = Glove(embedding_params=embedding_params, semcat_dir=semcat_dir,
                  eval_params=eval_params,
                  calculation_type=calculate, calculation_args=calculation_args)
