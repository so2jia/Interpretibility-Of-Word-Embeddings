from Utils.Plots.interpretability_score import interpretability_scores


def main():
    # ["comp", "comp", "Complementer"],
    # paths_raw = [["comp", "comp", "Complementer"],
    #              ["exponential", "dense", "Exponential Kernel"],
    #              ["hellinger", "hellinger", "Hellinger Distance"],
    #              ["norm", "norm", "L2 Normed Space"],
    #              ["bandwidth", "bw", "Bandwidth Estimation"]]
    #
    # paths_semantic = [["comp_semantic", "comp", "Complementer Semantic"],
    #                   ["exponential_semantic", "exp", "Exponential Kernel Semantic"],
    #                   ["hellinger_semantic", "hellinger", "Hellinger Distance Semantic"],
    #                   ["norm_semantic", "norm", "L2 Normed Space Semantic"],
    #                   ["bandwidth_semantic", "bw", "bandwidth Estimation Semantic"]]

    paths_raw = [["gauss", "glove.6B.300d", "Gauss Kernel KDE"]]

    paths_semantic = [["gauss_semantic", "glove.6B.300d", "Gauss Kernel Semantic"]]

    for path in paths_raw:
        folder = path[0]
        prefix = path[1]

        glove = "../../data/glove/glove.6B.300d.txt"

        norm = [True, False]
        ws = ["w_b", "w_nb", "w_nsb"]
        for n in norm:
            for w in ws:
                params = {
                    "embedding_path": glove,
                    "embedding_space": f"../../out/{folder}/{prefix}_e_s.npy",
                    "semcat_dir": "../../data/semcat/Categories/",
                    "distance_space": f"../../out/{folder}/{prefix}_{w}.npy",
                    "dense": True,
                    "lines_to_read": 50000,
                    "lamb": 10,
                    "norm": n,
                    "man_space": False,
                    "name": path[2],
                    "output": f"../../out/{folder}/results/{w}_stats{'-norm' if n else ''}.txt"
                }
                interpretability_scores(**params)


    for path in paths_semantic:
        folder = path[0]
        prefix = path[1]

        custom_space = f"../../out/{folder}/{prefix}_I.embedding.100d.txt"

        norm = [True, False]
        ws = ["w_b", "w_nb", "w_nsb"]
        for n in norm:
            for w in ws:
                params = {
                    "embedding_path": custom_space,
                    "embedding_space": f"../../out/{folder}/{prefix}_e_s.npy",
                    "semcat_dir": "../../data/semcat/Categories/",
                    "distance_space": f"../../out/{folder}/{prefix}_{w}.npy",
                    "dense": True,
                    "lines_to_read": 50000,
                    "lamb": 10,
                    "norm": n,
                    "man_space": False,
                    "name": path[2],
                    "output": f"../../out/{folder}/results/{w}_stats{'-norm' if n else ''}.txt"
                }
                interpretability_scores(**params)


if __name__ == '__main__':
    main()