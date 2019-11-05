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

    paths_raw = [["gauss/kernel_01", "01", "Gauss Kernel 0.1 KDE"],
                 ["gauss/kernel_02", "02", "Gauss Kernel 0.2 KDE"],
                 ["gauss/kernel_05", "05", "Gauss Kernel 0.5 KDE"],
                 ["gauss/kernel_1", "1", "Gauss Kernel 1.0 KDE"],
                 ["gauss/hellinger", "h", "Hellinger Distance Kernel 0.2"],
                 ["original", "o", "Closed Bhattacharyya Distance"]]

    paths_semantic = [["gauss/kernel_01", "s01", "Gauss Kernel 0.1 Semantic"],
                      ["gauss/kernel_02", "s02", "Gauss Kernel 0.2 Semantic"],
                      ["gauss/kernel_05", "s05", "Gauss Kernel 0.5 Semantic"],
                      ["gauss/kernel_1", "s1", "Gauss Kernel 1.0 Semantic"],
                      ["gauss/hellinger", "sh", "Hellinger Distance Kernel 0.2 Semantic"],
                      ["original", "so", "Closed Bhattacharyya Distance Semantic"]]

    for path in paths_raw:
        folder = path[0]
        prefix = path[1]

        glove = "../../data/glove/glove.6B.300d.txt"

        norm = [True, False]
        ws = ["w_b", "w_nb", "w_nsb"]

        w = ws[0]
        params = {
            "embedding_path": glove,
            "embedding_space": f"../../out/{folder}/{prefix}_e_s.npy",
            "semcat_dir": "../../data/semcat/Categories/",
            "distance_space": f"../../out/{folder}/{prefix}_{w}.npy",
            "dense": True,
            "lines_to_read": 50000,
            "lamb": 10,
            "norm": False,
            "man_space": False,
            "name": path[2],
            "output": f"../../out/{folder}/results/{w}_stats{'-norm' if False else ''}.txt"
        }
        interpretability_scores(**params)

    for key, path in enumerate(paths_semantic):
        folder = path[0]
        prefix = path[1]

        custom_space = f"../../out/{paths_raw[key][0]}/{paths_raw[key][1]}_I.embedding.100d.txt"

        norm = [True, False]
        ws = ["w_b", "w_nb", "w_nsb"]
        n = False
        w = ws[0]
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
            "output": f"../../out/{folder}/results/semantic_{w}_stats{'-norm' if n else ''}.txt"
        }
        interpretability_scores(**params)


if __name__ == '__main__':
    main()