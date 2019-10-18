import subprocess

mat = ["w_b", "w_nb", "w_nsb"]
nor = [False, True]

semcat = "../../data/semcat/Categories/"
norm_type = "ks"
dense = False
lamb = 10
line_to_read = 50000

python = "/home/tamas/repos/Interpretibility-Of-Word-Embeddings/venv/bin/python"

<<<<<<< Updated upstream
<<<<<<< Updated upstream
embedding = f"../../data/glove/glove.6B.300d.txt"
e_s = f"../../out/kde/sparse_model/semantic/l_0.{reg}_e_s.npy"
=======
embedding = f"../../out/comp_semantic/comp_I.embedding.100d.txt"
e_s = f"../../out/comp_semantic/comp_e_s.npy"
>>>>>>> Stashed changes
=======
embedding = f"../../out/comp_semantic/comp_I.embedding.100d.txt"
e_s = f"../../out/comp_semantic/comp_e_s.npy"
>>>>>>> Stashed changes

for m in mat:
    for n in nor:
        norm = n
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        w = f"../../out/kde/sparse_model/semantic/l_0.{reg}_{m}.npy"
        output_file = f"../../out/kde/sparse_results/semantic/l0{reg}/l_0.{reg}_{m}.png"
=======
        w = f"../../out/comp_semantic/comp_{m}.npy"
        output_file = f"../../out/comp_semantic/result/comp_{m}.png"
>>>>>>> Stashed changes
=======
        w = f"../../out/comp_semantic/comp_{m}.npy"
        output_file = f"../../out/comp_semantic/result/comp_{m}.png"
>>>>>>> Stashed changes
        args = [python, 'interpretability_correlation.py',
                embedding, e_s, w, semcat, norm_type,
                "--lamb=10", f"--output_file={output_file}"]
        if n:
            args.append("-norm")

        with subprocess.Popen(args=args) as proc:
            pass

# for i in range(1, 6, 1):
#
#     reg = str(i)
#
#     embedding = f"../../data/glove/sparse/semantic/l_0.{reg}I.embedding.100d.txt.gz"
#     e_s = f"../../out/kde/sparse_model/semantic/l_0.{reg}_e_s.npy"
#
#     for m in mat:
#         for n in nor:
#             norm = n
#             w = f"../../out/kde/sparse_model/semantic/l_0.{reg}_{m}.npy"
#             output_file = f"../../out/kde/sparse_results/semantic/l0{reg}/l_0.{reg}_{m}.png"
#             args = [python, 'interpretability_correlation.py',
#                     embedding, e_s, w, semcat, norm_type,
#                     "--lamb=10", f"--output_file={output_file}"]
#             if n:
#                 args.append("-norm")
#
#             with subprocess.Popen(args=args) as proc:
#                 pass
