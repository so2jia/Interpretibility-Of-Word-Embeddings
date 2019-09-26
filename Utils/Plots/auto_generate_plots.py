import subprocess

mat = ["w_b", "w_nb", "w_nsb"]
nor = [False, True]

reg = "3"

embedding = f"../../data/glove/glove.6B.300d.txt"
e_s = f"../../out/kde/dense_model/e_s.npy"
semcat = "../../data/semcat/Categories/"
norm_type = "ks"
dense = False
lamb = 10
line_to_read = 50000

python = "/home/tamas/repos/Interpretibility-Of-Word-Embeddings/venv/bin/python"

for m in mat:
    for n in nor:
        norm = n
        w = f"../../out/kde/dense_model/{m}.npy"
        output_file = f"../../out/kde/dense_results/dense.6B.300d_{m}.png"
        args = [python, 'interpretability_correlation.py',
                embedding, e_s, w, semcat, norm_type, "-dense",
                "--lamb=10", f"--output_file={output_file}"]
        if n:
           args.append("-norm")

        with subprocess.Popen(args=args) as proc:
            pass
