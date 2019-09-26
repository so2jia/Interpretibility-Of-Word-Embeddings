import subprocess

mat = ["w_b", "w_nb", "w_nsb"]
nor = [False, True]

reg = "3"

embedding = f"../../data/glove/sparse/glove300d_l_0.{reg}_DL_top50000.emb.gz"
e_s = f"../../out/l_0.{reg}_e_s.npy"
semcat = "../../data/semcat/Categories/"
norm_type = "ks"
dense = False
lamb = 10
line_to_read = 50000

python = "/home/tamas/repos/Interpretibility-Of-Word-Embeddings/venv/bin/python"

for m in mat:
    for n in nor:
        norm = n
        w = f"../../out/l_0.{reg}_{m}.npy"
        output_file = f"../../out/results/l0{reg}/l0{reg}_{m}.png"
        args = [python, 'interpretability_correlation.py',
                embedding, e_s, w, semcat, norm_type,
                "--lamb=10", f"--output_file={output_file}"]
        if n:
           args.append("-norm")

        with subprocess.Popen(args=args) as proc:
            pass
