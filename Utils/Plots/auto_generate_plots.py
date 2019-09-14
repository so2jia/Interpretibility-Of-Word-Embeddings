import subprocess

mat = ["w_b", "w_nb", "w_nsb"]
nor = [False, True]

embedding = "../../data/glove/glove.6B.300d.txt"
e_s = "../../out/e_s.npy"
semcat = "../../data/semcat/Categories/"
norm_type = "ks"
dense = True
lamb = 10
line_to_read = 50000

python = "E:/Research/Interpretibility-Of-Word-Embeddings/venv/Scripts/python"

for m in mat:
    for n in nor:
        norm = n
        w = f"../../out/{m}.npy"
        output_file = f"../../out/test_data_I/cor_val_interpret_{m}.png"
        args = [python, 'interpretability_correlation.py',
                embedding, e_s, w, semcat, norm_type, "-dense",
                "--lamb=10", f"--output_file={output_file}"]
        if n:
           args.append("-norm")

        with subprocess.Popen(args=args) as proc:
            pass
