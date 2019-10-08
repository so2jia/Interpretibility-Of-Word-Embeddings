#!/bin/bash
# Dense raw
nice python interpret_cli.py "data/glove/glove.6B.300d.txt" "data/semcat/Categories/" -dense --lines_to_read=50000 --weights_dir="out/dense/raw/" -save_weights --processes=10 --name="glove.6B.300d" --calculate="score" --calculation_args 10;

# Sparse raw
nice python interpret_cli.py "data/glove/sparse/glove300d_l_0.1_DL_top50000.emb.gz" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/sparse/raw/l01/" -save_weights --processes=10 --name="l_0.1" --calculate="score" --calculation_args 10;
nice python interpret_cli.py "data/glove/sparse/glove300d_l_0.2_DL_top50000.emb.gz" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/sparse/raw/l02/" -save_weights --processes=10 --name="l_0.2" --calculate="score" --calculation_args 10;
nice python interpret_cli.py "data/glove/sparse/glove300d_l_0.3_DL_top50000.emb.gz" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/sparse/raw/l03/" -save_weights --processes=10 --name="l_0.3" --calculate="score" --calculation_args 10;
nice python interpret_cli.py "data/glove/sparse/glove300d_l_0.4_DL_top50000.emb.gz" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/sparse/raw/l04/" -save_weights --processes=10 --name="l_0.4" --calculate="score" --calculation_args 10;
nice python interpret_cli.py "data/glove/sparse/glove300d_l_0.5_DL_top50000.emb.gz" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/sparse/raw/l05/" -save_weights --processes=10 --name="l_0.5" --calculate="score" --calculation_args 10;

# Dense semantical
nice python interpret_cli.py "out/dense/raw/glove.6B.300d_I.embedding.100d.txt" "data/semcat/Categories/" -dense --lines_to_read=50000 --weights_dir="out/dense/semantical/" -save_weights --processes=10 --name="glove.6B.300d" --calculate="score" --calculation_args 10;

# Sparse semantical
gzip out/sparse/raw/l01/l_0.1_I.embedding.100d.txt;
nice python interpret_cli.py "out/sparse/raw/l01/l_0.1_I.embedding.100d.txt.gz" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/sparse/semantical/l01/" -save_weights --processes=10 --name="l_0.1" --calculate="score" --calculation_args 10;

gzip out/sparse/raw/l01/l_0.2_I.embedding.100d.txt;
nice python interpret_cli.py "out/sparse/raw/l01/l_0.2_I.embedding.100d.txt.gz" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/sparse/semantical/l02/" -save_weights --processes=10 --name="l_0.2" --calculate="score" --calculation_args 10;

gzip out/sparse/raw/l01/l_0.3_I.embedding.100d.txt;
nice python interpret_cli.py "out/sparse/raw/l01/l_0.3_I.embedding.100d.txt.gz" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/sparse/semantical/l03/" -save_weights --processes=10 --name="l_0.3" --calculate="score" --calculation_args 10;

gzip out/sparse/raw/l01/l_0.4_I.embedding.100d.txt;
nice python interpret_cli.py "out/sparse/raw/l01/l_0.4_I.embedding.100d.txt.gz" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/sparse/semantical/l04/" -save_weights --processes=10 --name="l_0.4" --calculate="score" --calculation_args 10;

gzip out/sparse/raw/l01/l_0.5_I.embedding.100d.txt;
nice python interpret_cli.py "out/sparse/raw/l01/l_0.5_I.embedding.100d.txt.gz" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/sparse/semantical/l05/" -save_weights --processes=10 --name="l_0.5" --calculate="score" --calculation_args 10;