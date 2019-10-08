#!/bin/bash
nice python interpret_cli.py "data/glove/sparse/l_0.$1_I.embedding.100d.txt" "data/semcat/Categories/" --lines_to_read=50000 --weights_dir="out/" -save_weights --processes=2 --name="l_0.$1" --calculate="score" --calculation_args 10;

