from Eval.glove import Glove
import numpy as np

embedding_path = "data/glove/glove.6B.300d.txt"
semcat_path = "data/semcat/Categories"

embedding_params = \
    {
        "input_file": embedding_path,
        "dense_file": True,
        "lines_to_read": 50000,
        "mcrae_dir": None,
        "mcrae_words_only": False
    }

eval_params = \
    {
        "weights_dir": "out/",
        "save_weights": False,
        "load_weights": True
    }

model = Glove(embedding_params=embedding_params, semcat_dir=semcat_path, eval_params=eval_params,
              calculation_type="score", calculation_args=[])

# scores = []
#
# for i in range(10):
#     score = model.calculate_score(i+1)
#     scores.append(score)
#
# np.savetxt('out/scores.txt', X=np.array(scores))
