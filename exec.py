from Eval.glove import Glove

embedding_path = "data/glove/glove.6B.300d.txt"
semcat_path = "data/semcat/Categories"

model = Glove(embedding_path, semcat_path)

scores = []

for i in range(10):
    score = model.calculate_score(i+1)
    scores.append(score)

with open("out/scores.txt", mode="w") as f:
    f.writelines(scores)
