# Interpretability Of Word Embeddings

### **Usage**

`
python glove_cli.py <embedding_path> <semcat_dir> [-h] [--lambda_value] [--weights_dir ] [--save]
`

embedding_path - Path to the Glove embedding file (f.e. "glove/glove.6B.300d.txt")<br>
semcat_dir - Path to the SemCat categories directory (f.e. "semcat/Categories")<br>
lambda_value - Lambda value is used to relax interpretability score<br>
weights_dir - A path where the weights going to saved to or read from (f.e. "weights/")<br>
save - A flag whether you want to save the weights or not<br>

### **Related papers:** 

[Semantic Structure and Interpretability of Word Embeddings](https://arxiv.org/pdf/1711.00331.pdf)
