# Interpretability Of Word Embeddings

### **Requirements**

Tested with Python 3.7.4.<br>

Every dependency can be found in the [Requirements file](requirements.txt).

##### **Pip:**

`pip install -r requirements.txt`

### **Usage**

`
glove_cli.py <embedding_path> <semcat_dir>
             [-h] [-dense_file] [--lines_to_read LINES_TO_READ]
                  [--mcrae_dir MCRAE_DIR] [-mcrae_words_only]
                  [--weights_dir WEIGHTS_DIR] [-save_weights]
                  [-load_weights] [--calculate CALCULATE]
                  [--calculation_args [CALCULATION_ARGS [CALCULATION_ARGS ...]]]
`

##### **Required parameters**
- **embedding_path** - Path to the Glove embedding file (f.e. "glove/glove.6B.300d.txt")
- **semcat_dir** - Path to the SemCat categories directory (f.e. "semcat/Categories")
##### **Embedding related parameters**
- **dense_file** - If embedding_path points to a dense embedding file, mark it with this parameter
- **lines_to_read** - Maximum vectors to read. Default -1 (All vector)
- **mcrae_dir** - McRae directory
- **mcrae_words_only** - Use McRae words only
##### **Model related parameters**
- **weights_dir** - A path where the weights going to be saved to or read from (f.e. "weights/"), 
                    Default "out/"
- **save_weights** - Save weights to weights_dir
- **load_weights** - Load weights from weights_dir
##### **validation related parameters**
- **calculate** - Calculation method \[_score_|_decomp_\]
- **calculation_args** - List of arguments for calculation:
  - _score_: <br>
    - \[int \] <- Lamda value which > 0. Optional: Default 1.
  - _decomp_: <br>
    - \[str \] <- The word to decompose <br>
    - \[int \] <- The top X category. Optional: Default 20. <br>
    - \[bool \] <- Save result into file. Optional: Default False.

**Example:**<br>
`python interpret_cli.py "data/glove/glove.6B.300d.txt"
"data/semcat/Categories"
-dense_file
--lines_to_read=50000
-load_weights
--calculate=decomp
--calculation_args
barrel
10
True`

### **Related papers:** 

[Semantic Structure and Interpretability of Word Embeddings](https://arxiv.org/pdf/1711.00331.pdf)

### **Related links**

- [Glove](https://nlp.stanford.edu/projects/glove/)<br>
- [Semcat Dataset](https://github.com/avaapm/SEMCATdataset2018)
