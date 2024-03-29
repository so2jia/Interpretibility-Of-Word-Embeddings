# Plotting utility scripts

## **Categorical Decomposition**


`
python categorical_decomposition.py <input_file> <dimension> [-h] [--output_file OUTPUT_FILE]
`

- **input_file** - Bhattacharya distance matrix (w_b.npy)
- **dimension** - The dimension to decompose
- **output_file** - Output PNG file (Optional)

**Output:**<br><br>
![Categorical Decomposition](../../images/plots/glove_dense/categorical_decomposition_dim2.png)

## **Dimensional Decomposition**

`
python dimensional_decomposition.py <semcat_dir> <input_file> <category> [-h] [--output_file OUTPUT_FILE]
`

- **semcat_dir** - Path to the SemCat categories directory (f.e. "semcat/Categories")
- **input_file** - Bhattacharya distance matrix (w_b.npy)
- **category** - The Category to decompose (f.e. 'math')
- **output_file** - Output PNG file (Optional)

**Output:**<br><br>
![Categorical Decomposition](../../images/plots/glove_dense/dimensional_decomposition_cat-math.png)

## **Interpretability Score**

`
python interpretability_score.py <input_file> [-h] [--output_file OUTPUT_FILE] 
`

- **input_file** - Input file which contains the list of the scores
- **output_file** - Output PNG file (Optional)

**Output:**<br><br>
![Categorical Decomposition](../../images/plots/glove_dense/interpretability_score.png)

## **Representation Strength**

`
python representation_strength.py <input_file> <semcat_dir> [-h] [--output_file OUTPUT_FILE]
`

- **input_file** - Bhattacharya distance matrix (w_b.npy)
- **semcat_dir** - Path to the SemCat categories directory (f.e. "semcat/Categories")
- **output_file** - Output PNG file (Optional)

**Output:**<br><br>
![Categorical Decomposition](../../images/plots/glove_dense/representation_strength.png)

## **Semantical Decomposition**

`
python semantic_decomposition.py <semcat_dir> <input_file> [-h] [--output_file OUTPUT_FILE]
`

- **semcat_dir** - Path to the SemCat categories directory (f.e. "semcat/Categories")
- **input_file** - Embedding-Category matrix (I.npy)
- **output_file** - Output PNG file (Optional)

**Output:**<br><br>
![Categorical Decomposition](../../images/plots/glove_dense/semantical_decomposition_window.png)

## **Dimensional interpretability according to KS-test value**

`
python interpretability_correlation.py <embedding_s> <bhatta> <test_type>  
[-h] [--lines_to_read LINES_TO_READ] [--output_file OUTPUT_FILE]
`

- **embedding** - Embedding file
- **embedding_s** - Standardised embedding vectors (.npy)
- **bhatta** - Bhattacharya matrix (.npy)
- **semcat_dir** - SemCat Categories directory
- **test_type** - Normalitytest types: 
[[ks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html), 
[normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html), 
[shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)]
- **norm** - L2 Normalize embedding space
- **dense** - Dense embedding
- **lines_to_read** - Maximum lines to read from embedding
- **output_file** - Output PNG file (Optional)

**Output:**<br><br>
![Categorical Decomposition](../../images/plots/glove_dense/ks_test.png)