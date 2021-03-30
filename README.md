# Modelling Lexical Ambiguity with Density Matrices

Train density matrices with BERT2DM, Word2DM, and multi-sense Word2DM. Evaluate density matrices and composition methods on disambiguation tasks.

All the trained density matrices evaluated in this paper are publicly avalailable in two formats:
* [Text files](https://drive.google.com/file/d/1-S2zblENDlnQ4tYRqbjp7bQy56JkLm6g/view?usp=sharing) - each line consists of a word followed by it's 17x17 density matrix in row-major order.
* [Pickled classes](https://drive.google.com/drive/folders/1_k6cxUSgAvvfXWPpkPz4huBK5C146lcc?usp=sharing) - trained instances of the DensityMatrices class defined in density_matrices.py.

### Dependencies (Python) ###

* NumPy
* SciPy
* Pandas
* PyTorch
* Scikit-Learn
* NLTK
* Transformers
