# DLNLP-CBOW
The goal is to implement the Conitnuous Bag of Words (CBOW) model in PyTorch and use it on a sentiment classification task.

# Getting started
Load data using [loader.py](loader.py) and [main.py](main.py).

## Running the tests
### Training Part
Create Data Loader using PyTorch, which includes dataset preprocessing and cleaning. Then, implment a training loop to try different hyper-parameters.
### CBOW Part
Implement Continuous Bag of Words model using [CBOW.py](CBOW.py). In the nn.Embedding(vocab size, embedding dim), vocab size is the dimension of the vocabulary and embedding dim is the embeddings dimension. The get word embedding takes a word and convert it to a vector. You need also to define the loss function and the optimizer (e.g. nn.torch.optim. etc.).
### Training CBOW
Using the DataLoader created in the Training part, implement the training loop. 
### Evaluation
Word similarity gets two word as input and the trained model, and it return the cosine similarity of the two respective vectors.
The relevant files are as follows: [loader_eval.py](loader_eval.py) and [eval.py](eval.py).
## Built With
* [PyTorch](-) - 
* [Numpy](-) - 
* [Tensorflow](-) -
