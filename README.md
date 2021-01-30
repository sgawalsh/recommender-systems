# recommender-systems

The `sgd_mf.py` file implements a class that uses matrix factorization to predict results from a sparsely filled table of user-item interactions. A set amount of factors is chosen to represent each user and item vector, and a table composed of these vectors is made for both users and items. The values for each of the vectors are determined by taking the loss between the dot product of the user and item vectors and the populated values from the initial dataset, and performing sigmoid gradient descent to adjust the vectors.

The `nn.py` file implements a fully connected neural net using the PyTorch library. Embeddings are generated for each user and item entry, and used as inputs to the neural net. The net can then be trained on the available data.

In the `analysis.py` file some brief analysis of the results from the previous to files is performed. We can predict user and item similarity by calculating euclidean distance between two emebedding vectors. We can also check the results of one predicition method against another by seeing whether highly ranked predictions determined by one method remain highly ranked when using the other.
