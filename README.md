# reccomender-systems

The `sgd_mf.py` file implements a class that uses matrix factorization to predict results from a sparsely filled table of user-item interactions. A set amount of factors is chosen to represent each user and item vector, and a table composed of these vectors is made for both users and items. The values for each of the vectors are determined by taking the loss between the dot product of the user and item vectors and the populated values from the initial dataset, and performing sigmoid gradient descent to adjust the vectors.
