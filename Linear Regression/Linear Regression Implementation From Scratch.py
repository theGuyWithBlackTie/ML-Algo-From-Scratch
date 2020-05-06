import random
from mxnet import autograd, np, npx

npx.set_np()

# Generating dataset
'''
Here we are creating the dataset for training and testing synthetically.
This dataset would contain only two features and a corresponding 'y' value. Generally real world dataset contains noise 
too. Noise is also added in fourth line of the function.
'''


def synthetic_data(w, b, num_examples):
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)  # Adding noise :)
    return X, y


true_w = np.array([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)  # Generating 1000 records for dataset.
print('features:', features[0:10], '\nlabel:', labels[0:10])  # Printing first 10 records


'''
Reading the dataset .
'''