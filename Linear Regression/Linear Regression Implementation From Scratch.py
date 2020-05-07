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
#print('features:', features[0:10], '\nlabel:', labels[0:10])  # Printing first 10 records


'''
Reading the above generated dataset .
The following function would read data in size of 10 and returns
'''
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))

    #The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

for X,y in data_iter(batch_size,features,labels):
    #print(X,'\n',y)
    break


'''
Initializing Model Parameters
'''
w = np.random.normal(0,0.01,(2,1)) #Randomly initializing weights by sampling random numbers from normal distribution with mean 0 and deviation 0.01.
b = np.zeros(1)                    #Here, bias (intercept) is set to zero.

'''
After initializing parameters, we need to update them until they fit the data well. Each update requires taking the 
gradient of the loss function with respect to parameters. Given this gradient, we can update each parameter in the 
direction that reduces loss.
'''
w.attach_grad()
b.attach_grad()


'''
Defining the Model
'''
def linreg(X,w,b):
    return np.dot(X,w)+b

'''
Defining the Loss Function
'''
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

'''
Defining the Optimization Algorithm
'''
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


lr = 0.03 #Learning Rate
num_epochs = 3 #Number of iterations over the dataset
net = linreg
loss = squared_loss # 0.5 (y-y`)^2

for epoch in range(num_epochs):
    #Assuming the number of examples can be divided by the batch size, all the examples in the training dataset are used
    #once in one epoch iteration.

    for X,y in data_iter(batch_size,features,labels):
        with autograd.record():
            l = loss(net(X,w,b),y)
        l.backward()  #Compute gradient on l with respect to [w,b]
        sgd([w,b], lr, batch_size) #update parameters using their gradient
    train_l = loss(net(features,w,b), labels)
    print('epoch %d, loss %f' % (epoch +1, train_l.mean().asnumpy()))


'''
As we synthesized the data ourselves, we know precisely what true parameters are. Thus, we can evaluate the success in 
training by comparing the true parameters with those that we learned through the training loop.
'''
print('Error in estimating w', true_w-w.reshape(true_w.shape))
print('Error in estimating b', true_b-b)