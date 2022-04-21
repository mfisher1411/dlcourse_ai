import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    orig_predictions = predictions.copy()
    
    if orig_predictions.ndim == 1:
        max_vals = np.max(orig_predictions)
        norm = orig_predictions - max_vals
        exps = np.exp(norm)
        sums = np.sum(exps)
    else:
        max_vals = np.max(orig_predictions, axis=-1)
        max_vals = max_vals[:,np.newaxis]
        norm = orig_predictions - max_vals
        exps = np.exp(norm)
        sums_axis = np.sum(exps, axis=-1)
        sums = np.repeat(sums_axis[:,np.newaxis],orig_predictions.shape[-1],axis=-1)
    
    return exps/sums
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
      
    '''
    if probs.ndim > 1:
        t_index_1d = target_index.reshape(-1)
        flat_index_array = np.ravel_multi_index(
            np.array([np.arange(t_index_1d.shape[0]),t_index_1d], dtype=np.int),
            probs.shape)
        loss_arr = -np.log(np.ravel(probs)[flat_index_array])
        return np.mean(loss_arr)
    return -np.log(probs[target_index])
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    '''
    orig_predictions = predictions.copy()
    probs = softmax(orig_predictions)
    loss = cross_entropy_loss(probs,target_index)
    #print(predictions.shape)
    #print(predictions)
    #orig_predictions_2 = predictions.copy()
    dprediction = probs + loss
    #print(dprediction.shape)
    #print(dprediction)
    '''
    probs = softmax(predictions)
    loss_val = cross_entropy_loss(probs, target_index)

    ground_trues = np.zeros(predictions.shape, dtype=np.float32)
    if predictions.ndim > 1:
        t_index_1d = target_index.reshape(-1)
        flat_index_array = np.ravel_multi_index(
            np.array([np.arange(t_index_1d.shape[0]),t_index_1d], dtype=np.int),
            ground_trues.shape)
        np.ravel(ground_trues)[flat_index_array] = 1.0
    else:
        ground_trues[target_index] = 1.0

    dprediction = probs - ground_trues
    if predictions.ndim > 1:
        dprediction /= dprediction.shape[0]
    loss = loss_val if predictions.ndim == 1 else np.mean(loss_val)

    return loss, dprediction
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    


# def l2_regularization(W, reg_strength):
#     '''
#     Computes L2 regularization loss on weights and its gradient

#     Arguments:
#       W, np array - weights
#       reg_strength - float value

#     Returns:
#       loss, single value - l2 regularization loss
#       gradient, np.array same shape as W - gradient of weight by l2 loss
#     '''

#     # TODO: implement l2 regularization and gradient
#     # Your final implementation shouldn't have any loops
#     raise Exception("Not implemented!")

#     return loss, grad
    

# def linear_softmax(X, W, target_index):
#     '''
#     Performs linear classification and returns loss and gradient over W

#     Arguments:
#       X, np array, shape (num_batch, num_features) - batch of images
#       W, np array, shape (num_features, classes) - weights
#       target_index, np array, shape (num_batch) - index of target classes

#     Returns:
#       loss, single value - cross-entropy loss
#       gradient, np.array same shape as W - gradient of weight by loss

#     '''
#     predictions = np.dot(X, W)

#     # TODO implement prediction and gradient over W
#     # Your final implementation shouldn't have any loops
#     raise Exception("Not implemented!")
    
#     return loss, dW


# class LinearSoftmaxClassifier():
#     def __init__(self):
#         self.W = None

#     def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
#             epochs=1):
#         '''
#         Trains linear classifier
        
#         Arguments:
#           X, np array (num_samples, num_features) - training data
#           y, np array of int (num_samples) - labels
#           batch_size, int - batch size to use
#           learning_rate, float - learning rate for gradient descent
#           reg, float - L2 regularization strength
#           epochs, int - number of epochs
#         '''

#         num_train = X.shape[0]
#         num_features = X.shape[1]
#         num_classes = np.max(y)+1
#         if self.W is None:
#             self.W = 0.001 * np.random.randn(num_features, num_classes)

#         loss_history = []
#         for epoch in range(epochs):
#             shuffled_indices = np.arange(num_train)
#             np.random.shuffle(shuffled_indices)
#             sections = np.arange(batch_size, num_train, batch_size)
#             batches_indices = np.array_split(shuffled_indices, sections)

#             # TODO implement generating batches from indices
#             # Compute loss and gradients
#             # Apply gradient to weights using learning rate
#             # Don't forget to add both cross-entropy loss
#             # and regularization!
#             raise Exception("Not implemented!")

#             # end
#             print("Epoch %i, loss: %f" % (epoch, loss))

#         return loss_history

#     def predict(self, X):
#         '''
#         Produces classifier predictions on the set
       
#         Arguments:
#           X, np array (test_samples, num_features)

#         Returns:
#           y_pred, np.array of int (test_samples)
#         '''
#         y_pred = np.zeros(X.shape[0], dtype=np.int)

#         # TODO Implement class prediction
#         # Your final implementation shouldn't have any loops
#         raise Exception("Not implemented!")

#         return y_pred



                
                                                          

            

                
