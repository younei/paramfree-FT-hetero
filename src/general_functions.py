import numpy as np
import math

# Frobenius inner product 
def Frobenius(A, B):
    
    return np.trace(A.T @ B) 

# Projection operator 
def unit_ball_projection(x):

    r = 1

    # if x is vector 
    if x.ndim == 1:
        return x / np.maximum(r, np.linalg.norm(x, ord=2))
    elif x.ndim == 2:
    # if x is matrix 
        # Frobenius norm 
        return x / np.maximum(r, np.linalg.norm(x, ord='fro'))
    else:
        raise ValueError("Unknown Input Type.")


# loss function's subgradient
def subgradient(x, y, w, loss_name=None):

    # absolute loss subgradient
    if loss_name == 'absolute':
        pred = x @ w 
        if y < pred:
            return 1
        elif y >= pred:
            return - 1
    else:
        raise ValueError("Unknown loss.")


# loss function
def loss(x, y, w, loss_name=None):

    # absolute loss
    
    if loss_name == 'absolute':
        if hasattr(y, '__len__'):
            return 1 / len(y) * np.sum(np.abs(y - x @ w))
        else:
            return np.abs(y - x @ w)
    
    else:
        raise ValueError("Unknown loss.")


# feature map Phi(.) for the side information
# memo: what shape will be this x ?? CHECK 
def feature_map(x, y, feature_map_name=None, r=None, W=None):


    if feature_map_name == 'linear':
        x_transformed = np.mean(x, axis=0)
    
    elif feature_map_name == 'circle_feature_map':
        x_transformed = []
        x_transformed.append(math.cos(2 * x * math.pi))
        x_transformed.append(math.sin(2 * x * math.pi))
        x_transformed = np.asarray(x_transformed)
    
    elif feature_map_name == 'circle_fourier':
        x_transformed = np.cos(W * x + r)
        x_transformed = np.asarray(x_transformed)
        x_transformed = np.append(x_transformed, [1.])

    elif feature_map_name == 'linear_with_labels':
        if y is None:
            x_transformed = np.mean(x, axis=0)
            x_transformed = np.append(x_transformed, [1])
        else:
            n_points, n_dims = x.shape
            x_transformed = np.zeros(2 * n_dims)
            for idx_n in range(n_points):
                x_tmp = np.tensordot(x[idx_n, :], [y[idx_n], 1.], 0)
                x_transformed = x_transformed + np.ravel(x_tmp)
            x_transformed = x_transformed / n_points
            x_transformed = np.append(x_transformed, [1])
    
    elif feature_map_name == 'fourier_vector':
        n_points, n_dims = x.shape
        k = W.shape[0]
        x_transformed = (1 / n_points) * np.sqrt(2 / k) * np.sum(np.cos(W @ x.T + r), 1)
        x_transformed = np.append(x_transformed, np.mean(x, axis=0))
        x_transformed = np.append(x_transformed, [1.])
    else:
        raise ValueError("Unknown feature map.")

    return x_transformed


