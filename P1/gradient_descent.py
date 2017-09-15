import numpy as np
import math

def batch_descent(initial_guess, step_size, threshold, gradient):
    '''
    initial_guess = vector
    step_size = number, the size of each step
    threshold = number
    f = function
    gradient = function computing gradient
    '''
    w = [np.array(initial_guess)] # list of iteratively computed coordinates
    
    while True:
        w_old = w[-1]
        gradient_at_w_old = gradient(w_old)
        # stops when the norm of the gradient falls below threshold
        print(np.linalg.norm(gradient_at_w_old))
        if np.linalg.norm(gradient_at_w_old) < threshold:
            break
        w_new = w_old - step_size * gradient_at_w_old
        w.append(w_new)
        
    return w[-1]

def f_neg_gaussian(mean, cov, vector):
    '''
    mean is a vector at the center of the gaussian
    cov is the covariance matrix
    vector is the vector at which we wish to evaluate the gaussian
    returns the evaluation of the negative gaussian at vector
    '''
    dim = len(vector)
    constant = - 1 / math.sqrt((2 * math.pi) ** dim * abs(np.linalg.det(cov)))
    displacement = vector - mean
    inv_cov = np.linalg.inv(cov)
    power = - 1 / 2 * np.transpose(displacement).dot(inv_cov).dot(displacement)
    return math.exp(power) * constant
    
        
def gradient_neg_gaussian(mean, cov, vector):
    f = f_neg_gaussian(mean, cov, vector)
    print(f)
    inv_cov = np.linalg.inv(cov)
    displacement = vector - mean
    return - f * inv_cov.dot(displacement)

def gradient_quad_bowl(A, b, vector):
    return A.dot(vector) - b

A = np.array([[10, 5], [5, 10]])
b = np.transpose(np.array([400, 400]))   
vector = np.transpose(np.array([2, 2]))
mean = np.transpose(np.array([1, 1]))
cov = np.array([[3, 0], [0, 3]])
print(batch_descent(np.transpose(np.array([10, 11])), .25, 0.001, lambda v: gradient_quad_bowl(A, b, v)))



