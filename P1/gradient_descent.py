import numpy as np
import math
import random
import loadParametersP1 as loadParams
import loadFittingDataP1 as loadData

def batch_descent(initial_guess, step_size, threshold, gradient):
    '''
    initial_guess = vector
    step_size = number, the size of each step
    threshold = number
    gradient = function computing gradient
    '''
    w = [np.array(initial_guess)] # list of iteratively computed coordinates
    
    while True:
        w_old = w[-1]
        gradient_at_w_old = gradient(w_old)
        # stops when the norm of the gradient falls below threshold
        if np.linalg.norm(gradient_at_w_old) < threshold:
            break
        w_new = w_old - step_size * gradient_at_w_old
        w.append(w_new)
        
    return w[-1]

def stochastic_descent(initial_guess, step_size, threshold, gradient, dataX, datay):
    '''
    initial_guess is a vector
    gradient is a function computing gradient at single data point
    returns stochastic gradient descent minimum
    '''
    cur_var = initial_guess
    cur_var = initial_guess
    
    while True:
        total_data = len(datay)
        data_index = random.randint(0, total_data - 1)
        xi = dataX[data_index]
        yi = datay[data_index]
        gradient_at_cur_var = gradient(xi, yi, cur_var)
        # stops when the norm of the gradient falls below threshold
        if np.linalg.norm(gradient_at_cur_var) < threshold:
            break
        cur_var -= step_size * gradient_at_cur_var
        
    return cur_var

def gradient(f, vector):
    '''
    f is a function
    vector is point at which we want to calculate the gradient of f
    returns the gradient of f at vector
    '''
    h = 0.01
    dim = len(vector)
    

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
    inv_cov = np.linalg.inv(cov)
    displacement = vector - mean
    return - f * inv_cov.dot(displacement)

def gradient_quad_bowl(A, b, vector):
    return A.dot(vector) - b

def gradient_lse_data_point(xi, yi, cur_theta):
    '''
    lse term = (xi*theta - yi)^2
    takes single data point (xi, yi) 
    and returns gradient of lse at cur_theta
    '''
    return 2 * xi.dot(xi).dot(cur_theta) - 2 * xi.dot(yi)

# negative Gauss and quadratic bowl functions
gaussMean, gaussCov, quadBowlA, quadBowlb = loadParams.getData()
gaussMean = np.array(gaussMean)
gaussCov = np.array(gaussCov)
quadBowlA = np.array(quadBowlA)
quadBowlb = np.array(quadBowlb)
quadBowlStart = np.array([10, 11])
step_size = 0.01
threshold = 0.25

print("BATCH DESCENT MIN:", batch_descent(quadBowlStart, step_size, threshold, lambda v: gradient_quad_bowl(quadBowlA, quadBowlb, v)))
print("ACTUAL MIN:", np.linalg.inv(quadBowlA).dot(quadBowlb))

# least square fitting problem
X, y = loadData.getData()
print("DATA")
print(X)
print(y)


