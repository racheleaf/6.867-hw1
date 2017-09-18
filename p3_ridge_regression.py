import numpy as np
import loadFittingDataP2 as loadData
import matplotlib.pyplot as plt
import math
import p2_poly_fit as poly_fit
import regressData

def ridge_regression(X, Y, M, l):
    '''
    X is a list of x values
    Y is a list of y values
    M is the highest power of x in the basis
    l is value of lambda
    returns coefficients of deg M poly
    '''
    X_poly_basis = poly_fit.get_poly_basis(X, M)
    XTXlI_inv = np.linalg.inv(np.transpose(X_poly_basis).dot(X_poly_basis) + l * np.identity(M + 1))
    XTY = np.transpose(X_poly_basis).dot(Y)
    return XTXlI_inv.dot(XTY)

def get_sse_from_params(M, l, training_data, test_data):
    '''
    M, l parameters: highest power of x in basis, and lambda, respectively
    training_data, test_data are each tuples of x and y lists
    returns sse of polynomial model from parameters and training data 
    when tested on test_data
    '''
    #print(training_data[0], training_data[1])
    training_data_X = training_data[0]
    training_data_Y = training_data[1]
    poly = ridge_regression(training_data_X, training_data_Y, M, l)
    
    test_data_X = test_data[0]
    test_data_Y = test_data[1]
    eval_poly_at = lambda a: poly_fit.eval_poly(a, poly)
    return poly_fit.eval_sse(test_data_X, test_data_Y, eval_poly_at)
    
    
def plot_poly_fit(X, Y, M, l):
    '''
    X is list of x values
    Y is list of y values
    M is maximum degree of polynomial
    plots graph of degree M fit
    '''
    poly = ridge_regression(X, Y, M, l)
    f = lambda a: poly_fit.eval_poly(a, poly)
    poly_fit.plot(X, Y, f)
    
def plot_just_curve(x, f, xlabel = "x", ylabel = "y"):
    '''
    x is list of x coordinates we want to plot
    f is function
    plots points for each x coordinate and draws curve
    '''
    y = list(map(f, x))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def run_things(): 
    M = 4
    l = 0.1
    
    # ridge regression for lambda = 0.1
    X, Y = loadData.getData(False)
    #plot_poly_fit(X, Y, M, l)
    
    def parseData(data):
        return list(map(lambda x: np.transpose(x)[0], data))
    
    AData = parseData(regressData.regressAData())
    BData = parseData(regressData.regressBData())
    vData = parseData(regressData.validateData()) 
    sse = get_sse_from_params(M, l, AData, BData)
    
    # SSE vs. M
    plot_just_curve(range(11), lambda m: get_sse_from_params(m, l, AData, BData), "M", "SSE")
    # SSE vs. lambda
    plot_just_curve(np.linspace(0, 0.05, 1000), lambda ell: get_sse_from_params(M, ell, AData, BData), "lambda", "SSE")
    
    
    

run_things()
