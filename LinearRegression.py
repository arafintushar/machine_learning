import numpy as np

#Linear Regression
class LinearRegression:
    def __init__(self, lr = 0.001, iteration =  1000):
        self.learning_rate = lr
        self.iteration = iteration
    
    #function for model training
    def fit(self, X, y):
        #no_of_training_exp, no_of_features
        self.m, self.n = X.shape

        self.W = np.zeros(self.n)
        self.b = 0

        self.X = X
        self.y = y

        #gradient decent
        for i in range(self.iteration):
            y_pred = self.predict(self.X)

            #calculating gradient
            dw = (2 * (self.X.T).dot(y_pred - self.y)) / self.m
            db = 2 * np.sum(y_pred - self.y) / self.m

            #update weights and bias
            self.W = self.W - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            # score = self.score(self.y, y_pred)

            # print("iteration = {} score = {}\n".format(i+1, score))
        
        return self
    
    #hypothetical function f(x)
    def predict(self, X):
        return X.dot(self.W) + self.b
    
    #score
    def score(self, y_test, y_pred):
        return np.mean((y_pred - y_test)**2)
    
