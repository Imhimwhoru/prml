import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionWithSlope():
    def __init__(self, learning_rate=0.01, no_of_itr=1000):
        self.learning_rate = learning_rate
        self.no_of_itr = no_of_itr

    def fit(self, X, Y):
        self.m = 0  # Slope
        self.c = 0  # Intercept
        self.X = X.flatten()  # Flatten X to a 1D array
        self.Y = Y.flatten()  # Flatten Y to a 1D array
        self.n = len(self.Y)  # Number of data points

        # Gradient descent
        for _ in range(self.no_of_itr):
            self.update_weights()

    def update_weights(self):
        # Predicted values for m and c
        Y_prediction = self.m * self.X + self.c

        # Compute gradients for m and c
        dm = -(2 / self.n) * np.sum(self.X * (self.Y - Y_prediction))
        dc = -(2 / self.n) * np.sum(self.Y - Y_prediction)

        # Update m,c
        self.m -= self.learning_rate * dm
        self.c -= self.learning_rate * dc

    def predict(self, X):
        # Predict values using m and c
        return self.m * X + self.c

    def print_weights(self):
        print("Slope (m):", self.m)
        print("Intercept (c):", self.c)


if __name__  == "__main__":
    
    df = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/refs/heads/master/datasets/housing/housing.csv')
    X = df[['median_income']].values
    y = df[['median_house_value']].values

    # train test split ratio 
    split_ratio = 0.8
    split_index = int(split_ratio * len(df))

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    # Train the model
    learning_rate = 0.01
    iterations = 1000
    model = LinearRegressionWithSlope(learning_rate, iterations)
    model.fit(X_train, y_train)

    model.print_weights()

    # Predict on test data
    y_pred = model.predict(X_test.flatten())

    # Evaluation of model
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    mse = mean_squared_error(y_test.flatten(), y_pred)
    r2 = r2_score(y_test.flatten(), y_pred)

    print("Mean Squared Error:", mse)
    print("R-Squared:", r2)

    # Visualize the results
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', label='Predicted')
    plt.title("Linear Regression Fit (y = mx + c)")
    plt.xlabel("Median Income")
    plt.ylabel("Median House Value")
    plt.legend()
    plt.show()


"""
# on the basis of weight calc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Linear_Regression(): 
    def __init__(self, learning_rate=0.01, no_of_itr=1000): 
        self.learning_rate = learning_rate 
        self.no_of_itr = no_of_itr 

    def fit(self, X, Y): 
        self.m, self.n = X.shape  
        self.w = np.zeros((self.n, 1))  
        self.b = 0                    
        self.X = X 
        self.Y = Y 

        for i in range(self.no_of_itr): 
            self.update_weights() 

    def update_weights(self): 
        Y_prediction = self.predict(self.X) 
        dw = -(self.X.T).dot(self.Y - Y_prediction) / self.m 
        db = -np.sum(self.Y - Y_prediction) / self.m 
        self.w = self.w - self.learning_rate * dw 
        self.b = self.b - self.learning_rate * db 

    def predict(self, X): 
        return X.dot(self.w) + self.b 

    def print_weights(self): 
        print('Weights for the respective features are :') 
        print(self.w) 
        print('Bias value for the regression is:', self.b)

df = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/refs/heads/master/datasets/housing/housing.csv')  
X = df[['median_income']].values  
y = df[['median_house_value']].values   

split_ratio = 0.8
split_index = int(split_ratio * len(df))

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]

learning_rate = 0.01
iterations = 1000
model = Linear_Regression(learning_rate, iterations)
model.fit(X_train, y_train)
model.print_weights()

y_pred = model.predict(X_test)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-Squared:", r2)

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title("Linear Regression Fit")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.legend()
plt.show()
"""