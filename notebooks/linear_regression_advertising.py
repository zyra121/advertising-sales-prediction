import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# We have loaded the dataset, but we encounter an issue:
# the index column was saved as a regular column in the CSV file.
# To resolve this, we use the usecols parameter in the read_csv function to exclude the unwanted index column.

df = pd.read_csv('github_dataset/Advertising.csv', usecols=[1, 2, 3, 4])

# Let's explore the dataset.

df.head()

df.info()

df.describe().T

# Let's now take a look at how we created the visualizations.

################################################################
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="TV", y="sales", color="blue")
plt.title("Relationship between TV Advertising and Sales")
plt.xlabel("TV Advertising Budget (in thousands of dollars)")
plt.ylabel("Sales (in thousands of units)")
plt.show()
#################################################################
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="radio", y="sales", color="red")
plt.title("Relationship between radio Advertising and Sales")
plt.xlabel("radio Advertising Budget (in thousands of dollars)")
plt.ylabel("Sales (in thousands of units)")
plt.show()
#################################################################
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="newspaper", y="sales", color="purple")
plt.title("Relationship between newspaper Advertising and Sales")
plt.xlabel("Newspaper Advertising Budget (in thousands of dollars)")
plt.ylabel("Sales (in thousands of units)")
plt.savefig("images/scatterplot_newspaper.png")
plt.show()
#################################################################
sns.pairplot(df, diag_kind='kde')
plt.show()
#################################################################
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
#################################################################

# Simple Linear Regression with OLS Using Scikit-Learn
# This is a basic example of simple linear regression to illustrate how the model works.

X = df[['TV']]
y = df[['sales']]

model = LinearRegression().fit(X, y)

g = sns.regplot(x = X, y =y, scatter_kws = {'color' : 'b', 's' : 9},
                ci = False, color = 'r')
g.set_title(f"Model Equation: Sales = {round(model.intercept_[0], 2)} + TV*{round(model.coef_[0][0], 2)}")
g.set_ylabel("Number of Sales")
g.set_xlabel("TV expenses")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

#MSE
y_pred = model.predict(X)
mean_squared_error(y, y_pred)

#RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# R-Square
model.score(X,y)


# Multiple Linear Regression

X = df.drop('sales', axis = 1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Sales = {round(model.intercept_, 2)} + {round(model.coef_[0], 2)}*TV + {round(model.coef_[1], 2)}*radio + {round(model.coef_[2], 3)}*newspaper")


# Plotting actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Train RMSE
y_pred = model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN R-KARE

model.score(X_train, y_train)

# Test RMSE

y_pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test R-KARE
model.score(X_test, y_test)

# 10-fold Cross-Validation RMSE

np.mean(np.sqrt(-cross_val_score(model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring = "neg_mean_squared_error")))


# Simple Linear Regression with Gradient Descent from Scratch

# Cost Function MSE

def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) **2
    mse = sse/m
    return mse

# update weights

def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b  + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1/m * b_deriv_sum)
    new_w = w - (learning_rate * 1/m * w_deriv_sum)
    return new_b, new_w

# train function

def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(
        initial_b, initial_w, cost_function(Y, initial_b, initial_w, X)
    ))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4f}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(
        num_iters, b, w, cost_function(Y, b, w, X)
    ))

    return cost_history, b, w

X = df['radio']
Y = df['sales']

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)