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

# Displaying the first five rows of the dataset for initial inspection.
df.head()

# Let's examine the structure of the dataset and the data types of the variables.
df.info()

#Let's examine the descriptive statistics for the numerical variables.
# Transposing the output for a more readable view
df.describe().T

# Let's now take a look at how we created the visualizations.

###############################################################################################
###############################################################################################
###############################################################################################
# This scatter plot visualizes the relationship between TV advertising budget and product sales.
# Each point represents a single observation from the dataset.
# This plot helps identify whether a linear relationship exists between the two variables.
sns.scatterplot(data=df, x="TV", y="sales", color="blue")
plt.title("Relationship between TV Advertising and Sales",
          fontweight='bold',
          color='darkred',
          fontsize=15)
plt.xlabel("TV Advertising Budget (in thousands of dollars)", fontsize=10, fontweight='bold')
plt.ylabel("Sales (in thousands of units)", fontsize=10, fontweight='bold')
plt.xlim(-20, 320)
plt.ylim(bottom=0, top = 30)
sns.set_style('darkgrid')
plt.show()
###############################################################################################
###############################################################################################
###############################################################################################
# This scatter plot illustrates the relationship between radio advertising budget and product sales.
# Each red point represents a data observation showing how much was spent on radio ads and the resulting sales.
# The goal is to observe whether a pattern or correlation exists between radio spending and sales volume.
# This visualization helps in evaluating the potential impact of radio advertisements on sales.
sns.scatterplot(data=df, x="radio", y="sales", color="red")
plt.title("Relationship between Radio Advertising and Sales",
          fontweight='bold',
          color='darkred',
          fontsize=15)
plt.xlabel("Radio Advertising Budget (in thousands of dollars)", fontsize=10, fontweight='bold')
plt.ylabel("Sales (in thousands of units)", fontsize=10, fontweight='bold')
sns.set_style('darkgrid')
plt.xlim(-5, 55)
plt.ylim(top=30)
plt.show()
###############################################################################################
###############################################################################################
###############################################################################################
# This scatter plot displays the relationship between newspaper advertising budget and product sales.
# Each purple point represents a data observation showing how newspaper spending relates to sales performance.
# By visualizing this relationship, we can evaluate whether newspaper advertising has a strong impact on sales.
# The plot helps identify trends or a lack of correlation between these two variables.
sns.scatterplot(data=df, x="newspaper", y="sales", color="purple")
plt.title("Relationship between Newspaper Advertising and Sales",
          fontweight='bold',
          color='darkred',
          fontsize=15)
plt.xlabel("Newspaper Advertising Budget (in thousands of dollars)", fontsize=10, fontweight='bold')
plt.ylabel("Sales (in thousands of units)", fontsize=10, fontweight='bold')
sns.set_style('darkgrid')
plt.xlim(-5, 95)
plt.ylim(top=30)
plt.show()
###############################################################################################
###############################################################################################
###############################################################################################
tv_corr = df['TV'].corr(df['sales'])
radio_corr = df['radio'].corr(df['sales'])
news_corr = df['newspaper'].corr(df['sales'])
df['sales_category'] = pd.cut(df['sales'], bins=3, labels=['low', 'mid', 'high'])
###############################################################################################
# This pairplot provides a comprehensive visualization of the relationships between all numerical features in the dataset.
# The plot includes scatter plots for each variable pair and kernel density estimates (KDE) on the diagonals.
# Data points are color-coded based on 'sales_category', which segments sales into three bins: low, mid, and high.
# Additionally, correlation values between TV, radio, and newspaper advertising budgets and sales are annotated on the figure.
# This visualization is useful for identifying patterns, trends, and the strength of linear relationships between variables.
g = sns.pairplot(df, diag_kind='kde',
                 corner=True,
                 plot_kws={'alpha': 0.9, 's': 60, 'edgecolor': 'k'},
                 hue='sales_category',
                 height=2.6 ,
                 aspect=1.1)
plt.suptitle("Advertising Dataset - Pairplot", fontsize=25, fontweight='bold', color='darkred')
g.figure.text(0.86, 0.61, f"TV-Sales Corr: {tv_corr:.2f}", fontsize=10, color='black', fontweight='bold')
g.figure.text(0.86, 0.59, f"Radio-Sales Corr: {radio_corr:.2f}", fontsize=10, color='black', fontweight='bold')
g.figure.text(0.86, 0.57, f"News-Sales Corr: {news_corr:.2f}", fontsize=10, color='black', fontweight='bold')
plt.show()
###############################################################################################
###############################################################################################
###############################################################################################
# This heatmap visualizes the correlation matrix for the numerical variables in the dataset.
# Each cell shows the Pearson correlation coefficient between a pair of variables.
# Color gradients (from blue to red) represent the strength and direction of the correlation:
# - Red indicates a strong positive correlation.
# - Blue indicates a strong negative correlation.
# - Values near zero indicate weak or no correlation.
# This visualization helps identify multicollinearity and relationships useful for predictive modeling.
sns.heatmap(df.corr(numeric_only=True),
            cmap="coolwarm",
            annot=True,
            linewidths=0.7,
            fmt=".2f",
            square=True,
            cbar_kws={'shrink' : 0.75},annot_kws={'fontsize':14})
plt.title("Correlation Heatmap", fontweight='bold', fontsize=16, color='darkred', pad=15)
plt.tight_layout()
plt.xticks(fontweight='bold', fontsize=12)
plt.yticks(fontweight='bold', fontsize=12)
plt.show()
###############################################################################################
###############################################################################################
###############################################################################################
# Simple Linear Regression with OLS Using Scikit-Learn
# This is a basic example of simple linear regression to illustrate how the model works.

X = df[['TV']]
y = df[['sales']]

simple_model = LinearRegression().fit(X, y)
y_pred = simple_model.predict(X)
###############################################################################################
###############################################################################################
###############################################################################################
#MSE (Mean Squared Error)
#It is the average of the squared differences between the predicted and actual values.
mean_squared_error(y, y_pred)
###############################################################################################
#RMSE (Root Mean Square Error)
np.sqrt(mean_squared_error(y, y_pred))
# It measures the square root of the average of squared differences between actual and predicted values. It penalizes large errors more than small ones.
###############################################################################################
# MAE (Mean Absolute Error)
mean_absolute_error(y, y_pred)
# It represents the average of the absolute differences between predicted and actual values. Less sensitive to outliers compared to MSE/RMSE.
###############################################################################################
# 𝑅2 (Coefficient of Determination (R-squared))
# It indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. Values range from 0 to 1.
simple_model.score(X,y)
###############################################################################################
###############################################################################################
###############################################################################################
# NOTE! : It is not appropriate to compare MSE, MAE, and RMSE against each other directly, as they are based on different error formulations.
# A single evaluation metric should be chosen beforehand, and all subsequent comparisons—such as between
# different models or after applying feature engineering techniques—should be made using that same metric.
###############################################################################################
###############################################################################################
###############################################################################################
#The code generates a regression plot using Seaborn to visualize
# the relationship between TV advertising expenses and sales figures based on a simple linear regression model.
# The scatter points are drawn in blue, while the regression line is displayed in dark red.
# The plot includes the regression equation and several key performance metrics such as
#𝑅2, MSE, RMSE, and MAE, positioned in the top-right corner for reference.
###############################################################################################
###############################################################################################
# This regression plot visualizes the linear relationship between TV advertising expenses and sales.
# The blue scatter points represent the actual data observations.
# The red regression line represents the best-fitting line estimated by the simple linear regression model.
# Key performance metrics such as R², MSE, RMSE, and MAE are annotated in the top-right corner of the plot.
# This plot provides both a visual and quantitative evaluation of model accuracy and fit quality.
g = sns.regplot(data = df, x = X, y= y, scatter_kws = {'color' : 'b', 's' : 9, 'alpha' : 1},
                ci = False, color = 'r',
                line_kws={'color': 'darkred', 'linewidth': 2})
g.set_title(f'Model Equation : {round(simple_model.intercept_[0], 2)} + TV*{round(simple_model.coef_[0][0], 2)}',
            fontweight = 'bold',
            color='darkred',
            fontsize=16)
g.set_xlabel("TV Expenses", fontsize=12, fontweight='bold')
g.set_ylabel('Number of Sales', fontsize=12, fontweight='bold')
sns.set_style('darkgrid')
g.set_xlim(-10, 310)
g.set_ylim(bottom=0)
g.text(313, 20, f'$R^2$: {simple_model.score(X, y):.2f}', fontsize=8, color='black')
g.text(313, 18, f'MSE: {mean_squared_error(y, y_pred):.2f}', fontsize=8, color='black')
g.text(313, 16, f'RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}', fontsize=8, color='black')
g.text(313, 14, f'MAE: {mean_absolute_error(y, y_pred):.2f}', fontsize=8, color='black')
plt.tight_layout()
plt.show()
###############################################################################################
###############################################################################################
###############################################################################################
# Multiple Linear Regression

X = df.drop(['sales', 'sales_category'], axis = 1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)
y_pred = multiple_model.predict(X_test)

print(f"Sales = {round(multiple_model.intercept_, 2)} "
      f"+ {round(multiple_model.coef_[0], 2)}*TV "
      f"+ {round(multiple_model.coef_[1], 2)}*radio "
      f"+ {round(multiple_model.coef_[2], 3)}*newspaper")
###############################################################################################
###############################################################################################
###############################################################################################
# This scatter plot compares the actual sales values (y_test) with the predicted values (y_pred) from the multiple linear regression model.
# Each point represents a prediction for a given test sample.
# The red dashed diagonal line (y = x) represents perfect prediction.
# Points close to this line indicate accurate predictions, while points farther away show prediction errors.
# This visualization helps assess the overall performance and reliability of the regression model on unseen test data.
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', )
plt.xlabel("Actual Sales", fontsize=12, fontweight='bold')
plt.ylabel("Predicted Sales", fontsize=12, fontweight='bold')
plt.title("Actual vs Predicted Sales",  fontsize=18, fontweight='bold', color = 'darkred')
plt.tight_layout()
plt.show()
###############################################################################################
###############################################################################################
###############################################################################################
# Train RMSE
y_pred = multiple_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
###############################################################################################
# TRAIN R-KARE
multiple_model.score(X_train, y_train)
###############################################################################################
# Test RMSE
y_pred = multiple_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
###############################################################################################
# Test R-KARE
multiple_model.score(X_test, y_test)
###############################################################################################
###############################################################################################
###############################################################################################
# 10-fold Cross-Validation RMSE

np.mean(np.sqrt(-cross_val_score(multiple_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring = "neg_mean_squared_error")))
###############################################################################################
###############################################################################################
###############################################################################################

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
###############################################################################################
###############################################################################################
###############################################################################################
#A line plot visualizing the Mean Squared Error (MSE) over each iteration during the gradient descent optimization process.
#It demonstrates how the model gradually minimizes the error, indicating convergence towards an optimal solution.
plt.plot(range(num_iters), cost_history)
plt.title("Gradient Descent - Cost Function over Iterations", fontsize=14, fontweight='bold', color = 'darkred', y=1.05)
plt.xlabel("Iteration", fontsize=12, fontweight='bold')
plt.ylabel("MSE", fontsize=12, fontweight='bold')
plt.grid(True)
plt.show()
###############################################################################################
###############################################################################################
###############################################################################################
#A comparison between the actual sales values and the predicted values generated by the linear regression model trained using gradient descent.
#The scatter points represent real data, while the dark red line shows the predicted trend, allowing visual evaluation of the model's fit on the radio feature.
plt.scatter(X, Y, label="Actual", alpha=0.7)
plt.plot(X, b + w * X, color="darkred", label="Predicted", linewidth=2)
plt.title("Gradient Descent Result - Radio vs Sales", fontsize=14, fontweight='bold', color = 'darkred', y=1.05)
plt.xlabel("Radio Budget", fontsize=12, fontweight='bold')
plt.ylabel("Sales", fontsize=12, fontweight='bold')
plt.legend()
plt.show()
###############################################################################################
###############################################################################################
###############################################################################################