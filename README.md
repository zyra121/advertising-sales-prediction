## Advertising Sales Prediction (Linear Regression Demo)

This project demonstrates a basic linear regression analysis using the Advertising dataset from the *An Introduction to Statistical Learning* book.  
It is a demo-level project designed to show how to perform simple and multiple linear regression in Python, visualize key relationships, and evaluate model performance.  
Additionally, it includes an implementation of Gradient Descent from scratch for educational purposes.

---

## 📂 Contents

- 📊 Exploratory Data Analysis  
  - Dataset overview  
  - Scatter plots & correlation matrix  
- 📈 Simple Linear Regression  
  - Model training with `scikit-learn`  
  - Regression line visualization  
- 🧮 Multiple Linear Regression  
  - Evaluation with metrics (MSE, RMSE, MAE, R²)  
  - 10-fold cross-validation  
- ⚙️ Gradient Descent (from scratch)  
  - Cost function  
  - Weight updates  
  - Learning curve visualization

---

## 🛠️ Tools & Libraries

- Python 3.x  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  

---

🚀 Getting Started


1. Clone the repository:

   ```bash
   git clone https://github.com/codelones/advertising-sales-prediction.git
   cd titanic-survival-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

 3. Run the project in PyCharm or any Python IDE.


## 📦 Dataset Source


[Advertising.csv (ISL book)](https://www.statlearning.com/resources-second-edition)


## 📊 Visualizations


![TV vs Sales](images/scatterplot_TV.png)

A scatter plot showing the relationship between TV advertising budget and sales. This visualization helps to analyze how TV advertising affects sales.


![Radio vs Sales](images/scatterplot_radio.png)

A scatter plot displaying the correlation between radio advertising budget and sales. Useful for evaluating the impact of radio ads on product sales.


![Newspaper vs Sales](images/scatterplot_newspaper.png)

This plot shows the relationship between newspaper advertising budget and sales. It helps assess how newspaper ads influence sales.


![Pairplot of Dataset](images/pairplot_df.png)

A pairplot illustrating the pairwise relationships between all features in the dataset. It provides a comprehensive view of the data distribution and feature interactions.


![Correlation Heatmap](images/correlation_heatmap.png)

A heatmap visualizing the correlation coefficients among all variables. It's essential for detecting multicollinearity before applying multiple linear regression.


![Model Equation](images/model_equation.png)

This plot visualizes the regression line for a simple linear regression model along with its equation. It demonstrates how the model predicts sales based on TV budget.


![Actual vs Predicted Sales](images/actual_vs_predict.png)

A scatter plot comparing actual vs predicted sales values. It visually shows the accuracy and performance of the regression model.
