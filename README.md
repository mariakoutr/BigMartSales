# Big Mart Sales Prediction 

## Introduction 

This project made using R language. It includes one .R code file, one report .Rmd file made using Rmarkdown and one HTML file, which was 
produced by the .Rmd report using knitr. 

### Problem Statement
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities.
Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of 
each product at a particular store.

Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.


### Data sets

We have train (8523) and test (5681) data set, train data set has both input and output variable(s). 
The goal is to predict the sales for test data set.

### Summary 

After we loaded the data and the required libaries, we explored the data and its variables and made some observations. 
Afterwards, we imputed the missing values, and we continued with the feature engineering process. We have built a baseline model and five
different other models to fit our data: Linear Regression, Ridge and Lasso Regression, Decision Tree, Random Forest and EXtreme Gradient
Boost. 
We divided the training set to secondary training and test set, to evaluate the Root Mean Squared Error (RMSE) between the prediction
and the actual values on them before training the whole training set and predict values for the actual given test set. The Root Mean
Squared Error (RMSE) was used as the evaluation metric in this analysis. We concluded that the model built using the XGBoost algorithm
was the most efficient (lowest RMSE), followed by the Random Forest model. At last, we trained the whole training data set using these
two models and we predicted the values of Item_Outlet_Sales, which was the target variable, of the given test set. 

