# Pour les noms des variables, nous conservons les memes notations que l'enonce

# Global parameters
train_set_size = 0.99

# Computes the Root Mean Square Error of the predicted y
compute_rmse <- function(y, y_est){
  return(sum((y-y_est)^2)/length(y))
}

# Question 2 : Multinlinear regression least squares estimator

# Estimate least sqaure OLS regression estimate of beta
estimate_beta <- function(X, y){
  M = t(X)%*%X
  M_inv = solve(M)
  beta = M_inv%*%t(X)%*%y
  return(beta)
}

# Returns the OLS regression prediction of y
predict_y <- function(X, beta){
  return(X%*%beta)
}

# Importating the data
ols_data_set = data.matrix(read.table("mysmalldata.txt", sep=','))

# Computing the train-test slicing coordinates
train_row_limit = train_set_size * nrow(ols_data_set)
test_row_start = train_row_limit + 1

# Splitting the data set into a train set and test set
X_train = ols_data_set[1:train_row_limit, 2:ncol(ols_data_set)]
y_train = ols_data_set[1:train_row_limit, 1]
X_test = ols_data_set[test_row_start:nrow(ols_data_set), 2:ncol(ols_data_set)]
y_test = ols_data_set[test_row_start:nrow(ols_data_set), 1]

# Estimating the least square OLS regression estimate of beta using the train set
beta_hat = estimate_beta(X=X_train, y=y_train)

# Predicting and computing the error for the train set and the test set
y_est_train = predict_y(X=X_train, beta=beta_hat)
rmse_lse_train = compute_rmse(y=y_train, y_est=y_est_train)

y_est_test = predict_y(X=X_test, beta=beta_hat)
rmse_lse_test = compute_rmse(y=y_test, y_est=y_est_test)

# Question 7 : Estimateur Lasso

# Package
install.packages("glmnet")
library("glmnet")

# Parameters
lambda = 0.1

# Importing the dataset
lasso_data_set = data.matrix(read.table("mydata.txt", sep=','))
X = lasso_data_set[, 2:ncol(lasso_data_set)]
y = lasso_data_set[, 1]

# Estimating the lasso regression parameters using the glmnet package
# The parameter alpha is the weight between Lasso and Ridge regression, we set it to one for Lasso regression
fit = glmnet(x=X, y=y, alpha = 1, lambda = lambda)

# Predicting y using the lasso estimated parameters
y_est_lasso = predict(fit, newx = X, type = "response", s=lambda)

# Computing the score for the lasso regression
rmse_lasso = compute_rmse(y_est_lasso, y)


# Question 10 : Cross Validation

# Parameters
lambda = 0.1
train_size = 0.8
cross_val_size = 0.1
test_size = 0.1

# 