# Pour les noms des variables, nous conservons les memes notations que l'enonce

# Global parameters
train_set_size = 0.90

# Computes the Root Mean Square Error of the predicted y
compute_rmse <- function(y, y_est){
  return(sum((y-y_est)^2)/length(y))
}

count_zeros <- function(x){
  return(sum(x == 0) )
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

nb_zeros_ols = count_zeros(beta_hat)

# Question 7 : Estimateur Lasso

# Package
install.packages("glmnet")
library("glmnet")

# Parameters
lambda = c(1)

# Importing the dataset
lasso_data_set = data.matrix(read.table("mydata.txt", sep=','))

# Computing the train-test slicing coordinates
train_row_limit = train_set_size * nrow(lasso_data_set)
test_row_start = train_row_limit + 1

# Splitting the data set into a train set and test set
X_train = lasso_data_set[1:train_row_limit, 2:ncol(lasso_data_set)]
y_train = lasso_data_set[1:train_row_limit, 1]
X_test = lasso_data_set[test_row_start:nrow(lasso_data_set), 2:ncol(lasso_data_set)]
y_test = lasso_data_set[test_row_start:nrow(lasso_data_set), 1]

# Estimating the lasso regression parameters using the glmnet package
# The parameter alpha is the weight between Lasso and Ridge regression, we set it to one for Lasso regression
fit = glmnet(x=X_train, y=y_train, alpha = 1, lambda = lambda)

# Predicting y using the lasso estimated parameters
y_est_lasso = predict(fit, newx = X_test, type = "response", s=lambda)

# Computing the score for the lasso regression
rmse_lasso = compute_rmse(y_est_lasso, y_test)

# Conting the number of zeros
nb_non_zeros_ols = fit$df
nb_zeros_ols = ncol(lasso_data_set) - nb_non_zeros_ols

# Question 10 : Cross Validation

# Parameters
lambda_array = c(0.001, 0.01, 0.1, 1, 10)
train_size = 0.8
val_size = 0.1
test_size = 0.1
train_row_limit = train_size * nrow(ols_data_set)
validation_row_start = train_row_limit + 1
validation_row_end = (train_size + val_size) * nrow(ols_data_set)
test_row_start = validation_row_start + 1

# Train, validation, test split
X_train = lasso_data_set[1:train_row_limit, 2:ncol(lasso_data_set)]
y_train = lasso_data_set[1:train_row_limit, 1]
X_val = lasso_data_set[validation_row_start:validation_row_end, 2:ncol(lasso_data_set)]
y_val = lasso_data_set[validation_row_start:validation_row_end, 1]
X_test = lasso_data_set[test_row_start:nrow(lasso_data_set), 2:ncol(lasso_data_set)]
y_test = lasso_data_set[test_row_start:nrow(lasso_data_set), 1]

rmse_min = Inf
lambda_min = c(0)
# Estimating the lasso regression parameters using the glmnet package
for (lambda in lambda_array){
  lambda = c(lambda)
  # The parameter alpha is the weight between Lasso and Ridge regression, we set it to one for Lasso regression
  fit = glmnet(x=X_train, y=y_train, alpha = 1, lambda = lambda)
  # Predicting y using the lasso estimated parameters
  y_est_lasso = predict(fit, newx = X_val, type = "response", s=lambda)
  # Computing the score for the lasso regression
  rmse_lasso = compute_rmse(y_est_lasso, y_val)
  if (rmse_lasso < rmse_min){
    rmse_min = rmse_lasso
    lambda_min = lambda
  }
}

# Merging the train and validation into a full train set
X_train = lasso_data_set[1:validation_row_end, 2:ncol(lasso_data_set)]
y_train = lasso_data_set[1:validation_row_end, 1]

# Predicting y using the lasso estimated parameters
y_est_lasso_cval = predict(fit, newx = X_test, type = "response", s=lambda)

# Computing the score for the lasso regression against the test set
rmse_lasso_cval = compute_rmse(y_est_lasso_cval, y_test)


# Alternative method for Cross Validation using the glmnet package
# Cross validation with 100 lambda trials
fit_auto = glmnet(X_train, y_train, alpha = 1, nlambda = 100)

# Predicting y using the lasso estimated parameters
y_est_lasso_auto = predict(fit_auto, newx = X_test, type = "response")

# Computing the score for the lasso regression
rmse_lasso_auto = compute_rmse(y_est_lasso_auto, y_test)

# The obtained value is slightly inferior to the value we obtained manually

# Plotting the number of zeros against lambdas
nb_zeros_array = sort(ncol(lasso_data_set) - fit_auto$df)
lambdas_array = sort(fit_auto$lambda)
