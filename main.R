# Data importation
train_set = data.matrix(read.table("Base.txt", sep=','))

X = train_set[,2:11]
y = train_set[,1]

# Question 2 : Multinlinear regression least squares estimator

estimate_beta <- function(X, y){
  beta_tild = solve(a = t(X)%*%X, b = t(X)%*%y)
  beta = solve(X%*%t(X), beta_tild)
  return(beta)
}

estimate_y <- function(X, beta){
  y_est = X * beta
  return(y_est)
}

compute_rmse <- function(y, y_est){
  return(sum((y-y_est)^2)/length(y))
  }

beta = estimate_beta(X=X, y=y)
y = estimate_y(X=X, beta=beta)
rmse = compute_rmse(y=y, y_est=y_est)

