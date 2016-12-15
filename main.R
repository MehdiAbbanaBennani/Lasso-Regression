##CODE QUI MARCHE CHEZ MOI, JAI PAS REUSSI A FAIRE TOURNER LE TIEN
train_set = data.matrix(read.table("/Users/Julien/Desktop/Lasso/Base.txt", sep=','))

X = train_set[,2:11]
y = train_set[,1]

# Question 2 : Multinlinear regression least squares estimator

estimate_beta <- function(X, y){
  a = t(X)%*%X
  b= solve(a)
  c= t(X)%*%y
  beta = b%*%c
  return(beta)
}

estimate_y <- function(X, beta){
  y_est = X%*%beta
  return(y_est)
}

compute_rmse <- function(y, y_est){
  return(sum((y-y_est)^2)/length(y))
}

beta = estimate_beta(X=X, y=y)
y_est = estimate_y(X=X, beta=beta)
rmse = compute_rmse(y=y, y_est)

#########CODE DE MEHDI##########################################
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

# Question 7 : Estimateur Lasso
library("genlasso")
n = 50
p = 100
X = matrix(rnorm(n*p), ncol=p)
y = X[,1] + rnorm(n)
D = diag(1,p)
out = genlasso(y, X=X, D=D)
summary(out)
plot(out)
betalasso = coef(out, lambda=sqrt(n*log(p)))
