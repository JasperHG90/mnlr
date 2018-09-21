#' normalize_x
#'
#' Center & scale variables in design matrix X
#'
#' @param X design matrix X
#'
#' @return X, centered & scaled
#'

normalize_X <- function(X) {

  m <- nrow(X)

  ## Calculate mean
  mu <- (1/m) * (t(X) %*% matrix(1, nrow = m))

  ## Calculate variance
  var <- (1/m) * apply(X, 2, function(x) sum((x - mean(x)) ^ 2))

  ## Subtract mean from each column and divide by variance
  X <- apply(X, 2, function(x) (x - mean(x)) / var(x))

  ## Return
  return(
    list(
      "train_mu" = mu,
      "train_var" = var,
      "X_normalized" = X
    )
  )

}

#' sigmoid
#'
#' Sigmoid function
#'
#' @param Z a scalar or matrix of any size
#'
#' @return sigmoid(Z) --> Z transformed by the sigmoid function
#'

sigmoid <- function(Z) {

  1 / (1+exp(-Z))

}

#' initialize_parameters
#'
#' Initialize the parameters for the logistic regression model
#'
#' @param columns number of columns ("variables") that feature in the model
#'
#' @return list containing matrix W (weights) and column vector b initialized at 0
#'

initialize_parameters <- function(columns) {

  # Initialize a matrix with zeros
  W <- matrix(0L, 1, columns)
  # Initialize a column vector with zeros
  b <- 0

  # Tests
  assertthat::are_equal(c(1, columns), dim(W))

  # To list
  params <- list(
    "W" = W,
    "b" = b
  )

  # Return
  return(params)

}

#' propagate
#'
#' Compute the cost function and its gradient
#'
#' @param w weights for the coefficients as a row vector
#' @param b scalar for the value b
#' @param X design matrix X
#' @param Y responses given as 1/0
#'
#' @return gradients and cost
#'

propagate <- function(w, b, X, Y, lambda) {

  ## Store number of rows
  m <- dim(X)[1]

  ## Create model and apply sigmoid
  Z <- w %*% t(X) + b
  A <- sigmoid(Z)

  ## Flatten matrix so that it has the same data type as Y
  A <- as.vector(A)

  ## Compute the cost using the cost function
  ## This function is basically calculating the deviance / log likelihood
  ## We want this to be as small as possible.
  cost <- -(1/m) * sum(Y * log(A) + (1-Y) * log(1-A)) + (lambda / (2*m)*norm(w))

  ## Compute gradients
  ## The gradient is the derivative of the cost function with respect to:
  ## - W (e.g. the weights)
  ## - b
  dw <- (1/m) * (t(X) %*% matrix(A-Y, ncol=1)) + t((lambda / m) * w)
  db <- (1/m) * sum(A-Y)

  ## Put gradients in list
  grads <- list(
    "dw" = dw,
    "db" = db
  )

  ## Return gradients and cost
  return(
    list(
      "gradients" = grads,
      "cost" = cost
    )
  )

}

#' norm
#'
#' Calculate the squared Euclidean norm of a column vector
#'
#' @param x column vector containing coefficients
#'
#' @return scalar value of the squared Euclidean norm
#'

norm <- function(x) {

  ## Make sure x is a column vector
  x <- matrix(x, ncol = 1)

  ## Calculate norm and return
  return(as.vector(t(x) %*% x))

}

#' Optimize
#'
#' optimize the parameters w and b by reducing the cost function
#'
#' @param w weights for the coefficients as a row vector
#' @param b scalar for the value b
#' @param X design matrix X
#' @param Y responses given as 1/0
#' @param current_iteration current iteration of the model. Will run until current_iteration == max_iterations
#' @param max_iterations maximum number of iterations the model will run
#' @param learning_rate learning rate alpha. Controls the speed of gradient descent
#' @param lambda Regularization parameter.
#' @param print_cost If TRUE, function will print the cost every 1000 iterations
#'
#' @return list containing final parameters, gradients and the costs for each iteration
#'

optim <- function(w, b, X, Y, current_iteration = 1,
                  max_iterations, learning_rate,
                  lambda = 0, print_cost = FALSE) {

  ## Keep track of cost
  costs <- c()

  ## If our current iteration <= max_iterations, run the program
  while(current_iteration <= max_iterations) {

    ## 1. call the propagate function
    prop <- propagate(w, b, X, Y, lambda)

    ## 2. Get derivatives
    dw <- prop$gradients$dw
    db <- prop$gradients$db

    ## 3. Update parameters by taking current parameters and subtracting learning_rate * gradient
    w <- w - t(learning_rate * dw)
    b <- as.vector(b - t(learning_rate * db))

    ## Keep track of the cost
    costs <- c(costs, prop$cost)

    ## Print the cost every 100 iterations
    if((current_iteration %% 1000 == 0) & (print_cost)) {

      print(paste0("Cost after iteration ", current_iteration, ": ", round(prop$cost, 3)))

    }

    ## Add 1 to the current iteration
    current_iteration <- current_iteration + 1

  }

  # return current gradients and the vector with costs

  list(

    "params" = list(
      "w" = w,
      "b" = b
    ),
    "gradients" = list(
      "dw" = dw,
      "db" = b
    ),
    "costs" = costs

  )

}

#' predict
#'
#' Predict function for logistic regression using maximum likelihood
#'
#' @param w final coefficients (slopes) for each of the variables
#' @param b final intercept for the model
#' @param X design matrix X
#'
#' @return list containing probabilities and class predictions
#'

predict_lr <- function(w, b, X) {

  ## Record number of rows
  m <- dim(X)[1]

  ## Compute probabilities using sigmoid transformation
  A <- sigmoid((w %*% t(X)) + b)

  ## Predict class based on probability
  Yhat <- ifelse(A >= 0.5, 1, 0)

  ## Return probabilities and class
  return(
    list(
      "probability" = A,
      "class" = as.vector(Yhat)
    )
  )

}

#' predict_mlr
#'
#' Predict classes for binomial/multinomial logistic regression
#'
#' @param w final coefficients (slopes) for each of the variables
#' @param b final intercept for the model
#' @param X design matrix X
#'
#' @return list containing probabilities and class predictions
#'
#' @export

predict_mlr <- function(model, X) {

  ## If model is binomial, then we're good
  if(model$type == "binomial") {

    ## Get parameters from model
    w <- model$model_information$params$w
    b <- model$model_information$params$b

    ## Predict
    predictions <- predict_lr(w, b, X)

    ## Return
    return(predictions)

  } else {

    ## Multinomial model

    # Retrieve models from list
    models <- model$models
    probabilities <- list()

    # Model names
    model_names <- names(models)

    # For each model, predict outcome
    for(model_name in model_names) {

      mod <- models[[model_name]]

      ## Get parameters
      w <- mod$model_information$params$w
      b <- mod$model_information$params$b

      ## Predict
      preds <- predict_lr(w, b, X)

      ## Append to probabilities
      probabilities[[model_name]] <- matrix(preds$probability, ncol=1)

    }

    ## Bind into matrix
    probability_matrix <- do.call(cbind, probabilities)
    colnames(probability_matrix) <- names(models)

    ## Find category for which probability is maximized
    max_prob <- apply(probability_matrix, 1, function(x) names(which(x == max(x))))

    ## Return
    return(
      list(
        "probability" = probability_matrix,
        "class" = max_prob
      )
    )

  }

}

#' accuracy
#'
#' Calculate accuracy for a binomial logistic regression model
#'
#' @param pred class predictions
#' @param Y actual classes
#'
#' @return scalar ranging from 0  to 100 indicating the accuracy in percentages
accuracy <- function(pred, Y) {

  ## Create pivot table
  pivot <- table(pred,Y)

  ## Calculate accuracy
  return((sum(diag(pivot)) / sum(pivot)) * 100)

}

#' log_likelihood
#'
#' Calculate the log likelihood for a model
#'
#' @param y actuall class values
#' @param y_pred probabilities for each of the predicted values
#'
#' @return log-likelihood
#'

log_likelihood <- function(y, y_pred) {

  sum(y * log(y_pred) + (1-y) * log(1-y_pred))

}

#' logistic regression
#'
#' Create a logistic regression model
#'
#' @param w weights for the coefficients as a row vector
#' @param b scalar for the value b
#' @param X design matrix X
#' @param Y responses given as 1/0
#' @param current_iteration current iteration of the model. Will run until current_iteration == max_iterations
#' @param max_iterations maximum number of iterations the model will run
#' @param learning_rate learning rate alpha. Controls the speed of gradient descent
#' @param lambda Regularization parameter.
#' @param print_cost If TRUE, function will print the cost every 1000 iterations
#' @param normalize If TRUE, we will center & scale the data prior to modeling.
#'
#' @return
#'

lrm <- function(X, Y, max_iterations = 2000, lambda = 0,
                learning_rate = 0.1, print_cost = FALSE,
                normalize = FALSE) {

  ## Make sure that variables passed make sense
  assertthat::are_equal(nrow(X), length(Y))

  ## Normalize X
  if(normalize) {

    Xnorm <- normalize_X(X)
    X <- Xnorm$X_normalized

  }

  ## Check how many outcome variables there are
  outcomes <- unique(Y)

  ## Initialize parameters
  params <- initialize_parameters(ncol(X))

  ## Retrieve from list
  w <- params$W
  b <- params$b

  ## Implement gradient descent
  gd_results <- optim(w, b, X, Y,
                      current_iteration = 1,
                      max_iterations,
                      learning_rate,
                      lambda = lambda,
                      print_cost = print_cost)

  ## Predict on training set
  pred <- predict_lr(gd_results$params$w,
                     gd_results$params$b,
                     X)

  ## Get probabilities and predictions
  pred_class <- pred$class
  pred_probability <- pred$probability

  ## Calculate accuracy
  acc <- accuracy(pred_class, Y)

  ## Return
  ret <- list(
    "inputs" = list(
      "X" = X,
      "Y" = Y,
      "max_iterations" = max_iterations,
      "learning_rate" = learning_rate,
      "lambda" = lambda,
      "normalize_X" = normalize_X
    ),
    "model_information" = gd_results,
    "predictions" = list(
      "probabilities" = pred_probability,
      "class" = pred_class
    ),
    "metrics" = list(
      "accuracy" = acc
    )
  )

}

#' mlogistic
#'
#' Perform Binomial/Multinomial regression model
#'
#' @param w weights for the coefficients as a row vector
#' @param b scalar for the value b
#' @param X design matrix X
#' @param Y outcome label. Must be a factor with at least 2 levels.
#' @param current_iteration current iteration of the model. Will run until current_iteration == max_iterations
#' @param max_iterations maximum number of iterations the model will run
#' @param learning_rate learning rate alpha. Controls the speed of gradient descent
#' @param lambda Regularization parameter.
#' @param print_cost If TRUE, function will print the cost every 1000 iterations
#'
#' @return
#'
#' @export

mlogistic <- function(X, Y, max_iterations = 2000, lambda = 0,
                      learning_rate = 0.1, print_cost = FALSE,
                      normalize = FALSE) {

  ## Assertions
  assertthat::assert_that(is(Y)[1] == "factor",
                          msg="Outcome variable Y must be a factor with at least two classes")
  assertthat::assert_that(length(unique(Y)) >= 1,
                          msg="Outcome variable Y must at least have two outcome classes")

  ## Get number of categories
  ncat <- unique(Y)

  ## If ncat is one, then just wrap around lrm
  if(length(ncat) == 2) {

    ## Turn factor to binary
    Y <- ifelse(Y == ncat[1], 1, 0)

    m <- lrm(X, Y, max_iterations, lambda, learning_rate, print_cost, normalize)
    m$type <- "binomial"

    ## Return
    return(m)

  } else {

    ## We will run ncat - 1 regression models
    models <- list()

    ## Loop
    for(cat in ncat) {

      ## Convert Y into binary
      Ycat <- ifelse(Y == cat, 1, 0)

      ## Model
      models[[cat]] <- lrm(X, Ycat, max_iterations, lambda, learning_rate, print_cost, normalize)

    }

    ## Retrieve probabilities for each class
    probs <- do.call(cbind, lapply(models, function(x) t(x$predictions$probabilities)))

    ## Add names
    colnames(probs) <- names(models)

    ## Choose the category for which the probability is maximized
    max_prob <- apply(probs, 1, function(x) names(which(x == max(x))))

    ## Construct model information
    output <- list(

      "type" = "multinomial",
      "models" = models,
      "probabilities" = probs,
      "predictions" = max_prob,
      "accuracy" = sum(Y == max_prob) / length(Y)

    )

  }

}


