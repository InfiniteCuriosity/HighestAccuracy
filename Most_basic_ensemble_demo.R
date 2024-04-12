library(MASS) # Need for Boston Housing data set
library(tree) # Need to make tree models

head(MASS::Boston, n = 10) # look at the first ten (out of 505) rows of the Boston Housing data set
dim(MASS::Boston)

# Break the data set into train (~80%) and test (~20%)
df <- MASS::Boston
train <- df[1:400, ] # the first 400 rows
test <- df[401:505, ] # the last 104 rows

# 1. Linear model
Boston_lm <- lm(medv ~ ., data = train)

# Predictions for the linear model using the test data (required for the ensemble)
Boston_lm_predictions <- predict(object = Boston_lm, newdata = test)

# Error rate for the linear model using actual vs predicted results
Boston_lm_RMSE <- Metrics::rmse(actual = test$medv, predicted = Boston_lm_predictions)

# 2. Tree model
Boston_tree <- tree(medv ~ ., data = train)

# Predictions for the tree model using the test data (required for the ensemble)
Boston_tree_predictions <- predict(object = Boston_tree, newdata = test)

# Error rate for the tree model using actual and predicted results
Boston_tree_RMSE <- Metrics::rmse(actual = test$medv, predicted = Boston_tree_predictions)

# 3. Create the ensemble
ensemble <- data.frame(
  'linear' = Boston_lm_predictions,
  'tree' = Boston_tree_predictions,
  'y_ensemble' = test$medv
)

head(ensemble)

ensemble_train <- ensemble[1:80, ]
ensemble_test <- ensemble[81:105, ]

head(ensemble_test)

# Ensemble linear modeling
ensemble_lm <- lm(y_ensemble ~ ., data = ensemble_train)

# Predictions for the ensemble linear model
ensemble_prediction <- predict(ensemble_lm, newdata = ensemble_test)

# Root mean squared error for the ensemble linear model
ensemble_lm_RMSE <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = ensemble_prediction)

results <- list(
  'Linear' = Boston_lm_RMSE,
  'Trees' = Boston_tree_RMSE,
  'Ensembles_Linear' = ensemble_lm_RMSE
  )

results
warnings()