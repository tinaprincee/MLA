# Load necessary libraries
library(psych)
library(ggplot2)
library(DataExplorer)
library(car)
install.packages("lmtest")
library(lmtest) # autocorrelation
library(Metrics)
library(MASS)

# MISSING ANALYSIS
str(forestfires)
summary(forestfires)
is.na(forestfires)
plot_missing(forestfires)

pairs.panels(forestfires)
plot_histogram(forestfires)
plot_density(forestfires)
plot_corelation(forestfires)

# TRAINING & TESTING SPLIT
set.seed(123)
train_index <- sample(1:nrow(forestfires), 0.7 * nrow(forestfires))

training <- forestfires[train_index, ]
testing <- forestfires[-train_index, ]
head(training)
head(testing)

# MODEL BUILDING

# Build a full model
fullmodel <- lm(area ~ ., data = training)
summary(fullmodel)

lm_step <- stepAIC(fullmodel, direction = 'backward')

# 1 Build a simple linear regression model using temperature to predict the burned area
lm_temp <- lm(area ~ temp, data = training)
summary(lm_temp)

# 2 Build a simple linear regression model using relative humidity to predict the burned area
lm_rh <- lm(area ~ RH, data = training)
summary(lm_rh)

# 3 Build a simple linear regression model using FFMC to predict the burned area
lm_ffmc <- lm(area ~ FFMC, data = training)
summary(lm_ffmc)

# Predict using the full model on the test data
fullmodel_pred <- predict(fullmodel, newdata = testing)
lm_temp_pred <- predict(lm_temp, newdata = testing)
lm_rh_pred <- predict(lm_rh, newdata = testing)
lm_ffmc_pred <- predict(lm_ffmc, newdata = testing)

# CALCULATE PERFORMANCE METRICS

# Full model
fullmodel_r2 <- summary(fullmodel)$r.squared
fullmodel_test_r2 <- cor(testing$area, fullmodel_pred)^2

# Temperature
lm_temp_r2 <- summary(lm_temp)$r.squared
lm_temp_test_r2 <- cor(testing$area, lm_temp_pred)^2

# Humidity
lm_rh_r2 <- summary(lm_rh)$r.squared
lm_rh_test_r2 <- cor(testing$area, lm_rh_pred)^2

# FFMC
lm_ffmc_r2 <- summary(lm_ffmc)$r.squared
lm_ffmc_test_r2 <- cor(testing$area, lm_ffmc_pred)^2

# Output the performance metrics
cat("Full Model - Train R2:", fullmodel_r2, "Test R2:", fullmodel_test_r2, "\n")
cat("Temperature Model - Train R2:", lm_temp_r2, "Test R2:", lm_temp_test_r2, "\n")
cat("Relative Humidity Model - Train R2:", lm_rh_r2, "Test R2:", lm_rh_test_r2, "\n")
cat("FFMC Model - Train R2:", lm_ffmc_r2, "Test R2:", lm_ffmc_test_r2, "\n")


#RIDGE REGRESSION


# Load necessary libraries
install.packages("glmnet")
library(glmnet)
install.packages("dplyr")
library(dplyr)

# Load forest fires dataset (assuming 'forestfires' is already loaded)
# Replace with your actual dataset loading method
# For example: data <- read.csv("forestfires.csv")

# Define predictors (X) and target (Y)
X <- model.matrix(area ~ ., data = forestfires)[, -1]
Y <- forestfires$area

# Define lambda sequence
lambda <- 10^seq(10, -2, length = 100)

# Split the data into training and validation sets
set.seed(123)
part <- sample(2, nrow(X), replace = TRUE, prob = c(0.7, 0.3))
X_train <- X[part == 1, ]
X_cv <- X[part == 2, ]
Y_train <- Y[part == 1]
Y_cv <- Y[part == 2]

# Perform Ridge regression
ridge_reg <- glmnet(X_train, Y_train, alpha = 0, lambda = lambda)

# Find the best lambda via cross-validation
ridge_reg_cv <- cv.glmnet(X_train, Y_train, alpha = 0)
bestlam_ridge <- ridge_reg_cv$lambda.min

# Predict on the validation set
ridge_pred <- predict(ridge_reg, s = bestlam_ridge, newx = X_cv)

# Calculate Mean Squared Error (MSE) and R-squared (R2)
mse_ridge <- mean((Y_cv - ridge_pred)^2)
sst_ridge <- sum((Y_cv - mean(Y_cv))^2)
r2_ridge <- 1 - (mse_ridge / sst_ridge)

# Output results
cat("Ridge Regression Results:\n")
cat("Best Lambda:", bestlam_ridge, "\n")
cat("Mean Squared Error:", mse_ridge, "\n")
cat("R-squared (R2):", r2_ridge, "\n\n")

# Get Ridge regression coefficients
ridge_coefs <- predict(ridge_reg, type = "coefficients", s = bestlam_ridge)
print("Ridge Coefficients:")
print(ridge_coefs)

#LOSSO REGRESSION

# Perform Lasso regression
lasso_reg <- glmnet(X_train, Y_train, alpha = 1, lambda = lambda)

# Find the best lambda via cross-validation
lasso_reg_cv <- cv.glmnet(X_train, Y_train, alpha = 1)
bestlam_lasso <- lasso_reg_cv$lambda.min

# Predict on the validation set
lasso_pred <- predict(lasso_reg, s = bestlam_lasso, newx = X_cv)

# Calculate Mean Squared Error (MSE) and R-squared (R2)
mse_lasso <- mean((Y_cv - lasso_pred)^2)
sst_lasso <- sum((Y_cv - mean(Y_cv))^2)
r2_lasso <- 1 - (mse_lasso / sst_lasso)

# Output results
cat("Lasso Regression Results:\n")
cat("Best Lambda:", bestlam_lasso, "\n")
cat("Mean Squared Error:", mse_lasso, "\n")
cat("R-squared (R2):", r2_lasso, "\n\n")

# Get Lasso regression coefficients
lasso_coefs <- predict(lasso_reg, type = "coefficients", s = bestlam_lasso)
print("Lasso Coefficients:")
print(lasso_coefs)




