# Install required packages
packages <- c(
  "tidyverse",
  "caret",
  "randomForest",
  "rpart",
  "rpart.plot",
  "corrplot",
  "pROC",
  "ROSE",
  "shiny"
)

# Install packages that are not already installed
new_packages <- packages[
  !(packages %in% installed.packages()[, "Package"])
]

if (length(new_packages) > 0) {
  install.packages(new_packages)
}

# Load libraries
library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(corrplot)
library(pROC)
library(ROSE)
library(shiny)
# Import dataset
machine_data <- read.csv(file.choose())

# Display first six records
head(machine_data)

# Dataset dimensions
dim(machine_data)

# Column names
colnames(machine_data)

# Dataset structure
str(machine_data)
dir.create("data", showWarnings = FALSE)

write.csv(
  machine_data,
  "data/ai4i2020.csv",
  row.names = FALSE
)
# Display column names
colnames(machine_data)

# Display first 10 records
head(machine_data, 10)

# Understand data types
str(machine_data)

# Statistical summary
summary(machine_data)
# Missing values in each column
colSums(is.na(machine_data))

# Total missing values
sum(is.na(machine_data))

# Duplicate records
sum(duplicated(machine_data))
# Count normal and failed machines
table(machine_data$Machine.failure)

# Calculate percentage
prop.table(
  table(machine_data$Machine.failure)
) * 100
#Vizualization
library(ggplot2)

ggplot(machine_data,
       aes(x = factor(Machine.failure))) +
  geom_bar(fill = "steelblue") +
  labs(
    title = "Machine Failure Distribution",
    x = "Machine Failure (0 = Normal, 1 = Failure)",
    y = "Number of Observations"
  ) +
  theme_minimal()

colnames(machine_data)

# Identify the air temperature column
air_col <- grep(
  "Air.temperature",
  colnames(machine_data),
  value = TRUE
)

# Create histogram
ggplot(machine_data,
       aes(x = .data[[air_col[1]]])) +
  geom_histogram(
    bins = 30,
    fill = "orange",
    color = "black"
  ) +
  labs(
    title = "Distribution of Air Temperature",
    x = "Air Temperature (Kelvin)",
    y = "Frequency"
  ) +
  theme_minimal()

speed_col <- grep("Rotational.speed",
                  names(machine_data),
                  value = TRUE)

torque_col <- grep("Torque",
                   names(machine_data),
                   value = TRUE)

ggplot(machine_data,
       aes(x = .data[[speed_col[1]]],
           y = .data[[torque_col[1]]],
           color = factor(Machine.failure))) +
  geom_point(alpha = 0.5) +
  labs(
    title = "Rotational Speed vs Torque",
    color = "Machine Failure"
  ) +
  theme_minimal()

table(machine_data$Machine.failure)

round(
  prop.table(table(machine_data$Machine.failure)) * 100,
  2
)

library(dplyr)

machine_data <- machine_data %>%
  mutate(
    Temperature_Difference =
      Process.temperature..K. -
      Air.temperature..K.,
    
    Mechanical_Power =
      (2 * pi * Rotational.speed..rpm. *
         Torque..Nm.) / 60
  )

# Display updated dataset
head(machine_data)

# Check new columns
colnames(machine_data)

# Create clean modelling dataset

model_data <- machine_data %>%
  select(
    -UDI,
    -Product.ID,
    -TWF,
    -HDF,
    -PWF,
    -OSF,
    -RNF
  )

# Convert categorical variables
model_data$Type <- as.factor(model_data$Type)

# Convert target into a classification factor
model_data$Machine.failure <- factor(
  model_data$Machine.failure,
  levels = c(0, 1),
  labels = c("Normal", "Failure")
)

# Inspect cleaned data
str(model_data)

# Check dimensions
dim(model_data)

failure_count <- table(model_data$Machine.failure)

print(failure_count)

failure_percentage <- prop.table(failure_count) * 100

round(failure_percentage, 2)

#Spliting data
library(caret)

set.seed(123)

# Stratified 80:20 split
train_index <- createDataPartition(
  model_data$Machine.failure,
  p = 0.80,
  list = FALSE
)

# Training dataset
train_data <- model_data[train_index, ]

# Testing dataset
test_data <- model_data[-train_index, ]

# Check dimensions
dim(train_data)
dim(test_data)
# Training distribution
table(train_data$Machine.failure)

# Testing distribution
table(test_data$Machine.failure)
# Select numerical variables
numeric_data <- model_data %>%
  select(where(is.numeric))

# Correlation matrix
cor_matrix <- cor(numeric_data)

# Visualize correlation
library(corrplot)

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.col = "black",
  tl.srt = 45,
  addCoef.col = "black",
  number.cex = 0.6
)
# Check dimensions
dim(model_data)
dim(train_data)
dim(test_data)

# Training class distribution
table(train_data$Machine.failure)

# Testing class distribution
table(test_data$Machine.failure)

# Check proportions
prop.table(table(train_data$Machine.failure)) * 100
prop.table(table(test_data$Machine.failure)) * 100

# Train Logistic Regression model

logistic_model <- glm(
  Machine.failure ~ .,
  data = train_data,
  family = binomial(link = "logit")
)

# Display model summary
summary(logistic_model)

# Predict failure probabilities
logistic_prob <- predict(
  logistic_model,
  newdata = test_data,
  type = "response"
)

# Convert probabilities into classes
logistic_pred <- ifelse(
  logistic_prob >= 0.5,
  "Failure",
  "Normal"
)

# Convert predictions into factor
logistic_pred <- factor(
  logistic_pred,
  levels = c("Normal", "Failure")
)

# Display first 10 predictions
head(data.frame(
  Actual = test_data$Machine.failure,
  Predicted = logistic_pred,
  Probability = round(logistic_prob, 4)
), 10)
# Confusion matrix
logistic_cm <- confusionMatrix(
  data = logistic_pred,
  reference = test_data$Machine.failure,
  positive = "Failure"
)

print(logistic_cm)

# Extract performance metrics
logistic_accuracy <- logistic_cm$overall["Accuracy"]

logistic_precision <- logistic_cm$byClass["Precision"]

logistic_recall <- logistic_cm$byClass["Recall"]

logistic_f1 <- logistic_cm$byClass["F1"]

# Calculate ROC-AUC
logistic_roc <- roc(
  test_data$Machine.failure,
  logistic_prob,
  levels = c("Normal", "Failure"),
  direction = "<"
)

logistic_auc <- as.numeric(auc(logistic_roc))

# Display results
cat("Logistic Regression Results\n")

cat("Accuracy:", round(logistic_accuracy, 4), "\n")
cat("Precision:", round(logistic_precision, 4), "\n")
cat("Recall:", round(logistic_recall, 4), "\n")
cat("F1 Score:", round(logistic_f1, 4), "\n")
cat("ROC-AUC:", round(logistic_auc, 4), "\n")

plot(
  logistic_roc,
  col = "blue",
  lwd = 2,
  main = "ROC Curve - Logistic Regression"
)

abline(
  a = 0,
  b = 1,
  lty = 2,
  col = "red"
)

# Remove redundant feature
model_data <- model_data %>%
  select(-Temperature_Difference)

# Recreate training and testing sets
set.seed(123)

train_index <- createDataPartition(
  model_data$Machine.failure,
  p = 0.80,
  list = FALSE
)

train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Train corrected Logistic Regression
logistic_model <- glm(
  Machine.failure ~ .,
  data = train_data,
  family = binomial()
)

summary(logistic_model)

logistic_prob <- predict(
  logistic_model,
  newdata = test_data,
  type = "response"
)

logistic_roc <- roc(
  test_data$Machine.failure,
  logistic_prob,
  levels = c("Normal", "Failure"),
  direction = "<"
)

plot(
  logistic_roc,
  legacy.axes = TRUE,
  col = "blue",
  lwd = 2,
  main = "ROC Curve - Logistic Regression"
)

abline(
  a = 0,
  b = 1,
  lty = 2,
  col = "red"
)

auc(logistic_roc)

library(rpart)
library(rpart.plot)

# Train Decision Tree
tree_model <- rpart(
  Machine.failure ~ .,
  data = train_data,
  method = "class",
  control = rpart.control(
    cp = 0.001,
    maxdepth = 5,
    minsplit = 20
  )
)

# Display model summary
print(tree_model)

rpart.plot(
  tree_model,
  type = 3,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Decision Tree - Machine Failure Prediction"
)

tree_pred <- predict(
  tree_model,
  newdata = test_data,
  type = "class"
)

tree_prob <- predict(
  tree_model,
  newdata = test_data,
  type = "prob"
)[, "Failure"]

tree_cm <- confusionMatrix(
  tree_pred,
  test_data$Machine.failure,
  positive = "Failure"
)

print(tree_cm)

tree_roc <- roc(
  test_data$Machine.failure,
  tree_prob,
  levels = c("Normal", "Failure"),
  direction = "<"
)

cat("Decision Tree AUC:",
    as.numeric(auc(tree_roc)))

# Confusion matrix
tree_cm <- confusionMatrix(
  tree_pred,
  test_data$Machine.failure,
  positive = "Failure"
)

print(tree_cm)

# Extract metrics
cat("Accuracy:",
    tree_cm$overall["Accuracy"], "\n")

cat("Precision:",
    tree_cm$byClass["Precision"], "\n")

cat("Recall:",
    tree_cm$byClass["Recall"], "\n")

cat("F1 Score:",
    tree_cm$byClass["F1"], "\n")

cat("ROC-AUC:",
    as.numeric(auc(tree_roc)), "\n")

# RANDOM FOREST
library(randomForest)

set.seed(123)

rf_model <- randomForest(
  Machine.failure ~ .,
  data = train_data,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

# Display model
print(rf_model)

# Predict classes
rf_pred <- predict(
  rf_model,
  newdata = test_data,
  type = "class"
)

# Predict probabilities
rf_prob <- predict(
  rf_model,
  newdata = test_data,
  type = "prob"
)[, "Failure"]

# Display predictions
head(data.frame(
  Actual = test_data$Machine.failure,
  Predicted = rf_pred,
  Probability = round(rf_prob, 4)
), 10)

rf_cm <- confusionMatrix(
  rf_pred,
  test_data$Machine.failure,
  positive = "Failure"
)

print(rf_cm)

rf_accuracy <- rf_cm$overall["Accuracy"]
rf_precision <- rf_cm$byClass["Precision"]
rf_recall <- rf_cm$byClass["Recall"]
rf_f1 <- rf_cm$byClass["F1"]

rf_roc <- roc(
  test_data$Machine.failure,
  rf_prob,
  levels = c("Normal", "Failure"),
  direction = "<"
)

rf_auc <- as.numeric(auc(rf_roc))

cat("Random Forest Results\n")
cat("Accuracy:", round(rf_accuracy, 4), "\n")
cat("Precision:", round(rf_precision, 4), "\n")
cat("Recall:", round(rf_recall, 4), "\n")
cat("F1 Score:", round(rf_f1, 4), "\n")
cat("ROC-AUC:", round(rf_auc, 4), "\n")

# Calculate feature importance
importance(rf_model)

# Visualize feature importance
varImpPlot(
  rf_model,
  main = "Random Forest - Feature Importance"
)

# Feature importance
importance(rf_model)

# Visualize feature importance
varImpPlot(
  rf_model,
  main = "Important Features for Machine Failure Prediction"
)

#Validation Dataset
set.seed(456)

validation_index <- createDataPartition(
  train_data$Machine.failure,
  p = 0.80,
  list = FALSE
)

rf_train <- train_data[validation_index, ]

validation_data <- train_data[-validation_index, ]

dim(rf_train)
dim(validation_data)

set.seed(123)

rf_validation_model <- randomForest(
  Machine.failure ~ .,
  data = rf_train,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

# Predict failure probabilities
validation_prob <- predict(
  rf_validation_model,
  newdata = validation_data,
  type = "prob"
)[, "Failure"]

#Experimenting with duff Thresholds
thresholds <- c(0.1, 0.2, 0.3, 0.4, 0.5)

threshold_results <- data.frame()

for (t in thresholds) {
  
  pred <- ifelse(
    validation_prob >= t,
    "Failure",
    "Normal"
  )
  
  pred <- factor(
    pred,
    levels = c("Normal", "Failure")
  )
  
  cm <- confusionMatrix(
    pred,
    validation_data$Machine.failure,
    positive = "Failure"
  )
  
  threshold_results <- rbind(
    threshold_results,
    data.frame(
      Threshold = t,
      Accuracy = as.numeric(cm$overall["Accuracy"]),
      Precision = as.numeric(cm$byClass["Precision"]),
      Recall = as.numeric(cm$byClass["Recall"]),
      F1 = as.numeric(cm$byClass["F1"])
    )
  )
}

print(round(threshold_results, 4))

library(tidyr)

threshold_long <- threshold_results %>%
  pivot_longer(
    cols = c(Accuracy, Precision, Recall, F1),
    names_to = "Metric",
    values_to = "Score"
  )

ggplot(
  threshold_long,
  aes(
    x = Threshold,
    y = Score,
    color = Metric
  )
) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  labs(
    title = "Performance at Different Classification Thresholds",
    x = "Classification Threshold",
    y = "Metric Score"
  ) +
  theme_minimal()


# Feature Importance using Random Forest

importance_values <- importance(rf_model)

print(importance_values)

# Plot feature importance
varImpPlot(
  rf_model,
  main = "Random Forest - Feature Importance",
  col = "steelblue"
)


# Get predicted probabilities
rf_prob <- predict(
  rf_model,
  newdata = test_data,
  type = "prob"
)[, "Failure"]

colnames(predict(
  rf_model,
  newdata = test_data,
  type = "prob"
))

# Test different classification thresholds
thresholds <- seq(0.1, 0.9, by = 0.1)

for (t in thresholds) {
  
  predicted <- ifelse(rf_prob >= t, 1, 0)
  
  cm <- confusionMatrix(
    factor(predicted, levels = c(0, 1)),
    factor(test_data$Machine.failure, levels = c(0, 1)),
    positive = "1"
  )
  
  cat("\nThreshold:", t)
  cat("\nAccuracy:", cm$overall["Accuracy"])
  cat("\nPrecision:", cm$byClass["Precision"])
  cat("\nRecall:", cm$byClass["Recall"])
  cat("\nF1 Score:", cm$byClass["F1"])
  cat("\n-------------------------")
}
# Check target variable classes
levels(test_data$Machine.failure)

# Check probability column names
colnames(predict(
  rf_model,
  newdata = test_data,
  type = "prob"
))

rf_prob <- predict(
  rf_model,
  newdata = test_data,
  type = "prob"
)[, "Failure"]

thresholds <- seq(0.1, 0.9, by = 0.1)

for (t in thresholds) {
  
  predicted <- ifelse(rf_prob >= t, "Failure", "Normal")
  
  cm <- confusionMatrix(
    factor(predicted, levels = c("Normal", "Failure")),
    factor(test_data$Machine.failure,
           levels = c("Normal", "Failure")),
    positive = "Failure"
  )
  
  cat("\nThreshold:", t)
  cat("\nAccuracy:", cm$overall["Accuracy"])
  cat("\nPrecision:", cm$byClass["Precision"])
  cat("\nRecall:", cm$byClass["Recall"])
  cat("\nF1 Score:", cm$byClass["F1"])
  cat("\n-------------------------")
}

# Threshold performance visualization

thresholds <- seq(0.1, 0.6, by = 0.1)

precision <- c(0.464, 0.7432432, 0.8813559,
               0.9107143, 0.9591837, 0.9777778)

recall <- c(0.8656716, 0.8208955, 0.7761194,
            0.761194, 0.7014925, 0.6567164)

f1 <- c(0.6041667, 0.7801418, 0.8253968,
        0.8292683, 0.8103448, 0.7857143)

plot(thresholds, precision, type = "b",
     col = "blue", ylim = c(0, 1),
     xlab = "Threshold", ylab = "Score",
     main = "Threshold Optimization Performance")

lines(thresholds, recall, type = "b", col = "red")
lines(thresholds, f1, type = "b", col = "darkgreen")

legend("bottomleft",
       legend = c("Precision", "Recall", "F1 Score"),
       col = c("blue", "red", "darkgreen"),
       lty = 1, pch = 1)

# Test higher thresholds

thresholds <- c(0.7, 0.8, 0.9)

for (t in thresholds) {
  
  predicted <- ifelse(rf_prob >= t, "Failure", "Normal")
  
  cm <- confusionMatrix(
    factor(predicted, levels = c("Normal", "Failure")),
    factor(test_data$Machine.failure,
           levels = c("Normal", "Failure")),
    positive = "Failure"
  )
  
  cat("\nThreshold:", t)
  cat("\nAccuracy:", cm$overall["Accuracy"])
  cat("\nPrecision:", cm$byClass["Precision"])
  cat("\nRecall:", cm$byClass["Recall"])
  cat("\nF1 Score:", cm$byClass["F1"])
  cat("\n-------------------------")
}

# Check training data distribution

table(train_data$Machine.failure)

# Calculate percentages

prop.table(
  table(train_data$Machine.failure)
) * 100

install.packages("recipes")
install.packages("themis")
install.packages("dplyr")
library(dplyr)
library(recipes)
library(themis)
library(randomForest)
library(caret)

# Create SMOTE recipe


library(recipes)
library(themis)
library(dplyr)

# Create updated SMOTE recipe
smote_recipe <- recipe(
  Machine.failure ~ .,
  data = train_data
) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(
    Machine.failure,
    over_ratio = 1,
    neighbors = 5
  )

# Prepare recipe
smote_model <- prep(
  smote_recipe,
  training = train_data
)

# Generate balanced training dataset
smote_train <- bake(
  smote_model,
  new_data = NULL
)

# Check class distribution
table(smote_train$Machine.failure)


# Apply SMOTE to training data

smote_model <- prep(
  smote_recipe,
  training = train_data
)

smote_train <- bake(
  smote_model,
  new_data = NULL
)

# Check new class distribution

table(smote_train$Machine.failure)


# Train Random Forest with SMOTE data

set.seed(123)

rf_smote <- randomForest(
  Machine.failure ~ .,
  data = smote_train,
  ntree = 500,
  importance = TRUE
)

# Display model
print(rf_smote)


# Predict using original test data

smote_predictions <- predict(
  rf_smote,
  newdata = test_data,
  type = "class"
)

# Confusion matrix

smote_cm <- confusionMatrix(
  smote_predictions,
  test_data$Machine.failure,
  positive = "Failure"
)

print(smote_cm)
# Check training columns
print(names(smote_train))

# Check testing columns
print(names(test_smote))

# Find missing columns
setdiff(names(smote_train), names(test_smote))

# Find extra columns
setdiff(names(test_smote), names(smote_train))

test_smote <- bake(
  smote_model,
  new_data = test_data
)

# Check training columns
names(smote_train)

# Check testing columns
names(test_smote)

# Check missing columns
setdiff(names(smote_train), names(test_smote))

smote_predictions <- predict(
  rf_smote,
  newdata = test_smote,
  type = "class"
)

smote_cm <- confusionMatrix(
  smote_predictions,
  test_smote$Machine.failure,
  positive = "Failure"
)

print(smote_cm)


# Get SMOTE model probabilities

smote_prob <- predict(
  rf_smote,
  newdata = test_smote,
  type = "prob"
)[, "Failure"]

# Test different thresholds

thresholds <- seq(0.1, 0.9, by = 0.1)

for (t in thresholds) {
  
  predicted <- ifelse(
    smote_prob >= t,
    "Failure",
    "Normal"
  )
  
  cm <- confusionMatrix(
    factor(predicted,
           levels = c("Normal", "Failure")),
    test_smote$Machine.failure,
    positive = "Failure"
  )
  
  cat("\nThreshold:", t)
  cat("\nAccuracy:", cm$overall["Accuracy"])
  cat("\nPrecision:", cm$byClass["Precision"])
  cat("\nRecall:", cm$byClass["Recall"])
  cat("\nF1 Score:", cm$byClass["F1"])
  cat("\n-------------------------")
}


# ============================================================
# ADDITIONAL VISUALIZATIONS FOR PROJECT REPORT
# Run this section after the SMOTE model evaluation above.
# ============================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(pROC)

dir.create("visualizations", showWarnings = FALSE)

# 1. Process Temperature vs Air Temperature
p1 <- ggplot(machine_data,
             aes(x = Air.temperature..K.,
                 y = Process.temperature..K.,
                 color = factor(Machine.failure))) +
  geom_point(alpha = 0.5) +
  labs(title = "Process Temperature vs Air Temperature",
       x = "Air Temperature (K)", y = "Process Temperature (K)",
       color = "Machine Failure") +
  theme_minimal()
print(p1)
ggsave("visualizations/01_temperature_relationship.png", p1,
       width = 8, height = 5)

# 2. Tool Wear by Machine Condition
p2 <- ggplot(machine_data,
             aes(x = factor(Machine.failure), y = Tool.wear..min.,
                 fill = factor(Machine.failure))) +
  geom_boxplot() +
  labs(title = "Tool Wear by Machine Condition",
       x = "Machine Condition (0 = Normal, 1 = Failure)",
       y = "Tool Wear (minutes)", fill = "Machine Failure") +
  theme_minimal() +
  theme(legend.position = "none")
print(p2)
ggsave("visualizations/02_tool_wear_boxplot.png", p2,
       width = 8, height = 5)

# 3. Torque Distribution
p3 <- ggplot(machine_data,
             aes(x = Torque..Nm., fill = factor(Machine.failure))) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  labs(title = "Torque Distribution by Machine Condition",
       x = "Torque (Nm)", y = "Number of Observations",
       fill = "Machine Failure") +
  theme_minimal()
print(p3)
ggsave("visualizations/03_torque_distribution.png", p3,
       width = 8, height = 5)

# 4. Compare ROC curves of all three models
png("visualizations/04_roc_curve_comparison.png",
    width = 900, height = 650)
plot(logistic_roc, col = "blue", lwd = 2,
     main = "ROC Curve Comparison of Models")
plot(tree_roc, col = "darkgreen", lwd = 2, add = TRUE)
plot(rf_roc, col = "red", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(
         paste0("Logistic Regression (AUC = ", round(as.numeric(auc(logistic_roc)), 3), ")"),
         paste0("Decision Tree (AUC = ", round(as.numeric(auc(tree_roc)), 3), ")"),
         paste0("Random Forest (AUC = ", round(as.numeric(auc(rf_roc)), 3), ")")
       ),
       col = c("blue", "darkgreen", "red"), lwd = 2, cex = 0.8)
dev.off()

# 5. Compare model evaluation metrics
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(as.numeric(logistic_cm$overall["Accuracy"]),
               as.numeric(tree_cm$overall["Accuracy"]),
               as.numeric(rf_cm$overall["Accuracy"])),
  Precision = c(as.numeric(logistic_cm$byClass["Precision"]),
                as.numeric(tree_cm$byClass["Precision"]),
                as.numeric(rf_cm$byClass["Precision"])),
  Recall = c(as.numeric(logistic_cm$byClass["Recall"]),
             as.numeric(tree_cm$byClass["Recall"]),
             as.numeric(rf_cm$byClass["Recall"])),
  F1_Score = c(as.numeric(logistic_cm$byClass["F1"]),
               as.numeric(tree_cm$byClass["F1"]),
               as.numeric(rf_cm$byClass["F1"]))
)

comparison_long <- model_comparison %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1_Score),
               names_to = "Metric", values_to = "Score")

p5 <- ggplot(comparison_long,
             aes(x = Model, y = Score, fill = Metric)) +
  geom_col(position = "dodge") +
  coord_cartesian(ylim = c(0, 1)) +
  labs(title = "Performance Comparison of Machine Learning Models",
       x = "Model", y = "Score", fill = "Metric") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))
print(p5)
ggsave("visualizations/05_model_performance_comparison.png", p5,
       width = 9, height = 5)

# 6. Random Forest confusion matrix
rf_cm_df <- as.data.frame(rf_cm$table)
p6 <- ggplot(rf_cm_df,
             aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Random Forest Confusion Matrix",
       x = "Actual Class", y = "Predicted Class", fill = "Count") +
  theme_minimal()
print(p6)
ggsave("visualizations/06_random_forest_confusion_matrix.png", p6,
       width = 7, height = 5)

# 7. Class counts before and after SMOTE
smote_counts <- data.frame(
  Dataset = rep(c("Before SMOTE", "After SMOTE"), each = 2),
  Class = rep(c("Normal", "Failure"), 2),
  Count = c(
    as.numeric(table(train_data$Machine.failure)["Normal"]),
    as.numeric(table(train_data$Machine.failure)["Failure"]),
    as.numeric(table(smote_train$Machine.failure)["Normal"]),
    as.numeric(table(smote_train$Machine.failure)["Failure"])
  )
)

p7 <- ggplot(smote_counts,
             aes(x = Class, y = Count, fill = Dataset)) +
  geom_col(position = "dodge") +
  labs(title = "Class Distribution Before and After SMOTE",
       x = "Machine Condition", y = "Number of Records",
       fill = "Dataset") +
  theme_minimal()
print(p7)
ggsave("visualizations/07_smote_class_distribution.png", p7,
       width = 8, height = 5)

write.csv(model_comparison,
          "visualizations/model_performance_comparison.csv",
          row.names = FALSE)

cat("\nAdditional plots saved in the 'visualizations' folder.\n")

smote_cm <- confusionMatrix(
  data = smote_predictions,
  reference = test_smote$Machine.failure,
  positive = "Failure"
)

print(smote_cm)
