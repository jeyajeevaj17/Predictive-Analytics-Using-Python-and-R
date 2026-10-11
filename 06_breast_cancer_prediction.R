# Experiment 6: Breast cancer prediction using R
# Demonstration using the built-in breast cancer Wisconsin diagnostic data
# from the mlbench package. This is educational, not a clinical diagnostic tool.
if (!requireNamespace("mlbench", quietly = TRUE)) {
  install.packages("mlbench", repos = "https://cloud.r-project.org")
}
library(mlbench)
data(BreastCancer)

df <- BreastCancer[, -1]  # remove sample ID
df <- df[complete.cases(df), ]
df$Class <- factor(df$Class, levels = c("benign", "malignant"))

set.seed(42)
idx <- sample(seq_len(nrow(df)), size = floor(0.75 * nrow(df)))
train <- df[idx, ]
test <- df[-idx, ]

# Remove the constant/ID-like column if present and use all remaining predictors.
train$Class <- NULL
test$Class <- NULL
# Rebuild data frames with outcome included
train <- df[idx, ]
test <- df[-idx, ]

model <- glm(Class ~ ., data = train, family = binomial())
prob <- predict(model, newdata = test, type = "response")
pred <- factor(ifelse(prob >= 0.5, "malignant", "benign"),
               levels = c("benign", "malignant"))
actual <- test$Class

cat("BREAST CANCER CLASSIFICATION (EDUCATIONAL DATASET)\\n")
cat("Test records:", nrow(test), "\\n")
cat("Accuracy:", round(mean(pred == actual) * 100, 2), "%\\n")
cat("Confusion matrix (rows=actual, columns=predicted):\\n")
print(table(Actual = actual, Predicted = pred))
cat("First 10 predictions:\\n")
print(data.frame(Actual = actual[1:min(10, length(actual))],
                 Predicted = pred[1:min(10, length(pred))],
                 RiskProbability = round(prob[1:min(10, length(prob))], 3)))
