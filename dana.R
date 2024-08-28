library(ggplot2)
library(tidyr)
library(dplyr)
library(caret)
library(glmnet)
library(kernlab)
library(pROC)
set.seed(1)


d <- read.csv("/Users/dana/Desktop/Statistical Learning/alzheimers_disease_data.csv")
d <- as_tibble(d)
head(d)
dim(d)

# checking for missing value
sum(is.na(d)) 
# check for duplicate ID
length(unique(d$PatientID)) == dim(d)[1]
# delete unnecessary columns and do some processing
d <- d %>% 
  select(- c(DoctorInCharge, PatientID)) %>%
  mutate(Diagnosis = as.factor(Diagnosis)) %>%
  mutate(value = 1)  %>% spread(Ethnicity, value, fill = 0) %>%
  rename(Caucasian = "0", African = "1", Asian = "2") %>%
  select( -c("3"))

# EDA
## histogram for selected variables 
df_long <- d %>%
  select() %>%
  tidyr::gather(key = "variable", value = "value")
# 
# library(ggplot2)
# # Assuming df_long is your data frame, you can create the plot with:
# ggplot(df_long, aes(x = value, fill = variable)) +
#   geom_histogram(alpha = 0.5, bins = 20) +
#   facet_wrap(~variable, scale = "free") +
#   labs(x = "X-axis Label", y = "Y-axis Label", title = "Main Title", subtitle = "Subtitle Here") +
#   theme(legend.position = "none")
# ggplot(df_long, aes(x = value, fill = variable)) +
#   geom_histogram(alpha = 0.5, bins=20) +
#   facet_wrap(~variable, scale = "free") +
#   labs(x = "", y = "", title = "", subtitle = "") +
#   theme(legend.position = "none")


# split training data and test date. test data is use to evaluate and 
# not used for training
# Splitting the dataset into training and testing sets
train_size <- floor(0.9 * nrow(d))
train_id <- sample(seq_len(nrow(d)), size = train_size)

train <- d[train_id, ]
test <- d[-train_id, ]

# Creating 'data' directory if it doesn't exist
if (!dir.exists("data")) {
  dir.create("data")
}

# Saving the training and testing sets
saveRDS(train, "data/train.rds")
saveRDS(test, "data/test.rds")

# Loading the data back into R
train <- readRDS("data/train.rds")
test <- readRDS("data/test.rds")

# Optional: Print the first few rows to verify
head(train)
head(test)



# we use CV for all estimation
trctrl <- trainControl(method = "cv", number = 10)

# (1) (simple) logistic regression
logi_reg <- train(Diagnosis ~., data = train,
                  method = "glm",
                  family = binomial(),
                  preProcess = c("center", "scale"),
                  trControl = trctrl)
## about caret::train https://topepo.github.io/caret/model-training-and-tuning.html

summary(logi_reg)

## predict test data

logi_reg_pred_prob <- logi_reg$finalModel %>%
  predict(test, type = "response")


observed <- test$Diagnosis

logi_reg_pred <- tibble(
  prob = logi_reg_pred_prob
) %>% mutate(predicted = ifelse(prob >= 0.5, "1", "0"))


## confusion matrix
# table(logi_reg_pred$predicted, observed)
# Assuming 'logi_reg_pred$predicted' and 'observed' are vectors with predicted and actual values
conf_matrix <- table(Predicted = logi_reg_pred$predicted, Observed = observed)
print(conf_matrix)

## accuracy
logi_reg_pred %>%
  summarise(perc_correct = mean(observed == predicted))



#visualisation
# Install and load necessary packages
#install.packages(c("pROC", "reshape2", "ggplot2"))

library(pROC)
library(reshape2)
library(ggplot2)

# Assuming `logi_reg`, `test`, and other variables are already available

# Predictions
logi_reg_pred_prob <- logi_reg$finalModel %>%
  predict(test, type = "response")
observed <- test$Diagnosis

# # ROC Curve
roc_obj <- roc(observed, logi_reg_pred_prob)
roc_data <- data.frame(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)

ggplot(roc_data, aes(x = fpr, y = tpr)) +
  geom_line() +
  geom_abline(linetype = "dashed", color = "grey") +
  labs(title = "ROC Curve", x = "False Positive Rate (FPR)", y = "True Positive Rate (TPR)") +
  theme_minimal()
# Calculate AUC
auc_value <- auc(roc_obj)
print(auc_value)


# Confusion Matrix Heatmap
logi_reg_pred <- tibble(
  prob = logi_reg_pred_prob
) %>% mutate(predicted = ifelse(prob >= 0.5, "1", "0"))

observed <- test$Diagnosis

# Convert confusion matrix to a data frame
conf_matrix <- table(logi_reg_pred$predicted, observed)
conf_matrix_melt <- melt(as.matrix(conf_matrix))

# Print the column names to confirm
print(names(conf_matrix_melt))

# Adjust ggplot code according to the actual column names
ggplot(conf_matrix_melt, aes(x = Var1, y = observed, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Observed") +
  theme_minimal()
# 
# logi_reg_pred <- tibble(
#   prob = logi_reg_pred_prob
# ) %>% mutate(predicted = ifelse(prob >= 0.5, "1", "0"))
# 
# conf_matrix <- table(logi_reg_pred$predicted, observed)
# conf_matrix_melt <- melt(as.matrix(conf_matrix))
# 
# ggplot(conf_matrix_melt, aes(x = Var1, y = Var2, fill = value)) +
#   geom_tile() +
#   scale_fill_gradient(low = "white", high = "blue") +
#   labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Observed") +
#   theme_minimal()

# Probability Distribution Plot
logi_reg_pred_df <- data.frame(
  observed = as.factor(observed),
  predicted_prob = logi_reg_pred_prob
)

ggplot(logi_reg_pred_df, aes(x = predicted_prob, fill = observed)) +
  geom_density(alpha = 0.5) +
  labs(title = "Probability Distribution by Class", x = "Predicted Probability", y = "Density") +
  scale_y_continuous(limits = c(0, 5)) +  # Increase the upper limit of the y-axis as needed
  theme_minimal()


# (2) logistic regression with L2 regularization

logi_reg_ridge <- train(Diagnosis ~., data = train,
                        method = "glmnet",
                        preProcess = c("center","scale"),
                        trControl = trctrl,
                        tuneGrid = expand.grid(alpha = 0, 
                                               lambda = c(seq(0.1, 2, by = 0.1),
                                                          seq(2, 10, by = 0.5))))



logi_reg_ridge_pred <- logi_reg_ridge %>%
  predict(test, type = "prob") %>%
  as_tibble() %>%
  setNames(c("False", "True")) %>%
  mutate(predicted = ifelse(True >= 0.5, "1", "0"),
         prob = True)

table(logi_reg_ridge_pred$predicted, observed)

logi_reg_ridge_pred %>%
  summarise(perc_correct = mean(observed == predicted))

## value of lambda
logi_reg_ridge$bestTune
#1
# # ROC Curve Ridge
roc_obj_ridge <- roc(observed, logi_reg_ridge_pred$prob)
roc_data_ridge <- data.frame(fpr = 1 - roc_obj_ridge$specificities, tpr = roc_obj_ridge$sensitivities)

ggplot(roc_data_ridge, aes(x = fpr, y = tpr)) +
  geom_line() +
  geom_abline(linetype = "dashed", color = "grey") +
  labs(title = "ROC Curve", x = "False Positive Rate (FPR)", y = "True Positive Rate (TPR)") +
  theme_minimal()
# Calculate AUC
auc_value_ridge <- auc(roc_obj_ridge)
print(auc_value_ridge)

