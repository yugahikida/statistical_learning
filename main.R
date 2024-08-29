library(ggplot2)
library(tidyr)
library(dplyr)
library(caret)
library(glmnet)
library(kernlab)
library(scoringRules)
library(jtools)
library(broom)
library(pROC)
set.seed(1)


d <- read.csv("data/alzheimers_disease_data.csv")
d <- as_tibble(d)
head(d)
dim(d)

# checking for missing value
sum(is.na(d))
dim(d)
# check for duplicate ID
length(unique(d$PatientID)) == dim(d)[1]
# delete unnecessary columns and do some processing
d <- d %>% 
  select(- c(DoctorInCharge, PatientID)) %>%
  mutate(Diagnosis = as.factor(Diagnosis)) %>%
  mutate(value = 1)  %>% spread(Ethnicity, value, fill = 0) %>%
  rename(Caucasian = "0", African = "1", Asian = "2") %>%
  select( -c("3"))

dim(d)


# split training data and test date. test data is use to evaluate and 
# not used for training
train_size <- floor(0.9 * nrow(d))
train_id <- sample(seq_len(nrow(d)), size = train_size)

train <- d[train_id, ]
test <- d[-train_id, ]

# save train data and test data such that we are using same set of data.
# saveRDS(train, "data/train.rds")
# saveRDS(test, "data/test.rds")

readRDS("data/train.rds")
readRDS("data/test.rds")

# we use CV for all estimation
trctrl <- trainControl(method = "cv", number = 10)

# (1) (simple) logistic regression
logi_reg <- train(Diagnosis ~., data = train,
                  method = "glm",
                  family = binomial(),
                  preProcess = c("center", "scale"),
                  trControl = trctrl)
## about caret::train https://topepo.github.io/caret/model-training-and-tuning.html


## visualize significant coefficients
final_model <- logi_reg$finalModel
tidy_model <- broom::tidy(final_model, conf.int = TRUE, conf.level = 0.95)
tidy_model <- tidy_model %>% 
  filter(term %in% c("MMSE", "FunctionalAssessment", "MemoryComplaints",
                     "BehavioralProblems", "ADL"))

ggplot(tidy_model, aes(x = term, y = estimate)) +
  geom_point(size = 4) +  # Plot the coefficients as points
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +  # Add error bars for the confidence intervals
  labs(title = "",
       x = "Predictor",
       y = "Coefficient Estimate") +
  coord_flip() +
  theme_minimal()

## predict test data
logi_reg_pred_prob <- logi_reg$finalModel %>%
  predict(test, type = "response")

observed <- test$Diagnosis

logi_reg_pred <- tibble(
  prob = logi_reg_pred_prob
) %>% mutate(predicted = ifelse(prob >= 0.5, "1", "0"))
  

## confusion matrix
table(logi_reg_pred$predicted, observed)

## accuracy
logi_reg_pred %>%
  summarise(perc_correct = mean(observed == predicted))

## ROC
roc_obj <- roc(observed, logi_reg_pred_prob)
roc_data <- data.frame(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)

ggplot(roc_data, aes(x = fpr, y = tpr)) +
  geom_line() +
  geom_abline(linetype = "dashed", color = "grey") +
  labs(title = "ROC Curve", x = "False Positive Rate (FPR)", y = "True Positive Rate (TPR)") +
  theme_minimal()
# Calculate AUC
auc_value <- auc(roc_obj)

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

## visualization of comparision with logistic regression
coef_logistic <- as.data.frame(logi_reg$finalModel$coefficients)
coef_ridge <- as.data.frame(as.matrix(coef(logi_reg_ridge$finalModel, s = 0.1)))
df1 <- data.frame(Variable = rownames(coef_logistic), Coefficient = coef_logistic[,1], Model = "Logistic")
df2 <- data.frame(Variable = rownames(coef_ridge), Coefficient = coef_ridge[,1], Model = "Ridge")
df <- rbind(df1, df2)

ggplot(df, aes(x = Variable, y = Coefficient, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "",
       x = "Variables",
       y = "Value of coefficeints") +
  theme_minimal() +
  coord_flip()


# (3) Support vector machine
trctrl_2 <- trainControl(method = "cv", number = 10, classProbs = TRUE)


svm_model <- train(make.names(Diagnosis) ~., data = train, 
                   method = "svmRadial",
                   preProcess = c("center","scale"),
                   trControl = trctrl_2, 
                   tuneLength = 10)

svm_pred_converted <- svm_model %>% predict(test)
levels(svm_pred_converted) <- c("0", "1")

svm_pred <- tibble(
  predicted = svm_pred_converted,
  prob = svm_model %>% predict(test, type = "prob") %>% 
    select(X1) %>% unname() %>% unlist()
)


# about probabilistic forecast with SVM 
# https://stackoverflow.com/questions/63749263/different-results-for-svm-using-caret-in-r-when-classprobs-true

table(svm_pred$predicted, observed)

svm_pred %>%
  summarise(perc_correct = mean(observed == predicted))

# (4) Gaussian process for classification

gp <- train(Diagnosis ~., data = train,
            method = "gaussprRadial",
            type = "classification",
            preProcess = c("center","scale"),
            trControl = trctrl,
            tuneLength = 10)

gp_pred <- tibble(
  predicted = gp %>% predict(test),
  prob = gp %>% predict(test, type = "prob")
)

table(gp_pred$predicted, observed)

gp_pred %>%
  summarise(perc_correct = mean(observed == predicted))

# Evaluation of all models
## label prediction
calculate_metrics <- function(predicted, observed) {
  predicted <- factor(predicted, levels = c(0, 1))
  observed <- factor(observed, levels = c(0, 1))
  
  cm <- table(Predicted = predicted, Observed = observed)
  
  tp <- cm[2, 2]; tn <- cm[1, 1]; fp <- cm[2, 1]; fn <- cm[1, 2]
  
  accuracy <- (tp + tn) / sum(cm)
  precision <- tp / (tp + fp)
  sensitivity <- tp / (tp + fn)  # Also known as recall or true positive rate
  specificity <- tn / (tn + fp)  # True negative rate

  metrics <- list(
    Accuracy = accuracy,
    Precision = precision,
    Sensitivity = sensitivity,
    Specificity = specificity
  )
  
  return(metrics)
}

calculate_metrics(logi_reg_pred$predicted, observed) # logistic regression
calculate_metrics(logi_reg_ridge_pred$predicted, observed) # Ridge logistic regression
calculate_metrics(svm_pred$predicted, observed) # SVM
calculate_metrics(gp_pred$predicted, observed) # GP

# scoringrule
## https://www.rdocumentation.org/packages/scoringRules/versions/1.1.1/topics/scores_binom

## Calculate CRPS for each forecast
crps_values <- crps(y = as.numeric(observed) - 1, family = "binom", size = 1, prob = logi_reg_pred$prob) # logistic regression
crps_values <- crps(y = as.numeric(observed) - 1, family = "binom", size = 1, prob = logi_reg_ridge_pred$prob) # Ridge logistic regression
crps_values <- crps(y = as.numeric(observed) - 1, family = "binom", size = 1, prob = svm_pred$prob) # SVM
crps_values <- crps(y = as.numeric(observed) - 1, family = "binom", size = 1, prob = gp_pred$prob[[2]]) #GP

## Calculate the average CRPS
average_crps <- mean(crps_values)
average_crps

## visualization of GP prior
simulate_gp_prior <- function(x, sigma_f = 1, length_scale = 1, n_samples = 5) {
  ### Define the RBF kernel function
  rbf_kernel <- function(x1, x2) {
    sigma_f^2 * exp(-sum((x1 - x2)^2) / (2 * length_scale^2))
  }
  
  ### Create covariance matrix using the RBF kernel
  cov_matrix <- outer(x, x, Vectorize(function(xi, xj) rbf_kernel(xi, xj)))
  
  ### Simulate samples from the multivariate normal distribution
  samples <- MASS::mvrnorm(n_samples, mu = rep(0, length(x)), Sigma = cov_matrix)
  
  return(samples)
}

## Define the input space
x <- seq(-5, 5, length.out = 100)

## Simulate GP priors with different values for the length_scale parameter
length_scales <- c(0.5, 1, 2)
gp_samples <- lapply(length_scales, function(ls) simulate_gp_prior(x, length_scale = ls))

plot_color <- rgb(64/255, 64/255, 64/255, alpha = 0.8)

## Plot the results
par(mfrow = c(1, length(length_scales)))

for (i in 1:length(length_scales)) {
  matplot(x, t(gp_samples[[i]]), type = "l", lty = 1, col = plot_color,
          main = paste("sigma =", length_scales[i]),
          ylab = "f(x)", xlab = "x")
}


