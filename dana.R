library(ggplot2)
library(tidyr)
library(dplyr)
library(caret)
library(glmnet)
library(kernlab)
set.seed(1)


d <- read.csv("data/alzheimers_disease_data.csv")
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

ggplot(df_long, aes(x = value, fill = variable)) +
  geom_histogram(alpha = 0.5, bins=20) +
  facet_wrap(~variable, scale = "free") +
  labs(x = "", y = "", title = "", subtitle = "") +
  theme(legend.position = "none")


# split training data and test date. test data is use to evaluate and 
# not used for training
# train_size <- floor(0.9 * nrow(d))
# train_id <- sample(seq_len(nrow(d)), size = train_size)

# train <- d[train_id, ]
# test <- d[-train_id, ]

# save train data and test data such that we are using same set of data.
saveRDS(train, "data/train.rds")
saveRDS(test, "data/test.rds")

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

summary(logi_reg)

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
