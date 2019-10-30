if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(readr)) install.packages("readr")

# Read in 1994 Census Data file from Kaggle

census_income <- read_csv("./censusincome.csv")

# Select and wrangle outcome variable (income)
# and 10 predictor variables/features
# 4 predictors (sex, race, age, marital.status) are demographic
# 4 predictors (education, occupation, hours.per.week, workclass) are more action/achievement-based
# 2 predictors (capital.gain, capital.loss) are monetary

income_data <- census_income %>%
  mutate(marital.status.cond=ifelse((marital.status=="Married-civ-spouse" |
                                      marital.status=="Married-AF-spouse" |
                                      marital.status=="Married-spouse-absent"),
                                    "Married", marital.status)) %>%
  mutate(education.cond=ifelse((education=="Preschool" |
                                  education=="1st-4th" |
                                  education=="5th-6th" |
                                  education=="7th-8th"),
                               "Below-HS", education)) %>%
  mutate(education.cond=ifelse((education.cond=="9th" |
                                  education.cond=="10th" |
                                  education.cond=="11th" |
                                  education.cond=="12th"),
                               "Some-HS", education.cond)) %>%
  mutate(workclass.cond=ifelse((workclass=="Federal-gov" |
                                  workclass=="State-gov" |
                                  workclass=="Local-gov"),
                               "Government", workclass)) %>%
  mutate(workclass.cond=ifelse((workclass.cond=="Without-pay" |
                                  workclass.cond=="Never-worked"),
                               "No-Pay", workclass.cond)) %>%
  mutate(workclass.cond=ifelse((workclass.cond=="Self-emp-inc" |
                                  workclass.cond=="Self-emp-not-inc"),
                               "Self-employed", workclass.cond)) %>%
  mutate(workclass.cond=ifelse((workclass.cond=="?"),
                               "Unknown", workclass.cond)) %>%
  mutate(occupation.cond=ifelse((occupation=="?" |
                                   occupation=="Armed-Forces"),
                               "Unknown-or-Armed-Forces", occupation)) %>%
  select(income, education.cond, hours.per.week, occupation.cond,
         race, sex, marital.status.cond, age, workclass.cond,
         capital.gain, capital.loss)

# Create train and validation sets

set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y=income_data$income, times=1, p=0.1, list=FALSE)
train <- income_data[-test_index, ]
validation <- income_data[test_index, ]

# Split the train set to create second train set for cross-validation purposes

set.seed(1, sample.kind="Rounding")

test_index2 <- createDataPartition(y=train$income, times=1, p=0.1, list=FALSE)
train1 <- train[-test_index2, ]
train2 <- train[test_index2, ]

# Investigate data
# Group by different variables to see group probabilities of making <=50K

train1 %>%
  mutate(income_num=ifelse(income=="<=50K",1,0)) %>%
  group_by(sex) %>%
  summarize(income_prob=mean(income_num))

train1 %>%
  mutate(income_num=ifelse(income=="<=50K",1,0)) %>%
  group_by(race) %>%
  summarize(income_prob=mean(income_num))

train1 %>%
  mutate(income_num=ifelse(income=="<=50K",1,0)) %>%
  group_by(marital.status.cond) %>%
  summarize(income_prob=mean(income_num))

train1 %>%
  mutate(income_num=ifelse(income=="<=50K",1,0)) %>%
  group_by(education.cond) %>%
  summarize(income_prob=mean(income_num))

train1 %>%
  mutate(income_num=ifelse(income=="<=50K",1,0)) %>%
  group_by(occupation.cond) %>%
  summarize(income_prob=mean(income_num))

train1 %>%
  mutate(income_num=ifelse(income=="<=50K",1,0)) %>%
  group_by(workclass.cond) %>%
  summarize(income_prob=mean(income_num))

# Use density plots to view income breakdown for continuous variables

train1 %>% ggplot(aes(hours.per.week, col=income)) + geom_density()

train1 %>% ggplot(aes(age, col=income)) + geom_density()

train1 %>% ggplot(aes(capital.gain, col=income)) + geom_density()

train1 %>% ggplot(aes(capital.loss, col=income)) + geom_density()

# Begin modeling
# Train rpart models to see how 3 different groups of predictors perform
# Demographics model

set.seed(1,sample.kind="Rounding")

train_rpart_dem <- train(income~sex+race+marital.status.cond+age,
                         method="rpart",
                         data=train1)
train_rpart_dem
train_rpart_dem$finalModel
varImp(train_rpart_dem$finalModel)

plot(train_rpart_dem$finalModel)
text(train_rpart_dem$finalModel)

predict_rpart_dem <- predict(train_rpart_dem, train2)
accuracy_rpart_dem <- mean(predict_rpart_dem==train2$income)

accuracy_rpart_dem

confusionMatrix(data=factor(predict_rpart_dem),
                reference=factor(train2$income))

# Use F-1 score as a measure of balanced accuracy

F_1_rpart_dem <- F_meas(data=factor(predict_rpart_dem),
                        reference=factor(train2$income))
F_1_rpart_dem

# Achievement model

set.seed(1, sample.kind="Rounding")

train_rpart_ach <- train(income~education.cond+occupation.cond
                         +hours.per.week+workclass.cond,
                         method="rpart",
                         data=train1)

train_rpart_ach
train_rpart_ach$finalModel
varImp(train_rpart_ach$finalModel)

plot(train_rpart_ach$finalModel)
text(train_rpart_ach$finalModel)

predict_rpart_ach <- predict(train_rpart_ach, train2)
accuracy_rpart_ach <- mean(predict_rpart_ach==train2$income)

accuracy_rpart_ach

confusionMatrix(data=factor(predict_rpart_ach),
                reference=factor(train2$income))

F_1_rpart_ach <- F_meas(data=factor(predict_rpart_ach),
                        reference=factor(train2$income))

F_1_rpart_ach

# Monetary model

set.seed(1, sample.kind="Rounding")

train_rpart_mo <- train(income~capital.gain+capital.loss,
                         method="rpart",
                         data=train1)

train_rpart_mo
train_rpart_mo$finalModel
varImp(train_rpart_mo$finalModel)

plot(train_rpart_mo$finalModel)
text(train_rpart_mo$finalModel)

predict_rpart_mo <- predict(train_rpart_mo, train2)
accuracy_rpart_mo <- mean(predict_rpart_mo==train2$income)

accuracy_rpart_mo

confusionMatrix(data=factor(predict_rpart_mo),
                reference=factor(train2$income))

F_1_rpart_mo <- F_meas(data=factor(predict_rpart_mo),
                        reference=factor(train2$income))

F_1_rpart_mo

# Combine all predictors into combined rpart model

set.seed(1, sample.kind="Rounding")

train_rpart <- train(income~.,
                      method="rpart",
                      tuneGrid=data.frame(cp=seq(0.0,0.2,len=25)),
                      data=train1)

train_rpart
train_rpart$finalModel
varImp(train_rpart$finalModel)

plot(train_rpart$finalModel)
text(train_rpart$finalModel)

predict_rpart <- predict(train_rpart, train2)
accuracy_rpart <- mean(predict_rpart==train2$income)

accuracy_rpart

confusionMatrix(data=factor(predict_rpart),
                reference=factor(train2$income))

F_1_rpart <- F_meas(data=factor(predict_rpart),
                        reference=factor(train2$income))

F_1_rpart

# Partition the data further to train random forest and k-nearest neighbors models

set.seed(1, sample.kind="Rounding")

test_index3 <- createDataPartition(y=train1$income, times=1, p=0.75, list=FALSE)
trainrf <- train1[-test_index3, ]
trainrf2 <- train1[test_index3, ]

# Train Random Forest model

set.seed(1, sample.kind="Rounding")

train_rf <- train(income~.,
                  method="rf",
                  data=trainrf)

train_rf$finalModel

varImp(train_rf)
plot(train_rf)

predict_rf<- predict(train_rf, train2)
accuracy_rf <- mean(predict_rf==train2$income)

accuracy_rf

F_1_rf <- F_meas(data=factor(predict_rf),
                        reference=factor(train2$income))

F_1_rf

# Train K-Nearest Neighbors model

train_knn <- train(income~.,
                   method="knn",
                   tuneGrid=data.frame(k=seq(1,15,1)),
                   data=trainrf)

train_knn$finalModel

predict_knn <- predict(train_knn, train2)
accuracy_knn <- mean(predict_knn==train2$income)

accuracy_knn

F_1_knn <- F_meas(data=factor(predict_knn),
                 reference=factor(train2$income))

F_1_knn

# Train additional models (glm, lda, qda) to see if accuracy improves
# Glm model

train_glm <- train(income~.,
                   method="glm",
                   data=train1)

train_glm$finalModel

predict_glm <- predict(train_glm, train2)
accuracy_glm <- mean(predict_glm==train2$income)

accuracy_glm

varImp(train_glm)

F_1_glm <- F_meas(data=factor(predict_glm),
                 reference=factor(train2$income))

F_1_glm

# Lda model

train_lda <- train(income~.,
                   method="lda",
                   data=train1)

train_lda$finalModel

predict_lda <- predict(train_lda, train2)
accuracy_lda <- mean(predict_lda==train2$income)

accuracy_lda

F_1_lda <- F_meas(data=factor(predict_lda),
                  reference=factor(train2$income))

F_1_lda

# Qda model
# Qda model won't work with workclass.cond variable even after wrangling in multiple ways
# Qda accepts all other variables as predictors

train_qda <- train(income~sex+marital.status.cond+race+age
                   +education.cond+hours.per.week+occupation.cond
                   +capital.gain+capital.loss,
                   method="qda",
                   data=train1)

train_qda$finalModel

predict_qda <- predict(train_qda, train2)
accuracy_qda <- mean(predict_qda==train2$income)

accuracy_qda

F_1_qda <- F_meas(data=factor(predict_qda),
                  reference=factor(train2$income))

F_1_qda

# Based on accuracy and F1 results, ensemble should include: glm, lda, knn, combined rpart, and rf
# Create ensemble model and compare accuracy and F-1 scores across all models

predictions <- bind_cols(glm=predict_glm, lda=predict_lda,
                         rpart=predict_rpart,
                         rf=predict_rf, knn=predict_knn)

predictions_num <- ifelse(predictions=="<=50K",1,0)
ensemble <- rowSums(predictions_num)
predict_ensemble <- ifelse(ensemble>=3, "<=50K", ">50K")
accuracy_ensemble <- mean(predict_ensemble==train2$income)
F_1_ensemble <- F_meas(data=factor(predict_ensemble),
                       reference=factor(train2$income))

accuracy_ensemble
F_1_ensemble

confusionMatrix(data=factor(predict_ensemble),
                            reference=factor(train2$income))

accuracy <- bind_cols(glm=accuracy_glm, lda=accuracy_lda,
                      rpart=accuracy_rpart,
                      rf=accuracy_rf, knn=accuracy_knn,
                      ensemble=accuracy_ensemble)

accuracy

F_1 <- bind_cols(glm=F_1_glm, lda=F_1_lda,
                               rpart=F_1_rpart,
                               rf=F_1_rf, knn=F_1_knn,
                               ensemble=F_1_ensemble)

F_1

# The ensemble model is an improvement for both accuracy and F-1 (balanced accuracy)
# Test final model (ensemble) against validation set

predict_glm_v <- predict(train_glm, validation)
accuracy_glm_v <- mean(predict_glm_v==validation$income)
F_1_glm_v <- F_meas(data=factor(predict_glm_v),
                       reference=factor(validation$income))

predict_lda_v <- predict(train_lda, validation)
accuracy_lda_v <- mean(predict_lda_v==validation$income)
F_1_lda_v <- F_meas(data=factor(predict_lda_v),
                    reference=factor(validation$income))

predict_rpart_v <- predict(train_rpart, validation)
accuracy_rpart_v <- mean(predict_rpart_v==validation$income)
F_1_rpart_v <- F_meas(data=factor(predict_rpart_v),
                    reference=factor(validation$income))

predict_rf_v <- predict(train_rf, validation)
accuracy_rf_v <- mean(predict_rf_v==validation$income)
F_1_rf_v <- F_meas(data=factor(predict_rf_v),
                      reference=factor(validation$income))

predict_knn_v <- predict(train_knn, validation)
accuracy_knn_v <- mean(predict_knn_v==validation$income)
F_1_knn_v <- F_meas(data=factor(predict_knn_v),
                   reference=factor(validation$income))

predictions_v <- bind_cols(glm=predict_glm_v, lda=predict_lda_v,
                         rpart=predict_rpart_v,
                         rf=predict_rf_v, knn=predict_knn_v)

predictions_num_v <- ifelse(predictions_v=="<=50K",1,0)
ensemble_v <- rowSums(predictions_num_v)
predict_ensemble_v <- ifelse(ensemble_v>=3, "<=50K", ">50K")
accuracy_ensemble_v <- mean(predict_ensemble_v==validation$income)
F_1_ensemble_v <- F_meas(data=factor(predict_ensemble_v),
                       reference=factor(validation$income))

confusionMatrix(data=factor(predict_ensemble_v),
                reference=factor(validation$income))

accuracy_ensemble_v
F_1_ensemble_v
