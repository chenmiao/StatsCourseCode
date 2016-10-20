#use naive bayes on the extracted features (with avg ratings as response/dependent variable)

library(e1071)
#library(caret)
#library(klaR)
#library(ROCR)

setwd("/home/miao/Documents/StatsCourses/S626_Spring2016/Stat626_TermProject_Spring2016")
#sampledata <- read.csv("sample_avg.csv")
sampledata <- read.csv("bayes.csv")
sampledata$stars <- factor(sampledata$stars)
#dim(sampledata)

#### code for training-test set
split_data <- split_sample(sampledata, 0.75)

model.nb <- naiveBayes(stars ~ topic1 + topic2 + topic3 + topic4 + topic5 + topic6 + topic7 + topic8 + topic9 + topic10 + state + sentiment_score + review_count + single_category ,data=split_data$trainset)

pred <- predict(model.nb, split_data$testset)

table(split_data$testset$stars, pred)


#building the naive bayes model, not using smoothing
# model.nb <- naiveBayes(stars ~ topic1 + topic2 + topic3 + topic4 + topic5 + topic6 + topic7 + topic8 + topic9 + topic10 + state + sentiment_score + review_count + single_category ,data=sampledata)
# 
# #make prediction using the NB model
# pred <- predict(model.nb, sampledata)
# #print out the confusion matrix
# table(sampledata$stars, pred)

cm <- as.matrix(table(observed=split_data$testset$stars, predicted=pred))

#evaluating the model
n <- sum(cm)
nc <- nrow(cm)
diag <- diag(cm)
rowsums <- apply(cm, 1, sum)
colsums <- apply(cm, 2, sum)
p <- rowsums / n
q <- colsums / n

accuracy <- sum(diag) / n
precision <- diag / colsums
recall <- diag / rowsums
f1 <- 2 * precision * recall / (precision + recall) 
#print out precision, recall, and F1-measure for each class
data.frame(precision, recall, f1)

#macro-averaged metrics
macroPrecision = mean(precision)
macroRecall = mean(recall)
macroF1 = mean(f1)
#print out the precision/recall/f1_measure by averaging out these measures over the classes
data.frame(macroPrecision, macroRecall, macroF1)


expAccuracy = sum(p*q)
kappa = (accuracy - expAccuracy) / (1 - expAccuracy)
kappa


split_sample <- function(sample, percentage) {
  ## 75% of the sample size
  smp_size <- floor(percentage * nrow(sample))
  
  ## set the seed to make your partition reproductible
  set.seed(1)
  train_ind <- sample(seq_len(nrow(sample)), size = smp_size)
  
  train <- sample[train_ind, ]
  test <- sample[-train_ind, ]
  
  list(trainset=train,testset=test)
}


#for unknown reason, KlaR package NaiveBayes method returns me many warnings, which is weird because it basically uses the same NaiveBayes method as in e1071 package.

#using cross-validation with NaiveBayes model, by using KlaR and caret packages
# train_control <- trainControl(method = "cv", number = 10)
# grid <- expand.grid(.fL = c(0), .usekernel = c(FALSE))
# model <-
#   train(
#     Species ~ .,
#     data = iris,
#     trControl = train_control,
#     method = "nb",
#     tuneGrid = grid
#   )
# print(model)
# 
# x = iris[, -5]
# y = iris$Species
# model = train(x, y, 'nb', trControl = trainControl(method = 'cv', number =
#                                                      10))
# predict(model$finalModel, x)$class
# 
# model.nb <-
#   train(
#     stars ~ topic1 + topic2 + topic3 + topic4 + topic5 + topic6 + topic7 + topic8 + topic9 + topic10 + state + sentiment_score + review_count + single_category,
#     data = sampledata,
#     method = 'nb'
#   )

#reference: http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html  
#https://cran.r-project.org/web/packages/e1071/e1071.pdf
#http://joshwalters.com/2012/11/27/naive-bayes-classification-in-r.html
#http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
