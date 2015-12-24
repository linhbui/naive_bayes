# Part 1
# k-nearest neighbors for wine-quality data. Use cross-validation to determine the best k
install.packages("class")
library(class)
rm(list=ls())
data <- read.csv('winequality-red.csv', header=TRUE, sep = ';')
attributes <- data[,1:11]
qualityScore <- data[,12]

errKNN <- rep(0,20)
for(k in seq(from=1,to=20)){
  predictedValues <- knn.cv(attributes,qualityScore,k=k)
  error <- 1-sum(abs(qualityScore == predictedValues))/length(predictedValues)
  errKNN[k] <- error   
}
errKNN
plot(errKNN)
minErrKNN <- min(errKNN) #0.3846154

# compare with ridge regression
library(ridge)
errRidge <- rep(0, 20)
lambdas <- rep(0, 20)
for(iLambda in seq(from = 0, to = 20)){ 
  exp <- (+2 -4*(iLambda/20))
  xlambda <- 10^exp
  model <- linearRidge(qualityScore~., data=as.data.frame(attributes),lambda=xlambda)
  predictedValues <- predict(model,attributes)
  rounded <- round(predictedValues)
  error <- 1-sum(abs(qualityScore == rounded))/length(predictedValues)
  errRidge[iLambda] <- error
  lambdas[iLambda] <- xlambda
}

errRidge
plot(lambdas,errRidge)
minErrRidge <- min(errRidge) # 0.4058787

# Conclusion: KNN is a better classifier for this data set

# Part 2
# K-nearest neighbor on Iris data set
rm(list=ls())
library(class)
data <- read.csv('irisdata.csv', header=FALSE, sep = ',')
data(iris)
attributes <- iris[,1:4]
labels <- iris$Species

errKNN <- rep(0,20)
for(k in seq(from=1,to=20)){
  predictedValues <- knn.cv(attributes,labels,k=k)
  error <- 1-sum(abs(labels == predictedValues))/length(predictedValues)
  errKNN[k] <- error   
}
errKNN
plot(errKNN)
minErrKNN <- min(errKNN) #0.02013423

# Naive-Bayes
install.packages('klaR')
library(klaR)
data(iris)
model <- NaiveBayes(Species~., data=iris)
predictedValues <- predict(model)

errNB <- 1 - sum(predictedValues$class == iris$Species)/length(iris$Species)
errNB # 0.04

# Here, KNN is a better model compared to Naive-Bayes

# Part 3
# Use Naive-Bayes for the wine-quality data set
rm(list=ls())
library(klaR)
data <- read.csv('winequality-red.csv', header=TRUE, sep = ';')
attributes <- data[,1:11]
qualityScore <- as.factor(data[,12])
data[,12] <- qualityScore

model <- NaiveBayes(qualityScore~., data=data)
predictedValues <- predict(model)

errNB <- 1 - sum(predictedValues$class == qualityScore)/length(qualityScore)
errNB #0.01813634

# Part 4
# Classify the sonar data using Naive Bayes
rm(list=ls())
train <- read.table("sonar_train.csv",sep = ",",header = FALSE)
test <- read.table("sonar_test.csv",sep = ",",header = FALSE)

attributes <- train[,1:60]
labels <- as.factor(train[,61])
train[,61] <- labels

model <- NaiveBayes(labels~., data=train)
predictedTrain <- predict(model)
predictedTest <- predict(model,test)

errNB <- 1 - sum(predictedTest$class == test[,61])/length(test[,61])
errNB # 0.7948718

# Naive Bayes did a much poorer job to classify this data set compared to other methods (decision trees, random forest etc.)
# because the attributes in this data set are highly correlate with each other

# Part 5
# Use KNN with mixtureSimData
rm(list=ls())
data <- read.table(file="mixtureSimData.data")

dataMat <- matrix(0.0,200,2)
dataMat[,1] <- data[1:200,1]
dataMat[,2] <- data[201:400,1]
labels <- rep(1.0,200)  #labels is the class or target
labels[101:200] <- 2.0

Err <- rep(0,50)
for(k in seq(from=1,to=50)){
  out <- knn.cv(dataMat, labels, k=k)
  Error <- 1-sum(abs(labels == out))/length(out)
  Err[k] <- Error   
}
Err
plot(Err)
min(Err)  #[1] 0.165
which(Err == min(Err)) #[1]  5 13 










ooo