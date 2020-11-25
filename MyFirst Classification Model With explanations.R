#Classification Model 

library(caret)

# attach the iris dataset to the environment
data(iris)
# rename the dataset
dataset <- iris
# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

            #Summarize Dataset 

# dimensions of dataset
dim(dataset)


# list types for each attribute
sapply(dataset, class)


# take a peek at the first 5 rows of the data
head(dataset)

# list the levels for the class
levels(dataset$Species)



# summarize the class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

# summarize attribute distributions
summary(dataset)

                      #VISUALIZE DATA
# Univariate Plots

# split input and output
x <- dataset[,1:4]
y <- dataset[,5]

#Given that the input variables are numeric, we can create box and whisker plots of each.


# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

#We can also create a barplot of the Species class variable to get a graphical representation  
# of the class distribution (generally uninteresting in this case because they're even).


# barplot for class breakdown
plot(y)


                    #Evaluate Some Algorithms

#This will split our dataset into 10 parts, train in 9 and test on 1 and release for 
#all combinations of train-test splits. We will also repeat the process 3 times for 
#each algorithm with different splits of the data into 10 groups, in an effort to get 
#a more accurate estimate.

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#We are using the metric of "Accuracy" to evaluate models. This is a ratio of the number of 
#correctly predicted instances in divided by the total number of instances in the dataset 
#multiplied by 100 to give a percentage (e.g. 95% accurate).


                   #BUILDING MODELS

#We don't know which algorithms would be good on this problem or what configurations to use. 
#We get an idea from the plots that some of the classes are partially linearly separable in some 
#dimensions, so we are expecting generally good results.

#Linear Discriminant Analysis (LDA)     -   simple linear
#Classification and Regression Trees (CART).    -   nonlinear
#k-Nearest Neighbors (kNN).   -     nonlinear
#Support Vector Machines (SVM) with a linear kernel.    -   complex nonlinear
#Random Forest (RF)   -   complex nonlinear

#We reset the random number seed before reach run to ensure that the evaluation of each 
#algorithm is performed using exactly the same data splits. It ensures the results are 
#directly comparable.

                #Lets build our 5 models
# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)


               #Select Best Model
#We now have 5 models and accuracy estimations for each. We need to compare the models 
#to each other and select the most accurate.

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)


#We can also create a plot of the model evaluation results and compare the spread and the mean 
#accuracy of each model. There is a population of accuracy measures for each algorithm because 
#each algorithm was evaluated 10 times (10 fold cross validation).

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)

              # Make Predictions

#The LDA was the most accurate model. Now we want to get an idea of the accuracy of the model 
#on our validation set.This will give us an independent final check on the accuracy of the best 
#model. It is valuable to keep a validation set just in case you made a slip during modelling  
#such as overfitting to the training set or a data leak. Both will result in an overly optimistic 
#result. We can run the LDA model directly on the validation set and summarize the results in a 
#confusion matrix.

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)