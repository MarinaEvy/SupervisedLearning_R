#This is a KNN algorithm with PPM imputation for the missing values
# I impute the missing values with PPM - I do not conduct listwise deletion.
# I standardize the dataset.


install.packages("caret") #Download te package for splitting the dataset into training and testing
install.packages("ggplot2") #Download this package in case I want to plot some features
install.packages("rattle") #Download this to plot the decision tree
install.packages("RColorBrewer") #Download this to make the tree look fancy
install.packages("lattice") #Care package requires it
install.packages("mice") # data imputation
install.packages("dummies") # binary transformation of categorical variables
install.packages("class") # KNN package
install.packages("e1071")
install.packages("rpart")


#Activate the packages
library(caret)
library(ggplot2)
library(rattle)
library(RColorBrewer)
library(lattice)
library(mice)
library(dummies)
library(class)
library(e1071)
library(rpart)


set.seed(4000) #Let's set the seed for reproducibility
setwd("C:/Users/mjohnson6/Desktop") #let's set the working directory where the dataset is
df=read.csv("CKD.csv", header = TRUE, dec=",", na.strings = c(""," ","NA")) #read the csv file and save it to the dataframe object df

#for some reason, R reads the variales below as factors. have to convert them to numericals
numericVectors<-c("bu", "sc", "sod", "pot", "hemo", "rbcc", "pcv") 
for (n in 1:7){
  x=numericVectors[n]
  df[,x]<-as.numeric(levels(df[,x]))[df[,x]] #set some factor variables as numeric variables 
}

#for some reason, R reads the variales below as number so have to convert them to factors.
factorVectors<-c("al", "su")
for (n in 1:2){
  x=factorVectors[n]
  df[, x]<-as.factor(df[,x])
}

summary(df) # let's check the dataset
str(df) #let's see if all the variables are orrectly identified as factors and numbers.
classColumnID1<-dim(df)[2]


#Let's do ppm imputation and binary transformation for categorical data
imputedDF<- mice(df,m=1,maxit=50,meth='pmm',seed=500) #imputation with pmm
df<-complete(imputedDF) #replacing the NA's
preObj <- preProcess(df[, -classColumnID1], method=c("center", "scale")) #standardization
df1 <- predict(preObj, df[, -classColumnID1]) #standardization
df1<-dummy.data.frame(df1, names = NULL, omit.constants=TRUE, dummy.classes = "factor") #categorical to binary variables
df1["class"]=df$class #add the class to the df1
df=df1 #save df1 to df
summary(df) #get the summary of the dataframe to make sure things are ok!

#let's get rid of some enironmental variables
remove(numericVectors)
remove(factorVectors)
remove(x) 
remove(n)
remove(df1)
remove(imputedDF)
remove(preObj)
remove(classColumnID1)

trainSet<-createDataPartition(y=df$class, p=0.7, list=FALSE) #create the testing training partitions
Train<-df[trainSet,] #subset the training set
Test<-df[-trainSet,] #subset the testing set
classColumnID2<-dim(df)[2]

#let's get rid of an enironmental variable
remove(trainSet)

#let's find the best k for knn with the tune function
#This function builds several models (k models), and I can pick the k that give the smallest cross validation error
tKNN=tune.knn(Train[, -classColumnID2], Train[, classColumnID2], k=1:10 ,tunecontrol = tune.control(sampling = "cross"), set.seed(523))
CVError<-round(tKNN$best.performance, digits = 3)
Pred=knn(train=Train[, -classColumnID2], test = Test[, -classColumnID2], cl= Train$class, k=tKNN$best.parameters)
cm=table(Pred, Test$class)
finalError=round((1-(cm[1,1]+cm[2,2])/sum(cm)), digits = 3)
par(mfrow=c(2,1))
plot(tKNN, main = "k vs. error")

#let's get rid of some enironmental variables
remove(classColumnID2)
remove(cm)



# My knn model didnt work very well. I think  I should reduce the number of features
# I will use a decision tree algoritm and use the important variables in my knn
dfForTree=read.csv("CKD.csv", header = TRUE, dec=",", na.strings = c(""," ","NA")) #read the csv file and save it to the dataframe object df
#for some reason, R reads the variales below as factors. We have to convert them to numericals
numericVectors<-c("bu", "sc", "sod", "pot", "hemo", "rbcc", "pcv") 
for (n in 1:7){
  x=numericVectors[n]
  dfForTree[,x]<-as.numeric(levels(dfForTree[,x]))[dfForTree[,x]] #set some factor variables as numeric variables 
}

#for some reason, R reads the variales below as numbers. We have to convert them to factors.
factorVectors<-c("al", "su")
for (n in 1:2){
  x=factorVectors[n]
  dfForTree[, x]<-as.factor(dfForTree[,x])
}

#let's get rid of some enironmental variables
remove(x)
remove(n)
remove(numericVectors)
remove(factorVectors)


#Let's build the tree and get the variable importances
trainSetForTree<-createDataPartition(y=dfForTree$class, p=0.7, list=FALSE) #create the testing training partitions
TrainForTree<-dfForTree[trainSetForTree,] #subset the training set
TestForTree<-dfForTree[-trainSetForTree,] #subset the testing set
treeModel<-rpart(class ~. , method = "class", data=TrainForTree, control = rpart.control(minsplit=0, cp = 0))
importantVariables<-data.frame(treeModel$variable.importance)
importantFeatures<-row.names(importantVariables)
importantFeatures<-append(importantFeatures, "class", after=length(importantFeatures))
DFLessFeatures<-dfForTree[, importantFeatures]
classColumnID3=dim(DFLessFeatures)[2]

#let's get rid of some enironmental variables
remove(trainSetForTree)
remove(TrainForTree)
remove(TestForTree)
remove(importantFeatures)
remove(importantVariables)
remove(dfForTree)

#impute the NA's and replace categorical variables
imputedDF2<- mice(DFLessFeatures,m=1,maxit=10,meth='pmm',seed=500) #imputation with pmm
DFLessFeatures<-complete(imputedDF2) #replacing the NA's
preObj <- preProcess(DFLessFeatures[, -classColumnID3], method=c("center", "scale")) #standardization
newDF1 <- predict(preObj, DFLessFeatures[, -classColumnID3]) #standardization
newDF1<-dummy.data.frame(newDF1, names = NULL, omit.constants=TRUE, dummy.classes = "factor") #categorical to binary variables
newDF1["class"]=DFLessFeatures$class #add the class to the newDF1
DFLessFeatures=newDF1
summary(DFLessFeatures) #get the summary of the dataframe to make sure things are ok!

#let's get rid of some enironmental variables
remove(imputedDF2)
remove(newDF1)
remove(preObj)

#getting ready for a knn with feature selection
trainSetLessFeatures<-createDataPartition(y=DFLessFeatures$class, p=0.7, list=FALSE) #create the testing training partitions
TrainLessFeatures<-DFLessFeatures[trainSetLessFeatures,] #subset the training set
TestLessFeatures<-DFLessFeatures[-trainSetLessFeatures,] #subset the testing set
classColumnID3<-dim(DFLessFeatures)[2]
remove(trainSetLessFeatures)

#build mulyiple knn's with fewer features and pick the best one
tKNNLessFeatures=tune.knn(TrainLessFeatures[, -classColumnID3], TrainLessFeatures[, classColumnID3], k=1:10, tunecontrol = tune.control(sampling = "cross"), set.seed(523))
CVErrorLessFeatures<-round(tKNNLessFeatures$best.performance, digits = 3)
plot(tKNNLessFeatures, main = "k vs. error with feature selection")
Pred2=knn(train=TrainLessFeatures[, -classColumnID3], test = TestLessFeatures[, -classColumnID3], 
          cl= TrainLessFeatures$class, k=tKNNLessFeatures$best.parameters)
cmLessFeatures=table(Pred2, TestLessFeatures$class)
finalErrorLessFeatures=round((1-(cmLessFeatures[1,1]+cmLessFeatures[2,2])/sum(cmLessFeatures)), digits = 3)

remove(classColumnID3)
remove(cmLessFeatures)
