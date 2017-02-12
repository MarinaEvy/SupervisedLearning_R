#This is an ADABOOST algorithm with PPM imputation for the missing values
# I impute the missing values with PPM
# I standardize the dataset.
# I do not conduct feature selection

install.packages("caret") #Download te package for splitting the dataset into training and testing
install.packages("ggplot2") #Download this package in case I want to plot some features
install.packages("rattle") #Download this to plot the decision tree
install.packages("RColorBrewer") #Download this to make the tree look fancy
install.packages("lattice") #Care package requires it
install.packages("mice") # data imputation
install.packages("dummies") # binary transformation of categorical variables
install.packages("class") # KNN package
install.packages("e1071")
install.packages("nnet")
install.packages("neuralnet")
install.packages("grid")
install.packages("MASS")
install.packages("adabag")
install.packages("ada")
install.packages("plyr")


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
library(nnet)
library(neuralnet)
library(grid)
library(MASS)
library(adabag)
library(ada)
library(plyr)

set.seed(4000) #Let's set the seed for reproducibility
#PLEASE SET THE DIRECTORY
setwd("C:/Users/mjohnson6/Desktop") #let's set the working directory where the dataset is
df=read.csv("CKD.csv", header = TRUE, dec=",", na.strings = c(""," ","NA")) #read the csv file and save it to the dataframe object df

#for some reason, R reads the variales below as factors. We have to convert them to numericals
numericVectors<-c("bu", "sc", "sod", "pot", "hemo", "rbcc", "pcv") 
for (n in 1:7){
  x=numericVectors[n]
  df[,x]<-as.numeric(levels(df[,x]))[df[,x]] #set some factor variables as numeric variables 
}

#for some reason, R reads the variales below as numbers. We have to convert them to factors.
factorVectors<-c("al", "su")
for (n in 1:2){
  x=factorVectors[n]
  df[, x]<-as.factor(df[,x])
}

summary(df) # let's check the dataset
str(df) #let's see if all the variables are orrectly identified as factors and numbers.

classColumnID1<-dim(df)[2] #we will use the number of columns later on.




#Let's do ppm imputation and binary transformation for categorical data
imputedDF<- mice(df,m=1,maxit=50,meth='pmm',seed=500) #imputation with pmm
df<-complete(imputedDF) #replacing the NA's


summary(df) #get the summary of the dataframe to make sure things are ok!

remove(df1) 
remove(imputedDF) 
remove(preObj) 
remove(n)
remove(x)
remove(classColumnID1)
remove(factorVectors)
remove(numericVectors)


trainSet<-createDataPartition(y=df$class, p=0.7, list=FALSE) #create the testing training partitions
Train<-df[trainSet,] #subset he training set
Test<-df[-trainSet,] #subset the testing set
remove(trainSet)

#let's check if the class labels are imbalanced. 
#According to the literature, if +/- ratio is above 0.5, the class os balanced.
balanceIndicatorTrain<-length(which(Train[, dim(Train)[2]]=="notckd"))/length(which(Train[, dim(Train)[2]]=="ckd"))
balanceIndicatorTest<-length(which(Test[, dim(Test)[2]]=="notckd"))/length(which(Test[, dim(Test)[2]]=="ckd"))
balanceIndicatorTrain<-round(balanceIndicatorTrain, digits = 3)
balanceIndicatorTest<-round(balanceIndicatorTest, digits = 3)

cv_opts = trainControl(method="cv", number=10)
Grid <- expand.grid(maxdepth=seq(5,25,5),nu=c(1,2),iter=100)
results_ada = train(class~., data=Train, method="ada", trControl=cv_opts,tuneGrid=Grid)
x<-predict(results_ada, Test[, -ncol(Test)])
cm<-table(x,Test[,ncol(Test)])
finalError<-1-(cm[1,1]+cm[2,2])/sum(cm)
bestTuningParameter<-results_ada$bestTune
remove(x)
remove(cm)

plot(results_ada)
