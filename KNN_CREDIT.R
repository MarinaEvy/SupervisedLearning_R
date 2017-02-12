#This is a KNN algorithm with PPM imputation for the missing values
# I impute the missing values with PPM - I do not conduct listwise deletion.
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
df=read.csv("Credit.csv", header = TRUE, dec=",", na.strings = c(""," ","NA"))

numericVectors<-c("A2", "A3", "A8") 
for (n in 1:3){
  x=numericVectors[n]
  df[,x]<-as.numeric(levels(df[,x]))[df[,x]] #set some factor variables as numeric variables 
}

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

#let's remove some environmental variables
remove(classColumnID1)
remove(n)
remove(x)
remove(df1) 
remove(imputedDF) 
remove(preObj) 
remove(numericVectors)



trainSet<-createDataPartition(y=df$class, p=0.7, list=FALSE) #create the testing training partitions
Train<-df[trainSet,] #subset the training set
Test<-df[-trainSet,] #subset the testing set
classColumnID2<-dim(df)[2]

#let's remove the environmental variable below
remove(trainSet)

#let's find the best k for knn with the tune function
#This function builds several models (k models), and you can pick the k that give the smallest cross validation error
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






