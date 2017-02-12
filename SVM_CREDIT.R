#This is a SVM algorithm with PPM imputation for the missing values
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

set.seed(4000) #Let's set the seed for reproducibility

setwd("C:/Users/mjohnson6/Desktop") #let's set the working directory where the dataset is
df=read.csv("Credit.csv", header = TRUE, dec=",", na.strings = c(""," ","NA")) #read the csv file and save it to the dataframe object df

#for some reason, R reads the variales below as factors. We have to convert them to numericals
numericVectors<-c("A2", "A3", "A8") 
for (n in 1:3){
  x=numericVectors[n]
  df[,x]<-as.numeric(levels(df[,x]))[df[,x]] #set some factor variables as numeric variables 
}

summary(df) # let's check the dataset
str(df) #let's see if all the variables are orrectly identified as factors and numbers.
classColumnID1<-dim(df)[2] #we will use the number of columns later on.




#Let's do ppm imputation and binary transformation for categorical data
imputedDF<- mice(df,m=1,maxit=50,meth='pmm',seed=500) #imputation with pmm
df<-complete(imputedDF) #replacing the NA's
preObj <- preProcess(df[, -classColumnID1], method=c("center", "scale")) #standardization
df1 <- predict(preObj, df[, -classColumnID1]) #standardization
df1<-dummy.data.frame(df1, names = NULL, omit.constants=TRUE, dummy.classes = "factor") #categorical to binary variables
df1["class"]=df$class #add the class to the df1
df=df1 #save df1 to df
remove(df1) #remove this object
remove(imputedDF) #remove this object
remove(preObj) #remove this object
summary(df) #get the summary of the dataframe to make sure things are ok!




trainSet<-createDataPartition(y=df$class, p=0.7, list=FALSE) #create the testing training partitions
Train<-df[trainSet,] #subset the training set
Test<-df[-trainSet,] #subset the testing set

#let's check if the class labels are imbalanced. 
#According to the literature, if +/- ratio is above 0.5, the class os balanced.
balanceIndicatorTrain<-length(which(Train[, dim(Train)[2]]=="Yes"))/length(which(Train[, dim(Train)[2]]=="No"))
balanceIndicatorTest<-length(which(Test[, dim(Test)[2]]=="Yes"))/length(which(Test[, dim(Test)[2]]=="No"))
balanceIndicatorTrain=round(balanceIndicatorTrain, digits=3)
balanceIndicatorTest=round(balanceIndicatorTest, digits=3)
#parameter tuning for svm with a radial based function
tRadial=tune(svm, train.x = class~., data=Train, tunecontrol=tune.control(sampling = "cross"), kernel="radial", 
             ranges = list(cost = 2^(2:6), epsilon = seq(0,1,0.1), gamma=2^(-1:1)))


#parameter tuning for svm with a sigmoid based function
tSigmoid=tune(svm, train.x = class~., data=Train, tunecontrol=tune.control(sampling = "cross"), kernel="sigmoid", 
              ranges = list(cost = 2^(2:6), epsilon = seq(0,1,0.1), gamma=2^(-1:1)))

tLinear=tune(svm, train.x = class~., data=Train, tunecontrol=tune.control(sampling = "cross"), kernel="linear", 
              ranges = list(cost = 2^(2:6), epsilon = seq(0,1,0.1)))

# print the best performances of each function
print(c(tRadial$best.performance, tSigmoid$best.performance, tLinear$best.performance))

#pick the best parameters for the best function and make the predictions
tunedModel <- tLinear$best.model
pred <- predict(tunedModel, Test)
cm<-table(pred, Test[, dim(Test)[2]])
error<-1-(cm[1,1]+cm[2,2])/sum(cm)
error=round(error, digits=3)
#Cleaning the environment little bit
remove(n)
remove(cm)
remove(x)
remove(numericVectors)
remove(classColumnID1)



