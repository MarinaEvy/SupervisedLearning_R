#DECISION TREES
# This is a decison tree w/ CART Algorithm. I use cross validation to get a better estimate of the error. 
# I do not impute the missing values - I do not conduct listwise deletion either.
# I do not standardize or normalize the dataset.
# I keep the dataset as is. (No preprocessing)

install.packages("caret") #Download te package for splitting the dataset into training and testing
install.packages("ggplot2") #Download this package in case I want to plot some features
install.packages("rpart") #Download this package for the CART
install.packages("rattle") #Download this to plot the decision tree
install.packages("rpart.plot") #Download this to plot the decision tree
install.packages("RColorBrewer") #Download this to make the tree look fancy
install.packages("lattice") #caret package requires it

#activate the packages below
library(ggplot2) 
library(caret)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(lattice)

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


#Let's do cross validation to get a better estimate of model accuracy
n<-nrow(Train)
K<-10 #number of folds - we do 10-fold cross validation
foldSize<-n%/%K
al<-runif(n) #we label eah row in the training set with random numebrs so we can assign them in one of the folds
ra<-rank(al) # we want to figure out which row in the training set the first random number is associated with
blocks<-(ra-1)%/%foldSize+1 # create the folds
blocks<-as.factor(blocks) #make them factors
print(summary(blocks))
errorCV <- vector(mode="numeric", length=K)
cp<-seq(0, 0.1, 0.01) # Learning rate values for tuning
minSplit<-seq(0,30,2) ## of hidden layers for tuning
meanError<-data.frame()
for(i in 1:length(cp)) {
  for(j in 1:length(minSplit)) {
    for(k in 1:K) {
      
      cvModel<-rpart(class ~. , method = "class", data=Train[blocks!=k,], 
                     control = rpart.control(minsplit=minSplit[j], cp = cp[i]))
      
      pred<-predict(cvModel, newdata=Train[blocks==k,], type="class")
      cmCV<-table(pred, Train[blocks==k,"class"])
      errorK<- 1- (cmCV[1,1] + cmCV[2,2])/sum(cmCV)
      errorCV[k]<-errorK
    }
    averageError<-mean(errorCV)
    meanError[i,j]<-averageError
  }
}
which(meanError==min(meanError)) #cp=0.03 and minsplit 6 gives the bes result

TreeModel<-rpart(class ~. , method = "class", data=Train, 
                 control = rpart.control(minsplit=0, cp = 0))

plot(TreeModel) #plots the tree
text(TreeModel, pretty=0) # brings teh text onto the plot
fancyRpartPlot(TreeModel, uniform=TRUE, sub="Classification Tree") # makes the plot look nicer

pTreeModel<-rpart(class ~. , method = "class", data=Train, 
                  control = rpart.control(minsplit=30, cp = 0.1))

pred<-predict(pTreeModel, newdata=Test, type="class")
cmCV<-table(pred, Test[,"class"])
Finalerror<- 1- (cmCV[1,1] + cmCV[2,2])/sum(cmCV)
plot(pTreeModel) #plots the tree
text(pTreeModel, pretty=0) # brings teh text onto the plot
fancyRpartPlot(pTreeModel, uniform=TRUE, sub="Pruned Classification Tree") # makes the plot look nicer



