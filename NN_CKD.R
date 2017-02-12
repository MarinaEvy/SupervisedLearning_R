#This is a NN algorithm with PPM imputation for the missing values
# I do not conduct listwise deletion.
# I standardize the dataset.
# I do not conduct feature selection.

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
library("grid")
library("MASS")


set.seed(4000) #Let's set the seed for reproducibility

setwd("C:/Users/mjohnson6/Desktop") #let's set the working directory where the dataset is
df=read.csv("CKD.csv", header = TRUE, dec=",", na.strings = c(""," ","NA")) #read the csv file and save it to the dataframe object df

#for some reason, R reads the variales below as factors. have to convert them to numericals
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

summary(df) #check the dataset
str(df) #see if all the variables are correctly identified as factors and numbers.

classColumnID1<-dim(df)[2] #will use the number of columns later on.




#ppm imputation and binary transformation for categorical data
imputedDF<- mice(df,m=1,maxit=50,meth='pmm',seed=500) #imputation with pmm
df<-complete(imputedDF) #replacing the NA's
preObj <- preProcess(df[, -classColumnID1], method=c("center", "scale")) #standardization
df1 <- predict(preObj, df[, -classColumnID1]) #standardization
df1<-dummy.data.frame(df1, names = NULL, omit.constants=TRUE, dummy.classes = "factor") #categorical to binary variables
df1["class"]=df$class #add the class to the df1
df=df1 #save df1 to df

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

#check if the class labels are imbalanced. 
#According to the literature, if +/- ratio is above 0.5, the class os balanced.
balanceIndicatorTrain<-length(which(Train[, dim(Train)[2]]=="notckd"))/length(which(Train[, dim(Train)[2]]=="ckd"))
balanceIndicatorTest<-length(which(Test[, dim(Test)[2]]=="notckd"))/length(which(Test[, dim(Test)[2]]=="ckd"))
balanceIndicatorTrain<-round(balanceIndicatorTrain, digits = 3)
balanceIndicatorTest<-round(balanceIndicatorTest, digits = 3)

Train2<-Train
Train2[,dim(Train)[2]]=as.numeric(Train[,dim(Train)[2]])-1
fmla <- as.formula(paste("class~ ", paste(names(Train2[,-dim(Train2)[2]]), collapse= "+")))

#Tune the parameters w/ cross validation
n<-nrow(Train2)
K<-10 #number of folds - do 10-fold cross validation
foldSize<-n%/%K
al<-runif(n) #label each row in the training set with random numbers
ra<-rank(al) # finds which row in the training set the first random number is associated with
blocks<-(ra-1)%/%foldSize+1 # create the folds
blocks<-as.factor(blocks) #make them factors
print(summary(blocks))
errorCV <- vector(mode="numeric", length=K)
learningRate<-seq(0.005, 0.02, 0.005) # Learning rate values for tuning
hiddenLayer<-c(2,3) ## of hidden layers for tuning
meanError<-data.frame()
for(i in 1:length(learningRate)) {
  for(j in 1:length(hiddenLayer)) {
    for(k in 1:K) {
      
      nn <- neuralnet(fmla,data=Train2[blocks!=k,], hidden=hiddenLayer[j], err.fct="ce",linear.output=FALSE, 
                      learningrate=learningRate[i], algorithm="backprop", act.fct = "logistic", threshold = 0.05) #train the nn
      
      predProb<-compute(nn, Train2[blocks==k, -dim(Train2)[2]]) # make the predictions
      classLab <- vector(mode="character", length=dim(Train2[blocks==k, -dim(Train2)[2]])[1]) # create a character vector
      pred=predProb$net.result #Transfer class probs to class laels
      for(ss in 1:dim(pred)[1]) {
        if(pred[ss]<0.5) {
          classLab[ss]="ckd"
        } else {
          classLab[ss]="notckd"
        }
      }
      cm<-table(classLab, Train[blocks==k, dim(Train)[2]])
      errorCV[k]<-1-(cm[1,1]+cm[2,2])/sum(cm)
    }
    
    meanError[i,j]<-mean(errorCV)
  }
}




nn <- neuralnet(fmla,data=Train2, hidden=2, err.fct="ce",linear.output=FALSE, 
                learningrate=0.015, algorithm="backprop", act.fct = "logistic", threshold = 0.01)

predProb<-compute(nn, Test[, -dim(Test)[2]])
classLab <- vector(mode="character", length=dim(Test)[1])
pred=predProb$net.result
for(i in 1:dim(pred)[1]) {
  if(pred[i]<0.5) {
    classLab[i]="ckd"
  } else {
    classLab[i]="notckd"
  }
}
cm2<-table(classLab, Test[, dim(Test)[2]])
finalError<-round(1-(cm2[1,1]+cm2[2,2])/sum(cm2), digits = 3)
plot(nn)
nn$result.matrix

remove(errorCV)
remove(blocks)
remove(foldSize)
remove(i)
remove(j)
remove(k)
remove(n)
remove(al)
remove(ss)
remove(ra)
remove(K)
remove(fmla)
remove(hiddenLayer)
remove(learningRate)
remove(cm)
remove(cm2)
remove(pred)



