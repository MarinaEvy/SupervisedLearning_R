#This an NN algorithm with PPM imputation for the missing values
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


set.seed(4000) #set the seed for reproducibility

setwd("C:/Users/mjohnson6/Desktop") #set the working directory where the dataset is
df=read.csv("Credit.csv", header = TRUE, dec=",", na.strings = c(""," ","NA")) #read the csv file and save it to the dataframe object df

#for some reason, R reads the variales below as factors. convert them to numericals
numericVectors<-c("A2", "A3", "A8") 
for (n in 1:3){
  x=numericVectors[n]
  df[,x]<-as.numeric(levels(df[,x]))[df[,x]] #set some factor variables as numeric variables 
}


summary(df) # check the dataset
str(df) #check if all the variables are orrectly identified as factors and numbers.

classColumnID1<-dim(df)[2] #use the number of columns later on.




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
remove(numericVectors)


trainSet<-createDataPartition(y=df$class, p=0.7, list=FALSE) #create the testing training partitions
Train<-df[trainSet,] #subset he training set
Test<-df[-trainSet,] #subset the testing set
remove(trainSet)

#check if the class labels are imbalanced. 
#According to the literature, if +/- ratio is above 0.5, the class os balanced.
balanceIndicatorTrain<-length(which(Train[, dim(Train)[2]]=="Yes"))/length(which(Train[, dim(Train)[2]]=="No"))
balanceIndicatorTest<-length(which(Test[, dim(Test)[2]]=="Yes"))/length(which(Test[, dim(Test)[2]]=="No"))
balanceIndicatorTrain<-round(balanceIndicatorTrain, digits = 3)
balanceIndicatorTest<-round(balanceIndicatorTest, digits = 3)

Train2<-Train
Train2[,dim(Train)[2]]=as.numeric(Train[,dim(Train)[2]])-1
fmla <- as.formula(paste("class~ ", paste(names(Train2[,-dim(Train2)[2]]), collapse= "+")))

n<-nrow(Train2)
K<-10 #number of folds - do 10-fold cross validation
foldSize<-n%/%K
al<-runif(n) #label eah row in the training set with random numebrs so we can assign them in one of the folds
ra<-rank(al) # figure out which row in the training set the first random number is associated with
blocks<-(ra-1)%/%foldSize+1 # create the folds
blocks<-as.factor(blocks) #make them factors
print(summary(blocks))
errorCV <- vector(mode="numeric", length=K)
learningRate<-seq(0.005, 0.02, 0.005)
hiddenLayer<-c(2,3)
meanError<-data.frame()
for(i in 1:length(learningRate)) {
  for(j in 1:length(hiddenLayer)) {
    for(k in 1:K) {
      
      nn <- neuralnet(fmla,data=Train2[blocks!=k,], hidden=hiddenLayer[j], err.fct="ce",linear.output=FALSE, 
                      learningrate=learningRate[i], algorithm="backprop", 
                      act.fct = "logistic", threshold = 0.1, stepmax = 1e+06)
      
      predProb<-compute(nn, Train2[blocks==k, -dim(Train2)[2]])
      classLab <- vector(mode="character", length=dim(Train2[blocks==k, -dim(Train2)[2]])[1])
      pred=predProb$net.result
      for(ss in 1:dim(pred)[1]) {
        if(pred[ss]<0.5) {
          classLab[ss]="No"
        } else {
          classLab[ss]="Yes"
        }
      }
      cm<-table(classLab, Train[blocks==k, dim(Train)[2]])
      errorCV[k]<-1-(cm[1,1]+cm[2,2])/sum(cm)
    }
    
    meanError[i,j]<-mean(errorCV)
  }
}


nn <- neuralnet(fmla,data=Train2, hidden=1, err.fct="ce",linear.output=FALSE, 
                learningrate=0.015, algorithm="backprop", act.fct = "logistic", stepmax = 1e+06, threshold = 0.1)

predProb<-compute(nn, Test[, -dim(Test)[2]])
classLab <- vector(mode="character", length=dim(Test)[1])
pred=predProb$net.result
for(i in 1:dim(pred)[1]) {
  if(pred[i]<0.5) {
    classLab[i]="No"
  } else {
    classLab[i]="Yes"
  }
}
cm2<-table(classLab, Test[, dim(Test)[2]])
finalError<-1-(cm2[1,1]+cm2[2,2])/sum(cm2)
plot(nn)

mean(meanError[,1])

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
