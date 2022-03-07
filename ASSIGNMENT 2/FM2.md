---
title: "FM2"
author: "Vijay"
date: '2022-21-02'
output:
  pdf_document: default
  word_document: default
  html_document: default
---


```r
getwd()
```

```
## [1] "C:/Users/vijay/Documents"
```

##Here I am Importing the Dataset

```r
bank.df= read.csv("universalBank.csv")
```


```r
summary(bank.df)
```

```
##        ID            Age          Experience       Income          ZIP.Code         Family     
##  Min.   :   1   Min.   :23.00   Min.   :-3.0   Min.   :  8.00   Min.   : 9307   Min.   :1.000  
##  1st Qu.:1251   1st Qu.:35.00   1st Qu.:10.0   1st Qu.: 39.00   1st Qu.:91911   1st Qu.:1.000  
##  Median :2500   Median :45.00   Median :20.0   Median : 64.00   Median :93437   Median :2.000  
##  Mean   :2500   Mean   :45.34   Mean   :20.1   Mean   : 73.77   Mean   :93153   Mean   :2.396  
##  3rd Qu.:3750   3rd Qu.:55.00   3rd Qu.:30.0   3rd Qu.: 98.00   3rd Qu.:94608   3rd Qu.:3.000  
##  Max.   :5000   Max.   :67.00   Max.   :43.0   Max.   :224.00   Max.   :96651   Max.   :4.000  
##      CCAvg          Education        Mortgage     Personal.Loan   Securities.Account   CD.Account    
##  Min.   : 0.000   Min.   :1.000   Min.   :  0.0   Min.   :0.000   Min.   :0.0000     Min.   :0.0000  
##  1st Qu.: 0.700   1st Qu.:1.000   1st Qu.:  0.0   1st Qu.:0.000   1st Qu.:0.0000     1st Qu.:0.0000  
##  Median : 1.500   Median :2.000   Median :  0.0   Median :0.000   Median :0.0000     Median :0.0000  
##  Mean   : 1.938   Mean   :1.881   Mean   : 56.5   Mean   :0.096   Mean   :0.1044     Mean   :0.0604  
##  3rd Qu.: 2.500   3rd Qu.:3.000   3rd Qu.:101.0   3rd Qu.:0.000   3rd Qu.:0.0000     3rd Qu.:0.0000  
##  Max.   :10.000   Max.   :3.000   Max.   :635.0   Max.   :1.000   Max.   :1.0000     Max.   :1.0000  
##      Online         CreditCard   
##  Min.   :0.0000   Min.   :0.000  
##  1st Qu.:0.0000   1st Qu.:0.000  
##  Median :1.0000   Median :0.000  
##  Mean   :0.5968   Mean   :0.294  
##  3rd Qu.:1.0000   3rd Qu.:1.000  
##  Max.   :1.0000   Max.   :1.000
```


```r
library(caret)
library(class)
library(dplyr)
library(ISLR)
library(psych)
library(FNN)
library(lattice)
```

##Now I am going to Remove ID ,  ZIP Code 

```r
bank.df$ID <- NULL
bank.df$ZIP.Code <- NULL
bank.df$Education = as.factor(bank.df$Education)
```

##later I am creating a dummy dataset

```r
dummyvar <- as.data.frame(dummy.code(bank.df$Education))
```


```r
names(dummyvar) <- c("Education_1", "Education_2","Education_3")
```

##Now I am setting  education to NULL

```r
bank.df$Education <- NULL
```


```r
FinalBank <- cbind(bank.df, dummyvar)
```

##Here I am going to divide the dataset into train and test

```r
set.seed(1)
train.index <- createDataPartition(FinalBank$Personal.Loan, p= 0.6 , list=FALSE)
valid.index <- setdiff(row.names(FinalBank), train.index)
train.df <- FinalBank[train.index,]
valid.df <- FinalBank[valid.index,]
```

##Now we are generating the Test data

```r
new_customer <- data.frame(Age = 40,
                       Experience = 10,
                       Income = 84,
                       Family = 2,
                       CCAvg = 2,
                       Mortgage = 0,
                       Securities.Account = 0,
                       CD.Account = 0,
                       Online = 1,
                       CreditCard = 1,
                       Education_1 = 0, 
                       Education_2 = 1, 
                       Education_3 = 0)
```
## normalisation

```r
train.norm.df <- train.df[,-7]
valid.norm.df <- valid.df[,-7]
new_customer.norm <- new_customer

norm.values <- preProcess(train.df[, -7], method=c("center", "scale"))
train.norm.df <- predict(norm.values, train.df[, -7])
valid.norm.df <- predict(norm.values, valid.df[, -7])
new_customer.norm <- predict(norm.values, new_customer.norm)
```


```r
summary(train.norm.df)
```

```
##       Age             Experience           Income            Family            CCAvg            Mortgage      
##  Min.   :-1.97257   Min.   :-2.03718   Min.   :-1.4240   Min.   :-1.2058   Min.   :-1.1059   Min.   :-0.5679  
##  1st Qu.:-0.82922   1st Qu.:-0.89531   1st Qu.:-0.7457   1st Qu.:-1.2058   1st Qu.:-0.7016   1st Qu.:-0.5679  
##  Median :-0.03767   Median :-0.01695   Median :-0.2206   Median :-0.3368   Median :-0.2396   Median :-0.5679  
##  Mean   : 0.00000   Mean   : 0.00000   Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.0000  
##  3rd Qu.: 0.84183   3rd Qu.: 0.86141   3rd Qu.: 0.5452   3rd Qu.: 0.5321   3rd Qu.: 0.3380   3rd Qu.: 0.4423  
##  Max.   : 1.89723   Max.   : 2.00328   Max.   : 3.3022   Max.   : 1.4010   Max.   : 4.6700   Max.   : 5.7216  
##  Securities.Account   CD.Account          Online          CreditCard       Education_1       Education_2     
##  Min.   :-0.3339    Min.   :-0.2381   Min.   :-1.1863   Min.   :-0.6431   Min.   :-0.8462   Min.   :-0.6509  
##  1st Qu.:-0.3339    1st Qu.:-0.2381   1st Qu.:-1.1863   1st Qu.:-0.6431   1st Qu.:-0.8462   1st Qu.:-0.6509  
##  Median :-0.3339    Median :-0.2381   Median : 0.8427   Median :-0.6431   Median :-0.8462   Median :-0.6509  
##  Mean   : 0.0000    Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.0000  
##  3rd Qu.:-0.3339    3rd Qu.:-0.2381   3rd Qu.: 0.8427   3rd Qu.: 1.5544   3rd Qu.: 1.1814   3rd Qu.: 1.5358  
##  Max.   : 2.9940    Max.   : 4.1985   Max.   : 0.8427   Max.   : 1.5544   Max.   : 1.1814   Max.   : 1.5358  
##   Education_3     
##  Min.   :-0.6312  
##  1st Qu.:-0.6312  
##  Median :-0.6312  
##  Mean   : 0.0000  
##  3rd Qu.: 1.5836  
##  Max.   : 1.5836
```
##Here I am Performing Knn classification, using K=1

```r
outputrate <- class::knn(train = train.norm.df,test = new_customer.norm,
                       cl = train.df$Personal.Loan, k = 1)

print(outputrate)
```

```
## [1] 0
## Levels: 0 1
```


##we are going to find best K

```r
accuracy.df <- data.frame(k = seq(1, 10, 1), accuracy = rep(0, 10))
```



```r
for(i in 1:10) {
  knn.prediction <- class::knn(train = train.norm.df,
                         test = valid.norm.df,
                         cl = train.df$Personal.Loan, k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.prediction,
                                       as.factor(valid.df$Personal.Loan))$overall[1]
}
which(accuracy.df[,2] == max(accuracy.df[,2]))
```

```
## [1] 3
```



```r
accuracy.df
```

```
##     k accuracy
## 1   1   0.9630
## 2   2   0.9565
## 3   3   0.9640
## 4   4   0.9595
## 5   5   0.9605
## 6   6   0.9575
## 7   7   0.9580
## 8   8   0.9575
## 9   9   0.9535
## 10 10   0.9550
```



##choosing k = 3

```r
knn.prediction <- class::knn(train = train.norm.df,
                       test = valid.norm.df,
                       cl = train.df$Personal.Loan, k = 3)

confusionMatrix(knn.prediction, as.factor(valid.df$Personal.Loan), positive = "1")
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    0    1
##          0 1786   63
##          1    9  142
##                                           
##                Accuracy : 0.964           
##                  95% CI : (0.9549, 0.9717)
##     No Information Rate : 0.8975          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7785          
##                                           
##  Mcnemar's Test P-Value : 4.208e-10       
##                                           
##             Sensitivity : 0.6927          
##             Specificity : 0.9950          
##          Pos Pred Value : 0.9404          
##          Neg Pred Value : 0.9659          
##              Prevalence : 0.1025          
##          Detection Rate : 0.0710          
##    Detection Prevalence : 0.0755          
##       Balanced Accuracy : 0.8438          
##                                           
##        'Positive' Class : 1               
## 
```

##Now Here confusion matrix for the best k value =3


```r
newcustomer <- data.frame(Age = 40,
                            Experience = 10,
                            Income = 84,
                            Family = 2,
                            CCAvg = 2,
                            Mortgage = 0,
                            Securities.Account = 0,
                            CD.Account = 0,
                            Online = 1,
                            CreditCard = 1,
                            Education_1 = 0, 
                            Education_2 = 1, 
                            Education_3 = 0)

best_knn <-class::knn(train = train.norm.df,
                          test = newcustomer,
                          cl = train.df$Personal.Loan, k = 3)

best_knn
```

```
## [1] 1
## Levels: 0 1
```

##k-NN model says that the new customer will accept a loan offer


```r
##Importing Dataset
bank.df= read.csv("universalBank.csv")
```

##Performing Knn classification using K=3

```r
knn.test.prediction <- class::knn(train = train.norm.df,
                       test = test.norm.df,
                       cl = train.df$Personal.Loan, k = 3)

knn.train.prediction <- class::knn(train = train.norm.df,
                             test = train.norm.df,
                             cl = train.df$Personal.Loan, k = 3)

knn.valid.prediction <- class::knn(train = train.norm.df,
                             test = valid.norm.df,
                             cl = train.df$Personal.Loan, k = 3)
```

##Here the Confusion matrix is best for k=3

```r
confusionMatrix(knn.test.prediction, as.factor(test.df$Personal.Loan), positive = "1")
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    0    1
##          0 1592   50
##          1    6  111
##                                           
##                Accuracy : 0.9682          
##                  95% CI : (0.9589, 0.9759)
##     No Information Rate : 0.9085          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7817          
##                                           
##  Mcnemar's Test P-Value : 9.132e-09       
##                                           
##             Sensitivity : 0.68944         
##             Specificity : 0.99625         
##          Pos Pred Value : 0.94872         
##          Neg Pred Value : 0.96955         
##              Prevalence : 0.09153         
##          Detection Rate : 0.06310         
##    Detection Prevalence : 0.06652         
##       Balanced Accuracy : 0.84284         
##                                           
##        'Positive' Class : 1               
## 
```

```r
confusionMatrix(knn.train.prediction, as.factor(train.df$Personal.Loan), positive = "1")
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    0    1
##          0 2719   60
##          1    6  215
##                                           
##                Accuracy : 0.978           
##                  95% CI : (0.9721, 0.9829)
##     No Information Rate : 0.9083          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8551          
##                                           
##  Mcnemar's Test P-Value : 6.853e-11       
##                                           
##             Sensitivity : 0.78182         
##             Specificity : 0.99780         
##          Pos Pred Value : 0.97285         
##          Neg Pred Value : 0.97841         
##              Prevalence : 0.09167         
##          Detection Rate : 0.07167         
##    Detection Prevalence : 0.07367         
##       Balanced Accuracy : 0.88981         
##                                           
##        'Positive' Class : 1               
## 
```

```r
confusionMatrix(knn.valid.prediction, as.factor(valid.df$Personal.Loan), positive = "1")
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    0    1
##          0 1786   63
##          1    9  142
##                                           
##                Accuracy : 0.964           
##                  95% CI : (0.9549, 0.9717)
##     No Information Rate : 0.8975          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7785          
##                                           
##  Mcnemar's Test P-Value : 4.208e-10       
##                                           
##             Sensitivity : 0.6927          
##             Specificity : 0.9950          
##          Pos Pred Value : 0.9404          
##          Neg Pred Value : 0.9659          
##              Prevalence : 0.1025          
##          Detection Rate : 0.0710          
##    Detection Prevalence : 0.0755          
##       Balanced Accuracy : 0.8438          
##                                           
##        'Positive' Class : 1               
## 
```
