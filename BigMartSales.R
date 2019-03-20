
## Load the libraries 

library(caret)
library(dplyr)
library(reshape2)
library(elasticnet)
library(glmnet)
library(rpart)
library(randomForest)
library(ranger)
library(xgboost)

## Load the data sets 

setwd("C:/Users/maria/Documents/data_science_specialization
      /DataProjects/Beginner/BigMart")
train <- read.csv("TrainSales.csv",header = TRUE, na.strings = "")
test <- read.csv("TestSales.csv", header = TRUE, na.strings = "")

## We combine the train and test data. 

train2 <- train
test2 <- test
test2$Item_Outlet_Sales <- rep(NA, times = 5681)
comb2 <- rbind(train2,test2)

## The number of missing values of every variable is calculated. 

temp <- lapply(comb2, is.na)
sapply(temp, sum)

## 

str(comb2)

## We summarize and examine the numerical variables of the data set. 

num_vars <- c("Item_Weight", "Item_Visibility","Item_MRP",
              "Outlet_Establishment_Year","Item_Outlet_Sales")
summary(comb2[,num_vars])

## We explore the categorical variables. 

summary(comb2[,!(colnames(comb2) %in% num_vars)])
v<- sapply(comb2[,!(colnames(comb2) %in% num_vars)], unique)
sapply(v, length)
lapply((comb2[,!(colnames(comb2) %in% num_vars
                 | colnames(comb2) %in% c("Item_Identifier", 
                                          "Outlet_Identifier"))]), 
       table)
       
## Impute the missing values
## Item_Weight
## 
## We find the average weight for each Item. 
AvweightByItem <- aggregate(Item_Weight~Item_Identifier, 
                            data = comb2, mean)

## Impute the missing values
for(i in 1:(dim(comb2)[1]))
{
        if(is.na(comb2$Item_Weight[i])){
                pr <- comb2$Item_Identifier[i]
                comb2$Item_Weight[i] <-  AvweightByItem[pr,2]
        }
}

## Outlet_Size

t <- table(comb2$Outlet_Type, comb2$Outlet_Size)

for (i in 1:(dim(comb2)[1])){
        if(is.na(comb2$Outlet_Size[i])){
                if(comb2$Outlet_Type[i] == "Grocery Store"){
                        comb2$Outlet_Size[i] <- "Small"
                }
                else if(comb2$Outlet_Type[i] == "Supermarket Type1"){
                        comb2$Outlet_Size[i] <- sample(c("High","Medium",
                                                         "Small"),size = 1,
                                                       replace = TRUE, prob = c(t["Supermarket Type1",1]/sum(t["Supermarket Type1",]),
                                                                                (t["Supermarket Type1",2]/sum(t["Supermarket Type1",])),
                                                                                (t["Supermarket Type1",3]/sum(t["Supermarket Type1",]))))
                        
                }
                else if(comb2$Item_Type[i] == "Supermarket Type2"|comb2$Item_Type[i] == "Supermarket Type3"){
                        comb2$Outlet_Size[i] <- "Medium"
                }
                        
                }
}

### Feature Engineering 

## Item_Visibility

## Zeros = missing values
## Impute with the average visibiliy of each product 

## We find the average visibility for each Item. 

AvVisibilityByItem <- aggregate(Item_Visibility~Item_Identifier, data = comb2, mean)

## Impute the missing values 

for(i in 1:(dim(comb2)[1]))
{
        if(comb2$Item_Visibility[i] == 0){
                pr2 <- comb2$Item_Identifier[i]
                comb2$Item_Visibility[i] <-  AvVisibilityByItem[pr,2]
        }
}
sum(is.na(comb2$Item_Visibility))

## New var - Item_Visibility_Means_ratio 
## (mean visibility of a product in an outlet)/(mean visibility of product)

comb2$Item_Visibility_Means_ratio <- rep(0, times = dim(comb2)[1])

for(i in 1:(dim(comb2)[1])){
        comb2$Item_Visibility_Means_ratio[i] <- (comb2[i,"Item_Visibility"])/(AvVisibilityByItem[AvVisibilityByItem$Item_Identifier == comb2$Item_Identifier[i],"Item_Visibility"])
}

### Create variable Item_Type_General 
### (a new variable with broader Item type categories)

comb2$Item_Type_General <- rep("a", times = dim(comb2)[1])

for(i in 1:(dim(comb2)[1])){
        if(startsWith(as.character(comb2$Item_Identifier[i]),"FD")){
                comb2$Item_Type_General[i] <- "Food"
        }
        else if(startsWith(as.character(comb2$Item_Identifier[i]),"DR")){
                comb2$Item_Type_General[i] <- "Drinks"
        }
        else if(startsWith(as.character(comb2$Item_Identifier[i]),"NC")){
                comb2$Item_Type_General[i] <- "Non-Consumable"
        }
}

### Item_Fat_Contents Modify 

### "LF" and "low fat" as "Low Fat" 
### "reg" as "Regular"
### New category: "Non-Edible 
comb2$Item_Fat_Content <- as.character(comb2$Item_Fat_Content)

for(i in 1:(dim(comb2)[1])){
        if(startsWith(as.character(comb2$Item_Identifier[i]),"NC")){
                comb2$Item_Fat_Content[i] <- "Non-Edible"
                
        }
        else{
                if(comb2$Item_Fat_Content[i]=="low fat" | comb2$Item_Fat_Content[i] == "LF"){
                        comb2$Item_Fat_Content[i] <- "Low Fat"
                }
                else if(comb2$Item_Fat_Content[i] == "reg"){
                        comb2$Item_Fat_Content[i] <- "Regular"
                }
        }
}
comb2$Item_Fat_Content <- as.factor(comb2$Item_Fat_Content)


### Create variable Years_Operate 

### A new variable which represents the time in years that a store has 
### been operated. 

comb2$Years_Operate <- 2013 - comb2$Outlet_Establishment_Year

## Create new var outlet with outlet numbers 0 - 9. 
## one hot coding categorical vars and make new dataframe. 

comb2$Outlet <- comb2$Outlet_Identifier
levels(comb2$Outlet)<- 0:9

comb3 <- comb2
levels(comb3$Outlet_Location_Type) <- 1:3
levels(comb3$Outlet_Type) <- 0:3
dm <- dummyVars(~ Item_Fat_Content + Outlet_Size + Outlet_Location_Type +
                        Outlet_Type+Item_Type_General+ Outlet, data = comb3)
tr <- data.frame(predict(dm, newdata = comb3))

dum <- c("Item_Fat_Content", "Outlet_Size", "Outlet_Location_Type",
                 "Outlet_Type", "Item_Type_General", "Outlet")
comb4 <- cbind(comb3[,!(colnames(comb3) %in% dum)],tr)
comb4 <- comb4[,-c(4,7)]

## Seperate the data in training and testing data 

train4 <- comb4[!(is.na(comb4$Item_Outlet_Sales)),]
test4 <- comb4[is.na(comb4$Item_Outlet_Sales),]

test4$Item_Outlet_SalesN <- test4$Item_Outlet_Sales
test4 <- test4[,-6]
train4$Item_Outlet_SalesN <- train$Item_Outlet_Sales
train4 <- train4[,-6]
test4 <- test4[,-34]
train4$Item_Outlet_Sales <- train4$Item_Outlet_SalesN
train4<- train4[,-34]

## Seperate the training data for cross - validation. 

inTrain <- createDataPartition(train4$Item_Outlet_Sales, p = 0.75, list = FALSE)
train4tr <- train4[inTrain,]
train4te <- train4[-inTrain,]

## Model Building 

### Baseline model 


## A baseline model is made using the mean of the overall sales. 

bmodel_meanS <- mean(train4tr$Item_Outlet_Sales)
bmodel_meanS_pred <- rep(bmodel_meanS, times = dim(train4tr)[1])
rmsetrS <- sqrt(mean((bmodel_meanS_pred - train4tr$Item_Outlet_Sales)^2))
rmsetrS

bmodel_meanS_predte <- rep(bmodel_meanS, times = dim(train4te)[1])
rmsetrSte <- sqrt(mean((bmodel_meanS_predte - train4te$Item_Outlet_Sales)^2))
rmsetrSte

##A baseline model is made using the mean of  the sales of each product. 


bmodel_mean <- aggregate(Item_Outlet_Sales ~ Item_Identifier, data = train4tr,
                         mean)
bmodel_mean_pred <- rep(0, times = dim(train4tr)[1])
for(i in 1:(dim(train4tr)[1])){
        bmodel_mean_pred[i]<- bmodel_mean[bmodel_mean$Item_Identifier == 
                                                  train4tr$Item_Identifier[i],"Item_Outlet_Sales"]
        
}
rmsetr <- sqrt(mean((bmodel_mean_predtr-train4tr$Item_Outlet_Sales)^2))

bmodel_mean_predte <- rep(0, times = dim(train4te)[1])
for(i in 1:(dim(train4te)[1])){
        pr3 <- train4te$Item_Identifier[i]
        bmodel_mean_predte[i]<- bmodel_mean[pr3,"Item_Outlet_Sales"]
        
}
sum(is.na(bmodel_mean_predte))

for(i in 1:length(bmodel_mean_predte)){
        if(is.na(bmodel_mean_predte[i]))
        {bmodel_mean_predte[i]<- 0}
}
sum(is.na(bmodel_mean_predte))
rmsete <- sqrt(mean((bmodel_mean_predte-train4te$Item_Outlet_Sales)^2))

### Linear regression model

comb3 <- comb3[,-c(5,8)]
Item_Outlet_Sales <- comb3$Item_Outlet_Sales
comb3 <- comb3[,-10]
comb3$Item_Outlet_Sales <- Item_Outlet_Sales

train3 <- comb3[!(is.na(comb3$Item_Outlet_Sales)),]
test3 <- comb3[is.na(comb3$Item_Outlet_Sales),]
train3te <- train3[-inTrain,]
train3tr <- train3[inTrain,]

control <- trainControl(method = "cv", number = 10)

set.seed(2207)
model_lin <- train(Item_Outlet_Sales ~ . -Item_Identifier -Outlet_Identifier
                   -Outlet, data = train3tr,trControl = control,method = "lm", 
                   metric = "RMSE")
summary(model_lin)
model_lin
png("barplot1.png")
barplot(model_lin$finalModel$coefficients, ylim = c(-2000, 3500))
dev.off()

set.seed(2207)
model_lin2 <- train(Item_Outlet_Sales ~ . -Item_Identifier -Outlet_Identifier 
                   -Item_Fat_Content -Outlet, data = train3tr,
                   trControl = control,method = "lm", metric = "RMSE")
summary(model_lin2)
model_lin2
png("barplot2.png")
barplot(model_lin2$finalModel$coefficients, ylim = c(-2000, 3500))
dev.off()

fil_l <- lm(Item_Outlet_Sales ~ . -Item_Identifier -Outlet_Identifier, 
            data = train3tr, metric = "RMSE")
bestmod <- step(model_lin2, direction = "backward")

set.seed(2207)
model_lin3 <- train(Item_Outlet_Sales ~ Item_Fat_Content + Item_MRP + Outlet,
                    data = train3tr,trControl = control,method = "lm",
                    metric = "RMSE")
model_lin3
png("barplot3.png")
barplot(model_lin3$finalModel$coefficients, ylim = c(-2000, 3500))
dev.off()

pred_lin3 <- predict(model_lin3, newdata = train3te)
RMSE(pred_lin3, train3te$Item_Outlet_Sales)

###  Ridge 

set.seed(2207)
ridge <- train(Item_Outlet_Sales ~. -Item_Identifier -Outlet_Identifier,
               data = train3tr, method='ridge',lambda = 4)
ridge.pred <- predict(ridge, train3te)
RMSE(ridge.pred,train3te$Item_Outlet_Sales)


## Tune the lambda value
set.seed(2207)
fitControl <- trainControl(method = "cv", number = 10)
lambdaGrid <- expand.grid(lambda = 10^seq(10, -2, length=100))

set.seed(2207)
ridge2 <- train(Item_Outlet_Sales~. -Item_Identifier -Outlet_Identifier, 
               data = train3tr, method='ridge',trControl = fitControl,
               tuneGrid = lambdaGrid)
ridge2.pred <- predict(ridge2, train3te)
RMSE(ridge2.pred,train3te$Item_Outlet_Sales)
ridge2

## Lasso 

set.seed(2207)
Grid = expand.grid(alpha = 1,lambda = seq(0.001,0.1,by = 0.001))
set.seed(2207)
lasso <- train(Item_Outlet_Sales ~. -Item_Identifier -Outlet_Identifier,
               data = train3tr, method='glmnet', trControl = fitControl, 
               tuneGrid = Grid)
lasso.pred <- predict(lasso, train3te)
RMSE(lasso.pred,train3te$Item_Outlet_Sales)

### Decision Tree 

set.seed(2207)
fitpart <- rpart(Item_Outlet_Sales ~.,data = train3tr[,-c(1,6)], method = "anova"
                 , model = TRUE)
fitpart.pred <- predict(fitpart, newdata = train3te[,-c(1,6)])
RMSE(fitpart.pred, train3te$Item_Outlet_Sales)

## Tune the model

set.seed(2207)
fitpart2 <- rpart(Item_Outlet_Sales ~.,data = train3tr[,-c(1,6)], method = "anova"
                 , model = TRUE, maxdepth = 15, minsplit = 100)
fitpart.pred2 <- predict(fitpart2, newdata = train3te[,-c(1,6)])
RMSE(fitpart.pred2, train3te$Item_Outlet_Sales) 

set.seed(2207)
fitpart3 <- rpart(Item_Outlet_Sales ~.,data = train3tr[,-c(1,6)], method = "anova"
                  , model = TRUE, maxdepth = 8, minsplit = 150)
fitpart.pred3 <- predict(fitpart3, newdata = train3te[,-c(1,6)])
RMSE(fitpart.pred3, train3te$Item_Outlet_Sales)


### Random Forest 

set.seed(2207)
rf_fit <- randomForest(Item_Outlet_Sales ~ . , data = train3tr[,-c(1,6)])
rf_fit.pred <- predict(rf_fit, newdata = train3te[,-c(1,6)])
RMSE(rf_fit.pred, train3te$Item_Outlet_Sales)
                                                  
## Tune the model 

set.seed(2207)
rf_fit2 <- randomForest(Item_Outlet_Sales ~ . , data = train3tr[,-c(1,6)], 
                       ntree = 500, nodesize = 100)
rf_fit.pred2 <- predict(rf_fit2, newdata = train3te[,-c(1,6)])
RMSE(rf_fit.pred2, train3te$Item_Outlet_Sales)

## Random forest using the caret package 

set.seed(2207)

tgrid <- expand.grid(.mtry = c(3:10), .splitrule = "variance", 
                     .min.node.size = c(10,15,20))

rf_fitC <- train(Item_Outlet_Sales~.,data = train3tr[,-c(1,6)], 
                 method = "ranger", trControl = control, 
                 tuneGrid = tgrid, num.trees = 400, importance = "permutation")

rf_fitC.pred <- predict(rf_fitC, newdata = train3te[,-1,6])
RMSE(rf_fitC.pred,train3te$Item_Outlet_Sales)


### Xgb (with the one-hot encoded data set)

param_list <- list(objective = "reg:linear", eta = 0.01, gamma = 1, 
                   max_depth = 6, subsample = 0.8, colsample_bytree = 0.5) 

dtrain <- xgb.DMatrix(data = data.matrix(train4tr[,!(names(train4tr) %in% 
                                                             c("Item_Identifier",
                                                               "Outlet_Identifier",
                                                               "Item_Outlet_Sales"))])
                      ,label = train4tr$Item_Outlet_Sales)

dtest <- xgb.DMatrix(data = data.matrix(train4te[,!(names(train4te) %in% 
                                                            c("Item_Identifier",
                                                              "Outlet_Identifier",
                                                              "Item_Outlet_Sales"))]))

## cross-validation 

set.seed(2207)
xgb.cv(params = param_list, data = dtrain, nrounds = 1000, nfold = 5, 
       print_every_n = 10, early_stopping_rounds = 30, maximize = FALSE)

## Best result in 529 

set.seed(2207)
xgb_fit <- xgb.train(data = dtrain, params = param_list, nrounds = 529)
xgb.pred <- predict(xgb_fit, newdata = dtest)
RMSE(xgb.pred, train4te$Item_Outlet_Sales)

## variance importance 

var_imp <- xgb.importance(feature_names = setdiff(names(train4tr),
                                                  c("Item_Identifier",
                                                    "Outlet_Identifier",
                                                    "Item_Outlet_Sales")), 
                          model = xgb_fit)

png("xgb_imp.png")
xgb.plot.importance(var_imp)
dev.off()

### Final Training with the whole training data set. 

## XGBoost 

### Xgb (with the one-hot encoded data set)

param_list <- list(objective = "reg:linear", eta = 0.01, gamma = 1, 
                   max_depth = 6, subsample = 0.8, colsample_bytree = 0.5) 

dtrain <- xgb.DMatrix(data = data.matrix(train4[,!(names(train4) %in% 
                                                           c("Item_Identifier","Outlet_Identifier",
                                                             "Item_Outlet_Sales"))]) ,label = train4$Item_Outlet_Sales)

dtest <- xgb.DMatrix(data = data.matrix(test4[,!(names(test4) %in%                                      c("Item_Identifier","Outlet_Identifier",
                                                                                                          "Item_Outlet_Sales"))]))

## cross-validation 

set.seed(2207)
xgb.cv(params = param_list, data = dtrain, nrounds = 1000, nfold = 5, 
       print_every_n = 10, early_stopping_rounds = 30, maximize = FALSE)

## Best result in iteration 590 

set.seed(2207)
xgb_fit_fin <- xgb.train(data = dtrain, params = param_list, nrounds = 590)
xgb.pred_fin <- predict(xgb_fit_fin, newdata = dtest)

## variance importance 

var_imp <- xgb.importance(feature_names = setdiff(names(train4),
                                                  c("Item_Identifier",
                                                    "Outlet_Identifier",
                                                    "Item_Outlet_Sales")), 
                          model = xgb_fit_fin)

png("imp_plot_fin.png")
xgb.plot.importance(var_imp)
dev.off()

## Random Forest 

set.seed(2207)
rf_fit_fin <- randomForest(Item_Outlet_Sales ~ . , data = train3[,-c(1,6)], 
                           ntree = 500, nodesize = 100)
rf_fit.pred_fin <- predict(rf_fit_fin, newdata = test3[,-c(1,6)])

### submission file 

sub_col <- c("Item_Identifier", "Outlet_Identifier")
subm <- test4[,sub_col]
subm_fin <- cbind(subm,xgb.pred_fin)
names(subm_fin)[3] <- "Item_Outlet_Sales" 

write.csv(subm_fin, file = "submission_file_mk.csv", row.names = FALSE)
