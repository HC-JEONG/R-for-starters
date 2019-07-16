setwd('C:/DataScience')
getwd()

#필요한 라이브러리 활성화
library(pastecs)
library(STAT)
library(glmnet)
library(lasso2)

#BostonHousing2.csv 불러오기
Boston<-read.csv('BostonHousing2.csv')
head(Boston)

#rm이 6.5보다 작고 medv가 50인 것은 이상치로 판단하여 제거한 데이터를 BostonOut으로 설정
outlier<-c(which(Boston$rm<6.5&Boston$medv==50))
Boston[outlier,]
plot(medv~rm,data=Boston)
points(Boston[outlier,]$rm,Boston[outlier,]$medv,pch=16,col='red')

BostonOut<-Boston[-outlier,]
head(BostonOut)

# OLS, Stepwise, Ridge, Lasso 모형 비교
# Ridge, Lasso를 하기 위해 범주형 변수(rad)를 binary coding으로 transform한다 (앞에서는 rad를 범주형으로 처리)


#rad를 category화
BostonCat<-BostonOut
BostonCat$rad<-as.factor(BostonCat$rad)
str(BostonCat) #rad가 factor인 것을 확인할 수 있음
BostonCat$rad
#rad에 2,3,4,5,6,7,8,24를 1, 0으로 이진화하여 BostonCat에 추가하고 BostonTrans로 데이터명 설정
rad2 <- ifelse(BostonCat$rad == 2, 1, 0)
rad3 <- ifelse(BostonCat$rad == 3, 1, 0)
rad4 <- ifelse(BostonCat$rad == 4, 1, 0)
rad5 <- ifelse(BostonCat$rad == 5, 1, 0)
rad6 <- ifelse(BostonCat$rad == 6, 1, 0)
rad7 <- ifelse(BostonCat$rad == 7, 1, 0)
rad8 <- ifelse(BostonCat$rad == 8, 1, 0)
rad24 <- ifelse(BostonCat$rad == 24, 1, 0)
BostonTrans <- cbind(BostonCat, rad2, rad3, rad4, rad5, rad6, rad7, rad8, rad24)
str(BostonTrans)


# rm->rm^2, nox->nox^2, lstat->ln(lstat), dis->ln(dis)로 변수변환해서 BostonTrans 데이터에 추가
BostonTrans <- with(BostonTrans, cbind(BostonTrans, rm2=rm^2, nox2=nox^2, 
                                       llstat=log(lstat), ldis=log(dis))) #with함수 : with(가져올 데이터, 불러올 변수명)
head(BostonTrans)
str(BostonTrans)

# set.seed(32)로 난수 고정하고 BostonTrans의 80%를 train data, 20%를 test data로 설정.
set.seed(32)
sample.no<-sample(1:nrow(BostonTrans),nrow(BostonTrans)*0.8)
BostonTrans.train<-BostonTrans[sample.no,]
BostonTrans.test<-BostonTrans[-sample.no,]


#################### OLS #########################

#rad는 binary로 변환했으므로 회귀식에서 rad는 뺀다
fit.Trans <- lm(log(medv) ~ .-rm-rad-nox-lstat-dis , data= BostonTrans.train) #log(medv)가 종속변수, 나머지 모든 변수를 독립변수로 설정, rm, rad, nox, lstat, dis는 변수변환을 했으므로 식에서 빼준다
summary(fit.Trans)

# 훈련집합의 R squared와 RMSE 계산
#R^2=1-(SSE/SST)
1 - sum( (BostonTrans.train$medv - exp(fit.Trans$fitted.values))^2 ) / 
  sum( (BostonTrans.train$medv - mean(BostonTrans.train$medv))^2 ) #종속변수를 로그화 했었으니까 다시 지수화 해줘야 함

#RMSE
sqrt(mean( (exp(fit.Trans$fitted.values) - BostonTrans.train$medv)^2 ))

#test set의 R squared와 RMSE
#R^2=1-(SSE/SST)
1-sum((BostonTrans.test$medv-exp(predict(fit.Trans,newdata=BostonTrans.test)))^2)/
  sum((BostonTrans.test$medv-mean(BostonTrans.test$medv))^2)
sqrt(mean( (exp(predict(fit.Trans,newdata=BostonTrans.test))-BostonTrans.test$medv)^2))


AIC(fit.Trans)

################## Stepwise ##################

fit.step <- step(fit.Trans, direction = "both")
summary(fit.step)

# 훈련집합의 R squared와 RMSE 계산
1 - sum( (BostonTrans.train$medv - exp(fit.step$fitted.values))^2 ) / 
  sum( (BostonTrans.train$medv - mean(BostonTrans.train$medv))^2 )
sqrt(mean( (exp(fit.step$fitted.values) - BostonTrans.train$medv)^2 ))

#test set의 R squared와 RMSE
1-sum((BostonTrans.test$medv-exp(predict(fit.step,newdata=BostonTrans.test)))^2)/
  sum( (BostonTrans.test$medv - mean(BostonTrans.test$medv))^2 )

sqrt(mean((BostonTrans.test$medv - exp(predict(fit.step, newdata=BostonTrans.test)))^2))

AIC(fit.step)

################## Ridge ##############################

fit.ridge <- glmnet(as.matrix(BostonTrans.train[,-14]), log(BostonTrans.train$medv), alpha = 0)
plot(fit.ridge, xvar= "lambda")

set.seed(1234)
fit.cv.ridge <- cv.glmnet(data.matrix(BostonTrans.train[,-14]), log(BostonTrans.train$medv), alpha = 0)
plot(fit.cv.ridge)

grid <- seq(fit.cv.ridge$lambda.min, fit.cv.ridge$lambda.1se, length.out = 5)
fit.ridge <- glmnet(as.matrix(BostonTrans.train[,-14]), log(BostonTrans.train$medv), alpha = 0, lambda=grid)
head(fit.ridge)

# 훈련집합의 R squared와 RMSE 계산
ridge.fitted.value <- predict(fit.ridge, newx=data.matrix(BostonTrans.train[,-14]))
1 - colSums( (BostonTrans.train$medv - exp(ridge.fitted.value))^2 ) / 
  sum( (BostonTrans.train$medv - mean(BostonTrans.train$medv))^2 ) # R squared
sqrt(colMeans( (exp(ridge.fitted.value) - BostonTrans.train$medv)^2 )) # RMSE

#test set의 R squared와 RMSE
# R squared #
ridge.fitted.value2 <- predict(fit.ridge, newx=data.matrix(BostonTrans.test[,-14]))
1 - colSums( (BostonTrans.test$medv - exp(ridge.fitted.value2))^2 ) / 
  sum( (BostonTrans.test$medv - mean(BostonTrans.test$medv))^2 ) 
# RMSE #      
sqrt(colMeans( (exp(ridge.fitted.value2) - BostonTrans.test$medv)^2 )) 


####################### Lasso #############################

fit.lasso <- glmnet(data.matrix(BostonTrans.train[,-14]), log(BostonTrans.train$medv), alpha = 1)
plot(fit.lasso, xvar= "lambda")
set.seed(1234)
fit.cv.lasso <- cv.glmnet(data.matrix(BostonTrans.train[,-14]), log(BostonTrans.train$medv), alpha = 1)
plot(fit.cv.lasso)

grid <- seq(fit.cv.lasso$lambda.min, fit.cv.lasso$lambda.1se, length.out = 5)
fit.lasso <- glmnet(data.matrix(BostonTrans.train[,-14]), log(BostonTrans.train$medv), alpha = 1, lambda=grid)
head(fit.lasso)

# 훈련집합의 R squared와 RMSE 계산
lasso.fitted.value <- predict(fit.lasso, newx=data.matrix(BostonTrans.train[,-14]))
1 - colSums( (BostonTrans.train$medv - exp(lasso.fitted.value))^2 ) / 
  sum( (BostonTrans.train$medv - mean(BostonTrans.train$medv))^2 ) # R squared
sqrt(colMeans( (exp(lasso.fitted.value) - BostonTrans.train$medv)^2 )) # RMSE


#test set의 R squared와 RMSE
# R squared #
lasso.fitted.value2<-predict(fit.lasso,newx=data.matrix(BostonTrans.test[,-14]))
1-colSums((BostonTrans.test$medv - exp(lasso.fitted.value2))^2) /
  sum((BostonTrans.test$medv - mean(BostonTrans.test$medv))^2)
# RMSE #
sqrt(colMeans((exp(lasso.fitted.value2) - BostonTrans.test$medv)^2))