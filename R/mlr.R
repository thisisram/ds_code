startup_data = read.csv(('50_Startups.csv'))

unique(startup_data$State)
startup_data$State = factor(startup_data$State, 
                              levels=c('New York', 'California', 'Florida'),
                              labels=c(1,2,3))

library(caTools)
set.seed(123)
split = sample.split(startup_data,SplitRatio = 0.8)
startup_train = subset(startup_data, split==TRUE)
startup_test = subset(startup_data, split==FALSE)

regressor = lm(formula = Profit ~ ., data = startup_train)
summary(regressor)

y_pred = predict(regressor, newdata = startup_test)

