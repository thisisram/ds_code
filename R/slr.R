salary_data = read.csv('Salary_Data.csv')

library(caTools)
set.seed(123)
split = sample.split(salary_data$Salary, SplitRatio = 2/3)
salary_train = subset(salary_data, split==TRUE)
salary_test = subset(salary_data, split==FALSE)

regressor = lm(formula = Salary ~ YearsExperience, data = salary_train)

summary(regressor)

y_pred = predict(regressor, newdata = salary_test)

str(y_pred)

library(ggplot2)

ggplot() +  geom_point(aes(x = salary_test$YearsExperience, y = salary_test$Salary), colour = 'blue') +
  geom_line(aes(x=salary_test$YearsExperience, y=y_pred), colour='red') +
  ggtitle("Learnign Curve") +
  xlab("Years of Experience") +
  ylab("Salary")