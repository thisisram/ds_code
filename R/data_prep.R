data_frame = read.csv('Data.csv')
X = data_frame[c(1:3)]
y = data_frame[4]

data_frame$Age = ifelse(is.na(data_frame$Age),
                        ave(data_frame$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                        data_frame$Age)
data_frame$Salary = ifelse(is.na(data_frame$Salary),
                           ave(data_frame$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                           data_frame$Salary)

library(caTools)
set.seed(123)
split = sample.split(data_frame$Purchased, SplitRatio = 0.8)
train_set = subset(data_frame, split==TRUE)
test_set = subset(data_frame, split==FALSE)

train_set[,2:3] = scale(train_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])
