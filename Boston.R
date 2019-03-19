library(tree)


path <- "/Users/jacquessham/Documents/SelfProjects/Boston_housing"
setwd(path)
# Load Data into dataframe
boston <- read.csv("boston.csv")

set.seed(49)
rows <- nrow(boston)
train_index <- sample(rows, size=0.8*rows)

train <- boston[train_index,]
test <- boston[-train_index,]


#### Do Model 1 in R #####
lm_slr <- lm(medv ~ rm, train)
summary(lm_slr)
sqrt(mean((predict(lm_slr, test, se.fit = F) - test$medv)**2))

#### Do Model 2 in R #####
lm_lr <- lm(medv ~ crim + zn + indus + factor(chas) + nox + rm + age +
              dis + factor(rad) + tax + ptratio + lstat, train)
summary(lm_lr)
sqrt(mean((predict(lm_lr, test, se.fit = F) - test$medv)**2))

#### Do Model 3 in R #####
lm_dt <- tree(medv ~ crim + zn + indus + factor(chas) + nox + rm + age +
                 dis + factor(rad) + tax + ptratio + lstat, data=train)
summary(lm_dt)
plot(lm_dt)
title("Tree Regressor Model")
text(lm_dt, cex = 0.7, use.n = TRUE, fancy = FALSE, all = TRUE)
sqrt(mean((predict(lm_dt, test, se.fit = F) - test$medv)**2))
