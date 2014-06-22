### Course Project - Practical Machine Learning
========================================================
### Building a machine learning algorithm to predict activity quality from activity monitors.

Human Activity Recognition - <b>HAR</b> - has emerged as a key research area in the last years and is gaining increasing attention by the pervasive computing research community, especially for the development of context-aware systems. Further details can be found <a href= http://groupware.les.inf.puc-rio.br/har>here</a>. <br>
The data <a href= http://groupware.les.inf.puc-rio.br/har>they</a> have provided was collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
trainingData <- read.csv("~/Courses/Practical Machine Learning//pml-training.csv", 
    na.strings = c("NA", ""))
testingData <- read.csv("~/Courses/Practical Machine Learning//pml-testing.csv")
```

The goal of this project is to predict the manner in which they did the exercise. The manner has been divided into following classes:

```r
levels(trainingData$classe)
```

```
## [1] "A" "B" "C" "D" "E"
```

<br>
### Cleaning and setting the data:
Removing columns with missing values:

```r
# trainingData <- trainingData[,!sapply(data,function(x) any(is.na(x)))]
trainingData <- trainingData[, which(colSums(is.na(trainingData)) == 0)]
```

Also, we remove any variables(columns) that doesn't seem relative(non-sensor data) to the prediction model:

```r
trainingData <- trainingData[, -grep("timestamp|X|user_name|new_window|num_window", 
    names(trainingData))]
```

After removing the columns with missing values, we get following number of variables:

```r
names(trainingData)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

The variable we are predicting is "<b>classe</b>".
Now, we partition the data into training set(70%) and cross-validation set(30%).

```r
set.seed(1)
inTrain <- createDataPartition(y = trainingData$classe, p = 0.7, list = FALSE)
training <- trainingData[inTrain, ]
cv <- trainingData[-inTrain, ]
```

Feature density plot:

```r
featurePlot(x = training[, -53], y = training$classe, plot = "density", auto.key = list(columns = 5))
```

![plot of chunk densityPlot](figure/densityPlot.png) 


<br>
### Building the prediction model:
Rather than using the default training method(bootstrapping), we are using <i>cross-validation</i> method with 5 folds to train the data as it seems to be more efficient and fast.

```r
tctrl <- trainControl(method = "cv", number = 5)
modFit <- train(classe ~ ., data = training, method = "rf", trControl = tctrl)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

<br>
Summary of our fitted model:

```r
modFit
```

```
## Random Forest 
## 
## 3927 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 3142, 3142, 3142, 3142, 3140 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.008        0.01    
##   30    1         1      0.008        0.01    
##   50    1         1      0.01         0.02    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
plot(modFit)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7.png) 

We can see from above that we have achieved accuracy of 100% using 5 fold Cross-Validated resampling with just 52 predictors.
<br>
### Prediction and errors:
As for the prediction statistics on cross-validation set:

```r
prediction <- predict(modFit, newdata = cv)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
confusionMatrix(prediction, cv$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1660   21    0    0    0
##          B   10 1097   29    0    8
##          C    1   20  991   19    7
##          D    2    1    6  942    6
##          E    1    0    0    3 1061
## 
## Overall Statistics
##                                         
##                Accuracy : 0.977         
##                  95% CI : (0.973, 0.981)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.971         
##  Mcnemar's Test P-Value : 0.000241      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.992    0.963    0.966    0.977    0.981
## Specificity             0.995    0.990    0.990    0.997    0.999
## Pos Pred Value          0.988    0.959    0.955    0.984    0.996
## Neg Pred Value          0.997    0.991    0.993    0.996    0.996
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.282    0.186    0.168    0.160    0.180
## Detection Prevalence    0.286    0.194    0.176    0.163    0.181
## Balanced Accuracy       0.993    0.977    0.978    0.987    0.990
```

With an accuracy of 97.7%, our <i><b>out-of-sample error</b> is about <b>2.3 %</b></i>
<br>
<br>
