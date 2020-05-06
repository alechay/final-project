# Predicting human activity using smartphone accelerometer data

## Purpose

The goal of this project is to use machine learning classification algorithms to predict the type of activity a human subject is performing, given a 2.56s window of accelerometer data sampled at 50Hz. I wanted to compare model performance on the features that I extracted from the raw data, and model performance on the pre-engineered features that came with the dataset. In addition, I plan to use dimensionality reduction to find the features that best predict the class of activity.

## Data

The movement traces were measured by smartphone accelerometers and were taken while 30 human subjects perfomed 6 different activities: walking, walking upstairs, walking downstairs, sitting, standing, or laying down. Measurements were taken in 2.56s long observation windows that had 50% overlap. In each observation window, 9 movement traces were measured: total acceleration, body acceleration, and body gyro in the x, y, and z directions. Each observation window was mapped to the class of activity that was being performed (my y-value), and the subject who performed it. This data was open source and was taken from https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones.

## Methods

I followed the article in this link, https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/, to explore the dataset. This helped me write functions to load the data and visualize it. For a given subject, I made graphs of the 9 movement traces overlayed with the activity ID that they were performing. I then made histograms that examined the distributions of acceleration measurements for one type of motion across subjects. Then, I made histograms that examined the distribution of acceleration measurements for one class of activity across subjects. This information was useful for which features I would select to create my X data. Finally, I made boxplots to confirm that subjects spent approximately equal amounts of time performing each class of activity.

I then tried to train models on the raw data, and on features that I extracted from the data. I decided to extract 8 features from the 9 movement traces: mean, median, variance, minimum, maximum, interquartile range, 20th percentile and 80th percentile. This created an array of 72 features for each observation. I made this decision by consulting the article in the following link: https://www.mdpi.com/1424-8220/17/9/2058/htm. I trained 5 classification models on this data: KNN, logistic regression, one-vs-rest, random forest, and gradient boosting. I used K fold cross validation and grid search to evaluate my model and optimize the hyperparameters. I also evaluated all the models on a set train-test split with a classification report. For some of the models, I also evaluated their performance using a confusion matrix, an ROC curve, and/or a precision-recall curve. I later compared model performance on the features that I extracted from the raw data, and model performance on the pre-engineered features that came with the dataset.

I then trained those same 5 models on the pre-engineered data in the dataset. This data had 561 features per observation vs mine which had 72. You can read more about the features that the authors of the dataset engineered in their README and features_info files. For this dataset, I used the exact same hyperparameter optimization and evaluation techniques as I did for my feature-extracted dataset. I compared the perfomance of the models on the different datasets to each other.

## Results

I found that the models I trained on the raw data performed suprisingly well, at a classification accuracy of about 0.87. However, I was able to improve the performance after extracting features from the data. My best performing model, the optimized KNN classifier, performed at about 0.96 accuracy, which was slightly worse than the best performing model on the pre-engineered dataset. The other models did not perform as well on my feature-engineered dataset. The logistic regression, one vs rest, random forest, and gradient boosting classifiers performed at 0.90, 0.90, 0.93, and 0.94 accuracy, respectively. The best performing models on the pre-engineered dataset were the logistic regression and one vs rest classifiers, which performed at 0.98 accuracy. The KNN, random forest, and gradient boosting classifiers performed at 0.96, 0.95, and 0.96 accuracy, respectively.

## Discussion

It is interesting that the model performed reasonably well on the raw data. Perhaps if you give any model a ton of data, even though that data may not be represented in a sensical way, you will get somewhat accurate classifications. It is also interesting that the KNN classifier was the best performing model on my feature-engineered data but not on pre-engineered data. I am not sure why that might be. In general, though, it makes sense that the models trained on the pre-engineered data were more accurate because they had more features to base their predictions on. If I want to improve the performance of the models on my data, I will need to extract better features. I could also try other, more complicated classification models such as convolutional neural networks (CNNs) or recurrent neural networks like long short-term memory networks (LSTMs).

## Future directions

I would like to do some dimensionality reduction to find which features are the most predictive of the data. This could potentially be done through principal component analysis (PCA), but there are other methods. Based on the data exploration, I would expect that the total acceleration movement traces are the best predictors of activity.

## Bugs

No bugs in the code at this time.

## Sources

* Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
* Data exploration: https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/
* Ideas on how to select features of the data: https://www.mdpi.com/1424-8220/17/9/2058/htm
* Class notes and sklearn documentation