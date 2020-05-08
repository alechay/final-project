# Predicting human activity using smartphone accelerometer data

## Purpose

The goal of this project is to use machine learning classification algorithms to predict the type of activity a human subject is performing, given a 2.56s window of accelerometer data sampled at 50Hz. I wanted to compare model performance on the features that I extracted from the raw data, and model performance on the pre-engineered features that came with the dataset. In addition, I used feature selection to find the features that best predict the class of activity.

## Repository Contents

1. Final-project description notebook
2. Data exploration notebook
3. Personally engineered features notebook
4. Pre-engineered features notebook
5. Feature selection notebook

## Data

The movement traces were measured by smartphone accelerometers and were taken while 30 human subjects perfomed 6 different activities: walking, walking upstairs, walking downstairs, sitting, standing, or laying down. Measurements were taken in 2.56s long observation windows that had 50% overlap. In each observation window, 9 movement traces were measured: total acceleration, body acceleration, and body gyro in the x, y, and z directions. Each observation window was mapped to the class of activity that was being performed (my y-value), and the subject who performed it. This data was open source and was taken from https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones.

## Methods

I followed the article in this link, https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/, to explore the dataset. This helped me write functions to load the data and visualize it. For a given subject, I made graphs of the 9 movement traces overlayed with the activity ID that they were performing. I then made histograms that examined the distributions of acceleration measurements for one type of motion across subjects. Then, I made histograms that examined the distribution of acceleration measurements for one class of activity across subjects. This information was useful for which features I would select to create my X data. Finally, I made boxplots to confirm that subjects spent approximately equal amounts of time performing each class of activity.

I then tried to train models on the raw data, and on features that I extracted from the data. I decided to extract 8 features from the 9 movement traces: mean, median, variance, minimum, maximum, interquartile range, 20th percentile and 80th percentile. This created an array of 72 features for each observation. I made this decision by consulting the article in the following link: https://www.mdpi.com/1424-8220/17/9/2058/htm. I trained 5 classification models on this data: KNN, logistic regression, one-vs-rest, random forest, and gradient boosting. I used K fold cross validation and grid search to evaluate my model and optimize the hyperparameters. I also evaluated all the models on a set train-test split with a classification report. For some of the models, I also evaluated their performance using a confusion matrix, an ROC curve, and/or a precision-recall curve. I later compared model performance on the features that I extracted from the raw data, and model performance on the pre-engineered features that came with the dataset.

I then trained those same 5 models on the pre-engineered data in the dataset. This data had 561 features per observation vs mine which had 72. You can read more about the features that the authors of the dataset engineered in their README and features_info files. For this dataset, I used the exact same hyperparameter optimization and evaluation techniques as I did for my feature-extracted dataset. I compared the perfomance of the models on the different datasets to each other.

I then did feature selection to find the features, for both my feature extracted dataset and the pre-engineered dataset, that best predicted the class of activity. I used variance thresholding and recursive feature elimination to do this.

## Results

I found that the models I trained on the raw data performed suprisingly well, at a classification accuracy of about 0.87. However, I was able to improve the performance after extracting features from the data. My best performing model, the optimized KNN classifier, performed at about 0.96 accuracy, which was slightly worse than the best performing model on the pre-engineered dataset. The other models did not perform as well on my feature-engineered dataset. The logistic regression, one vs rest, random forest, and gradient boosting classifiers performed at 0.90, 0.90, 0.93, and 0.94 accuracy, respectively. The best performing models on the pre-engineered dataset were the logistic regression and one vs rest classifiers, which performed at 0.98 accuracy. The KNN, random forest, and gradient boosting classifiers performed at 0.96, 0.95, and 0.96 accuracy, respectively. Using variance thresholding and recursive feature elimination, I also found the features that best predicted the class of activity. I found that using more features, in both my feature extracted dataset and the pre-engineered dataset, only increased performance of the model up to a certain point. After a while, in fact, it seems that using more features decreases performance of the model.

## Discussion

It is interesting that the model performed reasonably well on the raw data. Perhaps if you give any model a ton of data, even though that data may not be represented in a sensical way, you will get somewhat accurate classifications. It is also interesting that the KNN classifier was the best performing model on my feature-engineered data but not on pre-engineered data. I am not sure why that might be. In general, though, it makes sense that the models trained on the pre-engineered data were more accurate because they had more features to base their predictions on. If I want to improve the performance of the models on my data, I will need to extract better features. Through variance thresholding and recursive feature elimination, I found that only about 50 features are needed to accurately predict the class of activity, and after that the accuaracy begins to decline. To improve classification accuracy I could also try other, more complicated classification models such as convolutional neural networks (CNNs) or recurrent neural networks like long short-term memory networks (LSTMs).

## Future directions

I would like to apply what I have learned about feature selection to better tune the models. I could do feature number optimization on other models than the random forest classifier, to produce the best performing model.

## Bugs

No bugs in the code at this time.

## Sources

* **Dataset:**
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
* **Data exploration:**
Brownlee, Jason. “How to model human activity from smartphone data.” Machine Learning Mastery, 17 September 2018, https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/.
* **Ideas on how to select features of the data:** 
Martin, et al. “Methods for Real-Time Prediction of the Mode of Travel Using Smartphone-Based GPS and Accelerometer Data.” MDPI, Multidisciplinary Digital Publishing Institute, 8 Sept. 2017, www.mdpi.com/1424-8220/17/9/2058/htm.
* **Class notes and sklearn documentation**