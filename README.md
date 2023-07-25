# The-Quora-Question-Pair-Similarity-Analysis
The Quora Question Pair Similarity Analysis

# Introduction

Quora is a platform for Q&A, just like StackOverflow. But quora is more of a general-purpose Q&A platform that means there is not much code like in StackOverflow.

One of the many problems that quora face is the duplication of questions. Duplication of question ruins the experience for both the questioner and the answerer. 
Since the questioner is asking a duplicate question, we can just show him/her the answers to the previous question. And the answerer doesn't have to repeat his/her answer for essentially the same questions.

For example, we have a question like "How can I be a good geologist?" and there are some answers to that question. 
Later someone else asks another question like "What should I do to be a great geologist?". We can see that both the questions are asking the same thing. Even though the wordings for the question are different, the intention of both questions is same.

So the answers will be same for both questions. That means we can just show the answers of the first question. That way the person who is asking the question will get the answers immediately and people who have answered already the first question don't have to repeat themselves.

This problem is available on Kaggle as a competition. https://www.kaggle.com/c/quora-question-pairs

So given two questions, our main objective is to find whether they are similar. So let’s do some magic with ML. 🪄

# Business Objectives and Constraints

There is no strict latency requirement.
We would like to have interpretability but it is not absolutely mandatory.
The cost of misclassification is medium.
Both classes (duplicate or not) are equally important.

#Data Overview
Available Columns: id, qid1, qid2, question1, question2, is_duplicate
Class labels: 0, 1
Total training data / No. of rows: 404290
No. of columns: 6
is_duplicate is the dependent variable.
No. of non-duplicate data points is 255027
No. of duplicate data points is 149263

We have 404290 training data points. And only 36.92% are positive. That means it is an imbalanced dataset.

# Business Metrics
It is a binary classification.

# We need to minimize the log loss for this challenge.
Basic EDA
est data don’t have question ids. So the independent variables are question1, question2 and the dependent variable is is_duplicate.

3 rows had null values. So We removed them and now We have 404287 question pairs for training.

36.92% of question pairs are duplicates and 63.08% of questions pair non-duplicate.
Out of 808574 total questions (including both question1 and question2), 537929 are unique.
Most of the questions are repeated very few times. Only a few of them are repeated multiple times.
One question is repeated 157 times which is the max number of repetitions.
There are some questions with very few characters, which does not make sense. It will be taken care of later with Data Cleaning.

# Data Cleaning
We have converted everything to lower case.
We have removed contractions.
We have replaced currency symbols with currency names.
We have also removed hyperlinks.
We have removed non-alphanumeric characters.
We have removed inflections with word lemmatizer.
We have also removed HTML tags.

# Feature Extraction

We have created 23 features from the questions.

We have created features q1_char_num, q2_char_num with count of characters for both questions.
We have created features q1_word_num, q2_word_num with count of characters for both questions.
We have created total_word_num feature which is equal to sum of q1_word_num and q2_word_num.
We have created differ_word_num feature which is absolute difference between q1_word_num and q2_word_num.
We have created same_first_word feature which is 1 if both questions have same first word otherwise 0.
We have created same_last_word feature which is 1 if both questions have same last word otherwise 0.
We have created total_unique_word_num feature which is equal to total number of unique words in both questions.
We have created total_unique_word_withoutstopword_num feature which is equal to total number of unique words in both questions without the stop words.
The total_unique_word_num_ratio is equal to total_unique_word_num divided by total_word_num.
We have created common_word_num feature which is count of total common words in both questions.
The common_word_ratio feature is equal to common_word_num divided by total_unique_word_num.
The common_word_ratio_min is equal to common_word_num divided by minimum number of words between question 1 and question 2.
The common_word_ratio_max is equal to common_word_num divided by maximum number of words between question 1 and question 2.
We have created common_word_withoutstopword_num feature which is count of total common words in both questions excluding the stopwords.
The common_word_withoutstopword_ratio feature is equal to common_word_withoutstopword_num divided by total_unique_word_withoutstopword_num.
The common_word_withoutstopword_ratio_min is equal to common_word_withoutstopword_num divided by minimum number of words between question 1 and question 2 excluding the stopwords.
The common_word_withoutstopword_ratio_max is equal to common_word_withoutstopword_num divided by maximum number of words between question 1 and question 2 excluding the stopwords.
Then we have extracted fuzz_ratio, fuzz_partial_ratio, fuzz_token_set_ratio and fuzz_token_sort_ratio features with fuzzywuzzy string matching tool. Reference:

# EDA with Features

(visualization ncan be seen in actual code)

# Featurization with SentenceBERT
We need to convert the questions to some numeric form to apply machine learning models. There are various options from basic like Bag of Words to Universal Sentence Encoder.

I tried InferSent sentence embeddings. But it returns 4096 dimension representation. And after applying it the train data became huge. So I discarded it. And I chose SentenceBERT for this problem.

SentenceBERT is a BERT based sentence embedding technique. We will use pre-trained SentenceBERT model paraphrase-mpnet-base-v2, which is recommended for best quality. The SentenceBERT produces an output of 768 dimensions. https://www.sbert.net/

We created two more features cosine_simlarity_bert and euclidean_distance_bert which measures similarity and distance between both pairs of questions with SentenceBert representation.

The total number of features till now is 25.

#EDA on new features related toSentenceBERT

(see in actual code)

#Data Pre-processing

We normalized (min-max scaling) the extracted features. We have not normalized the embeddings because it is not recommended.

We have 1561 features (25 + 768 + 768).

25 are extracted features.
768+768 for sentence embedding of question 1 and question 2.
Since the dataset was imbalanced. We did oversample by sampling from the minority class.
Now we have 510048 data points for training. 255024 from each class.

Note that I have not set aside any data for testing locally. Because our main goal is to get a good score on Kaggle.

# Training Models

#Support Vector Classifier

While training Halving Grid Search CV with param grid,

svc_param_grid = {‘C’:[1e-2, 1e-1, 1e0, 1e1, 1e2]}
We have used LinearSVC because it is recommended for large datasets. We have used the L2 penalty and the loss function is squared of hinge loss. Also, it is recommended to use primal formulation for large datasets. For some values of C it was not conversing so I increased max_iter to 3000.

svc_clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, max_iter=3000)
For cross-validation in halving grid search cv, I have used 1 shuffle split with a 70:30 split. Also, the scoring for selection is accuracy.

svc_clf_search = HalvingGridSearchCV(svc_clf, svc_param_grid, cv=splits, factor=2, scoring='accuracy', verbose=3)
The halving grid search cv found C=100 to be the best param. And the best accuracy is 85.79%. So the best estimator looks like,

LinearSVC(C=100.0, dual=False, max_iter=3000)
Now since we need to minimize log loss for the competition. We would want a good predicted probability. Calibrated Classifier can be used to get a good predicted probability.

svc_calibrated = CalibratedClassifierCV(base_estimator=svc_clf_model, method="sigmoid", cv=splits)
After calibration of the model for probabilities. I predicted probabilities of test data and submitted on Kaggle. The public leader board score for the Kaggle submission is 0.36980. It is very good considering that the model assumes linear separability.

#Random Forest

You know Quora itself usage Random Forest for this problem. Or at least they did when they first posted the competition on Kaggle in June 2017.

Same as before we are using halving grid search cv with following param grid,

rf_param_grid = {
    'n_estimators':[200, 500, 800],
    'min_samples_split':[5, 15],
    'max_depth': [70, 150, None]
}
And the rest of the params are the default for the Random Forest Classifier.

rf_clf = RandomForestClassifier()
We have used the very similar halving grid search cv as before,

rf_clf_search = HalvingGridSearchCV(rf_clf, rf_param_grid, cv=splits, factor=2, scoring='accuracy', verbose=3)
The halving grid search cv found {‘max_depth’: 150, ‘min_samples_split’: 5, ‘n_estimators’: 800} to be the best params. And the best accuracy is 90.53%. So the accuracy has increased by 5% as compared to SVM. The best estimator looks like,

RandomForestClassifier(max_depth=150, min_samples_split=5, n_estimators=800)
Now at this point, I should have used calibration but because it has already taken a lot of time I skipped it. I should have used Bayesian Optimisation technique 😞.

The public leader board score for the Kaggle submission is 0.32372, which slightly better than SVC. I was expecting a little less logloss but remember we have not done calibration (due to time constraints). We will try to better with XGBoost — the holy grail of ml models for the Kaggle competition.

# XGBoost

Due to time and system configuration constrained, I decided to use 200000 data points to estimate a few of the params.
At first, I was using Optuna for hyperparameter tuning but it had some issues because of which it was not releasing memory after the trials. So the system was crash after few trials.
Later on, I decided to use HyperOpt for the tuning.

With HyperOpt, I tuned only max_depth and learning_rate. It was not a fine-tune because I used only 5 trials. But it gave a rough idea.

Finally, I choose the following params for training the model on whole data,

params = dict(
    objective = "binary:logistic",
    eval_metric = "logloss",
    booster = "gbtree",
    tree_method = "hist",
    grow_policy = "lossguide",
    max_depth = 4,
    eta = 0.14
)
The objective = “binary:logistic” because we are trying to get probabilities. I have used tree_method = “hist” for faster training. grow_policy = “lossguide” is inspired from LightGBM for better accuracy.

The num_boost_round is set to 600 with early_stopping_rounds as 20.

The public leader board score for the Kaggle submission is 0.32105, which slightly better than the other models. I was expecting a better result than this. Which is possible with more fine-tuning the hyperparameters. XGBoost have tons of hyperparameters https://xgboost.readthedocs.io/en/latest/parameter.html

# Another XGBoost

I was not happy with the result of the XGBoost model so I decided to tune the parameters with gut feeling.

The first thing I did is that I got rid of oversampled data by removing the duplicate rows.

This time I added a few more parameters to generalize better,

params = dict(
    objective = "binary:logistic",
    eval_metric = "logloss",
    booster = "gbtree",
    tree_method = "hist",
    grow_policy = "lossguide",
    max_depth = 4,
    eta = 0.15,
    subsample = .8,
    colsample_bytree = .8,
    reg_lambda = 1,
    reg_alpha = 1
)
Also, I decreased the number of boosting round to 500.

# This submission resulted in public LB score of 0.28170. This seems a very good result.

# Final Thoughts
I learned a lot from this case study. I took some shortcuts either because of system configuration constraints or some time constraints.

I also experienced firsthand that machine learning is not all about model building but steps before that take more time. The hyperparameter tuning can be automated but things like feature extraction or deciding on what featurization to use need to be done manually.

I spent almost two weeks 😅 and half of that time I was waiting for some execution to complete. So I think it’s a good idea to use things like Amazon SageMaker if you have resource-intensive tasks.

In the future, we can try some deep learning-based models.
