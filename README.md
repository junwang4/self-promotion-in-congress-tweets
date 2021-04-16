## Get started

**NOTE**: My running environment is **Linux box (Ubuntu 16.04) with a 1080Ti GPU**.

### STEP 1. Prerequisite

Install bert-sklearn --
a scikit-learn wrapper to finetune BERT model based on the huggingface's pytorch transformer.
```
git clone -b master https://github.com/charles9n/bert-sklearn
cd bert-sklearn
pip install .
```

Install other packages
```
pip install fire
```


### STEP 2. Get the repo, including code and data
```
git clone https://github.com/junwang4/self-promotion-in-congress-tweets
cd self-promotion-in-congress-tweets
```

### STEP 3

#### 3.1 To evaluate the performance of the model, say, 5-fold cross-validation 

First, generate a prediction file as the result of training and testing each of the 5 folds
```
python run.py tweet_classifier --task=train_KFold_model
```
This will take as input the annotated dataset `data/annotations.csv`,
and put together the prediction results from each fold and output to `code/working/pred/[20210331]_train_K5_epochs3.csv`

Second, display the evaluation results

```
python run.py tweet_classifier --task=evaluate_and_error_analysis
```
In the case of using the default setting given in file `code/settings.ini`,
we have the following result:
```
              precision    recall  f1-score   support

           0      0.951     0.949     0.950      3089
           1      0.828     0.835     0.831       914

    accuracy                          0.923      4003
   macro avg      0.889     0.892     0.890      4003
weighted avg      0.923     0.923     0.923      4003

```

#### 3.2 Create a fully-trained BERT model to classify a tweet as self-promoting or not
```
cd code
python run.py tweet_classifier --task=train_one_full_model
```
This will take as input the annotated dataset `data/annotations.csv`,
and output a BERT model at `code/working/model/[20210331]_full_epochs3.bin`


#### 3.3 Apply the above trained model to the 2 million tweets 

```
python run.py tweet_classifier --task=apply_one_full_model_to_new_sentences
```
This will take as input the file `data/id_date_text.csv`, 
and output a prediction csv file in folder `code/working/pred/[20210331]_apply_epochs3.csv`

The file `data/id_date_text.csv`  (over 100M) is too much to be a regular file of github, 
so we only upload to github our final result file `data/final_data_for_regression_analysis.csv`.

// If you are interested in the above 2 million tweets data, email me for how to access it.


### STEP 4. Logistic Linear Mixed-effect Regression Analysis in R
(I wish there was such a package in Python)

Open your Rstudio, and load the R markdown file `code/regression_analysis.Rmd`,
which contains the instructions of how to run the regression analysis as well as the diagnosis of the key assumptions of linear regression.
