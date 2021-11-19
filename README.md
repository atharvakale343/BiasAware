# BiasAware

## Goal
* A model to determine the “bias label” of each paragraph in a news article
* This project would focus only on Political Bias 
* Output Labels of the Model: Biased, Unbiased
* Project does not consider other types of bias:
  * Overview of Different types of Bias (https://github.com/amazon-research/bold)
  
## Outcomes 
* Spread awareness of the legitimacy of a text 
* Help readers make informed decisions and form accurate opinions on current issues
* Reduce the epidemic of misinformation

## Dataset
* Dataset used: NELA-GT-2019 (Harvard dataset)
* 1.12M news articles from 260 sources 
* Collected between January 1st 2019 and December 31 2019 
* Label that is important to this project from this dataset
  * Aggregate Label: Reliable, mixed, or Unreliable categorized by article source

# Plan
1) Use Web Scraping Library to extract cleaned article content and replace the “content” column in SQLite database. 
2) Create a dataframe from the SQLite Database 
   1) Columns:
      1) Content (article body)
      2) Label 
         1) 1 for biased (corresponds to 2 in the dataset column labeled aggregate data)
         2) 0 for unbiased (corresponds to 0 in the dataset column labeled aggregate data)
3) Download the BERTSentence pre-trained model (need to determine the specific model)
4) Finetune the BERTSentence model using our dataset 
5) Encode all article sentences into vectors using the BERTSentence model 
6) Feed encoded sentences into a CNN 
7) Train CNN model based on labels with the training data 
8) Test CNN model with the testing data

# Model:
* Sentence Embedding: Vector for each sentence in the article; Finding the semantics in a context
  * SentenceBERT 
    * BERT is a pre-trained model that understands nuances of the English language 
    * Fine Tuning: We train BERT again on our data to find bias-specific features of the English language 
  * Label Detection and Classification using CNN 
    * Extract the “bias” features from a paragraph
  * Below is example of the workflow:
    ![Model Workflow](pictures/model_image.png) 

# Supplementary Tools Used:
* Python Script to make Dataframe from SQLite DB
  * https://github.com/mgruppi/nela-gt-2019 
* Web Scraping Library for scraping News Articles
  * Reason: NELA-GT-2019 Dataset scraping is poorly executed
  * https://github.com/arnavn101/WebXplore
* SentenceBERT
  * https://github.com/UKPLab/sentence-transformers 
  * https://github.com/Meelfy/pytorch_pretrained_BERT 
* CNN
  * https://pytorch.org/docs/stable/nn.html#convolution-layers 