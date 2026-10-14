# pt 1 code to train models/estimate performance

# build logistic regression classifier that uses unigram features
# design and test 2 modifications (custom features or preprocessing changes) to unweighted unigram features
# possible features:

#   Tf-idf transformed bag-of-words features. See J+M 11.1.2 for a description of tf-idf
#   Higher order n-gram features (bigrams, trigrams, or combinations of them) beyond the unigrams used for the bag-of-words features
#   Different preprocessing (stemming, different tokenizations, stopword removal)
#   Changing count bag-of-words features to binary 0 or 1 for the presence of unigrams
#   Incorporating features from columns in the dataset other than text
#   Reducing noisy features with feature selection
#   Counts or added weight from custom word lists
#   Any other custom-designed feature (such as length of input, number of capitalized words, neural embeddings of text, etc)

# total of 3 logistic regression models
# evaluate each with accuracy, precision, recall and f-1 score for positive (lying) class on diplomacy_dev.csv

# for one fo the feature-based models, extract 5 informative features that are mostly strongly positively and 5 that are most negatively associated with deception

import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

train_path = 'diplomacy_train.csv'
dev_path = 'diplomacy_dev.csv'

train = pd.read_csv(train_path, encoding='latin-1')
dev = pd.read_csv(dev_path, encoding='latin-1')

print(train.head(30))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\W', ' ', text)   # keep only word chars
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text.strip()

train = train[['text', 'intent']].copy()
dev = dev[['text', 'intent']].copy()
train['text'] = train['text'].apply(clean_text)
dev['text'] = dev['text'].apply(clean_text)

print(train.head(10)) 

# set the positive (lying) 
print(train['intent'].value_counts())
POS = 1  

# unweighted unigram counts, fit on train only
vectorizer = CountVectorizer(ngram_range=(1,1))
X_train = vectorizer.fit_transform(train['text'])
X_dev = vectorizer.transform(dev['text'])

# train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, train['intent'])

# evaluate on dev
pred = model.predict(X_dev)
p, r, f1, _ = precision_recall_fscore_support(
    dev['intent'], pred, pos_label=POS, average='binary')
print(f"Accuracy:  {accuracy_score(dev['intent'], pred):.3f}")
print(f"Precision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1:        {f1:.3f}")


#test confusion matrix
labels = [l for l in model.classes_ if l != POS] + [POS]
cm = confusion_matrix(dev['intent'], pred, labels=labels)
print("Confusion Matrix:")
print(f"[[{cm[0,0]} {cm[0,1]}]")
print(f" [{cm[1,0]} {cm[1,1]}]]")

# MODIFICATION 1: add feature "winning?" -- determine if speaker is winning or not winning (tie counts as not winning)



