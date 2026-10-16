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
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


train_path = 'diplomacy_train.csv'
dev_path = 'diplomacy_dev.csv'

train = pd.read_csv(train_path, encoding='latin-1')
dev = pd.read_csv(dev_path, encoding='latin-1')

#print(train.head(30))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\W', ' ', text)   # keep only word chars
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    words = [w for w in text.strip().split() if w not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

train = train[['text', 'intent']].copy()
dev = dev[['text', 'intent']].copy()
train['text'] = train['text'].apply(clean_text)
dev['text'] = dev['text'].apply(clean_text)

print(train.head(10)) 

# set the positive (lying) 
#print(train['intent'].value_counts())
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

# MODIFICATION 1: winning? feature (1 if delta > 0, else 0; ties and missing = 0)

train_1 = pd.read_csv(train_path, encoding='latin-1')
dev_1 = pd.read_csv(dev_path, encoding='latin-1')

# convert score delta to 1 (winning) / 0 (tie, losing, or missing)
col = 'meta.game_score_delta'
train_1[col] = (pd.to_numeric(train_1[col], errors='coerce') > 0).astype(int)
dev_1[col] = (pd.to_numeric(dev_1[col], errors='coerce') > 0).astype(int)
train_1 = train_1.rename(columns={col: 'winning?'})
dev_1 = dev_1.rename(columns={col: 'winning?'})

# clean and add marker token when winning? == 1
train_1['text_m1'] = train_1['text'].apply(clean_text) + train_1['winning?'].map({1: ' winningflag', 0: ''})
dev_1['text_m1'] = dev_1['text'].apply(clean_text) + dev_1['winning?'].map({1: ' winningflag', 0: ''})

# unigram counts + winning feature
vectorizer_m1 = CountVectorizer(ngram_range=(1, 1))
X_train_m1 = vectorizer_m1.fit_transform(train_1['text_m1'])
X_dev_m1 = vectorizer_m1.transform(dev_1['text_m1'])

# train
model_m1 = LogisticRegression(max_iter=1000, random_state=42)
model_m1.fit(X_train_m1, train_1['intent'])

# evaluate on dev
pred_m1 = model_m1.predict(X_dev_m1)
p, r, f1, _ = precision_recall_fscore_support(
    dev_1['intent'], pred_m1, pos_label=POS, average='binary')
print("Modification 1 (unigrams + winning?)")
print(f"Accuracy:  {accuracy_score(dev_1['intent'], pred_m1):.3f}")
print(f"Precision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1:        {f1:.3f}")

# MODIFICATION 2: make unigrams binary + winning?
 
train_2 = train_1.copy()
dev_2 = dev_1.copy()

# binary=True: 1 if a word appears 
vectorizer_m2 = CountVectorizer(ngram_range=(1, 1), binary=True)
X_train_m2 = vectorizer_m2.fit_transform(train_2['text_m1'])
X_dev_m2 = vectorizer_m2.transform(dev_2['text_m1'])

# train
model_m2 = LogisticRegression(max_iter=1000, random_state=42)
model_m2.fit(X_train_m2, train_2['intent'])

# evaluate on dev
pred_m2 = model_m2.predict(X_dev_m2)
p, r, f1, _ = precision_recall_fscore_support(
    dev_2['intent'], pred_m2, pos_label=POS, average='binary')
print("Modification 2 (binary unigrams + winning?)")
print(f"Accuracy:  {accuracy_score(dev_2['intent'], pred_m2):.3f}")
print(f"Precision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1:        {f1:.3f}")


from sklearn.preprocessing import StandardScaler

# Standardize features 
scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train_m2)
X_dev_s = scaler.transform(X_dev_m2)

# train separate model on standardized features
model_m2_s = LogisticRegression(max_iter=1000, random_state=42)
model_m2_s.fit(X_train_s, train_2['intent'])

# top and bottom 5 weights (positive = associated with deception)
names = vectorizer_m2.get_feature_names_out()
ranked = sorted(zip(model_m2_s.coef_[0], names))

print("\nTop 5 most negative (more truthful):")
for c, n in ranked[:5]:
    print(f"  {n:20s} {c:.3f}")

print("\nTop 5 most positive (more deceptive):")
for c, n in reversed(ranked[-5:]):
    print(f"  {n:20s} {c:.3f}")

#confusion matrix 
cm = confusion_matrix(dev_2['intent'], pred_m2, labels=[0, 1])
print("Confusion Matrix (rows = actual, cols = predicted):")
print(f"[[{cm[0,0]} {cm[0,1]}]")
print(f" [{cm[1,0]} {cm[1,1]}]]")
