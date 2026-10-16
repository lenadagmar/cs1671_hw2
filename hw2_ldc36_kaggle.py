import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

train_path = 'diplomacy_train.csv'
dev_path = 'diplomacy_dev.csv'
test_path = 'diplomacy_kaggle.csv'
score_col = 'meta.game_score_delta'
POS = 1  # lying class

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\W', ' ', text)   # replace non-word characters with a space
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    words = [w for w in text.strip().split() if w not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

def prepare(df):
    """Add winning? (1 if score delta > 0, else 0) and the cleaned text with a marker token."""
    df = df.copy()
    df['winning?'] = (pd.to_numeric(df[score_col], errors='coerce') > 0).astype(int)
    df['text_m1'] = df['text'].apply(clean_text) + df['winning?'].map({1: ' winningflag', 0: ''})
    return df

# Load and prepare
train = prepare(pd.read_csv(train_path, encoding='latin-1'))
dev = prepare(pd.read_csv(dev_path, encoding='latin-1'))
test = prepare(pd.read_csv(test_path, encoding='latin-1'))

# Binary unigram features, fit on train only
vectorizer = CountVectorizer(ngram_range=(1, 1), binary=True)
X_train = vectorizer.fit_transform(train['text_m1'])
X_dev = vectorizer.transform(dev['text_m1'])
X_test = vectorizer.transform(test['text_m1'])

# Train Model 3
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, train['intent'])

# Sanity check on dev (should match your reported Model 3 scores)
pred_dev = model.predict(X_dev)
p, r, f1, _ = precision_recall_fscore_support(
    dev['intent'], pred_dev, pos_label=POS, average='binary')
print(f"Dev: acc={accuracy_score(dev['intent'], pred_dev):.3f} "
      f"P={p:.3f} R={r:.3f} F1={f1:.3f}")

# Predict on test and write the submission (only id and intent)
test_pred = model.predict(X_test)
submission = pd.DataFrame({'id': test['id'], 'intent': test_pred.astype(int)})
submission.to_csv('diplomacy_test_predictions.csv', index=False)

print(submission.head())