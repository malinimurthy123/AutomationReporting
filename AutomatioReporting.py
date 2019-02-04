
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import base64
import string
import sys
import seaborn as sns
import spacy
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
#nltk.download()
stopwords = stopwords.words('english')
df = pd.read_csv(r'C:\Users\krmal\Desktop\Seattle Testing Conference\AutomationSmokeReport.csv')
df.head(9)


# In[22]:


df.shape


# In[23]:


df.isnull().sum()


# In[24]:


df['error_description'].nunique()


# In[25]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.33, random_state=10)


# In[26]:


print('Error Description:', train['error_description'].iloc[0])
print('Training Data Shape:', train.shape)
print('Testing Data Shape:', test.shape)


# In[27]:


fig = plt.figure(figsize=(8,4))
sns.barplot(x = train['error_description'].unique(), y=train['error_description'].value_counts())
plt.show()


# In[28]:


import spacy

punctuations = string.punctuation

# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# In[29]:


INFO_text = [text for text in train[train['error_description'] == 'id']['error_description']]

IS_text = [text for text in train[train['error_description'] == 'header']['error_description']]

INFO_clean = cleanup_text(INFO_text)
INFO_clean = ' '.join(INFO_clean).split()

IS_clean = cleanup_text(IS_text)
IS_clean = ' '.join(IS_clean).split()

INFO_counts = Counter(INFO_clean)
IS_counts = Counter(IS_clean)

df.head(9)
# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
import re
import spacy

from spacy.lang.en import English
parser = English()


# In[31]:


STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]


# In[32]:


class CleanTextTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}
    
def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text

def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens


# In[33]:


import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
#import seaborn as sns
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher

value1=nlp("valid car rental rate")
value2=nlp("searchResults")
value3=nlp("tabSelectionTable")
value1.similarity(value3)
testresult=nlp("valid car rental rate searchResults tabSelectionTable")

for result1 in testresult:
    for result2 in testresult:
        print((result1.text,result2.text),"similarity lies in",result1.similarity(result2))
        
finalreport=[(result1.text,result2.text,result1.similarity(result2))for result2 in testresult for result1 in testresult]

df=pd.DataFrame(finalreport)
df.head()

df.corr()

df.columns=["result1","result2","similarity"]

plt.figure(figsize=(20,10))
#sns.heatmap(df_viz.corr(),annot=True)
plt.show()
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

ax = df.plot.barh(title="Test Result by category", legend=False, figsize=(25,7), stacked=True)

labels = []
for j in df:
    for i in df.index:
        label = str(j)+": " + str(df.loc[i][j])
        labels.append(label)
patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2., y + height/2., label, ha='center', va='center')
plt.show()
resultfile1=open(r"C:\NLP\errorfile1.csv").read()
file1=nlp(resultfile1)

resultfile2=open(r"C:\NLP\errorfile1.csv").read()

fig = plt.figure(figsize=(8,4))
#plt.barplot(x = train['404'].unique(), y=train['404'].value_counts())
plt.show()
def printNMostInformative(vectorizer, clf, N):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
clf = LinearSVC()
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])


# In[38]:


def printNMostInformative(vectorizer, clf, N):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Group 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Group 2 best: ")
    for feat in topClass2:
        print(feat)

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
clf = LinearSVC()
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])

# data
train1 = train['error_description'].tolist()
labelsTrain1 = train['error_description'].tolist()

# train
pipe.fit(train1,labelsTrain1)
# test
preds = pipe.predict(train1)
print("accuracy:", accuracy_score(labelsTrain1, preds))
print("Top 5 Label Values used to predict:-")

printNMostInformative(vectorizer, clf, 10)

pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
transform = pipe.fit_transform(train1, labelsTrain1)
vocab = vectorizer.get_feature_names()

for i in range(len(train1)):
    s = ""
    indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
    numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
    for idx, num in zip(indexIntoVocab, numOccurences):
        s += str((vocab[idx], num))


# In[48]:


from sklearn import metrics
print(metrics.classification_report(labelsTrain1,preds))

