"""first import all library"""
import pandas as pd
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


"""read the dataset"""
data_fake_news = pd.read_csv("E:\dataset\\fake news\\archive\Fake.csv")
data_True_news = pd.read_csv("E:\dataset\\fake news\\archive\True.csv")

"""insert target column for classification."""
data_fake_news['target'] = "Fake"
data_True_news['target'] = "True"

"""concat both the dataframe into one dataframe"""
data = pd.concat([data_fake_news,data_True_news],axis=0,ignore_index=True)

# print(data)

"""perform preprocessing of dataset
1. check missing value.
2. any given link remove it.
3. any empty text after removing link from text column.
"""

def remove_link(text):
    txt =""
    text = text.split(" ")
    for word in text:
        if ('http' in word) or ('.com' in word) or ('https' in word) or ('.in' in word):
            continue
        else:
            txt += word
    return txt

data['text'] = data['text'].apply(remove_link)

punc = string.punctuation
punc += '\n \n\n \t \t\t \r \b\b'

"""remove punctuation mark"""
def remove_punctuation(text):
    text = text.split()
    txt = [word.lower() for word in text if word not in punc]
    return ' '.join(txt)

data['text'] = data['text'].apply(remove_punctuation)


"""split independent and dependent data"""
X = data['text']
y = data["target"]

y = pd.get_dummies(y,drop_first=True).values.reshape(-1,)

"""create a pipeline to preproce and for model"""
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=10)

tfidf = TfidfVectorizer(stop_words='english',max_df=0.5)
pipe = Pipeline(steps=[('vectorize',tfidf),('model',LogisticRegression())])
#X_train = tfidf.fit_transform(X_train)
#X_test = tfidf.transform(X_test)
pipe.fit(X_train,y_train)
pred_data = pipe.predict(X_test)

print(accuracy_score(y_test,pred_data))
print(classification_report(y_test,pred_data))