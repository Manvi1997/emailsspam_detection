#!/usr/bin/env python
# coding: utf-8

# ### Spam-Classifier-using-Naive-Bayes

# 1) The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research.
# 
# 2) It contains one set of SMS messages in English of 5,574 messages, tagged according being ham (legitimate) or spam.
# 
# 3) The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.
# 
# 
# Addition of Additional Feature TF–IDF
# 
# 
# 1) Tf–idf stands for "Term Frequency–Inverse Document Frequency" is a numerical statistic used to reflect how important a word is to a document in a collection or corpus of documents.
# 
# 2) TFIDF is used as a weighting factor during text search processes and text mining.
# 
# 3) The intuition behing the TFIDF is as follows: if a word appears several times in a given document, this word might be meaningful (more important) than other words that appeared fewer times in the same document. However, if a given word appeared several times in a given document but also appeared many times in other documents, there is a probability that this word might be common frequent word such as 'I' 'am'..etc. (not really important or meaningful!).
# 
# 4) TF: Term Frequency is used to measure the frequency of term occurrence in a document:
# 
# 5) TF(word) = Number of times the 'word' appears in a document / Total number of terms in the document
# 
# 6) IDF: Inverse Document Frequency is used to measure how important a term is:
# 
# 7) IDF(word) = log_e(Total number of documents / Number of documents with the term 'word' in it).
#  Example: Let's assume we have a document that contains 1000 words and the term “John” appeared 20 times, the Term-Frequency for the word 'John' can be calculated as follows: TF|john = 20/1000 = 0.02
#  
# 8) Let's calculate the IDF (inverse document frequency) of the word 'john' assuming that it appears 50,000 times in a 1,000,000 million documents (corpus). IDF|john = log (1,000,000/50,000) = 1.3 Therefore the overall weight of the word 'john' is as follows TF-IDF|john = 0.02 * 1.3 = 0.026
# 

# In[1]:


import os
import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string


# In[2]:


os.chdir("C:/Users/Hp/Desktop/spam")


# In[3]:


spam_df = pd.read_csv("emails.csv")


# In[4]:


spam_df.shape


# In[5]:


spam_df.columns


# In[6]:


spam_df.head(10)


# In[7]:


spam_df.describe()


# In[8]:


spam_df.info()


# In[9]:


# Let's see which message is the most popular ham/spam message
spam_df.groupby('spam').describe()


# In[10]:


# Let's divide the messages into spam and ham
ham = spam_df[spam_df['spam']==0]
spam = spam_df[spam_df['spam']==1]


# In[11]:


import seaborn as sns
print( 'Spam percentage =', (len(spam) / len(spam_df) )*100,"%")
print( 'Ham percentage =', (len(ham) / len(spam_df) )*100,"%")
sns.countplot(spam_df['spam'], label = "Count")


# ### Data Cleaning

# In[12]:


from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[13]:


spam_df['tokenized'] = spam_df.apply(lambda row: nltk.sent_tokenize(row['text']), axis=1)


# In[14]:


STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    
spam_df["text_wo_stop"] = spam_df["text"].apply(lambda text: remove_stopwords(text))


# In[15]:


stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

spam_df["text_stemmed"] = spam_df["text"].apply(lambda text: stem_words(text))


# In[16]:


spam_df.head()


# ### Split the data set into train and test 

# In[17]:


X = spam_df.text.values
y = spam_df.spam.values


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)


# ### TF_IDF Vectorization 

# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=3500)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[20]:


print(X_train_tfidf.shape,X_test_tfidf.shape)


# ### For Imbalced data : SMOTE TECHNIQUE 

# In[29]:


pip install imblearn


# In[30]:


from imblearn.over_sampling import SMOTE


# In[32]:


smt = SMOTE()
X_train_tfidf, y_train = smt.fit_sample(X_train_tfidf, y_train)


# ### Navie Bayes

# In[33]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)


# In[46]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


# In[47]:


y_pred = classifier.predict(X_test_tfidf)


# In[48]:


accuracy_score(y_test,y_pred)


# In[49]:


precision_score(y_test,y_pred)


# In[37]:


confusion_matrix(y_test,y_pred)


# In[38]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))


# In[40]:


import pickle


# In[56]:


# Save the Modle to file in the current working directory

Pkl_Filename = "Pickle_NB_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(classifier, file)


# In[57]:


# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_NB_Model = pickle.load(file)

Pickled_NB_Model


# In[58]:


# Use the Reloaded Model to 
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = Pickled_NB_Model.score(X_train_tfidf, y_train)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_NB_Model.predict(X_train_tfidf)  

Ypredict


# In[69]:


import streamlit as st
import pickle


# In[73]:


model = pickle.load(open('Pickle_NB_Model.pkl','rb'))


# In[ ]:




