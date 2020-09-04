# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:59:29 2020

@author: emree
"""
#%% 
import pandas as pd

#%%
data = pd.read_csv(r"gender_classifier.csv",encoding = "latin1")
data = pd.concat([data.gender,data.description],axis=1)
data.dropna(axis = 0,inplace = True)
data.gender = [1 if each == "female" else 0 for each in data.gender]
#İf gender is male then 0,else 1  

#%% regular expression 

import re

first_description=data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description)  # a dan z ye ve A dan Z ye kadar olan harfleri bulma geri kalanları " " (space) ile degistir
description = description.lower()   # buyuk harftan kucuk harfe cevirme


# %% stopwords (irrelavent words) gereksiz kelimeler
import nltk
nltk.download("stopwords")  # corpus diye bir klasöre indiriliyor
from nltk.corpus import stopwords

# description = description.split()

# split yerine tokenizer kullanabiliriz
description=nltk.word_tokenize(description)


#%% 
# gereksiz kelimeleri çıkar
description = [word for word in description if not word in set(stopwords.words("english"))]

#%% Lemmatization Köklerini bul

import nltk as nlp
nltk.download('wordnet')
lemma=nlp.WordNetLemmatizer()
description=[lemma.lemmatize(word) for word in description]
description=" ".join(description)

#%%

description_list=[]
# if  u want to name of the varieble u should change all of the desctions' name
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)  # a dan z ye ve A dan Z ye kadar olan harfleri bulma geri kalanları " " (space) ile degistir
    description = description.lower()
    description=nltk.word_tokenize(description)
    
    #description = [word for word in description if not word in set(stopwords.words("english"))]
    # ı blocked above one because its taking too much time
    
    description=[lemma.lemmatize(word) for word in description]
    description=" ".join(description)
    description_list.append(description)


#%% bag of words
    
from sklearn.feature_extraction.text import CountVectorizer
max_features=5000

count_vectorizer= CountVectorizer(max_features=max_features,stop_words="english")

sparce_matrix=count_vectorizer.fit_transform(description_list).toarray()

print("en sık kullanılan {} kelime: {}".format(max_features,count_vectorizer.get_feature_names()))



# %%
y = data.iloc[:,0].values   # male or female classes
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)


# %% naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

#%% prediction
y_pred = nb.predict(x_test)

print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))

















