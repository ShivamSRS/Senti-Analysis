

import nltk
import numpy as np
import pandas as pd
import re

# Import dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')

#preprocessing

#now we will remove punctuations and find root words for all words ie stemming
#ie fished, fisher becomes fish

nltk.download('stopwords')

#to remove stopwords
from nltk.corpus import stopwords

#A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that
# search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query.

#e would not want these words taking up space in our database, or taking up valuable processing time.
#fr this, we can remove them easily, by storing a list of words that you consider to be stop words.

#for stemming
from nltk.stem.porter import PorterStemmer
 
# Initialize empty array 
# to append clean text  
corpus = []

dataset.head()
dataset
# 1000 (reviews) rows to clean 
for i in range(0, 1000):  
      
    # column : "Review", row ith 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #removed punctuations amd stored the ith review 
    review.lower() #lowercased the whole thing
    
    # split to array(default delimiter is " ") 
    review = review.split()
    if i==0 : print("now the first review is \n",review)  
    
    # creating PorterStemmer object to 
    # take main stem of each word 
    ps = PorterStemmer()
    
    # loop for stemming each word 
    # in string array at ith row 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #we stemmed every word in review which is not a stop word
    
    if i==0 : print("now the first review is \n",review) 
    # rejoin all string array elements 
    # to create back into a string 
    review = ' '.join(review)
    if i==0 : print("now the first review is \n",review) 
      
    # append each string to create 
    # array of clean text  
    corpus.append(review)

#Tokenization, is splitting sentences and words from the body of the text

#now making mixed bags of words

#Take all the different words of reviews in the dataset without repeating of words.
#One column for each word, therefore there are going to be many columns.
#Rows are reviews
#If word is there in row of dataset of reviews, 
#then the count of word will be there in row of bag of words under the column of the word. this is the approach we will take

# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 
  
# To extract max 1500 feature. We can also set max number of features 
#(max no. features which help the most via attribute “max_features”)
# "max_features" is attribute to 
# experiment with to get better results 
cv = CountVectorizer(max_features = 1500)  

#Do the training on corpus and then apply the same transformation to the corpus
#“.fit_transform(corpus)” 
#and then convert it into array
# X contains corpus (dependent variable) 
#X is the bag of words
X = cv.fit_transform(corpus).toarray()

#If review is positive or negative that answer is in second column of : dataset[:, 1] 
# y contains answers if review 
# is positive(1) or negative(0) 
y = dataset.iloc[:, 1].values


# Splitting the dataset into 
# the Training set and Test set 
from sklearn.cross_validation import train_test_split 
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) 

# Fitting Random Forest Classification 
# to the Training set 
from sklearn.ensemble import RandomForestClassifier 
  
# n_estimators can be said as number of 
# trees, experiment with n_estimators 
# to get better results  
model = RandomForestClassifier(n_estimators = 501,criterion = 'entropy') 
                              
model.fit(X_train, y_train)

# Predicting the Test set results 
y_pred = model.predict(X_test)
print(y_pred)

"""To know the accuracy, confusion matrix is needed.

Confusion Matrix is a 2X2 Matrix.

TRUE POSITIVE : measures the proportion of actual positives that are correctly identified.
TRUE NEGATIVE : measures the proportion of actual positives that are not correctly identified.
FALSE POSITIVE : measures the proportion of actual negatives that are correctly identified.
FALSE NEGATIVE : measures the proportion of actual negatives that are not correctly identified.
"""

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix 
  
cm = confusion_matrix(y_test, y_pred)

print(cm)

#output = [[123 23]
          #[57 97]]
#The result obtained with such parameters gave us an accuracy of
# (123+97)/(300)*100 = 73.333% on a dataset of 300

