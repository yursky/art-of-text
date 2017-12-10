#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import pandas as pd  # To create a dataframe of data
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
#import gensim #For Word2Vec
# NLTK is an interesting library that was used in a Kaggle kernel and helps with a bunch of NLP stuff
import nltk

nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("stopwords")
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import ngrams
from nltk.corpus import stopwords  # for removing stopwords

print("Hi, I started!")

#######################Config Options (Enter None to process everything)#########
NUM_OF_AUTHORS = None #Pick the first n authors whose books you want to analyze #
NUM_OF_BOOKS = 1 #Number of books to iterate through when creating CSV          #
NUM_OF_PARAS = 4000 #Pick the number of paragraphs to analyze per book          #
NUM_OF_SENT = 5 #PIck the number of sentences per paragraph                     #
#################################################################################

###########################Function for removing punctuations from string########################
def remPuncFromStr(string1):
    string1 = string1.lower() #changing to lower case
    translation_table = dict.fromkeys(map(ord, string.punctuation), ' ') #creating dictionary of punc & None
    string2 = string1.translate(translation_table) #apply punctuation removal
    return string2
############################End of Function###############################################

#######################################Remove stopwords from string##############################
def remStopwordsFromStr(string1):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*') #compiling all stopwords.
    string2 = pattern.sub('', string1) #replacing the occurrences of stopwords in string1
    return string2
###########################End of Function###############################################

##################################Build ngrams from the data#####################################
def ngramListFromString(string1,count_of_words_in_ngram):
    string1 = string1.lower()
    string1 = string1.replace('.','. ')
    all_grams = ngrams(string1.split(), count_of_words_in_ngram)
    grams_list = []
    for grams in all_grams:
        grams_list.append(grams)
    return(grams_list)
##################################End of Function################################################

# Get all the Authors to go through:
dirList = list()
for root, dirs, files in os.walk("../Processed/trainBook/train", topdown=False):
    for name in dirs:
        if (name != ".ipynb_checkpoints"):
            dirList.append(name)

# First create the dataFrame that will hold the results:
finalDFDict = dict()
numFeaturesDF = pd.DataFrame(columns=["authors", "wordsInASent", "lexicalDiv", "stopWordCount", "commasPerSent",
                                      "semiPerSent", "colonsPerSent"])

#Iterate through directory and process the data:
id = 0 #To create an index column for every paragraph
if (NUM_OF_AUTHORS != None): #Limit # of authors if specified
    dirList = dirList[:NUM_OF_AUTHORS]
for dirs in dirList:
    fileList = list()
    if (NUM_OF_BOOKS != None): #Limit # of books if specified
        fileList = os.listdir("../Processed/trainBook/train/" + str(dirs))[:NUM_OF_BOOKS]
    else:
        fileList = os.listdir("../Processed/trainBook/train/" + str(dirs))
    for file in fileList:
        fileName = os.fsencode(file).decode("utf-8")
        print("This book will be processed: " + dirs + " " + str(fileName.replace(".txt", "")))

        rawDataFile = open("../Processed/trainBook/train/" + dirs + "/" + str(fileName), "r", encoding="utf-8")
        rawData = rawDataFile.read()  # type(rawData) == string
        paragraphs = rawData.split("\n\n")
        if (NUM_OF_PARAS != None): #Limit # of paragraphs if specified
            paragraphs = paragraphs[:NUM_OF_PARAS]
        for para in paragraphs:
            if (NUM_OF_SENT != None):
                allSentences = sent_tokenize(para)[:NUM_OF_SENT]
            for sent in allSentences:
                words = word_tokenize(sent)
                wordCount = len(words)
                #print("Tokenizing complete!")

                #Number of words in each sentence
                wordsPerSent = len(words)
                numFeaturesDF.at[id, "wordsInASent"] = wordsPerSent
                numFeaturesDF.at[id, "authors"] = dirs
                #print("Words Per Sentence found!")

                #Lexical diversity
                lexicalDiv = len(set(words)) / float(len(words))
                numFeaturesDF.at[id, "lexicalDiv"] = lexicalDiv
                numFeaturesDF.at[id, "authors"] = dirs
                #print("LexicalDiv found!")

                #Count num of stopwords:
                stop_words = set(stopwords.words("english"))
                stopWordCount = len([w for w in str(sent).lower().split() if w in stop_words])
                numFeaturesDF.at[id, "stopWordCount"] = stopWordCount
                numFeaturesDF.at[id, "authors"] = dirs
                #print("Number of stopwords found!")

                #Commas per sentence
                commaPerSent = len(sent.split(","))
                numFeaturesDF.at[id, "commasPerSent"] = commaPerSent
                numFeaturesDF.at[id, "authors"] = dirs
                #print("Number of commas per sentence found!")

                #Semicolons per sentence
                semiPerSent = len(sent.split(";"))
                numFeaturesDF.at[id, "semiPerSent"] = semiPerSent
                numFeaturesDF.at[id, "authors"] = dirs
                #print("Semicolons Per Sentence found!")

                #Colons per sentence
                colonsPerSent = len(sent.split(":"))
                numFeaturesDF.at[id, "colonsPerSent"] = colonsPerSent
                numFeaturesDF.at[id, "authors"] = dirs
                #print("NumOfColonsPerSentence found!")

                #Attempt at Word2Vec using Gensim library
                #First create a list of sentences where every item is a list of words in the sentence:
                #newListOfSent = list()
                #for i in allSentences:
                #    print(i)
                #print(newListOfSent)

                # Bag of Words features:
                #Get most common words in the paragraph:
                # allTokens = word_tokenize(para)
                # fDist = nltk.FreqDist(allTokens)
                # numTopWords = 10
                # vocab = list(fDist.keys())[:numTopWords]
                #
                # #Remove stopwords:
                # for word in vocab:
                #     value = word.lower()
                #     if (value in stopwords.words("english")):
                #         vocab.remove(word)
                #
                # # Use sklearn to create the bag of words feature vector for each paragraph
                # vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=word_tokenize)
                # numBag = vectorizer.fit_transform(para.split("\n")).toarray().astype(np.float64)
                # vocabulary = vectorizer.get_feature_names() #Get a column for every word in the paragraph
                #
                # #Clear out punctuation and other trash data
                # for word in vocabulary:
                #     if (word.isalpha() == False):
                #         print(word)
                #         # print("Index " + str(list(numBag.columns.values).index(col)))
                #         np.delete(numBag, vocabulary.index(word), axis=1) #Must come before previous line (don't delete what is assumed to exist)
                #         vocabulary.remove(word)
                #
                # numBag = np.sum(numBag[:, i] for i in range(numBag.shape[1])) #Sum over the columns for a total count per word per paragraph
                #
                # # Normalise by dividing the row by its Euclidean norm
                # norm = np.linalg.norm(numBag, axis=0)
                # for i in range(len(numBag)):
                #     numBag[i] = numBag[i] / float(norm)
                #
                # #Now add to the dataframe:
                # for word in vocabulary:
                #     index = 0 #To get access to the values in fvsBow
                #     columns = set(numFeaturesDF.columns.values)
                #     if (word in columns): #Word is already in dataframe
                #         numFeaturesDF.at[id, word] = numFeaturesDF.at[id, word] + numBag[index] #Add to the existing value
                #         numFeaturesDF.at[id, "authors"] = dirs
                #     else: #This is a new word to add to the bag
                #         numFeaturesDF.at[id, word] = numBag[index]  #Add new value
                #         numFeaturesDF.at[id, "authors"] = dirs
                #     index = index + 1
                #
                # #Also insert 0's for any word in the DF but not in the above vocab list:
                # numFeaturesDF = numFeaturesDF.fillna(value=0)

                #Feature Vector using POS count:
                #Get POS for each token in each sentence
                sentPos = [sentPos[1] for sentPos in nltk.pos_tag(words)]

                #Count frequencies for common POS types
                posList = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
                syntaxFeat = np.array([[sentToken.count(pos) for pos in posList] for sentToken in sentPos]).astype(np.float64)
                newSyntaxFeat = list()
                for i in range(len(posList)):
                    newSyntaxFeat.append(np.sum(syntaxFeat[:, i]))

                #Normalise by dividing each row by number of tokens in the chapter
                norm = np.linalg.norm(newSyntaxFeat, axis=0)
                for i in range(len(newSyntaxFeat)):
                    newSyntaxFeat[i] = newSyntaxFeat[i] / float(norm)

                #Now add to the dataframe:
                for pos in posList:
                    numFeaturesDF.at[id, pos] = newSyntaxFeat[posList.index(pos)]  # Add new value
                    numFeaturesDF.at[id, "authors"] = dirs

                #Take care of NaN's by taking the average of the column:
                for pos in posList:
                    numFeaturesDF[pos] = numFeaturesDF[pos].fillna(numFeaturesDF[pos].mean())

                print(numFeaturesDF)
                id = id + 1 #Update the id to only update when a new row is added

        #Friendly message to give you sense of how long it took for 1 book:
        print("Hey, I finished with the book " + str(fileName.replace(".txt", "")) + "!")

    #Friendly message for each author:
    print("Finished with " + str(dirs) + "'s books!")

# Push data to a CSV to read from later:
numFeaturesDF.to_csv("../Processed/trainBook/featuresWithSentences.csv", header=True, index=True)
