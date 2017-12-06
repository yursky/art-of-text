#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import pandas as pd  # To create a dataframe of data
import string
# NLTK is an interesting library that was used in a Kaggle kernel and helps with a bunch of NLP stuff
import nltk
from nltk import sent_tokenize

nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("stopwords")

#######################Config Options (Enter None to process everything)#########
NUM_OF_AUTHORS = None #Pick the first n authors whose books you want to analyze #
NUM_OF_BOOKS = 1 #Number of books to iterate through when creating CSV          #
NUM_OF_PARAS = 4000 #Pick the number of paragraphs to analyze per book          #
#NUM_OF_SENT = 5 #Pick the number of sentences per paragraph                     #
#################################################################################

print("Hi, I started!")

# Get all the Authors to go through:
dirList = list()
for root, dirs, files in os.walk("../Processed/trainBook/train", topdown=False):
    for name in dirs:
        if (name != ".ipynb_checkpoints"):
            dirList.append(name)

# First create the dataFrame that will hold the results:
finalDFDict = dict()
sentCSVdf = pd.DataFrame(columns=["text", "author"])

# Iterate through directory and process the data:
id = 0 #To create an index column for every paragraph
if (NUM_OF_AUTHORS != None): #Limit # of authors if specified
    dirList = dirList[:NUM_OF_AUTHORS]

for dirs in dirList:
    fileList = list()
    if (NUM_OF_BOOKS != None):  # Limit # of books if specified
        fileList = os.listdir("../Processed/trainBook/train/" + str(dirs))[:NUM_OF_BOOKS]
    else:
        fileList = os.listdir("../Processed/trainBook/train/" + str(dirs))

    for file in fileList:
        fileName = os.fsencode(file).decode("utf-8")
        print("This book will be processed: " + dirs + " " + str(fileName))

        rawDataFile = open("../Processed/trainBook/train/" + dirs + "/" + str(fileName), "r", encoding="utf-8")
        rawData = rawDataFile.read()  # type(rawData) == string

        sentences = sent_tokenize(rawData)

        #Add to the dataframe:
        for sent in sentences[:10000]:
            if (sent != "" and sent.isspace() == False): #Because null sentences are a thing
                sent = sent.replace("\n", " ")
                sent = sent.replace("\'", " ")
                sent = sent.replace("_", " ")
                sent = sent.replace("--", " ")
                for i in sent:
                    if i.isalpha() == False and i is not " ":
                        sent = sent.replace(i, "")
                #print(sent)
                print(sentCSVdf)
                sentCSVdf.loc[id] = ["\"" + str(sent) + "\"", dirs]
                id = id + 1

sentCSVdf.to_csv("../Processed/sentCSV.csv", header=True, index=True)