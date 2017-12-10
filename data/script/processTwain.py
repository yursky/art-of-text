import os
import pandas as pd  # To create a dataframe of data
# NLTK is an interesting library that was used in a Kaggle kernel and helps with a bunch of NLP stuff
import nltk

#######################Config Options (Enter None to process everything)#########
NUM_OF_BOOKS = None #Number of books to iterate through when creating CSV       #
#################################################################################

fileList = list()
if (NUM_OF_BOOKS != None): #Limit # of books if specified
    fileList = os.listdir("../Processed/trainBook/train/Twain")[:NUM_OF_BOOKS]
else:
    fileList = os.listdir("../Processed/trainBook/train/Twain")
for file in fileList:
    fileName = os.fsencode(file).decode("utf-8")
    print("This book will be processed: Twain " + str(fileName.replace(".txt", "")))

    rawDataFile = open("../Processed/trainBook/train/Twain/" + str(fileName), "r", encoding="utf-8")
    rawData = rawDataFile.read()  # type(rawData) == string
    rawDataFile.close()
    #processedText = ""

    if (rawData.find("HUCKLEBERRY FINN") != -1):
        processedText = rawData[rawData.find("HUCKLEBERRY FINN"):] #Get everything after this
    elif (rawData.find("The River and Its History") != -1):
        processedText = rawData[rawData.find("The River and Its History"):]  # Get everything after this
    elif (rawData.find("Concerning a Frightful Assassination") != -1):
        processedText = rawData[rawData.find("Concerning a Frightful Assassination"):]  # Get everything after this
    elif (rawData.find("1876") != -1):
        processedText = rawData[rawData.find("1876"):]  # Get everything after this
    elif (rawData.find("SAN FRANCISCO") != -1):
        processedText = rawData[rawData.find("SAN FRANCISCO"):]
    elif (rawData.find("A CONNECTICUT YANKEE") != -1):
        processedText = rawData[rawData.find("A CONNECTICUT YANKEE"):]

    processedText = processedText.replace("_", " ")
    finalFile = open("../Processed/trainBook/train/Twain/" + str(fileName), "w", encoding="utf-8")
    finalFile.seek(0)
    finalFile.truncate()
    finalFile.write(processedText)
    finalFile.close()