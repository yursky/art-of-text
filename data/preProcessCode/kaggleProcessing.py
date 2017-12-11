import os
import numpy as np
import pandas as pd
import seaborn as sns

#Read in the CSV:
df = pd.read_csv("../kaggle/train.csv")

#Read in all the rows that are assigned to HPL
listOfAuthors = df["author"].tolist()
textFromHPL = list()
textFromEAP = list()
textFromMWS = list()
for i in range(len(listOfAuthors)):
    auth = df.iloc[i]["author"]
    if (auth == "HPL"):
        sent = df.iloc[i]["text"].lower()

        for char in list(sent):
            if (char == "\'"):
                sentencePunc = sent[sent.index(char):]  #Everything from punc onward
                sentence = sent[:sent.index(char)]  #Beginning stuff
                sent = sentence + " " + sentencePunc
            elif (char == "\""):
                sentencePunc = sent[sent.index(char)+1:]  # Everything from punc onward
                sentence = sent[:sent.index(char)]  # Beginning stuff
                sent = sentence + " \'\'" + sentencePunc
            elif (char == "."):
                sentencePunc = sent[sent.index(char):]  # Everything from punc onward
                sentence = sent[:sent.index(char)]  # Beginning stuff
                sent = sentence + " " + sentencePunc

        textFromHPL.append(sent)
    if (auth == "EAP"):
        sent = df.iloc[i]["text"].lower()

        for char in list(sent):
            if (char == "\'"):
                sentencePunc = sent[sent.index(char):]  # Everything from punc onward
                sentence = sent[:sent.index(char)]  # Beginning stuff
                sent = sentence + " " + sentencePunc
            elif (char == "\""):
                sentencePunc = sent[sent.index(char) + 1:]  # Everything from punc onward
                sentence = sent[:sent.index(char)]  # Beginning stuff
                sent = sentence + " \'\'" + sentencePunc
            elif (char == "."):
                sentencePunc = sent[sent.index(char):]  # Everything from punc onward
                sentence = sent[:sent.index(char)]  # Beginning stuff
                sent = sentence + " " + sentencePunc

        textFromEAP.append(sent)
    if (auth == "MWS"):
        sent = df.iloc[i]["text"].lower()

        for char in list(sent):
            if (char == "\'"):
                sentencePunc = sent[sent.index(char):]  # Everything from punc onward
                sentence = sent[:sent.index(char)]  # Beginning stuff
                sent = sentence + " " + sentencePunc
            elif (char == "\""):
                sentencePunc = sent[sent.index(char) + 1:]  # Everything from punc onward
                sentence = sent[:sent.index(char)]  # Beginning stuff
                sent = sentence + " \'\'" + sentencePunc
            elif (char == "."):
                sentencePunc = sent[sent.index(char):]  # Everything from punc onward
                sentence = sent[:sent.index(char)]  # Beginning stuff
                sent = sentence + " " + sentencePunc

        textFromMWS.append(sent)

print(len(textFromHPL))
print(len(textFromEAP))
print(len(textFromMWS))
hplDevFile = open("hplMws.dev.0", "w")
for val in textFromHPL[:3381]:
    try:
        hplDevFile.write(val)
        hplDevFile.write("\n")
    except:
        continue
hplDevFile = open("hplMws.train.0", "w")
for val in textFromHPL[3381:]:
    try:
        hplDevFile.write(val)
        hplDevFile.write("\n")
    except:
        continue

eapTargetFile = open("eapInput.txt", "w")
for val in textFromEAP:
    try:
        eapTargetFile.write(val)
        eapTargetFile.write("\n")
    except:
        continue

mwsDevFile = open("hplMws.dev.1", "w")
for val in textFromMWS[:3626]:
    try:
        mwsDevFile.write(val)
        mwsDevFile.write("\n")
    except:
        continue

mwsDevFile = open("hplMws.train.1", "w")
for val in textFromMWS[3626:]:
    try:
        mwsDevFile.write(val)
        mwsDevFile.write("\n")
    except:
        continue