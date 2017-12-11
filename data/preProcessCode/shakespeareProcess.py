import os
import re
import numpy as np
import pandas as pd
import seaborn as sns

#Read in the file:
corpus = open("../shakespeare.txt", "r")
data = corpus.read()
final = list()
#Remove all the new lines

word = re.sub(r'\n+', '\n', data).strip()
final = word.split("\n")
count = 0
for sent in final:
    for char in list(sent):
        if (char == "\'"):
            sentencePunc = sent[sent.index(char):]  # Everything from punc onward
            sentence = sent[:sent.index(char)]  # Beginning stuff
            end = sentence + " " + sentencePunc
            final[count] = end
        elif (char == "\""):
            sentencePunc = sent[sent.index(char) + 1:]  # Everything from punc onward
            sentence = sent[:sent.index(char)]  # Beginning stuff
            end = sentence + " \'\'" + sentencePunc
            final[count] = end
    count = count + 1
#print(final)

targetFile = open("finalShakespeareData.dev.0", "w")
for val in final[:56000]:
    try:
        targetFile.write(val)
        targetFile.write("\n")
    except:
        continue

otherTargetFile = open("finalShakespeareData.train.0", "w")
for val in final:
    try:
        otherTargetFile.write(val)
        otherTargetFile.write("\n")
    except:
        continue


#targetFile.write(endString[])
#values = data.split("\n")
#print(values)
#Get rid of null values:
#for pieces in values:
#    if (pieces == "" or pieces == "\n"):
#        values.remove(pieces)

#Reconstruct everything with only one new line in between each:
#final = "\n".join(values)
