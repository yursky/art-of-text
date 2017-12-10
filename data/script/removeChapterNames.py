import os

# Get all the Authors to go through:
dirList = list()
for root, dirs, files in os.walk("../Processed", topdown=False):
    for name in dirs:
        if (name != ".ipynb_checkpoints"):
            dirList.append(name)


################################Delete all occurrences of this string##################################
def axeAllInstances(instance, arrToRemFrom):
    indexIntoRawArr = 0  # A way to avoid the dynamically changing size of arrToRemFrom

    while indexIntoRawArr < len(arrToRemFrom):  # While we can avoid outOfBounds errors
        if (arrToRemFrom[indexIntoRawArr].find(instance) != -1):
            arrToRemFrom.remove(arrToRemFrom[indexIntoRawArr])
        indexIntoRawArr = indexIntoRawArr + 1

    return arrToRemFrom
################################End of Function########################################################

# Iterate through directory and process the data:
for dirs in dirList:
    if (dirs != "Shakespeare"):
        for file in os.listdir(os.fsencode(dirs)):
            filename = os.fsdecode(file)
            print("This book will be processed: " + dirs + " " + str(filename))

            # Read in the raw data as a string
            rawDataFile = open("../Processed/" + dirs + "/" + str(filename), "r", encoding="utf-8")
            rawData = rawDataFile.read()  # type(rawData) == string
            rawDataLines = []
            rawDataLines = rawData.split("\n")  # Easier to do this, if we store text as a list

            # Now we need to find every occurence of a useless word (for example: CHAPTER)
            # Big hint: Useless words are usually all caps...usually
            # Made a function to do this, so we just call that a bunch of times
            rawDataLines = axeAllInstances("PREFACE", rawDataLines)
            rawDataLines = axeAllInstances("CHAPTER", rawDataLines)
            rawDataLines = axeAllInstances("Chapter", rawDataLines)

            # Make a string out of the list:
            processedText = ''  # Final Result
            for i in range(len(rawDataLines)):
                if (i < (len(rawDataLines) - 1)):  # Append new line chars to all but the last line
                    processedText = processedText + rawDataLines[i] + "\n"
                else:
                    processedText = processedText + rawDataLines[i]

            # Write the processed text to a new file and store the processed file:
            finalFile = open("../Processed/" + dirs + "/" + str(filename), "w", encoding="utf-8")
            finalFile.write(processedText)
            finalFile.close()