import os

# Get all the Authors to go through:
dirList = list()
for root, dirs, files in os.walk("../RawBooks", topdown=False):
    for name in dirs:
        if (name != ".ipynb_checkpoints"):
            dirList.append(name)

# Iterate through directory and process the data:
for dirs in dirList:
    if (dirs != "Shakespeare"):
        for file in os.listdir(os.fsencode("../RawBooks/" + dirs)):
            filename = os.fsdecode(file)
            print("This book will be processed: " + dirs + " " + str(filename))

            # Read in the raw data as a string
            rawDataFile = open("../RawBooks/" + dirs + "/" + str(filename), "r", encoding="utf-8")
            rawData = rawDataFile.read()  # type(rawData) == string

            # Now we need to find the first occurence of what we would expect to see
            # For a book, look for "Preface" or "Chapter 1"
            if (rawData.find("PREFACE") != -1):
                if (rawData.find("End of the Project") != -1):
                    processedText = rawData[rawData.find("PREFACE"):rawData.find("End of the Project")]
                elif (rawData.find("End of Project") != -1):
                    processedText = rawData[rawData.find("PREFACE"):rawData.find("End of Project")]
                elif (rawData.find("END OF THE PROJECT") != -1):
                    processedText = rawData[rawData.find("PREFACE"):rawData.find("END OF THE PROJECT")]
                elif (rawData.find("END OF THIS PROJECT") != -1):
                    processedText = rawData[rawData.find("PREFACE"):rawData.find("END OF THIS PROJECT")]
            elif (rawData.find("CHAPTER I") != -1):
                if (rawData.find("End of the Project") != -1):
                    processedText = rawData[rawData.find("CHAPTER I"):rawData.find("End of the Project")]
                elif (rawData.find("End of Project") != -1):
                    processedText = rawData[rawData.find("CHAPTER I"):rawData.find("End of Project")]
                elif (rawData.find("END OF THE PROJECT") != -1):
                    processedText = rawData[rawData.find("CHAPTER I"):rawData.find("END OF THE PROJECT")]
                elif (rawData.find("END OF THIS PROJECT") != -1):
                    processedText = rawData[rawData.find("CHAPTER I"):rawData.find("END OF THIS PROJECT")]
            elif (rawData.find("CHAPTER 1") != -1):
                if (rawData.find("End of the Project") != -1):
                    processedText = rawData[rawData.find("CHAPTER 1"):rawData.find("End of the Project")]
                elif (rawData.find("End of Project") != -1):
                    processedText = rawData[rawData.find("CHAPTER 1"):rawData.find("End of Project")]
                elif (rawData.find("END OF THE PROJECT") != -1):
                    processedText = rawData[rawData.find("CHAPTER 1"):rawData.find("END OF THE PROJECT")]
                elif (rawData.find("END OF THIS PROJECT") != -1):
                    processedText = rawData[rawData.find("CHAPTER 1"):rawData.find("END OF THIS PROJECT")]
            elif (rawData.find("Chapter 1") != -1):
                if (rawData.find("End of the Project") != -1):
                    processedText = rawData[rawData.find("Chapter 1"):rawData.find("End of the Project")]
                elif (rawData.find("End of Project") != -1):
                    processedText = rawData[rawData.find("Chapter 1"):rawData.find("End of Project")]
                elif (rawData.find("END OF THE PROJECT") != -1):
                    processedText = rawData[rawData.find("Chapter 1"):rawData.find("END OF THE PROJECT")]
                elif (rawData.find("END OF THIS PROJECT") != -1):
                    processedText = rawData[rawData.find("Chapter 1"):rawData.find("END OF THIS PROJECT")]
            elif (rawData.find("Chapter I") != -1):
                if (rawData.find("End of the Project") != -1):
                    processedText = rawData[rawData.find("Chapter I"):rawData.find("End of the Project")]
                elif (rawData.find("End of Project") != -1):
                    processedText = rawData[rawData.find("Chapter I"):rawData.find("End of Project")]
                elif (rawData.find("END OF THE PROJECT") != -1):
                    processedText = rawData[rawData.find("Chapter I"):rawData.find("END OF THE PROJECT")]
                elif (rawData.find("END OF THIS PROJECT") != -1):
                    processedText = rawData[rawData.find("Chapter I"):rawData.find("END OF THIS PROJECT")]

            finalFile = open("../Processed/" + dirs + "/" + str(filename), "w", encoding="utf-8")
            finalFile.write(processedText)
            finalFile.close()