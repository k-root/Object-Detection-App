import os
import shutil

destinationDatasetFolder = "ggbFlanged/"
sourceImagesFolder = r"C:\Users\Kaushik Koilada\Documents\Desk\Projects\ggbearings\250 of each type prints\Flanged MP Images/"
xmlFilesList = os.listdir(destinationDatasetFolder+"annots/")
filesList = os.listdir(sourceImagesFolder)
existingImages = os.listdir(destinationDatasetFolder+"images")
print(xmlFilesList)
print(filesList)

for xmlFile in xmlFilesList:
    fileName = xmlFile[:-4]
    imageFileName = fileName+".jpg"
    if (imageFileName in filesList) and (imageFileName not in existingImages):
        shutil.copy(sourceImagesFolder+imageFileName, destinationDatasetFolder+"images")
