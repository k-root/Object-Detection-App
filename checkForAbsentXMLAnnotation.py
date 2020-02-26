import os
import shutil


sourceImagesFolder = r"C:\Users\Kaushik Koilada\Documents\Desk\Projects\ggbearings\250 of each type prints\Flanged MP Images/"
# xmlFilesList = os.listdir(destinationDatasetFolder+"annots/")
xmlFilesList = os.listdir(r"C:\Users\Kaushik Koilada\Documents\Desk\Projects\ggbearings\250 of each type prints\dataset/annots/")
# filesList = os.listdir(sourceImagesFolder)
# existingImages = os.listdir(destinationDatasetFolder+"images")
existingImages = os.listdir(r"C:\Users\Kaushik Koilada\Documents\Desk\Projects\ggbearings\250 of each type prints\dataset/images")
# print(xmlFilesList)
# print(filesList)

for imgFile in existingImages:
    fileName = imgFile[:-4]
    xmlFileName = fileName+".xml"
    if(xmlFileName not in xmlFilesList):
        print(xmlFileName)
    # if (imageFileName in filesList) and (imageFileName not in existingImages):
    #     shutil.copy(sourceImagesFolder+imageFileName, destinationDatasetFolder+"images")
