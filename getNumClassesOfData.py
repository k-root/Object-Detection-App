import csv 
import requests 
import xml.etree.ElementTree as ET 
import os
        
  
def parseXML(xmlfile): 
  
    # create element tree object 
    tree = ET.parse(xmlfile) 
  
    # get root element 
    root = tree.getroot() 
  
    # create empty list for news items 
    classes = [] 
    objects = root.findall('.//object')
    for obj in objects:
        name = obj.find('name').text
        classes.append(name)
    
    return classes
  
      
def main(): 
    datasetDir = "datasets/ggbDatasetStraightFlangedFRC/"
    classesList = []
    xmlFileList = os.listdir(datasetDir+"annots")
    for xmlFile in xmlFileList:
        data = parseXML(datasetDir+"annots/"+xmlFile)
        classesList.extend(data)  
    uniqueClasses = set(classesList)
    numClasses = len(uniqueClasses)
    print(uniqueClasses,numClasses)
      
if __name__ == "__main__": 
  
    # calling main function 
    main() 