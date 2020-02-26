import pandas as pd
df = pd.read_csv('ggbDataset.csv')
###############***###############
'''added the field names
[Type, File, Label, x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]"
to the csv file
'''
###############***###############

def processLabel(label):
    mapping = { "ID_CornerBreak":"ID Corner Break" ,"OD_ChamferLength":"OD Chamfer Length", "OD_ChamferAngle":"OD Chamfer Angle" , "Flange_Diameter":"Flange Diameter", "Length_And_FlangeThickness":"Flange Length and Thickness", "Flange_BendRadius":"Flange Bend Radius"}
    return mapping[label]

def convert_row(row):
    # print(row)
    # print(row.Label)
    if row.Label not in ['Index_And_Date', 'table', 'Assembly_Dimensions', 'Inspection_Procedure']:
        labelName = processLabel(row.Label)
        return """
        <object>
            <name>%s</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>%s</xmin>
                <ymin>%s</ymin>
                <xmax>%s</xmax>
                <ymax>%s</ymax>
            </bndbox>
        </object>""" % (labelName, row.x_min, row.y_min, row.x_max, row.y_max)

    return ""

groupedDF = df.groupby("File")
fileList = groupedDF["File"].agg(['unique'])
fileNamesList = []
for name in fileList.iterrows():
    # print(name[1]['unique'])
    filePathInCSV = name[1]['unique'][0]
    fileName = filePathInCSV.split("/")[-1]
    fileName = fileName[:-29]+".jpg"  
    fileNamesList.append(fileName)
    xmlObjectString = ''.join(groupedDF.get_group(filePathInCSV).apply(convert_row, axis=1))
    # print(xmlObjectString)
    folder = "test"
    filePath = " "
    imageWidth = "1"
    imageHeight = "1"
    imageDepth = "1"
    finalXML = """<annotation>
        <folder>%s</folder>
        <filename>%s</filename>
        <path>%s</path>
        <size>
            <width>%s</width>
            <height>%s</height>
            <depth>%s</depth>
        </size>
        <segmented></segmented>
    """ %(folder, fileName, filePath, imageWidth, imageHeight, imageDepth) + xmlObjectString + "\n</annotation>"
    print(finalXML)
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    from xml.etree import ElementTree as ET
    tree = ET.XML(finalXML)
    with open("annots/"+fileName[:-3]+"xml", "wb") as f:
        f.write(ET.tostring(tree))  