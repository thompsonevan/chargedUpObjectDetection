import os
import xml.etree.ElementTree as ET
import untangle

dir = 'xmls/'

# for idx, example in enumerate(os.listdir()):
#     if(example.endswith('.xml')):
#         xmlObj = untangle.parse(os.path.join(dir, example))
#         xmlObj


for dx, file in enumerate(os.listdir(dir)):
    # print(dir + file)
    if file != 'desktop.ini':
        tree = ET.parse(dir + file)
        root = tree.getroot()

        for child in root:
            if(child.tag == 'path'):
                text = child.text.split('\\')[4].split('img')[1]
                print(text)
                newText = r"C:\Users\evant\trainingData\imgs\img" + text
                print(newText)
                child.text = newText
                print(child.text)
        tree.write(dir + file)