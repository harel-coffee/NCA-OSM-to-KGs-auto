import textdistance
import pandas as pd
import fasttext
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
model = fasttext.load_model("cc.en.300.bin")

inputFile = sys.argv[1]
country = sys.argv[2]
mappings = sys.argv[3]

data = pd.read_csv(inputFile, sep='\t', encoding='utf-8',)

labelName = []
tagName = []
for col in data.columns:
    if 'cls_' in col:
        labelName.append(col)
    if '=' in col:
        tagName.append(col)
        
confOSM = []
for i in labelName:
    for j in tagName:
        #print(j)
        try:
            temp1 = len(data[(data[i] ==1) & (data[j] ==1)])
        except KeyError:
            temp1 = 0
        temp2 = len(data[(data[j] ==1)])
        try:
            temp3 = temp1/temp2
        except ZeroDivisionError:
            temp3 = 0
        confOSM.append((i.replace('cls_',''),j.replace('osmTagKey_',''),temp3))
df_tag_cls= pd.DataFrame(confOSM, columns=['cls', 'tag', 'value'])

dfprerec = pd.read_csv(mappings,sep='\t', encoding='utf-8',)
dfprerecSD = pd.merge(dfprerec, df_tag_cls,  how='left', left_on=['tag','cls'], right_on = ['tag','cls'])
dfprerecSD.to_csv('SDTypeMappings'+country+'.csv', sep='\t', encoding='utf-8')

source = list(dfprerec['tag'].values)
target = list(dfprerec['cls'].values) 
fasttextDist = []
for i in range(len(source)):
    osm = model.get_word_vector(source[i])
    wiki = model.get_word_vector(target[i])
    fasttextDist.append((1 - spatial.distance.cosine(osm, wiki)))
dfprerec['fasttextDist'] = fasttextDist

levenshteinSim = []
for i in range(len(source)):
    levenshteinSim.append(textdistance.levenshtein.normalized_similarity(source[i], target[i]))
dfprerec['leven'] = levenshteinSim

dfprerec.to_csv('fastTextLevenShteinMatches'+country+'.csv', sep='\t', encoding='utf-8')