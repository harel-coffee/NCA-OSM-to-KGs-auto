import json
import pandas as pd

inputFile = sys.argv[1]
country = sys.argv[2]
mappings = sys.argv[3]

data = pd.read_csv(inputFile, sep='\t', encoding='utf-8',)

labelName = []
colNameOsm = []
colNameWiki = []
for col in data.columns:
    if 'cls_' in col:# and col in clsRm:
        labelName.append(col)
    elif 'osmTagKey_' in col:
        #print(col)
        try:
            if data[col].value_counts()[1]>50:
                colNameOsm.append(col)
        except KeyError:
            KeyError
    elif 'prop_' in col and 'prop_instance of' not in col:
        colNameWiki.append(col)
labels =  data[labelName]
columnsOSM = data[colNameOsm]
columnsWiki = data[colNameWiki]


columnsWiki.columns = columnsWiki.columns.str.replace(r'prop_', '').str.replace(' ','_').str.replace(r'<http://dbpedia.org/ontology/', '')
columnsWiki.columns = columnsWiki.columns.str.replace(r'>', '').str.replace('<http://www.w3.org/2003/01/geo/wgs84_pos#','')
columnsOSM.columns = columnsOSM.columns.str.replace(r'osmTagKey_', '')
labels.columns = labels.columns.str.replace(r'cls_', '').str.replace(' ','_')

columnsWiki = pd.concat([columnsWiki, labels], axis=1)

columnsOSM.to_csv('OsmTable'+country+'_source.csv', sep=',', encoding='utf-8', index=False)
columnsWiki.to_csv('WikiTable'+country+'_target.csv', sep=',', encoding='utf-8', index=False)

source = {}
for i in columnsOSM.columns:
    #print(dummiesOSM[i].dtypes)
    temp = columnsOSM[i].dtypes
    if temp == 'object':
        typ = 'text'
    else:
        typ = 'integer'
    source[i] = {'type': typ}

target = {}
for i in columnsWiki.columns:
    #print(i)
    temp = columnsWiki[i].dtypes
    if temp == 'object':
        typ = 'text'
    else:
        typ = 'integer'
    target[i] = {'type': typ}

with open('osmTable'+country+'_source.json', 'w') as outfile:
    json.dump(source, outfile)
with open('wikiTable'+country+'_target.json', 'w') as outfile:
    json.dump(target, outfile)


dfprerec = pd.read_csv(mappings,sep='\t', encoding='utf-8',)


SM = list(dfprerec[dfprerec['real'] == 't'].tag.values)
tm = list(dfprerec[dfprerec['real'] == 't'].cls.values)

data = {}
data['matches'] = []
for i in range(len(SM)):
    data['matches'].append({
    "source_table": 'osmTable'+country+'_source',
    "source_column": SM[i],
    "target_table": 'wikiTable'+country+'_target',
    "target_column": tm[i].replace('cls_','')
    })


with open('osm_wiki_mapping.json', 'w') as outfile:
    json.dump(data, outfile)



