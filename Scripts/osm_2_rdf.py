#!/usr/bin/env python3


import osmium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys

class osm2rdf_handler(osmium.SimpleHandler):
    def __init__(self, outpath):
        osmium.SimpleHandler.__init__(self)    
        self.counts=0
        self.fo = open(outpath, 'w', encoding="utf-8")
    
    def printTriple(self, s, p, o):
            print("\t".join([s, p, o, "."]), file=self.fo)
    
    def close(self):
        print(str(self.counts))
        self.fo.close()
    
    def node(self, n):
        if not ("wikidata" in n.tags):
        #comment above line and uncomment following line for wikipedia data
        #if not ("wikipedia" in n.tags):
            return
        
     #desired format
    # :24010109189 a geo:SpatialThing .
    # :24010109189 geo:latitude 52.5170365 .
   #  :24010109189 geo:longitude 13.3888599 .


        lat = '"'+str(n.location.lat)+'"'+'^^<http://www.w3.org/2001/XMLSchema#decimal>'
        lon = '"'+str(n.location.lon)+'"'+'^^<http://www.w3.org/2001/XMLSchema#decimal>'
        id = "<https://www.openstreetmap.org/node/"+str(n.id)+">"
        
        self.printTriple(id, "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                    "<http://www.w3.org/2003/01/geo/wgs84_pos#SpatialThing>")
            
            
            
        point = '"Point('+str(n.location.lat)+' '+str(n.location.lon)+')"^^<http://www.opengis.net/ont/geosparql#wktLiteral>'
    
        self.printTriple(id, "<http://www.w3.org/2003/01/geo/wgs84_pos#lat>", lat)
        self.printTriple(id, "<http://www.w3.org/2003/01/geo/wgs84_pos#long>", lon)
        self.printTriple(id, "<http://www.w3.org/2003/01/geo/wgs84_pos#Point>", point)

                
        for k,v in n.tags:
            #if k == "wikidata" or k == "wikipedia":
            #    continue
                
                
            val = str(v)
 
            val=val.replace("\\", "\\\\")
            val=val.replace('"', '\\"')
            val=val.replace('\n', " ")
            
            k = k.replace(" ", "")
                
            self.printTriple(id, "<https://wiki.openstreetmap.org/wiki/Key:"+k+">", '"'+val+'"')
            
            
            
h = osm2rdf_handler(sys.argv[2])
h.apply_file(sys.argv[1])

