import xml.etree.ElementTree as ET
import csv

dict={}

i=1
while i<6998:
    tree=ET.parse('xml_files/'+str(i)+'.xml')
    
    root=tree.getroot()
    
    present=0
    val=""
    for x in root:
            if x.tag=="us-bibliographic-data-grant":
                for y in x:
                    if y.tag=="classifications-ipcr":
                        present=1
                        for z in y:
                            #print(z.tag)
                            for a in z:
                                if a.tag=="section":
                                    if(len(val)==0):
                                        val+=str(a.text)
    
    for x in root:
        #print(x.tag)
        s=""
        if x.tag=="claims":
        #print(x.tag)
            for y in x:
                #print(y.text)
                s+=str(y.text)
                for z in y:
                    #print(z.text)
                    s+=str(z.text)
                    for a in z:
                        #print(a.text)
                        s+=str(a.text)
                        for b in a:
                            s+=str(b.text)
                            #print(b.text)
        elif x.tag=="description":
            #print(x.tag)
            for y in x:
                if y.tag=='p':
                    #print(y.text)
                    s+=str(y.text)
        elif x.tag=="abstract":
            #print(x.tag)
            for y in x:
                if y.tag=='p':
                    #print(y.text)
                    s+=str(y.text)
        elif x.tag=="us-bibliographic-data-grant":
            for y in x:
                if y.tag=="invention-title":
                    #print(y.text)
                    s+=str(y.text)
        if present!=0 and s!="" and val!="":
            #print(s,val)
            dict[s]=val;
    print("file No:"+str(i))
    i=i+1
    
with open('ipc_file1.tsv','wt') as file:
    tsv_writer=csv.writer(file,delimiter='\t')
    for key,value in dict.items():
        tsv_writer.writerow([key.encode('utf8'),value])
        
