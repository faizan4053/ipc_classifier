
k=0
file = open('ipg180227.xml', 'r') 
for each in file:
    print(each)
    if each.find("<?xml version")>-1:
        k=k+1
        print(k)
        f=open("xml_files/"+str(k)+'.xml','a')
    #print(each,k)
    f.write(each)