dataFile = open("LDPCGraphValues.txt")
dataString = dataFile.read()

dataStringLines = dataString.splitlines()

lDPCGraphsValue = dict()

for line in dataStringLines:
    attr = line.split(",")
    lDPCGraphsValue[(attr[0],int(attr[1]),int(attr[2]),int(attr[3]))] = int(attr[4])

print("fuck")