#######################################################
# Script:
#    trainDev.py
# Usage:
#    python trainDev.py <input_file> <output_file>
# Description:
#    Build the prediction model based on training data
#    Pass 1: prediction based on hours in a week
# Authors:
#    Jasmin Nakic, jnakic@salesforce.com
#    Jackie chu,   jchu@salesforce.com
#######################################################

import sys
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
import graphviz 

# debugFlag = False
debugFlag = True
# Feature list
graphCols = ("dummy","M1","M2","M3","M4","M5","ERR")
targetVals = ("BUG","VALID","REGRESSION","INVALID")
perfCols = ["M1","M2","M3","M4","M5","ERR"]

def addColumns(dest, src, colNames):
    # Initialize temporary array
    tmpArr = np.empty(src.shape[0])
    cols = 0
    # Copy column content
    for name in colNames:
        if cols == 0: # first column
            tmpArr = np.copy(src[name])
            tmpArr = np.reshape(tmpArr,(-1,1))
        else:
            tmpCol = np.copy(src[name])
            tmpCol = np.reshape(tmpCol,(-1,1))
            tmpArr = np.append(tmpArr,tmpCol,1)
        cols = cols + 1
    return np.append(dest,tmpArr,1)
#end addColumns

def genModel(data,colList,modelName):
    # Initialize array
    X = np.zeros(data.shape[0])
    X = np.reshape(X,(-1,1))

    # Add columns
    X = addColumns(X,data,colList)

    if debugFlag:
        print("X 0: ", X[0:5])

    Y = np.copy(data["STATUS"])
    if debugFlag:
        print("Y 0: ", Y[0:5])

    model = tree.DecisionTreeClassifier()
    print(model.fit(X, Y))

    print("NAMES: ", data.dtype.names)
    print("TREE: ", model.tree_)
    print("NODE_COUNT: ", model.tree_.node_count)
    print("CHILDREN_LEFT: ", model.tree_.children_left)
    print("CHILDREN_RIGHT: ", model.tree_.children_left)
    print("FEATURE: ", model.tree_.feature)
    print("THRESHOLD: ", model.tree_.threshold)
    print("SCORE values: ", model.score(X,Y))

    P = model.predict(X)
    if debugFlag:
        print("P 0-5: ", P[0:5])

    dot_data = tree.export_graphviz(model, out_file=None, 
                         feature_names=graphCols,
                         class_names=targetVals,
                         filled=True, rounded=True,  
                         special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render(modelName) 

    joblib.dump(model,modelName)
    return P
#end genModel

def writeResult(output,data,p):
    # generate result file
    result = np.array(
       np.empty(data.shape[0]),
       dtype=[
           ("M1",float),
           ("M2",float),
           ("M3",float),
           ("M4",int),
           ("M5",float),
           ("ERR",int),
           ("STATUS","|U20"),
           ("PREDICTION","|U20")
        ]
    )

    result["M1"]     = data["M1"]
    result["M2"]     = data["M3"]
    result["M3"]     = data["M3"]
    result["M4"]     = data["M4"]
    result["M5"]     = data["M5"]
    result["ERR"]    = data["ERR"]
    result["STATUS"] = data["STATUS"]
    result["PREDICTION"] = p

    if debugFlag:
        print("R 0-5: ", result[0:5])
    hdr = "M1,M2,M3,M4,M5,ERR,STATUS,PREDICTION"
    np.savetxt(output,result,fmt="%s",delimiter=",",header=hdr,comments="")
#end writeResult

# Start
inputFileName = sys.argv[1]
outputFileName = sys.argv[2]
# All input columns - data types are strings, float and int
trainData = np.genfromtxt(
    inputFileName,
    delimiter=',',
    names=True,
    dtype=(float,float,float,int,float,int,"U20")
)

P = genModel(trainData,perfCols,"perfTree")

writeResult(outputFileName,trainData,P)
