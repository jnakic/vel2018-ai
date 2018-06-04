#######################################################
# Script:
#    trainDecisionTree.py
# Usage:
#    python trainDecisionTree.py
# Description:
#    Build the prediction model based on training data
# Authors:
#    Jackie Chu,   cchu@salesforce.com
#    Jasmin Nakic, jnakic@salesforce.com
#######################################################

import sys
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
import graphviz 

# Enable or disable debug printing
debugFlag = True

# Feature list
perfCols = ["PageTime_ms","TotalServerTime_ms","TotalBrowserTime_ms","Action_count","Api_count","Db_count","DbTime_ms","Xhr_count"]

# GraphViz metadata
graphCols = ("dummy","PageTime_ms","TotalServerTime_ms","TotalBrowserTime_ms","Action_count","Api_count","Db_count","DbTime_ms","Xhr_count")
targetVals = ("Invalid","Regression","Success","Error")

# Model options
maxDepth = 3

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
    # Prepare the data for the model
    X = np.zeros(data.shape[0])
    X = np.reshape(X,(-1,1))
    X = addColumns(X,data,colList)
    if debugFlag:
        print("X 0: ", X[0:5])
    Y = np.copy(data["Status"])
    if debugFlag:
        print("Y 0: ", Y[0:5])

    # Build the model based on training data
    model = tree.DecisionTreeClassifier(max_depth=maxDepth)
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

    # Visualize the decision tree model
    dot_data = tree.export_graphviz(model, out_file=None, 
                         feature_names=graphCols,
                         class_names=targetVals,
                         filled=True, rounded=True,  
                         special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render(modelName) 

    # Write the model to the file
    modelFileName = modelName+".model"
    joblib.dump(model,modelFileName)
    return P
#end genModel

def writeResult(output,data,p):
    result = np.array(
       np.empty(data.shape[0]),
       dtype=[
           ("Page","|U20"),
           ("PageTime_ms",int),
           ("TotalServerTime_ms",int),
           ("TotalBrowserTime_ms",int),
           ("Action_count",int),
           ("Api_count",int),
           ("Db_count",int),
           ("DbTime_ms",int),
           ("Xhr_count",int),
           ("Status","|U20"),
           ("PREDICTION","|U20")
        ]
    )
    result["PageTime_ms"]     = data["PageTime_ms"]
    result["TotalServerTime_ms"]     = data["TotalServerTime_ms"]
    result["TotalBrowserTime_ms"]     = data["TotalBrowserTime_ms"]
    result["Action_count"]     = data["Action_count"]
    result["Api_count"]    = data["Api_count"]
    result["Db_count"]    = data["Db_count"]
    result["DbTime_ms"]    = data["DbTime_ms"]
    result["Xhr_count"]    = data["Xhr_count"]
    result["Status"] = data["Status"]
    result["PREDICTION"] = p
    if debugFlag:
        print("R 0-5: ", result[0:5])
    hdr = "PageTime_ms,TotalServerTime_ms,TotalBrowserTime_ms,Action_count,Api_count,Db_count,DbTime_ms,Xhr_count,Status,PREDICTION"
    np.savetxt(output,result,fmt="%s",delimiter=",",header=hdr,comments="")
#end writeResult

# Start
inputFileName = "PerfRun_TrainingData.csv"
outputFileName = "PerfRun_TrainingResult.txt"
modelName = "PerfRun"

# All input columns - data types are strings, float and int
trainData = np.genfromtxt(
    inputFileName,
    delimiter=',',
    names=True,
    dtype=("|U20",int,int,int,int,int,int,int,int)
)
if debugFlag:
    print("trainData 0: ", trainData[0:5])

P = genModel(trainData,perfCols,modelName)

writeResult(outputFileName,trainData,P)
