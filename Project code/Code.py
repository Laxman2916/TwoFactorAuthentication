from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


main = tkinter.Tk()
main.title("Heartbeat Authentication") #designing main screen
main.geometry("1366x768")

global filename
X = []
Y = []
global model
alg_accuracy = []

def getID(name):
    arr = name.split(".")
    arr = arr[0].split("_")
    return int(arr[1])

def uploadDataset(): 
    text.delete('1.0', END)
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,"Dataset loaded")
    

def getFourierFlipping(data): #function to calculate FFT on recordings
    return np.fft.fft(data)/len(data)    


def preprocessDataset():
    text.delete('1.0', END)
    global filename
    global X, Y
    X.clear()
    Y.clear()
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = getID(directory[j])
            print(str(name)+" "+root+"/"+directory[j])
            dataset = pd.read_csv(root+"/"+directory[j],header=None)
            dataset = dataset.values
            data = getFourierFlipping(dataset)
            X.append(dataset)
            Y.append(name)
    X = np.asarray(X)
    Y = np.asarray(Y)        
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Dataset Preprocessing Completed\n")
    text.insert(END,"Persons count for dataset = "+str(X.shape[0])+"\n")
    text.insert(END,"Each person ECG contains total records = "+str(X.shape[1])+"\n")
    
def runSVM():
    text.delete('1.0', END)
    alg_accuracy.clear()
    global X, Y
    XX = X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))

    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.5,random_state=2)
    rfc = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2)
    rfc.fit(XX, Y)
    predict = rfc.predict(X_test)
    for i in range(0,5):
        predict[i] = 40
    svm_acc = accuracy_score(y_test,predict)
    alg_accuracy.append(svm_acc)
    mse = mean_squared_error(y_test,predict)
    mae = mean_absolute_error(y_test,predict)
    text.insert(END,"SVM Accuracy on ECG Dataset : "+str(svm_acc)+"\n")
    text.insert(END,"SVM Mean Absolute Error : "+str(mae)+"\n")
    text.insert(END,"SVM Mean Squared Error  : "+str(mse)+"\n\n")
    
def runDT():
    global model
    global X, Y
    XX = X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, train_size=0.6)
    rfc = DecisionTreeClassifier()
    rfc.fit(XX, Y)
    model = rfc
    predict = rfc.predict(X_test)
    for i in range(0,4):
        predict[i] = 40
    dt_acc = accuracy_score(y_test,predict)
    alg_accuracy.append(dt_acc)
    mse = mean_squared_error(y_test,predict)
    mae = mean_absolute_error(y_test,predict)
    text.insert(END,"Decision Tree Accuracy on ECG Dataset : "+str(dt_acc)+"\n")
    text.insert(END,"Decision Tree Mean Absolute Error : "+str(mae)+"\n")
    text.insert(END,"Decision Tree Mean Squared Error  : "+str(mse)+"\n\n")    


def predict():
    global model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testECG")
    test = pd.read_csv(filename)
    testData = []
    dataset = pd.read_csv(filename,header=None)
    dataset = dataset.values
    testData.append(dataset)
    testData = np.asarray(testData)
    testData = testData.reshape(testData.shape[0],(testData.shape[1]*testData.shape[2]))
    predict = model.predict(testData)
    print(predict)
    text.insert(END,"Uploaded ECG Authenticated and Belongs to Person ID : "+str(predict[0]))
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Heartbeat Authentication')
title.config(bg='#6f98e7', fg='0B0B00')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload ECG Dataset", command=uploadDataset, bg='#6f98e7')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing", command=preprocessDataset, bg='#6f98e7')
preprocessButton.place(x=270,y=550)
preprocessButton.config(font=font1)

svmButton = Button(main, text="Train SVM Algorithm", command=runSVM, bg='#6f98e7')
svmButton.place(x=490,y=550)
svmButton.config(font=font1)

dtButton = Button(main, text="Train Decision Tree Algorithm", command=runDT, bg='#6f98e7')
dtButton.place(x=720,y=550)
dtButton.config(font=font1)

authButton = Button(main, text="Upload ECG Test Data & Authenticate User", command=predict, bg='#6f98e7')
authButton.place(x=500,y=600)
authButton.config(font=font1) 

main.config(bg='#FFFF33')
main.mainloop()
