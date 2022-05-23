from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

global X,Y
global dataset
global auc
global X_train, X_test, y_train, y_test
global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10

labels = ['110/pop3', '143/imap', '21/ftp', '22/ssh', '23/telnet', '2323/telnet', '25/smtp', '3306/mysql',
          '443/https', '445/smb', '465/smtp', '53/dns', '587/smtp', '7547/cwmp', '80/http', '8080/http', '8888/http', '993/imaps', '995/pop3s', 'not_respond']

main = tkinter.Tk()
main.title("Smart Internet Probing: Scanning Using Adaptive Machine Learning") #designing main screen
main.geometry("1300x1200")

   
#fucntion to upload dataset
def uploadDataset():
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    lbl = dataset['label']
    unique = np.unique(lbl)
    print(unique.tolist())
    text.insert(END,"Dataset before preprocessing\n\n")
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('label').size()
    label.plot(kind="bar")
    plt.tight_layout()
    plt.title("Different Port Scan Found in Dataset")
    plt.show()
    
#function to perform dataset preprocessing
def DataPreprocessing():
    global X,Y
    global dataset
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    dataset.drop(columns=['frame_info.encap_type', 'frame_info.time'],inplace=True)
    cols = ['eth.type','ip.id','ip.flags','ip.checksum','ip.src','ip.dst','ip.dsfield','tcp.flags','tcp.checksum','label']
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()
    le5 = LabelEncoder()
    le6 = LabelEncoder()
    le7 = LabelEncoder()
    le8 = LabelEncoder()
    le9 = LabelEncoder()
    le10 = LabelEncoder()
    dataset[cols[0]] = pd.Series(le1.fit_transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le2.fit_transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le3.fit_transform(dataset[cols[2]].astype(str)))
    dataset[cols[3]] = pd.Series(le4.fit_transform(dataset[cols[3]].astype(str)))
    dataset[cols[4]] = pd.Series(le5.fit_transform(dataset[cols[4]].astype(str)))
    dataset[cols[5]] = pd.Series(le6.fit_transform(dataset[cols[5]].astype(str)))
    dataset[cols[6]] = pd.Series(le7.fit_transform(dataset[cols[6]].astype(str)))
    dataset[cols[7]] = pd.Series(le8.fit_transform(dataset[cols[7]].astype(str)))
    dataset[cols[8]] = pd.Series(le9.fit_transform(dataset[cols[8]].astype(str)))
    dataset[cols[9]] = pd.Series(le10.fit_transform(dataset[cols[9]].astype(str)))
    text.insert(END,"Dataset after preprocessing\n\n")
    text.insert(END,str(dataset.head()))
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")
    
    X = X[0:20000]
    Y = Y[0:20000]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
def runMetrics(predict, y_test, algorithm):
    fpr, tpr, _ = roc_curve(y_test, predict, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr) * 100
    tpr = accuracy_score(y_test, predict) * 100
    auc.append(tpr)
    text.insert(END,algorithm+" TPR = "+str(tpr)+"\n\n")
    
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runLogisticRegression():
    global X_train, X_test, y_train, y_test, X, Y
    global auc
    auc = []
    text.delete('1.0', END)
    if os.path.exists('model/lr.txt'):
        with open('model/lr.txt', 'rb') as file:
            lr = pickle.load(file)
        file.close()        
    else:
        lr = LogisticRegression() 
        lr.fit(X, Y)
        with open('model/lr.txt', 'wb') as file:
            pickle.dump(lr, file)
        file.close()
    predict = lr.predict(X_test)
    runMetrics(predict, y_test, "Logistic Regression")
    
def runSVM():
    if os.path.exists('model/svm.txt'):
        with open('model/svm.txt', 'rb') as file:
            svm_cls = pickle.load(file)
        file.close()        
    else:
        svm_cls = svm.SVC() 
        svm_cls.fit(X, Y)
        with open('model/svm.txt', 'wb') as file:
            pickle.dump(svm_cls, file)
        file.close()
    predict = svm_cls.predict(X_test)
    runMetrics(predict, y_test, "SVM")
    

def runDT():
    if os.path.exists('model/dt.txt'):
        with open('model/dt.txt', 'rb') as file:
            dt = pickle.load(file)
        file.close()        
    else:
        dt = DecisionTreeClassifier() 
        dt.fit(X, Y)
        with open('model/dt.txt', 'wb') as file:
            pickle.dump(dt, file)
        file.close()
    predict = dt.predict(X_test)
    runMetrics(predict, y_test, "Decision Tree")

    
def runRandomForest():
    if os.path.exists('model/rf.txt'):
        with open('model/rf.txt', 'rb') as file:
            rf = pickle.load(file)
        file.close()        
    else:
        rf = RandomForestClassifier() 
        rf.fit(X, Y)
        with open('model/rf.txt', 'wb') as file:
            pickle.dump(rf, file)
        file.close()
    predict = rf.predict(X_test)
    runMetrics(predict, y_test, "Random Forest")

def runGraidentBoosting():    
    if os.path.exists('model/gb.txt'):
        with open('model/gb.txt', 'rb') as file:
            gb = pickle.load(file)
        file.close()        
    else:
        gb = GradientBoostingClassifier() 
        gb.fit(X, Y)
        with open('model/gb.txt', 'wb') as file:
            pickle.dump(gb, file)
        file.close()
    predict = gb.predict(X_test)
    runMetrics(predict, y_test, "Gradient Boosting")    
    
def runXGBoost():
    if os.path.exists('model/xgb.txt'):
        with open('model/xgb.txt', 'rb') as file:
            xgb = pickle.load(file)
        file.close()        
    else:
        xgb = XGBClassifier() 
        xgb.fit(X, Y)
        with open('model/xgb.txt', 'wb') as file:
            pickle.dump(xgb, file)
        file.close()
    predict = xgb.predict(X_test)
    runMetrics(predict, y_test, "Gradient Boosting")

def runDeepLearning():
    if os.path.exists('model/dl.txt'):
        with open('model/dl.txt', 'rb') as file:
            dl = pickle.load(file)
        file.close()        
    else:
        dl = MLPClassifier() 
        dl.fit(X, Y)
        with open('model/dl.txt', 'wb') as file:
            pickle.dump(dl, file)
        file.close()
    predict = dl.predict(X_test)
    runMetrics(predict, y_test, "Deep Learning Neural Network")
    

def graph():
    height = auc
    bars = ('Logistic Regression','SVM','Decision Tree','Random Forest','Gradient Boosting','XGBoost','Deep Learning')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("All Algorithms TPR (True Positive Rate) Comparison Graph")
    plt.show()

def GUI():
    global main
    global text
    font = ('times', 16, 'bold')
    title = Label(main, text='Smart Internet Probing: Scanning Using Adaptive Machine Learning')
    title.config(bg='darkviolet', fg='gold')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)

    font1 = ('times', 12, 'bold')
    text=Text(main,height=30,width=110)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=100)
    text.config(font=font1)

    font1 = ('times', 13, 'bold')
    uploadButton = Button(main, text="Upload Internet Port Scan Dataset", command=uploadDataset, bg='#ffb3fe')
    uploadButton.place(x=900,y=100)
    uploadButton.config(font=font1)  

    processButton = Button(main, text="Dataset Preprocessing", command=DataPreprocessing, bg='#ffb3fe')
    processButton.place(x=900,y=150)
    processButton.config(font=font1) 

    lrButton = Button(main, text="Run Logistic Regression Algorithm", command=runLogisticRegression, bg='#ffb3fe')
    lrButton.place(x=900,y=200)
    lrButton.config(font=font1) 

    svmButton = Button(main, text="Run SVM Algorithm", command=runSVM, bg='#ffb3fe')
    svmButton.place(x=900,y=250)
    svmButton.config(font=font1)

    dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDT, bg='#ffb3fe')
    dtButton.place(x=900,y=300)
    dtButton.config(font=font1) 

    rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest, bg='#ffb3fe')
    rfButton.place(x=900,y=350)
    rfButton.config(font=font1)

    gbButton = Button(main, text="Run Gradient Boosting Algorithm", command=runGraidentBoosting, bg='#ffb3fe')
    gbButton.place(x=900,y=400)
    gbButton.config(font=font1)

    xgButton = Button(main, text="Run XGBoost Algorithm", command=runXGBoost, bg='#ffb3fe')
    xgButton.place(x=900,y=450)
    xgButton.config(font=font1)

    dlButton = Button(main, text="Run Deep Learning  Algorithm", command=runDeepLearning, bg='#ffb3fe')
    dlButton.place(x=900,y=500)
    dlButton.config(font=font1)

    graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
    graphButton.place(x=900,y=550)
    graphButton.config(font=font1)

    main.config(bg='forestgreen')
    main.mainloop()
    
if __name__ == "__main__":
    GUI()


    
