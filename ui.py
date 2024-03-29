import csv
import pandas as pd
from os import listdir
from os import stat
import classifier
from tkinter import filedialog
from tkinter import *
from tkinter import ttk
import tkinter as tk


class windowHandler:

    def __init__(self, root):
        # creat window
        top = Frame(root)
        top.pack(side=TOP)
        middle = Frame(root)
        middle.pack(fill=X)
        middle2 = Frame(root)
        middle2.pack(fill=X)
        bottom = Frame(root)
        bottom.pack(side=BOTTOM, expand=True)
        self.lbl1 = Label(middle, text="Directory Path:")
        self.lbl1.pack(padx=5, pady=15, side=LEFT)
        self.lbl2 = Label(middle2, text="Discretization Bins:")
        self.lbl2.pack(padx=5, pady=15, side=LEFT)
        self.dta = StringVar()
        self.dataPath = Entry(middle, textvariable=self.dta)
        self.dataPath.pack(padx=5, side=LEFT)
        self.bins = StringVar()
        self.bins.trace("w", lambda name, index, mode, bins=self.bins: self.checkReady())
        self.binsNum = Entry(middle2, textvariable=self.bins)
        self.binsNum.pack(padx=5, side=LEFT)
        self.openDataPath = Button(middle, text="Browse", fg='black', command=lambda: self.openPath())
        self.openDataPath.pack(padx=5,  pady=15, side=LEFT)
        self.buildButton = Button(bottom, text="Build", fg='black', state=DISABLED, command=lambda: self.build())
        self.buildButton.pack(padx=5, pady=5)
        self.classifyButton = Button(bottom, text="Classify", fg='black', state=DISABLED, command=lambda: self.classify())
        self.classifyButton.pack(padx=5, pady=5)
        self.lbl3 = Label(bottom, state=NORMAL, fg="red", text="Invalid Path")
        self.lbl3.pack(padx=5, pady=5)
        self.lbl4 = Label(bottom, state=NORMAL, fg="red", text="Invalid Bins")
        self.lbl4.pack(padx=5, pady=5)


    def build(self):
        print("running....")
        classifier.buildModel(self.binsNum.get(), self.dataPath.get()) #build the model
        self.popupmsg("Building classifier using train-set is done!")

    def classify(self):
        print("running....")
        classifier.predict(self.binsNum.get(), self.dataPath.get()) #classify the data in the test file
        self.popupmsg2("Classifying done!")

    def openPath(self):
        directory = tk.filedialog.askdirectory(initialdir="/") # open window to search folder
        self.dataPath.delete(0, END)
        self.dataPath.insert(0, directory)
        self.checkReady() # check if path is valid


    def validBins(self):
        #check if bins input valid
        try:
            b = int(self.binsNum.get())
            if b > 1:
                return True
        except:
            return False

    def validPath(self):
        i = 0
        arr = []
        for f in listdir(self.dataPath.get()):
            if f == "Structure.txt" or f == "train.csv" or f == "test.csv": #check if the folder contains this files
                arr.append(True)
                i = i + 1
                if f == "train.csv" or f == "test.csv":
                    filename = self.dataPath.get()+"/"+f
                    with open(filename) as file:
                        csvreader = csv.reader(file)
                        l = 0
                        for row in csvreader:
                            if l > 0: #check if file is empty or not
                                break
                            if row in (None, "", []):
                                return False
                            l += 1
                    df = pd.read_csv(filename)
                    if df.empty:
                        return False
                if f == "Structure.txt" and stat(self.dataPath.get()+"/"+f).st_size == 0:
                    return False

        for c in arr:
            if c is False:
                return False
        if i < 3:
            return False
        return True

    def checkReady(self):
        if len(self.dataPath.get()) > 0 and self.validPath(): #check if there is path and if valid
            self.buildButton.config(state=NORMAL)
            self.classifyButton.config(state=NORMAL)
            self.lbl3.config(text="Valid Path", fg="green")
        else:
            self.buildButton.config(state=DISABLED)
            self.classifyButton.config(state=DISABLED)
            self.lbl3.config(text="Invalid Path", fg="red")
        if len(self.binsNum.get()) > 0 and self.validBins(): #check if there is a number of bins and if valid
            self.lbl4.config(text="Valid Bins", fg="green")
            if len(self.dataPath.get()) > 0 and self.validPath():
                self.buildButton.config(state=NORMAL)
                self.classifyButton.config(state=NORMAL)
        else:
            self.buildButton.config(state=DISABLED)
            self.classifyButton.config(state=DISABLED)
            self.lbl4.config(text="Invalid Path", fg="red")



    def popupmsg(self, msg):
        #pop up message
        popup = tk.Tk()
        popup.wm_title("Naive Bayes Classifier")
        label = ttk.Label(popup, text=msg)
        label.pack(side="top", fill="x", pady=10)
        B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
        B1.pack()
        popup.mainloop()

    def popupmsg2(self, msg):
        # pop up message
        popup = tk.Tk()
        popup.wm_title("Naive Bayes Classifier")
        label = ttk.Label(popup, text=msg)
        label.pack(side="top", fill="x", pady=10)
        B1 = ttk.Button(popup, text="Okay", command=lambda: self.dest(popup))
        B1.pack()
        popup.mainloop()

    def dest(self, popup):
        popup.destroy()
        root.destroy()


root = tk.Tk()
whd = windowHandler(root)
root.title("Naive Bayes Classifier")
root.geometry('500x300')
root.resizable(width=False, height=False)
root.mainloop()
