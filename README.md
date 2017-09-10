# DecisionKNN

Implement decision tree algorithm and KNN algorithm to classify documents as Tech, Political, sports or word and compare performance of each algorithm 

**execution:** python3 DecisionKNN.py

**PreRequisites:**
- Presence of Data.csv File.
- Modules: Numpy, Seaborn, Pandas, Matplotlib, sklearn, stringIO, Ipython

**Outputs:**
- Graphs_Decision_Tree - Each run generates graph in the same folder where the script is executed.
- Terminal Output - Decision tree -- Accuracy, Recall, F-Measure, Precision, graph, Confusion Matrix
		     KNN -- Accuracy, Recall, F-Measure, Precision, Confusion Matrix

**Analysis:** 
- Two cumulative bar graphs are generated 
     1. Comparing Accuracy for decision tree and KNN in 5 folder cross validation
     2. Comparing F-Measure for decision tree and KNN in 5 folder cross validation
- Error Rate: There is a module to find the perfect K value by calculating error rate for k values from 1 to 20. Once a Perfect K value is identified the module is commented and K value has been used to run KNN algorithm. If required the Module can be uncommented and executed which will show the graph for Error rate V/S K value.
