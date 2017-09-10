# @Author Manish Jaisinghani 

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import IPython
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO
from IPython.display import Image
import time
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data.csv", encoding = "ISO-8859-1")
data_clean = df.dropna()
predictors = data_clean[['kids',	'say',	'things',	'president',	'diet',	'fitnessliving',	'wellparenting',	'tv',	'search',	'crime',	'east',	'digital',	'shows',	'kelly',	'wallace',	'november',	'chat',	'facebook',	'messenger',	'find',	'world',	'many',	'want',	'videos',	'must',	'watch',	'run',	'according',	'large',	'family',	'life',	'read',	'parents',	'twitter',	'school',	'interest',	'much',	'also',	'absolutely',	'ever',	'office',	'land',	'thing',	'go',	'could',	'told',	'america',	'march',	'presidential',	'campaign',	'end',	'million',	'order',	'get',	'money',	'first',	'take',	'time',	'might',	'american',	'times',	'way',	'election',	'children',	'inc',	'country',	'leader',	'free',	'high',	'thought',	'know',	'good',	'candidates',	'definitely',	'part',	'white',	'house',	'four',	'years',	'vice',	'top',	'young',	'really',	'call',	'public',	'service',	'show',	'beyond',	'vote',	'artist',	'model',	'someone',	'cancer',	'helping',	'animals',	'asked',	'make',	'better',	'place',	'latest',	'share',	'comments',	'health',	'hillary',	'clinton',	'female',	'even',	'actually',	'chance',	'lady',	'content',	'pay',	'card',	'save',	'enough',	'reverse',	'risk',	'paid',	'partner',	'cards',	'around',	'next',	'generation',	'big',	'network',	'system',	'rights',	'reserved',	'terms',	'mexican',	'meeting',	'trump',	'january',	'mexico',	'different',	'route',	'border',	'immigrants',	'trying',	'donald',	'wall',	'billion',	'signs',	'executive',	'actions',	'building',	'along',	'southern',	'nowstory',	'believe',	'fruitless',	'thursday',	'set',	'week',	'plan',	'tuesday',	'something',	'recently',	'wednesday',	'needed',	'tweet',	'trade',	'nafta',	'massive',	'@realdonaldtrump',	'jobs',	'companies',	'remarks',	'gathering',	'congressional',	'republicans',	'planned',	'together',	'unless',	'senate',	'gop',	'lawmakers',	'security',	'national',	'problem',	'illegal',	'immigration',	'see',	'need',	'statement',	'back',	'two',	'leaders',	'last',	'year',	'days',	'called',	'action',	'begin',	'process',	'announced',	'move',	'level',	'foreign',	'representatives',	'come',	'since',	'officials',	'including',	'staff',	'minister',	'government',	'team',	'car',	'department',	'homeland',	'work',	'help',	'united',	'states',	'forces',	'number',	'officers',	'visit',	'try',	'able',	'related',	'monday',	'migrants',	'home',	'city',	'conversation',	'made']]
# print(predictors)
targets = data_clean.SITE

scaler = StandardScaler()

scaler.fit(df.drop('SITE', axis = 1))

scaled_features = scaler.transform(df.drop('SITE', axis = 1))
# print(scaled_features)

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
#
#
# print(df.columns[:-1])

# print(df_feat.head())
# sns.pairplot(df, hue='SITE', palette='coolwarm')
# sns.show()
accuracy_list = []
accuracy_list_decision = []
fMeasure_list = []
fMeasure_list_decision = []
kf = KFold(n_splits=5)
i = 1
for training, testing in kf.split(df['SITE']):
    print ('xxxxxxxxxxxxxxxxxxxxxxxxxx Run {} xxxxxxxxxxxxxxxxxxxxxxxxxx'.format(i))

    pred_train = df_feat.ix[training]
    tar_train = df['SITE'][training]
    pred_test = df_feat.ix[testing]
    tar_test = df['SITE'][testing]
    knn = KNeighborsClassifier(n_neighbors = 4)
    knn.fit(pred_train, tar_train)
    pred = knn.predict(pred_test)
    print('----------------------------KNN Classification------------------------------------')
    print("Accuracy is :")
    print(sklearn.metrics.accuracy_score(tar_test,pred))
    accuracy_list.append(sklearn.metrics.accuracy_score(tar_test,pred))
    print(confusion_matrix(tar_test,pred))
    print(classification_report(tar_test,pred))
    fMeasure_list.append(f1_score(tar_test,pred, average="macro"))

    #Decision tree code
    print('----------------------------Decision Tree------------------------------------')
    pred_train_decision = predictors.ix[training]
    # print (pred_train)
    tar_train_decision = targets[training]
    pred_test_decision = predictors.ix[testing]
    tar_test_decision = targets[testing]

    #Build model on training data
    classifier_decision=DecisionTreeClassifier()
    classifier_decision=classifier_decision.fit(pred_train_decision,tar_train_decision)

    predictions_decision=classifier_decision.predict(pred_test_decision)

    print(sklearn.metrics.confusion_matrix(tar_test_decision,predictions_decision))
    print(classification_report(tar_test_decision,predictions_decision))
    accuracy_list_decision.append(sklearn.metrics.accuracy_score(tar_test_decision,predictions_decision))
    fMeasure_list_decision.append(f1_score(tar_test_decision,predictions_decision, average="macro"))

    #Displaying the decision tree

    out = StringIO()
    tree.export_graphviz(classifier_decision, out_file=out)
    import pydotplus
    graph=pydotplus.graph_from_dot_data(out.getvalue())
    #Create graph pdf 1 for each run
    millis = int(round(time.time() * 1000))  # Generate time system time in milliseconds
    Image(graph.write_pdf("graph"+str(millis)+".pdf"))

    #Calculate accuracy

    print("Accuracy Score for graph"+str(millis)+".pdf is")
    print(sklearn.metrics.accuracy_score(tar_test_decision, predictions_decision))
    i+=1
    # error_rate = []
    # for i in range(1,20):
    #     knn = KNeighborsClassifier(n_neighbors = i)
    #     knn.fit(pred_train, tar_train)
    #     pred_i = knn.predict(pred_test)
    #     error_rate.append(np.mean(pred_i != tar_test))
    # plt.figure(figsize=(10,6))
    # plt.plot(range(1,20),error_rate, color = 'blue',linestyle = 'dashed', marker = 'o', markerfacecolor='red',markersize=10)
    # plt.title('Error rate V/S K Value')
    # plt.xlabel('k')
    # plt.ylabel('error rate')
    # plt.show()
print ("Accuracy for 5 folds KNN Classifier {}".format(sum(accuracy_list) / len(accuracy_list)))
print ("Accuracy for 5 folds decision tree {}".format(sum(accuracy_list_decision) / len(accuracy_list_decision)))
print ("F-Measure for 5 folds KNN Classifier {}".format(sum(fMeasure_list) / len(fMeasure_list)))
print ("F-Measure for 5 folds Decision tree {}".format(sum(fMeasure_list_decision) / len(fMeasure_list_decision)))
x = [1,2,3,4,5]
ax = plt.subplot(111)
plt.title('Accuracy Comparison')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

bar1 = ax.bar([float(y)-.1 for y in x], accuracy_list,width=0.1,color='r',align='center')
bar2 = ax.bar(x, accuracy_list_decision,width=0.1,color='g',align='center')
ax.legend((bar1[0], bar2[0]), ('KNN Classifier', 'Decision tree'))
# ax.xaxis()
plt.show()
ax1 = plt.subplot(111)
plt.title('FMeasure comparison')
plt.xlabel('Iteration')
plt.ylabel('Fmeasure')

bar3 = ax1.bar([float(y)-.1 for y in x], fMeasure_list,width=0.1,color='r',align='center')
bar4 = ax1.bar(x, fMeasure_list_decision,width=0.1,color='g',align='center')
ax1.legend((bar3[0], bar4[0]), ('KNN Classifier', 'Decision tree'),loc=2)

plt.show()
# combined_list = [accuracy_list]
