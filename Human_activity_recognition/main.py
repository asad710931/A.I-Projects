#importing all necessary libriries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import seaborn as sbn
import matplotlib.pyplot as plt
import pickle

#Loading and proccessing data
data=pd.read_csv('humun_activity.csv')
X=data.drop(["Activity"], axis=1)
y=data["Activity"]

#splititing data for training and tesing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

#initilizing model and start training data
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)


#save trained model for future use
with open('model3.pkl','wb') as f:
    pickle.dump(model,f)


#tesing with prediction 
y_pred=model.predict(X_test)

accur_score=accuracy_score(y_test,y_pred)
prec_score=precision_score(y_test,y_pred)
rec_score=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)

# print(f"Accuracy : {accur_score}\nPrecision : {prec_score}\nRecall : {rec_score}\nF1-score : {f1}")

cm=confusion_matrix(y_test,y_pred)

sbn.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title('confusion_matrix')
plt.show()
