#this class from sklearn will split data for training and testing
from sklearn.model_selection import train_test_split
#This is model that we will use to train datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

spamData=pd.read_csv('spambase.csv')

trainData=spamData.drop(["spam"],axis=1)
labels=spamData["spam"]

X_train,X_test,y_train,y_test=train_test_split(trainData,labels, test_size=0.2, random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
precision =precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)

print(f"accuracy :{accuracy}")
print(f"precision :{precision}")
print(f"recall :{recall}")
print(f"f1-score :{f1}")

cm=confusion_matrix(y_test,y_pred)


sb.heatmap(cm,annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.show()