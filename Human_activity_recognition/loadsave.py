import pickle
import pandas as pd



p_data=pd.read_csv("./humun_activity.csv")
pred_data=p_data.drop(['Activity'],axis=1)
expectetion=p_data["Activity"]
#print(f"{pred_data.iloc[0]}, \n {expectetion}")


with open('model2.pkl','rb') as f:
    mymodel=pickle.load(f)


#test one single prediction with first datarow in pred.csv
y_pred=mymodel.predict(pred_data.iloc[56:57])
print(y_pred)

y_pred_all=mymodel.predict(pred_data)
print(y_pred_all)