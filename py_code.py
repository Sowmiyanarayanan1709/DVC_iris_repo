import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import subprocess

df = pd.read_csv("C:\\Users\\s123\\Desktop\\Iris_project\\Iris.csv")

flower_mapping = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2 }
df["Species"] = df["Species"].map(flower_mapping)

X = df.drop('Species', axis = 1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=7)

model_rf = RandomForestClassifier(ccp_alpha=0.1, min_samples_split=10)
model_rf.fit(X_train,y_train)

filename = 'Random_Forest_model'
pickle.dump(model_rf,open(filename,'wb'))

cmd = "git rev-list --count HEAD"
try:
    output = subprocess.check_output(cmd.split()).decode().strip()
    git_count = int(output)
except:
    git_count = 0
accuracy = model_rf.score(X_test,y_test)

params = model_rf.get_params()
df_rf = pd.DataFrame(params, index = [0])
df_rf = df_rf.iloc[:,[1,3,8,10]]
df_rf["Accuracy"]  = accuracy
df_rf.insert(0,'model_name','Random Forest_'+str(git_count))
df_rf.insert(len(df_rf.columns), 'Number of rows', len(df) )
df_rf.to_csv("C:\\Users\\s123\\Desktop\\Iris_project\\parameters.csv",mode = 'a', index=False, header = False)

model_dt = DecisionTreeClassifier(ccp_alpha=0.1, min_samples_split=10)
model_dt.fit(X_train,y_train)

filename='Decision_Tree_model'
pickle.dump(model_dt,open(filename,'wb'))

accuracy = model_dt.score(X_test,y_test)

params = model_dt.get_params()
df_dt = pd.DataFrame(params, index = [0])
df_dt = df_dt.iloc[:,[0,2,6,8]]
df_dt["Accuracy"]  = accuracy
df_dt.insert(0,'model_name','Decision Tree_'+str(git_count))
df_dt.insert(len(df_dt.columns), 'Number of rows', len(df))
df_dt.to_csv("C:\\Users\\s123\\Desktop\\Iris_project\\parameters.csv",mode = 'a', index=False, header = False)


