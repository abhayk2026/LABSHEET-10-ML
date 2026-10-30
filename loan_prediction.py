"""Loan approval prediction; synthetic data for educational demonstration."""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
base=Path(__file__).parent;df=pd.read_csv(base/'loan_sample.csv')
X=df.drop(columns=['approved']);y=df['approved']
cat=X.select_dtypes(include='object').columns.tolist();num=X.select_dtypes(exclude='object').columns.tolist()
pre=ColumnTransformer([('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat),('num',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
best=None;best_score=-1
for name,est in [('Logistic Regression',LogisticRegression(max_iter=1000)),('Random Forest',RandomForestClassifier(n_estimators=100,random_state=42))]:
 pipe=Pipeline([('preprocess',pre),('model',est)]);pipe.fit(X_train,y_train);pred=pipe.predict(X_test);score=accuracy_score(y_test,pred)
 print(name,'accuracy:',round(score,3));print(classification_report(y_test,pred,zero_division=0))
 if score>best_score:best,best_score=pipe,score
joblib.dump(best,base/'loan_model.joblib')
print('Saved model. Example prediction:',best.predict(X.iloc[[0]])[0]);print('For educational use only.')
