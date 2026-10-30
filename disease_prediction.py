"""Symptom-based demonstration classifier; not a medical diagnostic tool."""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
base=Path(__file__).parent;df=pd.read_csv(base/'symptoms_sample.csv')
X=df.drop(columns=['disease']);y=df['disease']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
best=None;best_score=-1
for name,model in [('Decision Tree',DecisionTreeClassifier(random_state=42)),('Random Forest',RandomForestClassifier(n_estimators=100,random_state=42))]:
 model.fit(X_train,y_train);pred=model.predict(X_test);score=accuracy_score(y_test,pred)
 print(name,'accuracy:',round(score,3));print(classification_report(y_test,pred,zero_division=0))
 if score>best_score:best,best_score=model,score
joblib.dump(best,base/'disease_model.joblib')
print('Example classification:',best.predict(X.iloc[[0]])[0]);print('WARNING: Synthetic examples; never use for diagnosis or treatment.')
