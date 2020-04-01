```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.tree import export_graphvis
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


data=pd.read_excel('/Users/Charles/Desktop/BostonHousing.xlsx', header=0)



col_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX' ,'PTRATIO','B','LSTAT', 'MEDV']

feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX' ,'PTRATIO','B','LSTAT']
X = data[feature_cols]
y = data['CAT. MEDV']

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.3, random_state=1)




clf = DecisionTreeClassifier(max_depth=2)


clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
    
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
```
# Accuracy: 0.9671052631578947

```python


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```
![Boston Housing](https://user-images.githubusercontent.com/61456930/78094351-5bc8bd80-73a2-11ea-9b9c-2c0d1c2cfc1c.png)
