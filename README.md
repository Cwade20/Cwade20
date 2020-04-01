## Question 1
Make and show a tree with 2 levels


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
                special_characters=True, class_names=True, feature_names=feature_cols)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```
![2 Level](https://user-images.githubusercontent.com/61456930/78187018-b3ffce00-743b-11ea-9302-9b4253f5a228.png)



## Question 2
Make and show a tree with 3 levels

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

clf = DecisionTreeClassifier(max_depth=3)

clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
    
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
```
# Accuracy: 0.9671052631578947

```python
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, class_names=True, feature_names=feature_cols)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```
![3 Level](https://user-images.githubusercontent.com/61456930/78186997-aba79300-743b-11ea-9049-e8585e5a64f0.png)

## Question 3
What is the optimal decision tree and why?

>Based on the results of our decision tree, I feel that a level of 3 is sufficient for this classification. At a tree depth of 3, we have 4 out of 7 of our leafs indicating a gini index score of 0.0. This correlates to our projection classifying all samples within its respective leaf. This equates to a 57% 100% prediction. When analyzing the gini score for the remaining 3 leafs, we have values of .444, .015, and .453 respectivly. While two of these leafs are relativly higher indicating that of the given sample size, there is a 44.4% chance and 45.3% chance to be incorrectly categorized within the leaf, our third value of 1.5% could almost be categorized as near perfect, giving us 5 out of 7 or 71.4% accuracy within our tree. If we review the data from our 2 depth tree, we can see that we only have 1 out of 4 leafs, or 25% 100% accuracy. The remaining leafs more often than not incorrectly cateogrize our information. However splitting to a greater value of depths, I feel adds too much clutter to our visual, and I do not feel it adds any extraordinary data of relevence. I have diaplayed a 4 level tree below for visual aid. 

![4](https://user-images.githubusercontent.com/61456930/78189443-09d67500-7440-11ea-8eb7-b4545f55b4be.png)
