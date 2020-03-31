Group work allowed.

 If you build a decision tree with the notes below successfully before I post the code, that will be the homework for that assignment with extra points for completion until I post the sample code 3 days later.  At that point, I will post the sample code and a new data set and that will be the assignment instead but no extra credit will be available at that point and the previous data set cannot be used.

The extra credit will expire April 02 11:59 PM.

This will just be creating a decision tree.  Here is some pseudocode and hints:

from sklearn.tree import DecisionTreeClassifier, plot_tree
 Import data set as data frame
 First columns are used as predictors

 The last column is whether its a cost neighborhood# 2nd to last col is the continuous version of this variable# And is thus excluded (hint: use df.icol)
 Make and show a tree with 2 levels (hint: set max depth, fit the tree, plot the tree)

 Now with three
 What is the optimal tree depth?# Overfit here, but don’t do this in real life# Use for statement, fit the tree, check accuracy, print accuracy)

```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Load the Boston housing dataset
data = pd.read_csv('/Users/Charles/Desktop/GMU/GBUS738/BostonHousing.csv', header=0)

col_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX' ,'PTRATIO','B','LSTAT', 'MEDV']

feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX' ,'PTRATIO','B','LSTAT']
X = data[feature_cols]
y = data.MEDV

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
    
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


from sklearn.tree import export_graphvis
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphvia(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = feature_cols, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Boston.png')
Image(graph.create_png())
```
