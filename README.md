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
import numpy as np
#utlized for linear algebra solving
import pandas as pd
#utilized for data processing
from sklearn.model_selection import ShuffleSplit

data = pd.read_csv("/Users/Charles/Desktop/GMU/GBUS738/BostonHousing.csv")
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
data.head()

import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
#uses pyplot for graphics in the below lines
for i, col in enumerate(features.columns):
    plt.subplot(1, 3, i+1)
    x=data[col]
    y=prices
    plt.plot(x, y, 'o')
    #generates regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prices')
```
![RM](https://user-images.githubusercontent.com/61456930/78031814-11a7f380-7332-11ea-8aaf-31ce52e74fec.png)

```python
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, prices, test_size = 0.2, random_state = 42)
print("Training and testing split was successful")
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve,GridSearchCV

def fit_model(X, y):
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=0)
    
    regressor = DecisionTreeRegressor(random_state = 1001)
    #creates a decision tree regressor object
    tree_range = range (1,11)
    params = dict(max_depth=[1,2,3,4,5,6,7,8,9,10])
    #creates a dictionary for the paramater 'max_depth' with a range from 1 to 10
    scoring_fnc = make_scorer(performance_metric)
    #this changes our performance metric into a score function utilize make scorer
    grid = GridSearchCV(regressor, params, scoring = scoring_fnc,cv=cv_sets)
    #creates our grid search cv object with values regressor, params, socring_fnc, and cv_sets
    grid = grid.fit(X,y)
    #fits the grid search object to the data, and makes the compute optimal model
    return grid.best_estimator_
#returns optimal model after fitting data
    
reg = fit_model(X_train, y_train)
#fits the training data to the model utilizing the grid search
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
#returns what the max depth is for our model and prints as a line
```
