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
