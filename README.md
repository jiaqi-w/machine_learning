# machine_learning

**Requirements**
python 3.4+,
sklearn,
pandas


## Lasso Linear Regression

Documentations: [sklearn lasso cv (cross-validation)](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV), [sklearn lasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

Run lasso regression:

'''
python3 lasso_regression.py -in data/lasso_data/toy_example.csv -split 0.1
'''

Run lasso regression 5 cross validation:

'''
python3 lasso_regression.py -in data/lasso_data/toy_example.csv -cv 5
'''

With Normalization:

'''
python3 lasso_regression.py -in data/lasso_data/toy_example.csv --normalize
'''

Convert to bool value:

'''
python3 lasso_regression.py -in data/lasso_data/toy_example.csv --bool
'''






