# Logistic_Regression_GD
 Python implemention of Logistic Regression using gradient descent

### How to use it

1. require **numpy, pandas, matplotlib**
2. type "python ./logistic_regression_GD.py" in terminal

----

**if you want to run this model with your data, make sure your dataset:**

1. *.csv
2. the first **n-1** columns are attributes, the last is label
3. label should be [1, 2, 3, ....]

**else**

​	remove line 186-188

```python
`   Y = train_label.copy()
    Y[Y != (i + 1)] = 0
    Y[Y == (i + 1)] = 1`
```

​	and line 198-200

```python
`   Y = train_label.copy()
    Y[Y != (i + 1)] = 0
    Y[Y == (i + 1)] = 1`
```



**and process dataset yourself**

---

**Dataset source: https://archive.ics.uci.edu/ml/datasets/seeds**
