# Sklearn训练的模型怎么保存

```python
from sklearn.externals import joblib
from sklearn import svm

X = [[0,0,0,0,0],
     [1,1,0,1,0],
     [0,1,0,1,0],
     [1,0,2,3,6],
     [3,2,1,3,5]]

y = [0,1,2,3,2]

f = svm.SVC()
f.fit(X,y)

# save mode
model = joblib.dump(f,'train_model.m')

# load mode
train_model = joblib.load('train_model.m')

# predict
train_model.predict(X)
```

