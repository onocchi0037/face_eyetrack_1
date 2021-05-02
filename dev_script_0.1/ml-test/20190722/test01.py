#各種ライブラリのImport
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
 
#決定木のモデルを描画するためのImport
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
 
#scikit-learnよりあやめのデータを抽出する
from sklearn import datasets
data = datasets.load_iris()
 
#あやめのデータの詳細
print(data.DESCR)
 
 
#あやめのデータ（説明変数）をdataXに格納する
dataX = pd.DataFrame(data=data.data,columns=data.feature_names)
dataX.head()
 
#あやめのデータ（目的変数）をdataYに格納する
dataY = pd.DataFrame(data=data.target)
dataY = dataY.rename(columns={0: 'Species'})
dataY.head()
 
#対応する名前に変換する
def name(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Veriscolour'
    else:
        return 'Virginica'
 
dataY['Species'] = dataY['Species'].apply(name)
dataY.head()
 
#データの分割を行う（訓練用データ 0.7 評価用データ 0.3）
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.3)
 
#線形モデル(決定木)として測定器を作成する
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
 
#訓練の実施
clf.fit(X_train,Y_train)
 
#決定木の描画を行う
export_graphviz(clf, out_file="tree.dot", feature_names=X_train.columns, class_names=["0","1","2"], filled=True, rounded=True)
#g = pydotplus.graph_from_dot_file(path="tree.dot")
#g.write_png('figure-decisionTree.png')
#Image(g.create_png())
 
#評価の実行
df = pd.DataFrame(clf.predict_proba(X_test))
df = df.rename(columns={0: 'Setosa',1: 'Veriscolour',2: 'Virginica'})
df.head()
 
#評価の実行（判定）
df = pd.DataFrame(clf.predict(X_test))
df = df.rename(columns={0: '判定'})
df = df.rename(columns={0: 'Setosa',1: 'Veriscolour',2: 'Virginica'})
df.head()
 
#混同行列
from sklearn.metrics import confusion_matrix
df = pd.DataFrame(confusion_matrix(Y_test,clf.predict(X_test).reshape(-1,1), labels=['Setosa','Veriscolour','Virginica']))
df = df.rename(columns={0: '予(Setosa)',1: '予(Veriscolour)',2: '予(Virginica)'}, index={0: '実(Setosa)',1: '実(Veriscolour)',2: '実(Virginica)'})
df
 
#評価の実行（正答率）
clf.score(X_test,Y_test)
 
#評価の実行（個々の詳細）
ng=0
for i,j in zip(clf.predict(X_test),Y_test.values.reshape(-1,1)):
    if i == j:
        print(i,j,"OK")
    else:
        print(i,j,"NG")
        ng += 1