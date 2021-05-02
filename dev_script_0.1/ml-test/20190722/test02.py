import cv2
import numpy as np
#scikit-learnからデータの取り出し
from sklearn import datasets
iris = datasets.load_iris()
 
#アヤメの分類に使用するデータの確認
print(iris.DESCR)
iris.data
iris.target
print(iris.data.shape)
print(iris.target)

npz_m1 = np.load('../dlib/-1.npz')
npz_0 = np.load('../dlib/0.npz')
npz_1 = np.load('../dlib/1.npz')

x_m1, y_m1 = npz_m1['x'], npz_m1['y']
print(len(x_m1))
x_0, y_0 = npz_0['x'], npz_0['y']
print(len(x_0))
x_1, y_1 = npz_1['x'], npz_1['y']

y_m1 = np.full((101),0,np.uint8)
y_0 = np.full((101),1,np.uint8)
y_1 = np.full((101),2,np.uint8)

x = np.vstack((x_m1, x_0)) 
x = np.vstack((x, x_1)) 
y = np.hstack((y_m1, y_0, y_1)) 

print(len(x),len(y))

x = np.reshape(x,(303,136))
print(x.shape)

#アヤメの分類の学習
from sklearn.model_selection import train_test_split as split
x_train, x_test, y_train, y_test = split(iris.data,iris.target,train_size=0.8,test_size=0.2)
print(x_train.shape, y_train.shape)

x_train, x_test, y_train, y_test = split(x,y,train_size=0.8,test_size=0.2)



import tensorflow as tf
import keras
from keras.layers import Dense,Activation

#saver = tf.train.Saver()
 
#ニュートラルネットワークで使用するモデル作成
model = keras.models.Sequential()
model.add(Dense(units=32,input_dim=136))
model.add(Activation('relu'))
model.add(Dense(units=3))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
 
#教師あり学習の実行
print(x_train.shape, y_train.shape)
model.fit(x_train,y_train,epochs=100)
 
#評価の実行
score = model.evaluate(x_test,y_test,batch_size = 1)
print(score[1])

model.save('model.h5', include_optimizer=False)
#1つのデータに対する評価の実行方法
'''
import numpy as np
x = np.array([[5.1,3.5,1.4,0.2]])
r = model.predict(x)
print(r)
r.argmax()
'''