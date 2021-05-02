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

epochs=3000
#npz_m1 = np.load('../dlib/-1.npz')
load_list = ['../dlib/0.npz','../dlib/1.npz','../dlib/2.npz','../dlib/3.npz','../dlib/4.npz','../dlib/5.npz']#,'../dlib/6.npz','../dlib/7.npz','../dlib/8.npz']
#load_list = ['../dlib/3.npz','../dlib/4.npz','../dlib/5.npz']

x = []
y = []

for npz_filename in load_list:
    data = np.load(npz_filename)
    data_x, data_y = data['x'], data['y']
    data_x_reshape = []
    data_y_reshape = []
    for data in data_x:
        flatten_data = data.flatten()
        for i in range(10):
            addnoise_data = []
            for point in flatten_data:
                addnoise_point = point + np.random.random_sample()/100
                if 0 <= addnoise_point <= 1:
                    addnoise_data.append(addnoise_point)
                else:
                    addnoise_data.append(point)
            if i%2 == 0:
                x_inv_data = []
                for j, point in enumerate(addnoise_data):
                    if j%2 == 0:
                        point = (point*(-1)) + 1
                    x_inv_data.append(point)
                data_x_reshape.append(x_inv_data)    
            else:
                data_x_reshape.append(addnoise_data)
                
                    
    for data in data_y:
        for i in range(10):
            data_y_reshape.append(data - 0)
    
    x.extend(data_x_reshape)
    y.extend(data_y_reshape)


'''
x_m1, y_m1 = npz_m1['x'], npz_m1['y']
print(len(x_m1))
x_0, y_0 = npz_0['x'], npz_0['y']
print(len(x_0))
x_1, y_1 = npz_1['x'], npz_1['y']

y_m1 = np.full((101),0,np.uint8)
y_0 = np.full((101),1,np.uint8)
y_1 = np.full((101),2,np.uint8)

#x = np.vstack((x_m1, x_0, x_1)) 
#x = np.vstack((x, x_1)) 
y = np.hstack((y_m1, y_0, y_1)) 
'''
print(len(x),len(y))

#x = np.reshape(x,(303,136))
#print(x.shape)

#アヤメの分類の学習
from sklearn.model_selection import train_test_split as split
#x_train, x_test, y_train, y_test = split(iris.data,iris.target,train_size=0.8,test_size=0.2)

x_train, x_test, y_train, y_test = split(x,y,train_size=0.8,test_size=0.2)



import tensorflow as tf
import keras
from keras.layers import Dense,Activation

#saver = tf.train.Saver()
 
#ニュートラルネットワークで使用するモデル作成
model = keras.models.Sequential()
model.add(Dense(units=16,input_dim=136))
model.add(Activation('relu'))
model.add(Dense(units=6))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
 
#教師あり学習の実行
history = model.fit([x_train], [y_train], epochs=epochs)
 
#評価の実行
score = model.evaluate([x_test], [y_test], batch_size = 1)
print(score[1])

model.save('model_h.h5', include_optimizer=False)
#1つのデータに対する評価の実行方法
'''
import numpy as np
x = np.array([[5.1,3.5,1.4,0.2]])
r = model.predict(x)
print(r)
r.argmax()
'''

from matplotlib import pyplot as plt

# 精度のplot
plt.plot(history.history['acc'], marker='.', label='acc')
#plt.plot(history.history['val_acc'], marker='.', label='val_acc')
plt.title('model accuracy')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.savefig('./figure_acc.png')

# 損失のplot
plt.plot(history.history['loss'], marker='.', label='loss')
#plt.plot(history.history['val_loss'], marker='.', label='val_loss')
plt.title('model loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.savefig('./figure_loss.png')
