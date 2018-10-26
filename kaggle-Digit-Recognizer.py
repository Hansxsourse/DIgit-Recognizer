%time
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from keras.layers import MaxPool2D, Flatten, Dropout, ZeroPadding2D, BatchNormalization
from keras.utils import np_utils
import keras
from keras.models import save_model, load_model
from keras.models import Model

#1 加载数据集，对数据集进行处理，把输入和结果分开
df = pd.read_csv('all.csv')
data = df.as_matrix() #data = np.array(df)

df = None

roundnum=5
for x in range(roundnum):
    #打乱顺序
    np.random.shuffle(data)

    x_train = data[:,1:]

    #把训练的图片数据转化成28*28的图片
    x_train = x_train.reshape(data.shape[0],28,28,1).astype('float32')
    x_train = x_train/255
    #把训练的图片进行one-hot编码
    y_train = np_utils.to_categorical(data[:,0],10).astype('float32')

    #2 设相关参数
    #设置对训练集的批次大小
    batch_size = 1024
    #设置卷积滤镜个数
    n_filter = 32
    #设置最大池化，池化核大小
    pool_size = (2,2)

    #3 定义网络，安照zeropadding,巻积层，规范层，池化层进行设置
    #这里用了relu激活函数
    cnn_net = Sequential()

    cnn_net.add(Conv2D(32, kernel_size = (3,3), strides = (1,1),input_shape = (28,28,1)))
    cnn_net.add(Activation('relu'))
    cnn_net.add(BatchNormalization(epsilon = 1e-6, axis = 1))
    cnn_net.add(MaxPool2D(pool_size = pool_size))

    cnn_net.add(ZeroPadding2D((1,1)))
    cnn_net.add(Conv2D(48, kernel_size = (3,3)))
    cnn_net.add(Activation('relu'))
    cnn_net.add(BatchNormalization(epsilon = 1e-6, axis = 1))
    cnn_net.add(MaxPool2D(pool_size = pool_size))

    cnn_net.add(ZeroPadding2D((1,1)))
    cnn_net.add(Conv2D(64, kernel_size = (2,2)))
    cnn_net.add(Activation('relu'))
    cnn_net.add(BatchNormalization(epsilon = 1e-6, axis = 1))
    cnn_net.add(MaxPool2D(pool_size = pool_size))

    cnn_net.add(Dropout(0.25))
    cnn_net.add(Flatten())

    cnn_net.add(Dense(3168))
    cnn_net.add(Activation('relu'))

    cnn_net.add(Dense(10))
    cnn_net.add(Activation('softmax'))

    #4 查看网络结构
    cnn_net.summary()

    #from keras.utils.vis_utils import plot_model, model_to_dot
    #from Ipython.display import Image, SVG
    ##可视化模型
    #SVG(model_to_dot(cnn_net).create(prog='dot',format='svg'))

    #5 训练模型，保存模型
    cnn_net.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print('开始训练')
    cnn_net.fit(x_train,y_train,batch_size=batch_size,epochs=100,verbose=1,validation_split=0.1)
    print('训练结束')
    cnn_net.save('model/cnn_net_model3.h5')
    %time

#训练用1-5，预测用67(分别注释掉不用的部分即可)

#6 加载模型
    cnn_net = load_model('model/cnn_net_model3.h5')

#7 生成提交预测结果
    df = pd.read_csv('test.csv')
    x_valid = df.values.astype('float32')
    n_valid = x_valid.shape[0]
    x_valid = x_valid.reshape(n_valid,28,28,1)
    x_valid = x_valid/255

    y_pred = cnn_net.predict_classes(x_valid,batch_size=32,verbose=1)
    np.savetxt('model/DeepConvNN2.csv',np.c_[range(1,len(y_pred)+1),y_pred],delimiter=','\
              ,header='ImageId,Label',comments='',fmt='%d')
    print('完成')
break
