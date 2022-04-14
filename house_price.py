import keras.layers
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import regularizers

data=pd.read_csv('C:/Users/sicario/Desktop/randy/python projects/Housing1.csv')
print(data)
dataset=data.values
dataset[:,1]=dataset[:,1]/16200
X=dataset[:,0:5]
dataset[:,5]=dataset[:,5]/13300000
Y=dataset[:,5]

min_max_scaler=preprocessing.MinMaxScaler()
X_scale=min_max_scaler.fit_transform(X)
print(X_scale)

X_train,X_val_and_test,Y_train,Y_val_and_test=train_test_split(X_scale,Y,test_size=0.3)
X_test,X_val,Y_test,Y_val=train_test_split(X_val_and_test,Y_val_and_test,test_size=0.5)
print(X_test.shape,X_val.shape,X_train,Y_test.shape,Y_val.shape,Y_test.shape)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epochs,logs={}):
        if logs.get('acc')>=1.0:
            print('\n Model has reached 90% accuracy canceling now')
            self.model.stop_training=True


callbacks=myCallback()
model=tf.keras.Sequential([
    keras.layers.Dense(32,activation='relu',input_shape=(5,)),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1,activation='sigmoid')
])
model.summary()
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['acc'])
hist=model.fit(X_train,Y_train,batch_size=32,epochs=100,validation_data=(X_val,Y_val),callbacks=[callbacks])
model.evaluate(X_test,Y_test)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper right')
plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='lower right')
plt.show()

model_2 = tf.keras.Sequential([
    keras.layers.Dense(1000, activation='relu', input_shape=(5,)),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model_2.summary()
model_2.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])

hist_2 = model_2.fit(X_train, Y_train,batch_size=32, epochs=100,validation_data=(X_val, Y_val),callbacks=[callbacks])


plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist_2.history['acc'])
plt.plot(hist_2.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


model_3 = tf.keras.Sequential([
    keras.layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(5,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
    ])
model_3.summary()
model_3.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

hist_3 = model_3.fit(X_train, Y_train,batch_size=32, epochs=100,validation_data=(X_val, Y_val),callbacks=[callbacks])

plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()

plt.plot(hist_3.history['acc'])
plt.plot(hist_3.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


y_results = model.predict(X_test)
print(y_results)
