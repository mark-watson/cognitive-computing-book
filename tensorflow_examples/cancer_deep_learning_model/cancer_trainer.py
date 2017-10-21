from keras.models import Sequential # Keras by default imports and uses Tensorflow
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import TensorBoard
import pandas

train = pandas.read_csv("train.csv", header=None).values
X_train = train[:,0:9].astype(float) # 9 inputs
Y_train = train[:,-1].astype(float)  # one target output (0 for no cancer, 1 for malignant)
test = pandas.read_csv("test.csv", header=None).values
X_test = test[:,0:9].astype(float)
Y_test = test[:,-1].astype(float)

model = Sequential()
model.add(Dense(12, input_dim=9, activation='relu'))
model.add(Dense(12, input_dim=12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=0.002),
              loss='mse',
              metrics=['accuracy'])

callbacks = [TensorBoard(log_dir='logdir',histogram_freq=0,write_graph=True, write_images=False)]

model.fit(X_train, Y_train, batch_size=50, epochs=10, callbacks=callbacks)

# no cancer and malignant test samples:
y_predict = model.predict([[4,1,1,3,2,1,3,1,1], [3,7,7,4,4,9,4,8,1]])

print("* y_predict (should be close to [0, 1]):", y_predict)
