import sys
import numpy as np
from util import load_record
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

with open(sys.argv[1], "r") as f:
    training_records = [load_record(r) for r in f.readlines()]
print("Training set size: " + str(len(training_records)))
with open(sys.argv[2], "r") as f:
    testing_records = [load_record(r) for r in f.readlines()]
print("Testing set size: " + str(len(testing_records)))

training_max = max([len(r.features.keys()) for r in training_records])
testing_max = max([len(r.features.keys()) for r in testing_records])
max_features = max([training_max, testing_max])

print("Max # features: " + str(max_features))

x_train = np.zeros([len(training_records), max_features])
y_train = np.zeros([len(training_records)])
x_test = np.zeros([len(testing_records), max_features])
y_test = np.zeros([len(testing_records)])

print("Dataset instantiated")

print(x_train.shape)

for i, r in enumerate(training_records):
    features = list(r.features.keys())
    x_train[i, :len(features)] = np.array(features)
    x_train[i, 0] = r.features[1]
    y_train[i] = (r.label+1)/2

for i, r in enumerate(testing_records):
    features = list(r.features.keys())
    x_test[i, :len(features)] = np.array(features)
    x_test[i, 0] = r.features[1]
    y_test[i] = (r.label+1)/2

print("Dataset loaded")

k_reg = regularizers.l2(0.005)

model = Sequential()
model.add(Dense(1023, input_dim=max_features, activation='relu', kernel_regularizer=k_reg))
model.add(Dropout(0.5))
model.add(Dense(1023, activation='relu', kernel_regularizer=k_reg))
model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu', kernel_regularizer=k_reg))
# model.add(Dropout(0.5))
model.add(Dense(1023, activation='relu', kernel_regularizer=k_reg))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_regularizer=k_reg))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_regularizer=None))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='nadam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model compiled successfully")

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

print()
print("Score: " + str(score))

model.save_weights("mlp.model")
