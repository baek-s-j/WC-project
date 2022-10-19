# data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime as dt
from datetime import datetime

df = pd.read_csv("C:/Users/qor27/OneDrive/바탕 화면/백승진/캡스톤 디자인/Teo_LSTM/2018_r.csv")

dataset_train_actual = df.copy()

dataset_train_actual.isnull().sum()

dataset_train_actual = dataset_train_actual.fillna(dataset_train_actual.mean())

dataset_train_actual = dataset_train_actual.reset_index()

dataset_train_actual.isnull().sum()

dataset_train_actual.head()

dataset_train_actual.info()

dataset_train_timeindex = dataset_train_actual.set_index('dt')

dataset_train = dataset_train_actual.copy()

# Select features (columns) to be involved intro training and predictions

cols = list(dataset_train)[2:8]

# Extract dates (will be used in visualization)

datelist_train = list(dataset_train['dt'])

datelist_train = [date for date in datelist_train]

print('Training set shape == {}'.format(dataset_train.shape))

print('All timestamps == {}'.format(len(datelist_train)))

print('Featured selected: {}'.format(cols))

dataset_train = dataset_train[cols].astype(str)

for i in cols:

    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

dataset_train = dataset_train.astype(float)

# Using multiple features (predictors)

training_set = dataset_train.values

print('Shape of training set == {}.'.format(training_set.shape))

training_set

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

training_set_scaled = sc.fit_transform(training_set)

sc_predict = StandardScaler()

sc_predict.fit_transform(training_set[:, 0:1])

# Creating a data structure with 72 timestamps and 1 output

X_train = []

y_train = []

n_future = 30  # Number of days we want top predict into the future

n_past = 72  # Number of past days we want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1]])

    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X_train.shape))

print('y_train shape == {}.'.format(y_train.shape))

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv1D(filters=32, kernel_size=3,

                           strides=1, padding="causal",

                           activation="relu",

                           input_shape=[None, 7]),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),

    # tf.keras.layers.LSTM(128),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),

    # tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(1),

    # tf.keras.layers.Dense(3,kernel_initializer=tf.initializers.zeros),

    tf.keras.layers.Lambda(lambda x: x * 200),

    # tf.keras.layers.Reshape([24, 3])

])

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(

#     lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mse"])

# %%time

# es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)

# rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)


# tb = TensorBoard('logs')

model.summary()

history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Generate list of sequence of days for predictions

datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

'''

Remeber, we have datelist_train from begining.

'''

# Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE

datelist_future_ = []

for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())

datelist_train[-1]

datelist_future_

# Perform predictions

predictions_future = model.predict(X_train[-n_future:])

predictions_train = model.predict(X_train[n_past:])


# Inverse the predictions to original measurements


# ---> Special function: convert <datetime.date> to <Timestamp>

def datetime_to_timestamp(x):
    '''

        x : a given datetime value (datetime.date)

    '''

    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


y_pred_future = sc_predict.inverse_transform(predictions_future)

y_pred_train = sc_predict.inverse_transform(predictions_train)

PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Global_active_power']).set_index(pd.Series(datelist_future))

PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Global_active_power']).set_index(
    pd.Series(datelist_train[2 * n_past + n_future - 1:]))

# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN

# PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)


PREDICTIONS_FUTURE

PREDICTION_TRAIN

# Set plot size 

# from pylab import rcParams

plt.rcParams['figure.figsize'] = 14, 5

# Plot parameters

START_DATE_FOR_PLOTTING = '2009-06-07'

plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Global_active_power'], color='r',
         label='Predicted Global Active power')

plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index,
         PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Global_active_power'], color='orange',
         label='Training predictions')

plt.plot(dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:].index,
         dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:]['Global_active_power'], color='b',
         label='Actual Global Active power')

plt.axvline(x=min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)

plt.title('Predcitions and Acutal Global Active power values', family='Arial', fontsize=12)

plt.xlabel('Timeline', family='Arial', fontsize=10)

plt.ylabel('Stock Price Value', family='Arial', fontsize=10)

# plt.xticks(rotation=45, fontsize=8)

# plt.show()


import matplotlib.pyplot as plt

plt.semilogx(history.history["lr"], history.history["loss"])

plt.axis([1e-8, 1e-4, 0, 30])
