import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import utils
import forecasting

matplotlib.rcParams['figure.figsize'] = (8, 6)
matplotlib.rcParams['axes.grid'] = False

OUT_STEPS = 12
INPUT_WIDTH = 12
MAX_EPOCHS = 20
BATCH_SIZE = 10
LSTM_UNITS = 50

TRAIN_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.2

df = utils.read_dataset('dataset.csv')
df = utils.preprocess_dataset(df)
df = utils.resample_dataset(df, 'MS')


train_df, val_df, test_df = utils.split_dataset(df, TRAIN_PERCENTAGE, VALIDATION_PERCENTAGE)

multi_window = utils.WindowGenerator(input_width=INPUT_WIDTH,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS, 
                               batch_size=BATCH_SIZE,
                               label_columns=['value'],
                               train_df=train_df, val_df=val_df, test_df=test_df)

multi_window.plot()

forecaster = forecasting.SingleShotForecaster(LSTM_UNITS, OUT_STEPS)
forecaster.compile()
history = forecaster.fit(multi_window, 10)

print(type(multi_window.val))
print(forecaster.raw_model.evaluate(multi_window.val))
print(forecaster.raw_model.evaluate(multi_window.test, verbose=0))
multi_window.plot(forecaster.raw_model)