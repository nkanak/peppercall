import pandas as pd
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset.

    Convert strings of dates to actual date object and set the correct index.

    Args:
        df (pd.DataFrame): The dataset.
    
    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    df.date = pd.to_datetime(df.date)
    df = pd.DataFrame(df).set_index('date')
    return df

def read_dataset(filepath: str) -> pd.DataFrame:
    """Read the dataset as a DataFrame.

    Args:
        filepath (str): The filepath of the dataset.
    
    Returns:
        pd.DataFrame: The dataset.
    """
    df = pd.read_csv(filepath)
    return df

def resample_dataset(df: pd.DataFrame, aggregation=Literal['QS', 'MS', 'W', 'D']) -> pd.DataFrame:
    """Resample the dataset.

    Apply general aggregation functions to the timeseries data.
    After the aggregation, the function sums the values.

    Args:
        df (pd.DataFrame): The dataset.
        aggregation (str): The type of aggregation. Accepted values: 'QS', 'MS', 'W', 'D'.
    
    Returns:
        pd.DataFrame: The aggregated DataFrame
    """
    return df.resample(aggregation).sum()

def split_dataset(df: pd.DataFrame, train_percentage: float, validation_percentage: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train validation and test.

    Args:
        df (pd.DataFrame): The dataset.
        train_percentage (float): The percentage of the dataset for training.
        validation_percentage (float): The percentage of the dataset for validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the three splits (train, validation, test).
    """
    n = len(df)
    train_df = df[0:int(n*train_percentage)]
    val_df = df[int(n*train_percentage):int(n*(train_percentage + validation_percentage))]
    test_df = df[int(n*(train_percentage + validation_percentage)):]
    return train_df, val_df, test_df


class WindowGenerator():
  """A model to generate windows of data for timeseries forecasting.

    This class is an optimized version of the one provided at: https://www.tensorflow.org/tutorials/structured_data/time_series.
  """

  def __init__(self, input_width: int, label_width:int, shift:int, batch_size: int,
               train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
               label_columns: list[str]) -> None:
    # Store the raw data.
    self.__train_df = train_df
    self.__val_df = val_df
    self.__test_df = test_df
    self.__batch_size = batch_size

    # Work out the label column indices.
    self.__label_columns = label_columns
    if label_columns is not None:
      self.__label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.__column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.__input_width = input_width
    self.__label_width = label_width
    self.__shift = shift

    self.__total_window_size = input_width + shift

    self.__input_slice = slice(0, input_width)
    self.__input_indices = np.arange(self.__total_window_size)[self.__input_slice]

    self.__label_start = self.__total_window_size - self.__label_width
    self.__labels_slice = slice(self.__label_start, None)
    self.__label_indices = np.arange(self.__total_window_size)[self.__labels_slice]

    
  def split_window(self, features):
    inputs = features[:, self.__input_slice, :]
    labels = features[:, self.__labels_slice, :]
    if self.__label_columns is not None:
      labels = tf.stack(
        [labels[:, :, self.__column_indices[name]] for name in self.__label_columns],
        axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.__input_width, None])
    labels.set_shape([None, self.__label_width, None])

    return inputs, labels


  @property
  def train(self):
    return self.make_dataset(self.__train_df)

  @property
  def val(self):
    return self.make_dataset(self.__val_df)

  @property
  def test(self):
    return self.make_dataset(self.__test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result

  def plot(self, model=None, plot_col='value', max_subplots=10):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.__column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col}')
      plt.plot(self.__input_indices, inputs[n, :, plot_col_index],
               label='Inputs', marker='.', zorder=-10)

      if self.__label_columns:
        label_col_index = self.__label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.__label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.__label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time')
    plt.show()

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.__total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=self.__batch_size,)

    ds = ds.map(self.split_window)

    return ds

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.__total_window_size}',
        f'Input indices: {self.__input_indices}',
        f'Label indices: {self.__label_indices}',
        f'Label column name(s): {self.__label_columns}'])