import tensorflow as tf
from utils import WindowGenerator
from typing import Literal

class SingleShotForecaster:
    """A model to forecast pepper recalls.

    This model follows a single-shot method, i.e. Make the predictions all at once.
    It consists of a simple lstm layer. The model automatically normalizes the input using a batch normalizer.
    For further information check the example at: https://www.tensorflow.org/tutorials/structured_data/time_series
    """

    def __init__(self, number_of_lstm_units: int, prediction_length: int) -> None:
        """Setup the model.

        Args:
            number_of_lstm_units (int): The number of units of the LSTM.
            prediction_length (int): The number of values to predict in the future.
        """
        self.__number_of_lstm_units = number_of_lstm_units
        self.__prediction_length = prediction_length
        self.__model = tf.keras.Sequential(
            [
                # Shape [batch, time, features] => [batch, lstm_units].
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LSTM(
                    self.__number_of_lstm_units, return_sequences=False
                ),
                # Shape => [batch, out_steps*features].
                tf.keras.layers.Dense(
                    self.__prediction_length, kernel_initializer=tf.initializers.zeros()
                ),
                # Shape => [batch, out_steps, features].
                tf.keras.layers.Reshape([self.__prediction_length, 1]),
            ]
        )

    def compile(self, learning_rate: int) -> None:
        """Configure the module.
        
        Args:
            learing_rate (int): The optimization learning rate
        """
        self.__model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

    def fit(
        self, window: WindowGenerator, epochs: int, epochs_to_stop: int = 5
    ) -> tf.keras.callbacks.History:
        """Fit the model to the given window of data.

        Args:
            window (WindowGenerator): The window generator object that contains the train dataset.
            epochs (int): The number of epochs to train the model.
            epochs_to_stop (int): The number of epochs to run before stop earlier, if possible.

        Returns:
            tf.kreas.callbacks.History: The training history.
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=epochs_to_stop, mode="min"
        )
        history = self.__model.fit(
            window.train,
            epochs=epochs,
            validation_data=window.val,
            callbacks=[early_stopping],
        )

        return history

    def predict(self, window: WindowGenerator):
        pass

    def evaluate(self, window: WindowGenerator, split: Literal['train', 'val', 'test']) -> list[float]:
        if split == 'train':
            return self.__model.evaluate(window.train)
        elif split == 'val':
            return self.__model.evaluate(window.val)

        return self.__model.evaluate(window.test)

    @property
    def raw_model(self) -> tf.keras.Model:
        """Access the internal tensorflow model.

        Returns:
            tf.keras.Model: The tensorflow model.
        """
        return self.__model
