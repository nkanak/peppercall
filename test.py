import unittest
import pandas as pd

import utils


class TestReadDataset(unittest.TestCase):
    def test_read_dataset_correct_length(self):
        df = utils.read_dataset("test_dataset.csv")
        self.assertEqual(len(df), 7242)

    def test_read_dataset_correct_type(self):
        df = utils.read_dataset("test_dataset.csv")
        self.assertIsInstance(df, pd.DataFrame)


class TestPreprocessDataset(unittest.TestCase):
    def test_correct_column_types(self):
        df = utils.read_dataset("test_dataset.csv")
        df = utils.preprocess_dataset(df)
        self.assertIsInstance(df.index[0], pd.Timestamp)


class TestResampleDataset(unittest.TestCase):
    def setUp(self) -> None:
        df = utils.read_dataset("test_dataset.csv")
        self.df = utils.preprocess_dataset(df)

    def test_resample_month(self):
        self.df = utils.resample_dataset(self.df, "MS")
        self.assertEqual(len(self.df), 239)

    def test_resample_week(self):
        self.df = utils.resample_dataset(self.df, "W")
        self.assertEqual(len(self.df), 1035)

    def test_resample_quarter(self):
        self.df = utils.resample_dataset(self.df, "QS")
        self.assertEqual(len(self.df), 80)

    def test_resample_day(self):
        self.df = utils.resample_dataset(self.df, "D")
        self.assertEqual(len(self.df), 7242)


class TestSplitDataset(unittest.TestCase):
    def test_correct_sizes(self):
        df = utils.read_dataset("test_dataset.csv")
        df = utils.preprocess_dataset(df)
        train_df, val_df, test_df = utils.split_dataset(df, 0.7, 0.2)

        self.assertEqual(len(train_df), 5069)
        self.assertEqual(len(val_df), 1448)
        self.assertEqual(len(test_df), 725)
