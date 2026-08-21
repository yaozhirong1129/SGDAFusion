import os
from dataloader.dataloader import DataLoaderTrain, DataLoaderTest

def get_training_data(rgb_dir, csv_dir):
    assert os.path.exists(rgb_dir) and os.path.exists(csv_dir)
    return DataLoaderTrain(rgb_dir, csv_dir)


def get_test_data(rgb_dir, csv_dir):
    assert os.path.exists(rgb_dir) and os.path.exists(csv_dir)
    return DataLoaderTest(rgb_dir, csv_dir)