from model.train import get_csvs_df, train_model
import os
import pytest
import numpy as np


def test_csvs_no_files():
    with pytest.raises(RuntimeError) as error:
        get_csvs_df("./")
    assert error.match("No CSV files found in provided data")


def test_csvs_no_files_invalid_path():
    with pytest.raises(RuntimeError) as error:
        get_csvs_df("/invalid/path/does/not/exist/")
    assert error.match("Cannot use non-existent path provided")


def test_csvs_creates_dataframe():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    datasets_directory = os.path.join(current_directory, 'datasets')
    result = get_csvs_df(datasets_directory)
    assert len(result) == 20


def test_train_model():
    xtrain = np.array([1,2,3,4,5,6]).reshape(-1,1)
    ytrain = np.array([0,0,0,1,1,1])
    reg_model = train_model(0.02,xtrain,ytrain)
    preds = reg_model.predict([[8],[2]])
    np.testing.assert_almost_equal(preds,[1,0])
