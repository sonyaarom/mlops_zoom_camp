if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd
@data_loader


def load_data(*args, **kwargs):
    # define the path to the Parquet file
    parquet_file_path = '/home/src/mlops/homework_03/data/yellow_tripdata_2023-03.parquet'

   # read the file
    df = pd.read_parquet(parquet_file_path)

    # return dataframe
    return df



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'