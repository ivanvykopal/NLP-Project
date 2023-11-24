import pandas as pd


class Dataset:
    def __init__(self, path) -> None:
        self.data = None
        self.load_data(path)

    def load_data(self, path: str) -> None:
        raise NotImplementedError

    def get_data(self) -> pd.DataFrame:
        return self.data
