from pathlib import Path
from typing import Union

from pyhocon import ConfigTree
import pandas as pd


class DataRepository:
    SUPPORTED_FORMATS = ["csv", "parquet"]

    def __init__(self, config: ConfigTree, base_data_path: Union[str, Path]):
        self._config = config
        self._base_data_path = Path(base_data_path)
    
    @property
    def config(self) -> ConfigTree:
        return self._config
    
    @property
    def data_path(self) -> Path:
        return self._base_data_path
    
    def load(self, key: str) -> pd.DataFrame:
        resource = self._config.get(f"data.{key}", None)

        assert resource is not None, \
            f"Couldn't find resource with key 'data.{key}' in the config file"
        
        path = self.data_path / resource.path
        file_format = resource.get("format", path.suffix.lstrip(".")).lower()
        read_options = resource.get("options", dict())

        assert file_format in self.SUPPORTED_FORMATS, \
            f"Can't handle format {file_format}"
        
        reader = pd.read_csv if file_format == "csv" else pd.read_parquet
        return reader(path, **read_options)
    
    def save(self, key: str, df: pd.DataFrame, save_index=False) -> bool:
        resource = self._config.get(f"data.{key}", None)

        assert resource is not None, \
            f"Couldn't find resource with key 'data.{key}' in the config file"
        
        path = self.data_path / resource.path
        file_format = resource.get("format", path.suffix.lstrip(".")).lower()
        write_options = resource.get("options", dict())

        assert file_format in self.SUPPORTED_FORMATS, \
            f"Can't handle format {file_format}"
        
        writer = df.to_csv if file_format == "csv" else df.to_parquet
        writer(path, index=save_index, **write_options)

        return path.exists()
