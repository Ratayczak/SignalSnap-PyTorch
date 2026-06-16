from typing import Iterable, Any

from multichss.config import DataConfig

def data_config_dic(
    data_config_list: Iterable["DataConfig"]
) -> dict[Any, "DataConfig"]:
    return {config.data: config for config in data_config_list}
