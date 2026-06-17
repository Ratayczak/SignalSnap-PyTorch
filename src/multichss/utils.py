from typing import Any, Iterable

import h5py

from multichss.config import DataConfig


def data_config_dic(
    data_config_list: Iterable["DataConfig"],
) -> dict[Any, "DataConfig"]:
    return {config.data: config for config in data_config_list}


def unit_conversion_freq_to_time(f_unit: str) -> str:
    mapping = {"Hz": "s", "kHz": "ms", "MHz": "us", "GHz": "ns", "THz": "ps"}

    try:
        return mapping[f_unit]
    except KeyError:
        raise ValueError(f"Unknown frequency unit: {f_unit}")


def data_to_hdf(dataconfig: DataConfig):
    with h5py.File(dataconfig.path, "w") as f:
        grp = f.create_group(name=dataconfig.group_key)
        d = grp.create_dataset(name=dataconfig.dataset, data=dataconfig.data)
        d.attrs['dt'] = dataconfig.dt
        print(f"Created hdf file at {dataconfig.path}.")