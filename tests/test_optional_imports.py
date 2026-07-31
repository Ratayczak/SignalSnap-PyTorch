import subprocess
import sys
import textwrap


def test_array_only_usage_does_not_import_h5py():
    script = textwrap.dedent(
        """
        import builtins

        import numpy as np

        real_import = builtins.__import__

        def import_without_h5py(name, *args, **kwargs):
            if name == "h5py" or name.startswith("h5py."):
                raise ModuleNotFoundError("No module named 'h5py'", name="h5py")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = import_without_h5py

        from signalsnap_pytorch import DataConfig, SampledChannel
        from signalsnap_pytorch._core.data_access import open_channels

        config = DataConfig(channels=[SampledChannel(data=np.arange(8), dt=1.0)])

        with open_channels(config, [0]) as channels:
            np.testing.assert_array_equal(channels[0], np.arange(8))
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_core_usage_does_not_import_matplotlib():
    script = textwrap.dedent(
        """
        import builtins
        import importlib

        import numpy as np

        real_import = builtins.__import__

        def import_without_matplotlib(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = import_without_matplotlib

        from signalsnap_pytorch import DataConfig, SampledChannel
        from signalsnap_pytorch._core.data_access import open_channels

        config = DataConfig(channels=[SampledChannel(data=np.arange(8), dt=1.0)])

        with open_channels(config, [0]) as channels:
            np.testing.assert_array_equal(channels[0], np.arange(8))

        try:
            importlib.import_module("signalsnap_pytorch.plotting")
        except ModuleNotFoundError as exc:
            assert 'pip install "signalsnap-pytorch[plotting]"' in str(exc)
        else:
            raise AssertionError(
                "Importing signalsnap_pytorch.plotting should require matplotlib"
            )
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
