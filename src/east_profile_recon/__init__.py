from importlib import import_module


_EXPORTS = {
    "run_single_timepoint": (".core", "run_single_timepoint"),
    "main_worker": (".core", "main_worker"),
    "get_IT": (".data", "get_IT"),
    "load_QVWdata": (".data", "load_QVWdata"),
    "load_vacuum_data": (".data", "load_vacuum_data"),
    "read_QVWdata": (".data", "read_QVWdata"),
    "plot_all_bands": (".plotting", "plot_all_bands"),
    "plot_density_profile": (".plotting", "plot_density_profile"),
    "plot_iq_data": (".plotting", "plot_iq_data"),
    "plot_spectrogram": (".plotting", "plot_spectrogram"),
    "plot_strategy_comparison": (".plotting", "plot_strategy_comparison"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
