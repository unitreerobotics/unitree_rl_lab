"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `unitree_rl_lab` extension. They start
with `Unitree` in their name.
"""

"""Launch Isaac Sim Simulator first."""


import importlib
import pathlib
import pkgutil
import sys


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror=None,
):
    """Yields ModuleInfo for all modules recursively on path, or, if path is None, all accessible modules.

    Note:
        This function is a modified version of the original ``pkgutil.walk_packages`` function. Please refer to the original
        ``pkgutil.walk_packages`` function for more details.
    """

    def seen(p, m={}):
        if p in m:
            return True
        m[p] = True  # noqa: R503

    for info in pkgutil.iter_modules(path, prefix):

        # yield the module info
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)


def import_packages():
    """Import all packages in the unitree_rl_lab module."""
    # get the path to the unitree_rl_lab module
    unitree_rl_lab_path = pathlib.Path(__file__).parent.parent / "unitree_rl_lab"
    # import all packages
    for info in _walk_packages([str(unitree_rl_lab_path)], "unitree_rl_lab."):
        importlib.import_module(info.name)


if __name__ == "__main__":
    # import packages
    import_packages()
    # print all environments
    for task_spec in gym.registry.values():
        if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
            print(f"Task: {task_spec.id}")
            print(f"\tEntry point: {task_spec.entry_point}")
            print(f"\tConfig: {task_spec.kwargs.get('cfg_entry_point', 'N/A')}")
            print()