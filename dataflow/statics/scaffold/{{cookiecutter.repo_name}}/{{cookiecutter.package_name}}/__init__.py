import importlib
import pkgutil


def _auto_import_operators():
    """
    Recursively import all modules under operators.* in this package.
    As long as a module is imported, any @OPERATOR_REGISTRY.register()
    inside it will be executed automatically.
    """
    base_pkg_name = __name__ + ".operators"

    try:
        operators_pkg = importlib.import_module(base_pkg_name)
    except ImportError:
        # If there is no operators directory, just return
        return

    # Recursively traverse all submodules / subpackages under the operators package
    for _, module_name, is_pkg in pkgutil.walk_packages(
        operators_pkg.__path__, operators_pkg.__name__ + "."
    ):
        # No need to distinguish is_pkg; if a package's __init__ contains operators,
        # they will also be imported
        importlib.import_module(module_name)


# 1. Simply importing dataflow_ext_cases will automatically import everything under operators
_auto_import_operators()

# 2. If you want to provide an explicit entry point for external use, you can also expose a main
def main():
    _auto_import_operators()


__all__ = ["main"]
