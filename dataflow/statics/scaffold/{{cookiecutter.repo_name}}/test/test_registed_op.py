import importlib
import sys
import pytest

from dataflow.utils.registry import OPERATOR_REGISTRY, PROMPT_REGISTRY


# -----------------------------
# Color utilities (ANSI escape)
# -----------------------------
class C:
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def green(msg): return f"{C.GREEN}{msg}{C.RESET}"
def cyan(msg): return f"{C.CYAN}{msg}{C.RESET}"
def yellow(msg): return f"{C.YELLOW}{msg}{C.RESET}"
def red(msg): return f"{C.RED}{msg}{C.RESET}"
def bold(msg): return f"{C.BOLD}{msg}{C.RESET}"


# ==================================================
# Single test that verifies both registries at once
# (Import extension package exactly once)
# ==================================================
@pytest.mark.cpu
def test_operator_and_prompt_registered_once():
    """Ensure importing the extension package registers new operators and prompts.
    The package is imported exactly once in this test.
    """
    # --- 1) snapshot registry contents before import ---
    op_before = set(OPERATOR_REGISTRY._obj_map.keys())
    pr_before = set(PROMPT_REGISTRY._obj_map.keys())

    print(cyan(f"[Before Import] Operator count = {len(op_before)}"))
    print(cyan(f"[Before Import] Prompt    count = {len(pr_before)}"))

    # --- 2) Import the extension package exactly once ---
    pkg_name = "{{cookiecutter.package_name}}"
    if pkg_name in sys.modules:
        # We will NOT reload — respect the requirement: import only once.
        print(yellow(f"[Warning] {pkg_name} already in sys.modules; not reloading."))
    else:
        importlib.import_module(pkg_name)
        print(cyan(f"[Import] Imported package '{pkg_name}'"))

    # --- 3) snapshot registry contents after import ---
    op_after = set(OPERATOR_REGISTRY._obj_map.keys())
    pr_after = set(PROMPT_REGISTRY._obj_map.keys())

    print(cyan(f"[After Import] Operator count = {len(op_after)}"))
    print(cyan(f"[After Import] Prompt    count = {len(pr_after)}"))

    # --- 4) compute deltas ---
    op_added = sorted(op_after - op_before)
    pr_added = sorted(pr_after - pr_before)

    print(yellow(f"[Delta] Newly registered operators ({len(op_added)}):"))
    for k in op_added:
        print(green(f"  + {k}"))

    print(yellow(f"[Delta] Newly registered prompts ({len(pr_added)}):"))
    for k in pr_added:
        print(green(f"  + {k}"))

    # --- 5) assertions ---
    # Require that both registries gained at least one new entry.
    # If the package was already loaded before the test (and thus no new items),
    # the test will fail — this enforces "import once and test after import" behavior.
    assert len(op_added) > 0, "No new operators were registered after importing the package."
    assert len(pr_added) > 0, "No new prompts were registered after importing the package."

    print(green(bold("[PASS] Both operator and prompt registry tests passed!")))


# Allow running the test file directly (useful during local debugging)
if __name__ == "__main__":
    test_operator_and_prompt_registered_once()
