import ast
import glob
from pathlib import Path
import re
import pytest

VAULT_KEYWORDS = {"DEFAULT_MODEL_DIR", "MASTER_SHARD", "MASTER_DIR", "VAULT_DIR", "SRC_DIR", "L0_PATH", "L1_PATH", "L9_PATH"}
# Match paths pointing to the live vault (e.g. H:\AI_Memory_Vault or H:\ models)
VAULT_REGEX = re.compile(r"AI_Memory_Vault|H:\\", re.IGNORECASE)


def test_meta_all_vault_referencing_tests_marked():
    """E4: Meta-test that fails if any test references a vault path without the @pytest.mark.vault marker.
    Includes direct string literals, module-level vault constants, or skipif condition references."""
    tests_dir = Path(__file__).parent
    violations = []

    for test_file in sorted(tests_dir.glob("test_*.py")):
        if test_file.name == "test_meta_vault.py":
            continue

        tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))

        # Check for module-level variables that hold vault paths
        module_vault_vars = set(VAULT_KEYWORDS)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # If assignment value mentions vault, track variable name
                        val_str = ast.unparse(node.value)
                        if VAULT_REGEX.search(val_str):
                            module_vault_vars.add(target.id)

        # Inspect each test function
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                # Check decorators
                is_marked_vault = any(
                    (isinstance(d, ast.Name) and d.id == "vault")
                    or (isinstance(d, ast.Attribute) and d.attr == "vault")
                    or (
                        isinstance(d, ast.Call)
                        and (
                            (isinstance(d.func, ast.Attribute) and d.func.attr == "vault")
                            or (isinstance(d.func, ast.Name) and d.func.id == "vault")
                        )
                    )
                    for d in node.decorator_list
                )

                # Check body + decorators for vault references
                full_code = ast.unparse(node)
                refs_vault = bool(VAULT_REGEX.search(full_code)) or any(
                    re.search(r"\b" + re.escape(v) + r"\b", full_code) for v in module_vault_vars
                )

                if refs_vault and not is_marked_vault:
                    violations.append(f"{test_file.name}::{node.name}")

    assert not violations, (
        f"Found tests referencing vault paths without @pytest.mark.vault marker:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_meta_pyproject_toml_deselects_vault_by_default():
    """Part F (d): Meta-test that fails if pyproject.toml's addopts no longer deselects vault tests."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), f"pyproject.toml not found at {pyproject_path}"

    content = pyproject_path.read_text(encoding="utf-8")
    assert "addopts" in content, "pyproject.toml missing addopts"
    assert 'not vault' in content, (
        "pyproject.toml addopts does not deselect vault tests (-m \"not vault\"). "
        "Vault tests must remain deselected by default while H: is defragmented."
    )

