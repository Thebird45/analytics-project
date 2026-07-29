"""Tests de verificacion del entorno y la estructura del repositorio."""

import importlib.util
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_dependencias_principales():
    """Las dependencias declaradas en pyproject.toml deben estar instaladas."""
    for paquete in ("pandas", "numpy", "pytest"):
        assert importlib.util.find_spec(paquete) is not None, f"falta el paquete {paquete}"


def test_estructura_del_proyecto():
    """El repositorio debe seguir el src layout esperado por el curso."""
    rutas_esperadas = [
        RAIZ / "pyproject.toml",
        RAIZ / "src" / "analytics" / "__init__.py",
        RAIZ / "tests",
        RAIZ / ".github" / "workflows" / "ci.yml",
    ]
    for ruta in rutas_esperadas:
        assert ruta.exists(), f"no existe {ruta}"


def test_paquete_importable():
    """El paquete analytics debe poder importarse desde el entorno activo."""
    import analytics

    assert analytics.__version__
