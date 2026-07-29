# Pipeline lab 1

Proyecto del semestre de Analitica de Datos. Pipeline construido de forma incremental
lab por lab sobre el dataset Olist.

## Estructura

```
.
├── .github/workflows/ci.yml    # CI: ruff + pytest en cada push
├── data/raw/                   # dataset Olist (no versionado)
├── src/analytics/
│   ├── __init__.py
│   └── extract.py              # ingesta de datos
├── tests/
│   ├── test_environment.py     # verificacion del entorno
│   └── test_extract.py         # tests de cargar_csv
└── pyproject.toml
```

## Entorno

```bash
uv sync --all-groups
```

## Verificacion local

```bash
uv run pytest            # 7 tests
uv run ruff check .      # lint
uv run ruff format --check .
```

## Estado del pipeline

| Lab | Modulo | Estado |
|-----|--------|--------|
| 01 | `extract.cargar_csv` | listo |
| 02 | `extract` completo + `quality.py` | pendiente |
| 03 | `transform.py` + `eda.py` | pendiente |

## Dataset

Los 9 CSV de Olist van en `data/raw/`. No se versionan, estan en `.gitignore`.
