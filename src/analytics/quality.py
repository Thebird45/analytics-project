"""Modulo de calidad de datos: contratos pandera y reporte de trazabilidad."""

import pandas as pd
import pandera.pandas as pa

SCHEMA_ORDERS = pa.DataFrameSchema(
    columns={
        "order_id": pa.Column(
            str,
            pa.Check.str_length(32, 32),
            nullable=False,
            unique=True,
        ),
        "order_status": pa.Column(
            str,
            pa.Check.isin(
                [
                    "delivered",
                    "shipped",
                    "canceled",
                    "unavailable",
                    "invoiced",
                    "processing",
                    "created",
                    "approved",
                ]
            ),
        ),
        "order_purchase_timestamp": pa.Column(pa.DateTime, nullable=False),
    },
    coerce=True,
)
SCHEMA_ITEMS = pa.DataFrameSchema(
    columns={
        "order_id": pa.Column(str, nullable=False),
        "order_item_id": pa.Column(int, pa.Check.greater_than_or_equal_to(1)),
        "product_id": pa.Column(str, nullable=False),
        "seller_id": pa.Column(str, nullable=False),
        "price": pa.Column(
            float,
            [pa.Check.greater_than(0), pa.Check.less_than_or_equal_to(7_000)],
            nullable=False,
        ),
        "freight_value": pa.Column(float, pa.Check.greater_than_or_equal_to(0), nullable=False),
    },
    coerce=True,
)


def validar_schema(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    nombre: str,
) -> list[str]:
    """Valida un DataFrame contra un schema y devuelve errores.

    Args:
        df: DataFrame a validar.
        schema: Schema de Pandera a aplicar.
        nombre: Nombre descriptivo del dataset.

    Returns:
        Lista de mensajes con cada error encontrado.
    """
    try:
        # lazy=True acumula TODOS los errores en vez de detenerse en el primero
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        # exc.failure_cases es un DataFrame con una fila por violación
        errores = [
            f"{nombre}: columna '{fila['column']}' — {fila['check']} "
            f"(fila índice {fila['index']}, valor: {fila['failure_case']})"
            for _, fila in exc.failure_cases.iterrows()
        ]
        return errores
    except pa.errors.SchemaError as exc:
        return [f"{nombre}: {exc}"]

    return []


def generar_reporte_calidad(
    df_entrada: pd.DataFrame,
    df_salida: pd.DataFrame,
    errores: list[str],
    nombre: str = "dataset",
) -> dict:
    """Genera un reporte resumido de calidad para un dataset.

    Args:
        df_entrada: DataFrame antes de validaciones o filtrado.
        df_salida: DataFrame después de validaciones o filtrado.
        errores: Lista de errores detectados.
        nombre: Nombre descriptivo del dataset.

    Returns:
        Diccionario con métricas y errores del reporte de calidad.
    """
    filas_entrada = len(df_entrada)
    filas_salida = len(df_salida)
    filas_descartadas = filas_entrada - filas_salida

    if filas_entrada > 0:
        tasa_rechazo = filas_descartadas / filas_entrada * 100
    else:
        tasa_rechazo = 0.0

    tasa_rechazo_str = f"{tasa_rechazo:.2f}%"

    print(
        f"[CALIDAD] {nombre}: {filas_entrada} entrada → {filas_salida} salida "
        f"({filas_descartadas} descartados, {tasa_rechazo_str})"
    )

    return {
        "nombre": nombre,
        "filas_entrada": filas_entrada,
        "filas_salida": filas_salida,
        "filas_descartadas": filas_descartadas,
        "tasa_rechazo": tasa_rechazo_str,
        "errores": errores,
    }
