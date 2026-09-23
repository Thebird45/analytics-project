"""Funciones de transformación para preparar datos analíticos."""

import pandas as pd


def agregar_features_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas de fechas de compra y entrega.

    Args:
        df: DataFrame con las columnas de compra, entrega y fecha estimada.

    Returns:
        Copia del DataFrame con columnas derivadas de fechas.
    """
    resultado = df.copy()

    compra = pd.to_datetime(resultado["order_purchase_timestamp"], errors="coerce")
    entrega = pd.to_datetime(resultado["order_delivered_customer_date"], errors="coerce")
    estimada = pd.to_datetime(resultado["order_estimated_delivery_date"], errors="coerce")

    # tiempo que tardo en llegar el pedido
    resultado["tiempo_entrega_dias"] = ((entrega - compra) / pd.Timedelta(days=1)).astype(float)

    # tiempo que se retraso el pedido respecto a la fecha estimada
    resultado["retraso_entrega_dias"] = ((entrega - estimada) / pd.Timedelta(days=1)).astype(float)

    resultado["entregado_tarde"] = resultado["retraso_entrega_dias"] > 0  # indica si llego tarde
    resultado["dia_semana_compra"] = compra.dt.dayofweek  # Dia que se compro
    resultado["es_fin_de_semana"] = resultado["dia_semana_compra"].isin([5, 6])
    resultado["mes_compra"] = compra.dt.month

    return resultado


def agregar_encoding_estado(
    df: pd.DataFrame,
    columna: str = "customer_state",
) -> pd.DataFrame:
    """Agrega frequency encoding para una columna categórica.

    Args:
        df: DataFrame que contiene la columna a codificar.
        columna: Nombre de la columna categórica a transformar.

    Returns:
        Copia del DataFrame con una columna adicional de frecuencia relativa.
    """
    resultado = df.copy()
    # convierte los resultados de texto a % de frecuencia
    frecuencias = resultado[columna].value_counts(normalize=True)
    resultado[f"{columna}_freq"] = resultado[columna].map(frecuencias)
    return resultado


def agregar_ticket_historico(
    df: pd.DataFrame,
    columna_cliente: str = "customer_id",
    columna_valor: str = "precio_total",
    columna_fecha: str = "order_purchase_timestamp",
) -> pd.DataFrame:
    """Agrega el ticket promedio histórico por cliente.

    Args:
        df: DataFrame con cliente, fecha y valor a agregar.
        columna_cliente: Nombre de la columna del cliente.
        columna_valor: Nombre de la columna numérica a promediar.
        columna_fecha: Nombre de la columna temporal para ordenar cronológicamente.

    Returns:
        Copia del DataFrame ordenada por fecha con el promedio histórico agregado.
    """
    resultado = df.sort_values(columna_fecha).copy()
    # Separa la tabla por cada cliente y calcula el promedio historico,
    # sin incluir el valor actual
    resultado["ticket_promedio_historico"] = resultado.groupby(columna_cliente, sort=False)[
        columna_valor
    ].transform(lambda s: s.expanding().mean().shift(1))
    return resultado
