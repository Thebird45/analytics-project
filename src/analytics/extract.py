"""Modulo de ingesta de datos: carga de fuentes externas hacia el pipeline."""

from pathlib import Path
from typing import Literal, cast

import pandas as pd

TipoMerge = Literal["left", "right", "inner", "outer", "cross", "left_anti", "right_anti"]


def cargar_csv(ruta: str) -> pd.DataFrame:
    """Carga un archivo CSV y verifica que contenga al menos una fila de datos.

    Args:
        ruta: ruta al archivo CSV a cargar.

    Returns:
        DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: si el archivo no existe en la ruta indicada.
        ValueError: si el archivo existe pero no contiene filas de datos.
    """
    try:
        dataframe = cast(pd.DataFrame, pd.read_csv(ruta))  # type: ignore[call-overload]

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"El archivo '{ruta}' no existe.") from exc

    if dataframe.empty:
        raise ValueError(f"El archivo '{ruta}' no contiene filas de datos.")

    return dataframe


def cargar_olist(data_dir: str) -> dict[str, pd.DataFrame]:
    """Carga las 5 tablas principales de Olist desde data_dir.

    Registra con print() la forma (filas × columnas) de cada tabla al cargarla.
    Las columnas de fecha deben cargarse como datetime64, no como object.

    Args:
        data_dir: ruta al directorio que contiene los archivos CSV de Olist.

    Returns:
        Dict con keys 'orders', 'items', 'customers', 'payments', 'reviews',
        cada uno mapeado a su DataFrame correspondiente.

    Raises:
        FileNotFoundError: si alguno de los 5 archivos no existe en data_dir.
    """
    base_path = Path(data_dir)

    # Definimos el mapeo de cada tabla: (nombre_archivo_csv, lista_de_columnas_de_fecha)
    config_tablas = {
        "orders": (
            "olist_orders_dataset.csv",
            [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        "items": (
            "olist_order_items_dataset.csv",
            ["shipping_limit_date"],
        ),
        "customers": (
            "olist_customers_dataset.csv",
            [],  # No tiene columnas de fecha
        ),
        "payments": (
            "olist_order_payments_dataset.csv",
            [],  # No tiene columnas de fecha
        ),
        "reviews": (
            "olist_order_reviews_dataset.csv",
            [
                "review_creation_date",
                "review_answer_timestamp",
            ],
        ),
    }

    tablas: dict[str, pd.DataFrame] = {}

    for clave, (archivo, cols_fecha) in config_tablas.items():
        ruta_archivo = base_path / archivo

        try:
            if cols_fecha:
                df = cast(
                    pd.DataFrame,
                    pd.read_csv(ruta_archivo, parse_dates=cols_fecha),  # type: ignore[call-overload]
                )
            else:
                df = cast(
                    pd.DataFrame,
                    pd.read_csv(ruta_archivo),  # type: ignore[call-overload]
                )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No se encontró el archivo requerido para '{clave}' en la ruta: '{ruta_archivo}'"
            ) from exc

        # Imprimimos la forma (filas x columnas)
        filas, columnas = df.shape
        print(f"Tabla '{clave}' cargada con éxito: {filas:,} filas × {columnas} columnas.")

        tablas[clave] = df

    return tablas


def join_verificado(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    on: str | list[str],
    how: TipoMerge = "left",
    nombre: str = "join",
) -> pd.DataFrame:
    """Realiza un merge y verifica que el resultado no multiplique filas.

    Un join que multiplica filas indica que la tabla derecha tiene duplicados
    en la columna clave — error silencioso sin esta verificación.

    Args:
        df_left: DataFrame izquierdo (el que define el número de filas esperado).
        df_right: DataFrame derecho.
        on: columna(s) clave del join.
        how: tipo de join ('left', 'inner', 'outer', 'right').
        nombre: nombre descriptivo para el mensaje de error.

    Returns:
        DataFrame resultante del merge.

    Raises:
        AssertionError: si el resultado tiene más filas que df_left.
    """
    # 1. Guardamos el número de filas esperadas (el total de la tabla izquierda)
    filas_esperadas = len(df_left)

    # 2. Hacemos el merge, validando que la tabla derecha no tenga claves duplicadas
    try:
        df_merged = pd.merge(df_left, df_right, on=on, how=how, validate="many_to_one")
    except pd.errors.MergeError as exc:
        raise AssertionError(
            f"Error en el join '{nombre}': la tabla derecha contiene claves duplicadas "
            f"en la columna de unión, lo cual multiplicaría las filas del resultado."
        ) from exc

    # 3. Guardamos el número de filas tras la unión
    filas_resultantes = len(df_merged)

    # 4. Validamos que no se hayan multiplicado las filas
    if filas_resultantes > filas_esperadas:
        raise AssertionError(
            f"Error en el join '{nombre}': La tabla derecha contiene claves duplicadas. "
            f"Se esperaban como máximo {filas_esperadas:,} filas, "
            f"pero el resultado produjo {filas_resultantes:,} filas."
        )

    return df_merged


def construir_dataset_base(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combina las tablas de Olist en un único DataFrame analítico.

    Estrategia de joins:
    - orders × customers → left join on customer_id
    - + payments_agg → left join on order_id (payments debe agregarse primero)
    - + items_agg → left join on order_id (items debe agregarse primero)

    Agrega payments antes del join: total_pago (sum) y n_cuotas (max) por order_id.
    Agrega items antes del join: n_items (count) y ticket_total (sum de price) por order_id.

    Args:
        tablas: dict retornado por cargar_olist().

    Returns:
        DataFrame con una fila por pedido (99,441 filas si los datos son completos).
    """
    # 1. Extraer DataFrames del diccionario
    dataframe_orders = tablas["orders"]
    dataframe_customers = tablas["customers"]
    dataframe_payments = tablas["payments"]
    dataframe_items = tablas["items"]

    # 2. Agregar payments por order_id
    payments_agg = dataframe_payments.groupby("order_id", as_index=False).agg(
        total_pago=("payment_value", "sum"),
        n_cuotas=("payment_installments", "max"),
    )

    # 3. Agregar items por order_id
    items_agg = dataframe_items.groupby("order_id", as_index=False).agg(
        n_items=("order_item_id", "count"),
        ticket_total=("price", "sum"),
    )

    # 4. Joins progresivos usando join_verificado
    # orders x customers -> left join on customer_id
    dataframe_base = join_verificado(
        df_left=dataframe_orders,
        df_right=dataframe_customers,
        on="customer_id",
        how="left",
        nombre="orders_x_customers",
    )

    # + payments_agg -> left join on order_id
    dataframe_base = join_verificado(
        df_left=dataframe_base,
        df_right=payments_agg,
        on="order_id",
        how="left",
        nombre="base_x_payments_agg",
    )

    # + items_agg -> left join on order_id
    dataframe_base = join_verificado(
        df_left=dataframe_base,
        df_right=items_agg,
        on="order_id",
        how="left",
        nombre="base_x_items_agg",
    )

    return dataframe_base
