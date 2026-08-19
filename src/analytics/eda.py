"""Funciones para explorar y resumir columnas de un DataFrame."""

import pandas as pd


def perfil_columna(df: pd.DataFrame, columna: str) -> dict:
    """Devuelve un perfil estadístico de una columna.

    Args:
        df: DataFrame que contiene la columna a analizar.
        columna: Nombre de la columna a perfilar.

    Returns:
        Diccionario con métricas de resumen y estadísticas relevantes.
    """
    serie = df[columna]
    n_total = len(df)
    n_nulos = int(serie.isna().sum())
    pct_nulos = round((n_nulos / n_total) * 100, 2) if n_total else 0.0

    perfil: dict[str, object] = {
        "columna": columna,
        "tipo": str(serie.dtype),
        "n_total": n_total,
        "n_nulos": n_nulos,
        "pct_nulos": pct_nulos,
        "n_unicos": int(serie.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(serie):
        numerica = serie.dropna()
        perfil["media"] = round(float(numerica.mean()), 2) if not numerica.empty else None
        perfil["mediana"] = round(float(numerica.median()), 2) if not numerica.empty else None
        perfil["std"] = round(float(numerica.std(ddof=1)), 2) if len(numerica) > 1 else 0.0
        perfil["min"] = float(numerica.min()) if not numerica.empty else None
        perfil["max"] = float(numerica.max()) if not numerica.empty else None
        return perfil

    no_nulos = serie.dropna()
    if no_nulos.empty:
        perfil["valor_mas_frecuente"] = None
        perfil["freq_valor_mas_frecuente"] = 0
        return perfil

    conteos = no_nulos.value_counts()
    valor = conteos.index[0]
    perfil["valor_mas_frecuente"] = valor
    perfil["freq_valor_mas_frecuente"] = int(conteos.iloc[0])
    return perfil


def distribucion_categorica(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Devuelve conteo y porcentaje por categoría.

    Args:
        df: DataFrame que contiene la columna categórica.
        columna: Nombre de la columna a analizar.

    Returns:
        DataFrame con conteo y porcentaje por categoría, ordenado descendentemente.
    """
    conteos = df[columna].value_counts(dropna=False)
    resultado = conteos.to_frame(name="conteo")
    resultado["porcentaje"] = (resultado["conteo"] / resultado["conteo"].sum() * 100).round(2)
    return resultado.sort_values("conteo", ascending=False)


def matriz_correlacion(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Devuelve la matriz de correlación de Pearson.

    Args:
        df: DataFrame que contiene las columnas de interés.
        columnas: Lista de columnas numéricas a correlacionar.

    Returns:
        Matriz cuadrada con los coeficientes de correlación entre columnas.
    """
    correlacion = df[columnas].corr(method="pearson")
    return correlacion.reindex(columnas).copy()
