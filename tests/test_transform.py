"""Tests del módulo transform.py — Lab 03."""

import pandas as pd
import pytest

from analytics.transform import (
    agregar_encoding_estado,
    agregar_features_fecha,
    agregar_ticket_historico,
)


@pytest.fixture
def df_fechas():
    """Retorna un DataFrame con fechas de compra, entrega y estimada."""
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "order_purchase_timestamp": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-09"]),
            "order_delivered_customer_date": pd.to_datetime(
                ["2021-01-10", "2021-01-20", "2021-01-11"]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2021-01-15", "2021-01-15", "2021-01-10"]
            ),
        }
    )


@pytest.fixture
def df_estados():
    """Retorna un DataFrame con estados de clientes."""
    return pd.DataFrame(
        {
            "customer_state": ["SP", "SP", "SP", "RJ", "MG"],
        }
    )


@pytest.fixture
def df_pedidos_cliente():
    """Retorna pedidos de clientes para validar ticket histórico."""
    return pd.DataFrame(
        {
            "customer_id": ["x", "x", "x", "y"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2021-01-01", "2021-02-01", "2021-03-01", "2021-01-15"]
            ),
            "precio_total": [100.0, 200.0, 300.0, 50.0],
        }
    )


def test_agrega_columnas_esperadas(df_fechas):
    """Verifica que se agreguen las columnas esperadas."""
    resultado = agregar_features_fecha(df_fechas)
    columnas_esperadas = {
        "tiempo_entrega_dias",
        "retraso_entrega_dias",
        "entregado_tarde",
        "dia_semana_compra",
        "es_fin_de_semana",
        "mes_compra",
    }
    assert columnas_esperadas.issubset(resultado.columns)


def test_tiempo_entrega_dias_correcto(df_fechas):
    """Valida el cálculo correcto del tiempo de entrega."""
    resultado = agregar_features_fecha(df_fechas)
    assert resultado.loc[0, "tiempo_entrega_dias"] == 6
    assert resultado.loc[1, "tiempo_entrega_dias"] == 15


def test_entregado_tarde_detecta_retraso(df_fechas):
    """Detecta pedidos entregados tarde."""
    resultado = agregar_features_fecha(df_fechas)
    assert resultado.loc[1, "entregado_tarde"]
    assert not resultado.loc[0, "entregado_tarde"]


def test_es_fin_de_semana_correcto(df_fechas):
    """Valida la marca de fin de semana."""
    resultado = agregar_features_fecha(df_fechas)
    assert resultado.loc[2, "es_fin_de_semana"]
    assert not resultado.loc[0, "es_fin_de_semana"]


def test_agregar_features_fecha_no_muta_original(df_fechas):
    """Verifica que la función no modifique el DataFrame original."""
    columnas_antes = list(df_fechas.columns)
    agregar_features_fecha(df_fechas)
    assert list(df_fechas.columns) == columnas_antes


def test_frecuencia_estado_mas_comun(df_estados):
    """Comprueba la frecuencia relativa del estado más común."""
    resultado = agregar_encoding_estado(df_estados)
    assert resultado.loc[0, "customer_state_freq"] == pytest.approx(0.6)


def test_frecuencia_estado_menos_comun(df_estados):
    """Comprueba la frecuencia relativa del estado menos común."""
    resultado = agregar_encoding_estado(df_estados)
    assert resultado.loc[3, "customer_state_freq"] == pytest.approx(0.2)


def test_encoding_no_muta_original(df_estados):
    """Verifica que encoding no modifica el DataFrame original."""
    columnas_antes = list(df_estados.columns)
    agregar_encoding_estado(df_estados)
    assert list(df_estados.columns) == columnas_antes


def test_primer_pedido_es_nulo(df_pedidos_cliente):
    """El primer pedido no tiene historial previo."""
    resultado = agregar_ticket_historico(df_pedidos_cliente)
    primer_pedido_x = resultado[resultado["customer_id"] == "x"].iloc[0]
    assert pd.isna(primer_pedido_x["ticket_promedio_historico"])


def test_segundo_pedido_usa_solo_el_primero(df_pedidos_cliente):
    """El segundo pedido usa solo el ticket previo."""
    resultado = agregar_ticket_historico(df_pedidos_cliente)
    segundo_pedido_x = resultado[resultado["customer_id"] == "x"].iloc[1]
    assert segundo_pedido_x["ticket_promedio_historico"] == pytest.approx(100.0)


def test_tercer_pedido_promedia_los_dos_anteriores(df_pedidos_cliente):
    """El tercer pedido promedia los dos pedidos anteriores."""
    resultado = agregar_ticket_historico(df_pedidos_cliente)
    tercer_pedido_x = resultado[resultado["customer_id"] == "x"].iloc[2]
    assert tercer_pedido_x["ticket_promedio_historico"] == pytest.approx(150.0)


def test_no_usa_pedidos_futuros_si_llegan_desordenados():
    """No usa pedidos futuros cuando el DataFrame viene desordenado."""
    df_desordenado = pd.DataFrame(
        {
            "customer_id": ["x", "x"],
            "order_purchase_timestamp": pd.to_datetime(["2021-03-01", "2021-01-01"]),
            "precio_total": [300.0, 100.0],
        }
    )
    resultado = agregar_ticket_historico(df_desordenado)
    fila_enero = resultado[resultado["order_purchase_timestamp"] == "2021-01-01"].iloc[0]
    assert pd.isna(fila_enero["ticket_promedio_historico"])
