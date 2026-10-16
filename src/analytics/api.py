"""API REST para servir los modelos del semestre."""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from analytics import train
from analytics.pipeline import (
    FEATURES_CLASIFICACION,
    FEATURES_REGRESION,
    NOMBRE_MODELO_ENTREGA,
    NOMBRE_MODELO_RESENA,
)


class PedidoCompra(BaseModel):
    """Datos de un pedido en el momento de la compra."""

    n_items: int = Field(ge=1)
    precio_total: float = Field(gt=0)
    flete_total: float = Field(ge=0)
    customer_state_freq: float = Field(ge=0, le=1)
    dia_semana_compra: int = Field(ge=0, le=6)
    mes_compra: int = Field(ge=1, le=12)
    ticket_promedio_historico: float | None = None


class PedidoEntregado(BaseModel):
    """Datos de un pedido ya entregado, antes de que el cliente escriba la reseña."""

    n_items: int = Field(ge=1)
    precio_total: float = Field(gt=0)
    flete_total: float = Field(ge=0)
    customer_state_freq: float = Field(ge=0, le=1)
    tiempo_entrega_dias: float = Field(ge=0)
    entregado_tarde: bool
    ticket_promedio_historico: float | None = None


def _a_dataframe(pedido: BaseModel, features: list[str]) -> pd.DataFrame:
    """Convierte un pedido validado en un DataFrame de una fila con las columnas en orden.

    Args:
        pedido: modelo pydantic ya validado.
        features: columnas en el orden exacto con el que se entrenó el pipeline.

    Returns:
        DataFrame de una fila; un ticket_promedio_historico nulo queda como NaN
        y lo imputa el SimpleImputer del pipeline.
    """
    return pd.DataFrame([pedido.model_dump()], columns=features).astype(float)


def crear_app(models_dir: str = "models") -> FastAPI:
    """Crea la aplicación FastAPI con los modelos cargados desde models_dir.

    Los modelos se cargan una sola vez, al crear la app — no en cada petición.

    Endpoints:
        GET /salud             → {"estado": "ok", "fecha_entrenamiento": str}
        POST /predecir/entrega → {"tiempo_entrega_dias": float}
        POST /predecir/resena  → {"probabilidad_resena_negativa": float,
                                  "resena_negativa": bool}

    Args:
        models_dir: directorio con los dos .joblib y metricas.json que
            produce pipeline.guardar_artefactos.

    Returns:
        Aplicación FastAPI lista para servir.

    Raises:
        FileNotFoundError: si falta alguno de los tres archivos.
    """
    directorio = Path(models_dir)
    ruta_entrega = directorio / f"{NOMBRE_MODELO_ENTREGA}.joblib"
    ruta_resena = directorio / f"{NOMBRE_MODELO_RESENA}.joblib"
    ruta_metricas = directorio / "metricas.json"

    for ruta in (ruta_entrega, ruta_resena, ruta_metricas):
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró '{ruta}'. Ejecuta primero: uv run python -m analytics.pipeline"
            )

    modelo_entrega = train.cargar_pipeline(str(ruta_entrega))
    modelo_resena = train.cargar_pipeline(str(ruta_resena))
    registro = json.loads(ruta_metricas.read_text(encoding="utf-8"))

    app = FastAPI(
        title="API Olist — Analítica de Datos",
        description="Predicción de tiempo de entrega y de reseña negativa.",
    )

    @app.get("/salud")
    def salud() -> dict:
        """Indica que la API está activa y cuándo se entrenaron los modelos."""
        return {"estado": "ok", "fecha_entrenamiento": registro["fecha_entrenamiento"]}

    @app.post("/predecir/entrega")
    def predecir_entrega(pedido: PedidoCompra) -> dict:
        """Predice los días que tardará en llegar un pedido recién comprado."""
        X = _a_dataframe(pedido, FEATURES_REGRESION)
        return {"tiempo_entrega_dias": float(modelo_entrega.predict(X)[0])}

    @app.post("/predecir/resena")
    def predecir_resena(pedido: PedidoEntregado) -> dict:
        """Predice la probabilidad de que un pedido entregado reciba una reseña negativa."""
        X = _a_dataframe(pedido, FEATURES_CLASIFICACION)
        # columna 1 de predict_proba = clase True (reseña negativa)
        probabilidad = float(modelo_resena.predict_proba(X)[0, 1])
        return {
            "probabilidad_resena_negativa": probabilidad,
            "resena_negativa": probabilidad >= 0.5,
        }

    return app
