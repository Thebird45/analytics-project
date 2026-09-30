"""Demo en vivo del Lab 04: entrena, evalua y compara contra baseline con datos reales.

Requiere los 9 CSV de Olist en data/raw/ (no se versionan, ver README).
Uso:
    uv run python scripts/verificacion_manual_lab04.py
"""

from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from analytics import evaluate, extract, train, transform


def construir_dataset_regresion(tablas: dict) -> tuple:
    """Arma X, y para el modelo de tiempo_entrega_dias."""
    df = extract.construir_dataset_base(tablas)
    df = transform.agregar_features_fecha(df)
    df = transform.agregar_encoding_estado(df)
    df = transform.agregar_ticket_historico(df)

    flete_agg = tablas["items"].groupby("order_id", as_index=False).agg(
        flete_total=("freight_value", "sum")
    )
    df = df.merge(flete_agg, on="order_id", how="left")

    features = [
        "n_items",
        "precio_total",
        "flete_total",
        "customer_state_freq",
        "dia_semana_compra",
        "mes_compra",
        "ticket_promedio_historico",
    ]
    y = df["tiempo_entrega_dias"]
    mask = y.notna()
    return df.loc[mask, features], y[mask]


def construir_dataset_clasificacion(tablas: dict) -> tuple:
    """Arma X, y para el modelo de review_negativa (1-2 estrellas)."""
    df = extract.construir_dataset_base(tablas)
    df = transform.agregar_features_fecha(df)
    df = transform.agregar_encoding_estado(df)
    df = transform.agregar_ticket_historico(df)

    flete_agg = tablas["items"].groupby("order_id", as_index=False).agg(
        flete_total=("freight_value", "sum")
    )
    df = df.merge(flete_agg, on="order_id", how="left")

    # Un pedido puede tener mas de una resena: se conserva la mas reciente por order_id.
    reviews_dedup = (
        tablas["reviews"]
        .sort_values("review_creation_date")
        .drop_duplicates("order_id", keep="last")[["order_id", "review_score"]]
    )
    df = df.merge(reviews_dedup, on="order_id", how="inner")
    df["review_negativa"] = df["review_score"] <= 2

    features = [
        "n_items",
        "precio_total",
        "flete_total",
        "customer_state_freq",
        "tiempo_entrega_dias",
        "entregado_tarde",
        "ticket_promedio_historico",
    ]
    return df[features], df["review_negativa"]


def demo_regresion(tablas: dict) -> None:
    """Entrena, evalua (val + test) y serializa el pipeline de regresión."""
    print("\n" + "=" * 60)
    print("REGRESION — tiempo_entrega_dias")
    print("=" * 60)

    X, y = construir_dataset_regresion(tablas)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    pipeline = train.entrenar_pipeline_regresion(X_train, y_train)
    metricas_val = evaluate.evaluar_regresion(y_val, pipeline.predict(X_val))
    metricas_test = evaluate.evaluar_regresion(y_test, pipeline.predict(X_test))

    print(f"Filas de entrenamiento: {len(X_train):,}")
    print(f"Metricas validation: {metricas_val}")
    print(f"Metricas test:       {metricas_test}")

    train.guardar_pipeline(pipeline, "models/pipeline_tiempo_entrega.joblib")
    print("Pipeline guardado en models/pipeline_tiempo_entrega.joblib")


def demo_clasificacion(tablas: dict) -> None:
    """Entrena, evalua y compara contra baseline el pipeline de clasificación."""
    print("\n" + "=" * 60)
    print("CLASIFICACION — review_negativa")
    print("=" * 60)

    X, y = construir_dataset_clasificacion(tablas)
    print(f"Distribucion de clases:\n{y.value_counts(normalize=True)}")

    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = train.entrenar_pipeline_clasificacion(X_train, y_train)
    y_pred = pipeline.predict(X_eval)
    y_proba = pipeline.predict_proba(X_eval)[:, 1]

    metricas = evaluate.evaluar_clasificacion(y_eval, y_pred, y_proba)
    comparacion = evaluate.comparar_con_baseline(y_train, y_eval, y_pred)

    print(f"Metricas del modelo: {metricas}")
    print(f"Comparacion vs. baseline: {comparacion}")

    # Pregunta 9.3: efecto de class_weight="balanced" sobre el recall
    pipeline_sin_balanceo = train.entrenar_pipeline_clasificacion(
        X_train, y_train
    )
    pipeline_sin_balanceo.named_steps["modelo"].set_params(class_weight=None)
    pipeline_sin_balanceo.fit(X_train, y_train)
    recall_con_balanceo = recall_score(y_eval, y_pred)
    recall_sin_balanceo = recall_score(y_eval, pipeline_sin_balanceo.predict(X_eval))
    print(f"Recall review_negativa con class_weight='balanced': {recall_con_balanceo:.3f}")
    print(f"Recall review_negativa con class_weight=None:       {recall_sin_balanceo:.3f}")

    train.guardar_pipeline(pipeline, "models/pipeline_review_negativa.joblib")
    print("Pipeline guardado en models/pipeline_review_negativa.joblib")


if __name__ == "__main__":
    tablas = extract.cargar_olist("data/raw")
    demo_regresion(tablas)
    demo_clasificacion(tablas)
