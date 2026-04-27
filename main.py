import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Cargaremos el modelo RF creado
with open("random_forest_dengue.pkl", "rb") as f:
    bundle = pickle.load(f)

modelo      = bundle["modelo"]
le_distrito = bundle["le_distrito"]
le_provincia = bundle["le_provincia"]
FEATURES    = bundle["features"]
CLASES      = bundle["nombres_clases"]



app = FastAPI(
    title="API Predicción Dengue — Lima Metropolitana",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema de entrada ──
class InputPrediccion(BaseModel):
    distrito: str
    provincia: str = "LIMA"
    semana_epidemiologica: int
    semana_sin: float
    semana_cos: float
    temp_max_c: float
    temp_min_c: float
    temp_media_c: float
    rango_termico_c: float
    humedad_pct: float
    precipitacion_total_mm: float
    precipitacion_max_dia_mm: float
    casos_lag1: float = 0.0
    casos_lag2: float = 0.0
    casos_lag3: float = 0.0
    casos_lag4: float = 0.0
    precip_lag1: float = 0.0
    precip_lag2: float = 0.0
    precip_lag3: float = 0.0
    precip_lag4: float = 0.0
    temp_lag1: float = 0.0
    temp_lag2: float = 0.0
    temp_lag3: float = 0.0
    temp_lag4: float = 0.0
    humedad_lag1: float = 0.0
    humedad_lag2: float = 0.0
    humedad_lag3: float = 0.0
    humedad_lag4: float = 0.0
    distancia_estacion_km: float = 10.0
    pct_masculino: float = 50.0
    pct_femenino: float = 50.0
    edad_media: float = 30.0
    casos_menores_1: float = 0.0
    casos_1_4: float = 0.0
    casos_5_11: float = 0.0
    casos_12_17: float = 0.0
    casos_18_29: float = 0.0
    casos_30_59: float = 0.0
    casos_60_mas: float = 0.0
    tasa_crecimiento: float = 0.0
    acumulado_4sem: float = 0.0
    indice_calor_humedad: float = 0.0
    tendencia_precip: float = 0.0

# ── Función auxiliar ──
def construir_vector(data: InputPrediccion):
    if data.distrito not in le_distrito.classes_:
        raise HTTPException(
            status_code=422,
            detail=f"Distrito '{data.distrito}' no reconocido."
        )
    dist_cod = le_distrito.transform([data.distrito])[0]
    prov_cod = le_provincia.transform([data.provincia])[0]

    vector = [
        dist_cod, prov_cod,
        data.semana_epidemiologica, data.semana_sin, data.semana_cos,
        data.temp_max_c, data.temp_min_c, data.temp_media_c, data.rango_termico_c,
        data.humedad_pct, data.precipitacion_total_mm, data.precipitacion_max_dia_mm,
        data.casos_lag1, data.casos_lag2, data.casos_lag3, data.casos_lag4,
        data.precip_lag1, data.precip_lag2, data.precip_lag3, data.precip_lag4,
        data.temp_lag1, data.temp_lag2, data.temp_lag3, data.temp_lag4,
        data.humedad_lag1, data.humedad_lag2, data.humedad_lag3, data.humedad_lag4,
        data.distancia_estacion_km,
        data.pct_masculino, data.pct_femenino, data.edad_media,
        data.casos_menores_1, data.casos_1_4, data.casos_5_11,
        data.casos_12_17, data.casos_18_29, data.casos_30_59, data.casos_60_mas,
        data.tasa_crecimiento, data.acumulado_4sem,
        data.indice_calor_humedad, data.tendencia_precip,
    ]
    return np.array(vector).reshape(1, -1)

# Endpoints
@app.get("/")
def raiz():
    return {
        "mensaje": "API Predicción Dengue — Lima Metropolitana",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/distritos")
def listar_distritos():
    return {"distritos": list(le_distrito.classes_)}

@app.post("/predecir")
def predecir(data: InputPrediccion):
    X = construir_vector(data)
    codigo   = int(modelo.predict(X)[0])
    probas   = modelo.predict_proba(X)[0]
    nivel    = CLASES[codigo]
    confianza = round(float(probas[codigo]) * 100, 2)

    return {
        "distrito":            data.distrito,
        "semana":              data.semana_epidemiologica,
        "nivel_alerta":        nivel,
        "nivel_alerta_codigo": codigo,
        "confianza_pct":       confianza,
        "probabilidades": {
            CLASES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probas)
        }
    }

@app.post("/predecir/multi-horizonte")
def predecir_multi(data: InputPrediccion, casos_actuales: float = 0.0):
    resultados = {}

    sem1 = data.model_copy(update={
        "semana_epidemiologica": min(data.semana_epidemiologica + 1, 53),
        "casos_lag1": casos_actuales,
        "casos_lag2": data.casos_lag1,
        "casos_lag3": data.casos_lag2,
        "casos_lag4": data.casos_lag3,
    })
    X1 = construir_vector(sem1)
    cod1 = int(modelo.predict(X1)[0])
    resultados["semana_1"] = {"nivel": CLASES[cod1], "codigo": cod1}

    sem2 = data.model_copy(update={
        "semana_epidemiologica": min(data.semana_epidemiologica + 2, 53),
        "casos_lag1": casos_actuales,
        "casos_lag2": casos_actuales,
        "casos_lag3": data.casos_lag1,
        "casos_lag4": data.casos_lag2,
    })
    X2 = construir_vector(sem2)
    cod2 = int(modelo.predict(X2)[0])
    resultados["semana_2"] = {"nivel": CLASES[cod2], "codigo": cod2}

    sem4 = data.model_copy(update={
        "semana_epidemiologica": min(data.semana_epidemiologica + 4, 53),
        "casos_lag1": casos_actuales,
        "casos_lag2": casos_actuales,
        "casos_lag3": casos_actuales,
        "casos_lag4": casos_actuales,
    })
    X4 = construir_vector(sem4)
    cod4 = int(modelo.predict(X4)[0])
    resultados["semana_4"] = {"nivel": CLASES[cod4], "codigo": cod4}

    return {
        "distrito":    data.distrito,
        "semana_base": data.semana_epidemiologica,
        "horizontes":  resultados,
    }