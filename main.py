import pickle
import numpy as np
import os
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

import csv
import io
from fastapi import UploadFile, File

# ── Cargar modelo ──
with open("random_forest_dengue.pkl", "rb") as f:
    bundle = pickle.load(f)

modelo       = bundle["modelo"]
le_distrito  = bundle["le_distrito"]
le_provincia = bundle["le_provincia"]
FEATURES     = bundle["features"]
CLASES       = bundle["nombres_clases"]

# ── Conexión Supabase ──
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── App FastAPI ──
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

# ── Schemas ──
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
    usuario_id: int = None
    distrito_id: int = None

class InputEscenario(BaseModel):
    usuario_id: int
    distrito_id: int
    semana_epidemiologica: int
    año: int
    delta_temperatura: float = 0.0
    delta_precipitacion: float = 0.0
    delta_humedad: float = 0.0
    casos_lag1: float = 0.0
    base: InputPrediccion

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

# ── Endpoints ──
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
    codigo    = int(modelo.predict(X)[0])
    probas    = modelo.predict_proba(X)[0]
    nivel     = CLASES[codigo]
    confianza = round(float(probas[codigo]) * 100, 2)

    # Guardar predicción en Supabase
    try:
        prediccion_data = {
            "distrito_id":          data.distrito_id,
            "usuario_id":           data.usuario_id,
            "semana_epidemiologica": data.semana_epidemiologica,
            "año":                  2024,
            "horizonte":            1,
            "nivel_alerta":         nivel,
            "nivel_alerta_codigo":  codigo,
            "confianza_pct":        confianza,
            "prob_bajo":            round(float(probas[0]) * 100, 2),
            "prob_moderado":        round(float(probas[1]) * 100, 2),
            "prob_alto":            round(float(probas[2]) * 100, 2),
            "prob_critico":         round(float(probas[3]) * 100, 2),
        }
        result = supabase.table("predicciones").insert(prediccion_data).execute()
        prediccion_id = result.data[0]["id"] if result.data else None

        # Generar alerta automática si es Alto o Crítico
        if codigo >= 2 and prediccion_id:
            alerta_data = {
                "prediccion_id": prediccion_id,
                "nivel":         nivel,
                "estado":        "activa",
                "observaciones": f"Alerta generada automáticamente — {nivel} en semana {data.semana_epidemiologica}",
            }
            supabase.table("alertas").insert(alerta_data).execute()

    except Exception as e:
        print(f"Error guardando en Supabase: {e}")

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
    X1   = construir_vector(sem1)
    cod1 = int(modelo.predict(X1)[0])
    resultados["semana_1"] = {"nivel": CLASES[cod1], "codigo": cod1}

    sem2 = data.model_copy(update={
        "semana_epidemiologica": min(data.semana_epidemiologica + 2, 53),
        "casos_lag1": casos_actuales,
        "casos_lag2": casos_actuales,
        "casos_lag3": data.casos_lag1,
        "casos_lag4": data.casos_lag2,
    })
    X2   = construir_vector(sem2)
    cod2 = int(modelo.predict(X2)[0])
    resultados["semana_2"] = {"nivel": CLASES[cod2], "codigo": cod2}

    sem4 = data.model_copy(update={
        "semana_epidemiologica": min(data.semana_epidemiologica + 4, 53),
        "casos_lag1": casos_actuales,
        "casos_lag2": casos_actuales,
        "casos_lag3": casos_actuales,
        "casos_lag4": casos_actuales,
    })
    X4   = construir_vector(sem4)
    cod4 = int(modelo.predict(X4)[0])
    resultados["semana_4"] = {"nivel": CLASES[cod4], "codigo": cod4}

    # Guardar los 3 horizontes en Supabase
    try:
        for horizonte, cod in [(1, cod1), (2, cod2), (4, cod4)]:
            supabase.table("predicciones").insert({
                "distrito_id":           data.distrito_id,
                "usuario_id":            data.usuario_id,
                "semana_epidemiologica":  data.semana_epidemiologica,
                "año":                   2024,
                "horizonte":             horizonte,
                "nivel_alerta":          CLASES[cod],
                "nivel_alerta_codigo":   cod,
            }).execute()
    except Exception as e:
        print(f"Error guardando multi-horizonte: {e}")

    return {
        "distrito":    data.distrito,
        "semana_base": data.semana_epidemiologica,
        "horizontes":  resultados,
    }

@app.post("/escenario")
def simular_escenario(data: InputEscenario):
    base = data.base

    # Aplicar deltas climáticos
    base_mod = base.model_copy(update={
        "temp_media_c":           base.temp_media_c + data.delta_temperatura,
        "temp_max_c":             base.temp_max_c + data.delta_temperatura,
        "temp_min_c":             base.temp_min_c + data.delta_temperatura,
        "precipitacion_total_mm": max(0, base.precipitacion_total_mm + data.delta_precipitacion),
        "humedad_pct":            min(100, max(0, base.humedad_pct + data.delta_humedad)),
    })

    # Predecir 3 horizontes
    resultados = {}
    for semanas, key in [(1, "semana_1"), (2, "semana_2"), (4, "semana_4")]:
        mod = base_mod.model_copy(update={
            "semana_epidemiologica": min(base.semana_epidemiologica + semanas, 53)
        })
        X   = construir_vector(mod)
        cod = int(modelo.predict(X)[0])
        resultados[key] = {"nivel": CLASES[cod], "codigo": cod}

    # Guardar escenario en Supabase
    try:
        supabase.table("escenarios").insert({
            "usuario_id":            data.usuario_id,
            "distrito_id":           data.distrito_id,
            "semana_epidemiologica":  data.semana_epidemiologica,
            "año":                   data.año,
            "delta_temperatura":     data.delta_temperatura,
            "delta_precipitacion":   data.delta_precipitacion,
            "delta_humedad":         data.delta_humedad,
            "casos_lag1":            data.casos_lag1,
            "resultado_sem1":        resultados["semana_1"]["nivel"],
            "resultado_sem2":        resultados["semana_2"]["nivel"],
            "resultado_sem4":        resultados["semana_4"]["nivel"],
        }).execute()
    except Exception as e:
        print(f"Error guardando escenario: {e}")

    return {
        "distrito_id":   data.distrito_id,
        "semana_base":   data.semana_epidemiologica,
        "deltas": {
            "temperatura":   data.delta_temperatura,
            "precipitacion": data.delta_precipitacion,
            "humedad":       data.delta_humedad,
        },
        "horizontes": resultados,
    }

@app.get("/alertas")
def listar_alertas():
    try:
        result = supabase.table("alertas").select("*").eq("estado", "activa").execute()
        return {"alertas": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predicciones")
def listar_predicciones():
    try:
        result = supabase.table("predicciones").select("*").order("id", desc=True).limit(50).execute()
        return {"predicciones": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#Interfaz Principal del Mapa
@app.get("/predecir/mapa")
async def predecir_mapa(semana: int = None, año: int = None):
    
    if not semana:
        from datetime import date
        semana = date.today().isocalendar()[1]
    if not año:
        año = datetime.now().year

    COLORES = {
        0: "#2196F3",
        1: "#FF9800",
        2: "#F44336",
        3: "#9C27B0",
    }

    WEATHER_KEY = os.environ.get("WEATHER_API_KEY")

    # Obtener distritos
    try:
        result    = supabase.table("distritos").select("*").execute()
        distritos = result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    predicciones_mapa = []
    import math

    for distrito in distritos:
        try:
            lat = distrito["latitud"]
            lon = distrito["longitud"]

            # Clima individual por distrito desde WeatherAPI
            try:
                async with httpx.AsyncClient() as client:
                    clima_resp = await client.get(
                        f"https://api.weatherapi.com/v1/current.json"
                        f"?key={WEATHER_KEY}&q={lat},{lon}&aqi=no",
                        timeout=10
                    )
                    clima_data = clima_resp.json()
                    temp    = clima_data["current"]["temp_c"]
                    humedad = clima_data["current"]["humidity"]
                    precip  = clima_data["current"]["precip_mm"]
            except Exception:
                temp    = 19.0
                humedad = 85.0
                precip  = 0.0

            # Casos históricos promedio
            hist = supabase.table("casos_dengue")\
                .select("casos_confirmados")\
                .eq("distrito_id", distrito["id"])\
                .eq("semana_epidemiologica", semana)\
                .execute()

            casos_promedio = 0.0
            if hist.data:
                casos_promedio = sum(
                    r["casos_confirmados"] for r in hist.data
                ) / len(hist.data)

            # Construir vector
            semana_rad = 2 * math.pi * semana / 52
            semana_sin = round(math.sin(semana_rad), 4)
            semana_cos = round(math.cos(semana_rad), 4)

            dist_cod = int(le_distrito.transform(
                [distrito["nombre"]])[0]
            ) if distrito["nombre"] in le_distrito.classes_ else 0
            prov_cod = int(le_provincia.transform(["LIMA"])[0])

            vector = [
                dist_cod, prov_cod,
                semana, semana_sin, semana_cos,
                temp + 3, temp - 3, temp, 6.0,
                humedad, precip, precip * 0.5,
                casos_promedio, casos_promedio * 0.8,
                casos_promedio * 0.6, casos_promedio * 0.4,
                precip, precip, precip, precip,
                temp, temp, temp, temp,
                humedad, humedad, humedad, humedad,
                10.0, 50.0, 50.0, 30.0,
                0.0, 0.0, 0.0, 0.0,
                casos_promedio * 0.3, casos_promedio * 0.4,
                casos_promedio * 0.2,
                casos_promedio / (casos_promedio * 0.8 + 1),
                casos_promedio * 2.8,
                temp * humedad / 100,
                0.0,
            ]

            import pandas as pd
            X = pd.DataFrame([vector], columns=FEATURES)
            codigo = int(modelo.predict(X)[0])
            probas = modelo.predict_proba(X)[0]
            nivel  = CLASES[codigo]
            confianza = round(float(probas[codigo]) * 100, 2)

            predicciones_mapa.append({
                "distrito_id":         distrito["id"],
                "nombre":              distrito["nombre"],
                "latitud":             lat,
                "longitud":            lon,
                "nivel_alerta":        nivel,
                "nivel_alerta_codigo": codigo,
                "confianza_pct":       confianza,
                "color":               COLORES[codigo],
                "clima": {
                    "temperatura": temp,
                    "humedad":     humedad,
                    "precipitacion": precip,
                },
                "casos_promedio_historico": round(casos_promedio, 1),
            })

        except Exception as e:
            print(f"Error en distrito {distrito['nombre']}: {e}")
            predicciones_mapa.append({
                "distrito_id":         distrito["id"],
                "nombre":              distrito["nombre"],
                "latitud":             distrito.get("latitud"),
                "longitud":            distrito.get("longitud"),
                "nivel_alerta":        "Sin datos",
                "nivel_alerta_codigo": -1,
                "color":               "#CCCCCC",
            })

    return {
        "semana":          semana,
        "año":             año,
        "total_distritos": len(predicciones_mapa),
        "distritos":       predicciones_mapa,
    }



@app.post("/admin/cargar-casos")
async def cargar_casos_csv(file: UploadFile = File(...)):
    """
    Recibe un CSV del MINSA con columnas:
    distrito_id, semana_epidemiologica, año, casos_confirmados
    
    FastAPI completa automáticamente:
    - Clima desde WeatherAPI
    - Lags de casos desde casos_dengue
    """
    
    # Leer el CSV
    contenido = await file.read()
    texto     = contenido.decode('utf-8')
    reader    = csv.DictReader(io.StringIO(texto))
    
    exitosos = 0
    errores  = []
    WEATHER_KEY = os.environ.get("WEATHER_API_KEY")

    for fila in reader:
        try:
            distrito_id = int(fila['distrito_id'])
            semana      = int(fila['semana_epidemiologica'])
            año         = int(fila['año'])
            casos       = int(fila['casos_confirmados'])

            # Obtener coordenadas del distrito desde Supabase
            dist_result = supabase.table("distritos")\
                .select("latitud, longitud, nombre")\
                .eq("id", distrito_id)\
                .execute()

            if not dist_result.data:
                errores.append({
                    "distrito_id": distrito_id,
                    "error": "Distrito no encontrado"
                })
                continue

            lat  = dist_result.data[0]["latitud"]
            lon  = dist_result.data[0]["longitud"]
            nombre = dist_result.data[0]["nombre"]

            # Obtener clima desde WeatherAPI
            try:
                async with httpx.AsyncClient() as client:
                    clima_resp = await client.get(
                        f"https://api.weatherapi.com/v1/current.json"
                        f"?key={WEATHER_KEY}&q={lat},{lon}&aqi=no",
                        timeout=10
                    )
                    clima_data  = clima_resp.json()
                    temperatura = clima_data["current"]["temp_c"]
                    humedad     = clima_data["current"]["humidity"]
                    precipitacion = clima_data["current"]["precip_mm"]
            except Exception:
                temperatura   = 20.0
                humedad       = 75.0
                precipitacion = 0.0

            # Calcular lags desde casos_dengue
            lags = []
            for lag_sem in range(1, 5):
                sem_lag = semana - lag_sem
                año_lag = año
                if sem_lag <= 0:
                    sem_lag = 52 + sem_lag
                    año_lag = año - 1

                lag_result = supabase.table("casos_dengue")\
                    .select("casos_confirmados")\
                    .eq("distrito_id", distrito_id)\
                    .eq("semana_epidemiologica", sem_lag)\
                    .eq("año", año_lag)\
                    .execute()

                if lag_result.data:
                    lags.append(lag_result.data[0]["casos_confirmados"])
                else:
                    # Si no hay dato real usa promedio histórico
                    hist = supabase.table("casos_dengue")\
                        .select("casos_confirmados")\
                        .eq("distrito_id", distrito_id)\
                        .eq("semana_epidemiologica", sem_lag)\
                        .execute()
                    if hist.data:
                        prom = sum(r["casos_confirmados"] for r in hist.data) / len(hist.data)
                        lags.append(round(prom, 1))
                    else:
                        lags.append(0.0)

            # Construir registro completo
            registro = {
                "distrito_id":           distrito_id,
                "semana_epidemiologica": semana,
                "año":                   año,
                "casos_confirmados":     casos,
                "temperatura":           temperatura,
                "precipitacion":         precipitacion,
                "humedad":               humedad,
                "fecha_registro":        f"{año}-01-01T00:00:00",
            }

            # Verificar si ya existe
            existente = supabase.table("casos_dengue")\
                .select("id")\
                .eq("distrito_id", distrito_id)\
                .eq("semana_epidemiologica", semana)\
                .eq("año", año)\
                .execute()

            if existente.data:
                supabase.table("casos_dengue")\
                    .update(registro)\
                    .eq("id", existente.data[0]["id"])\
                    .execute()
                accion = "actualizado"
            else:
                supabase.table("casos_dengue")\
                    .insert(registro)\
                    .execute()
                accion = "insertado"

            print(f"✅ {nombre} sem{semana}/{año} — {casos} casos — {accion}")
            exitosos += 1

        except Exception as e:
            errores.append({
                "distrito_id": fila.get("distrito_id"),
                "semana":      fila.get("semana_epidemiologica"),
                "error":       str(e)
            })

    return {
        "exitosos": exitosos,
        "errores":  errores,
        "total":    exitosos + len(errores),
        "mensaje":  f"✅ {exitosos} registros procesados correctamente"
    }

@app.get("/escenarios")
def listar_escenarios():
    try:
        result = supabase.table("escenarios")\
            .select("*")\
            .order("id", desc=True)\
            .limit(50)\
            .execute()
        return {"escenarios": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))