import pickle
import numpy as np
import os
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

from typing import Optional
from uuid import UUID

import csv
import io
from fastapi import UploadFile, File
from fastapi import BackgroundTasks



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
    usuario_id: Optional[str] = None
    distrito_id: int = None

class InputEscenario(BaseModel):
    usuario_id: Optional[str] = None
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
        result = supabase.table("predicciones").select("*, distritos(nombre)").order("created_at", desc=True).limit(100).execute()
        return {"predicciones": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#Interfaz Principal del Mapa
@app.get("/predecir/mapa")
async def predecir_mapa(semana: int = None, anio: int = None):
    
    if not semana:
        from datetime import date
        semana = date.today().isocalendar()[1]
    if not anio:
        anio = datetime.now().year

    COLORES = {
        0: "#2196F3",
        1: "#FF9800", 
        2: "#F44336",
        3: "#9C27B0",
    }

    WEATHER_KEY = os.environ.get("WEATHER_API_KEY")

    # UNA sola llamada de clima para Lima centro
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.weatherapi.com/v1/current.json"
                f"?key={WEATHER_KEY}&q=-12.0464,-77.0428&aqi=no",
                timeout=8
            )
            clima_data = resp.json()
            temp    = clima_data["current"]["temp_c"]
            humedad = clima_data["current"]["humidity"]
            precip  = clima_data["current"]["precip_mm"]
    except:
        temp, humedad, precip = 19.0, 80.0, 0.0

    # UNA sola consulta para distritos
    distritos = supabase.table("distritos").select("*").execute().data

    # UNA sola consulta para casos históricos
    hist_result = supabase.table("casos_dengue")\
        .select("distrito_id, casos_confirmados")\
        .eq("semana_epidemiologica", semana)\
        .execute()

    casos_por_distrito = {}
    for r in hist_result.data:
        did = r["distrito_id"]
        if did not in casos_por_distrito:
            casos_por_distrito[did] = []
        casos_por_distrito[did].append(r["casos_confirmados"])

    import math
    import pandas as pd
    predicciones_mapa = []

    for distrito in distritos:
        try:
            casos_list = casos_por_distrito.get(distrito["id"], [])
            casos_promedio = sum(casos_list) / len(casos_list) if casos_list else 0.0

            semana_rad = 2 * math.pi * semana / 52
            semana_sin = round(math.sin(semana_rad), 4)
            semana_cos = round(math.cos(semana_rad), 4)

            dist_cod = int(le_distrito.transform(
                [distrito["nombre"]])[0]
            ) if distrito["nombre"] in le_distrito.classes_ else 0
            prov_cod = int(le_provincia.transform(["LIMA"])[0])

            vector = pd.DataFrame([{
                'Distrito_cod': dist_cod, 'Provincia_cod': prov_cod,
                'Semana_Epidemiologica': semana, 'Semana_Sin': semana_sin, 'Semana_Cos': semana_cos,
                'Temp_Max_C': temp + 3, 'Temp_Min_C': temp - 3, 'Temp_Media_C': temp, 'Rango_Termico_C': 6.0,
                'Humedad_Pct': humedad, 'Precipitacion_Total_mm': precip, 'Precipitacion_Max_Dia_mm': precip * 0.5,
                'Casos_Lag1_Semanas': casos_promedio, 'Casos_Lag2_Semanas': casos_promedio * 0.8,
                'Casos_Lag3_Semanas': casos_promedio * 0.6, 'Casos_Lag4_Semanas': casos_promedio * 0.4,
                'Precip_Lag1_mm': precip, 'Precip_Lag2_mm': precip, 'Precip_Lag3_mm': precip, 'Precip_Lag4_mm': precip,
                'Temp_Lag1_C': temp, 'Temp_Lag2_C': temp, 'Temp_Lag3_C': temp, 'Temp_Lag4_C': temp,
                'Humedad_Lag1_Pct': humedad, 'Humedad_Lag2_Pct': humedad, 'Humedad_Lag3_Pct': humedad, 'Humedad_Lag4_Pct': humedad,
                'Distancia_Estacion_km': 10.0, 'Pct_Masculino': 50.0, 'Pct_Femenino': 50.0, 'Edad_Media': 30.0,
                'Casos_Menores_1': 0.0, 'Casos_1_4': 0.0, 'Casos_5_11': 0.0, 'Casos_12_17': 0.0,
                'Casos_18_29': casos_promedio * 0.3, 'Casos_30_59': casos_promedio * 0.4, 'Casos_60_mas': casos_promedio * 0.2,
                'tasa_crecimiento': casos_promedio / (casos_promedio * 0.8 + 1),
                'acumulado_4sem': casos_promedio * 2.8,
                'indice_calor_humedad': temp * humedad / 100,
                'tendencia_precip': 0.0,
            }], columns=FEATURES)

            codigo = int(modelo.predict(vector)[0])
            probas = modelo.predict_proba(vector)[0]
            nivel  = CLASES[codigo]
            confianza = round(float(probas[codigo]) * 100, 2)

            predicciones_mapa.append({
                "distrito_id":         distrito["id"],
                "nombre":              distrito["nombre"],
                "latitud":             distrito["latitud"],
                "longitud":            distrito["longitud"],
                "nivel_alerta":        nivel,
                "nivel_alerta_codigo": codigo,
                "confianza_pct":       confianza,
                "color":               COLORES[codigo],
                "clima": {
                    "temperatura": temp,
                    "humedad":     humedad,
                    "precipitacion": precip
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
        "anio":            anio,
        "total_distritos": len(predicciones_mapa),
        "clima_lima":      {"temperatura": temp, "humedad": humedad, "precipitacion": precip},
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
            .select("*, distritos(nombre)")\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute()
        return {"escenarios": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/clima/{distrito_id}")
async def obtener_clima_distrito(distrito_id: int):
    WEATHER_KEY = os.environ.get("WEATHER_API_KEY")
    
    try:
        # Obtener coordenadas del distrito
        result = supabase.table("distritos")\
            .select("latitud, longitud, nombre")\
            .eq("id", distrito_id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Distrito no encontrado")
        
        lat  = result.data[0]["latitud"]
        lon  = result.data[0]["longitud"]
        nombre = result.data[0]["nombre"]

        # Llamada a WeatherAPI
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.weatherapi.com/v1/current.json"
                f"?key={WEATHER_KEY}&q={lat},{lon}&aqi=no",
                timeout=8
            )
            clima = resp.json()

        return {
            "distrito_id": distrito_id,
            "nombre":      nombre,
            "temperatura": clima["current"]["temp_c"],
            "humedad":     clima["current"]["humidity"],
            "precipitacion": clima["current"]["precip_mm"],
            "sensacion_termica": clima["current"]["feelslike_c"],
            "condicion":   clima["current"]["condition"]["text"],
        }

    except Exception as e:
        return {
            "distrito_id":   distrito_id,
            "temperatura":   19.0,
            "humedad":       80.0,
            "precipitacion": 0.0,
            "error":         str(e)
        }

@app.get("/estadisticas")
async def obtener_estadisticas():
    try:
        casos_result     = supabase.table("casos_dengue").select("distrito_id, casos_confirmados, semana_epidemiologica, año").execute()
        preds_result     = supabase.table("predicciones").select("nivel_alerta, nivel_alerta_codigo").execute()
        alertas_result   = supabase.table("alertas").select("id, estado").execute()
        distritos_result = supabase.table("distritos").select("id, nombre").execute()

        casos     = casos_result.data
        preds     = preds_result.data
        alertas   = alertas_result.data
        distritos = distritos_result.data
        nombres   = {d['id']: d['nombre'] for d in distritos}

        # Casos por semana ordenados
        casos_por_semana = {}
        for c in casos:
            sem = f"Sem {c['semana_epidemiologica']}"
            casos_por_semana[sem] = casos_por_semana.get(sem, 0) + c['casos_confirmados']
        casos_por_semana = dict(sorted(casos_por_semana.items(), key=lambda x: int(x[0].split()[1])))

        # Semanas con datos
        semanas_unicas = set(c['semana_epidemiologica'] for c in casos)

        # Top 10 distritos con nivel promedio
        casos_por_distrito = {}
        for c in casos:
            did = c['distrito_id']
            casos_por_distrito[did] = casos_por_distrito.get(did, 0) + c['casos_confirmados']

        top10_raw = sorted(casos_por_distrito.items(), key=lambda x: x[1], reverse=True)[:10]
        
        top10 = []
        for did, total in top10_raw:
            # Calcular nivel promedio basado en total de casos
            if total > 5000:
                nivel = "Crítico"
            elif total > 2000:
                nivel = "Alto"
            elif total > 500:
                nivel = "Moderado"
            else:
                nivel = "Bajo"
            
            top10.append({
                "nombre": nombres.get(did, f"Dist {did}"),
                "casos":  total,
                "nivel":  nivel,
            })

        # Distribución niveles de predicciones
        niveles = {"Bajo": 0, "Moderado": 0, "Alto": 0, "Crítico": 0}
        for p in preds:
            nivel = p['nivel_alerta']
            if nivel in niveles:
                niveles[nivel] += 1

        # Mapa de calor temporal — top 8 distritos x semanas agrupadas
        top8_ids = [did for did, _ in top10_raw[:8]]
        heatmap = {}
        for c in casos:
            if c['distrito_id'] not in top8_ids:
                continue
            nombre = nombres.get(c['distrito_id'], '')
            sem    = c['semana_epidemiologica']
            grupo  = f"W{((sem-1)//4)*4+1}-{((sem-1)//4)*4+4}"
            key    = f"{nombre}_{grupo}"
            if key not in heatmap:
                heatmap[key] = {"nombre": nombre, "grupo": grupo, "casos": 0}
            heatmap[key]["casos"] += c['casos_confirmados']

        # Convertir heatmap a nivel
        heatmap_list = []
        for item in heatmap.values():
            casos_val = item["casos"]
            if casos_val > 500:
                color = "#9C27B0"
            elif casos_val > 200:
                color = "#F44336"
            elif casos_val > 50:
                color = "#FF9800"
            else:
                color = "#2196F3"
            heatmap_list.append({
                "nombre": item["nombre"],
                "grupo":  item["grupo"],
                "casos":  casos_val,
                "color":  color,
            })

        # Grupo etario desde casos_dengue
        # Usamos distribución típica del dengue en Lima
        casos_totales = sum(c['casos_confirmados'] for c in casos)
        etario = {
            "menores_1":  round(casos_totales * 0.02),
            "1_4":        round(casos_totales * 0.05),
            "5_11":       round(casos_totales * 0.08),
            "12_17":      round(casos_totales * 0.10),
            "18_29":      round(casos_totales * 0.28),
            "30_59":      round(casos_totales * 0.35),
            "60_mas":     round(casos_totales * 0.12),
        }

        return {
            "total_predicciones":   len(preds),
            "total_alertas":        len(alertas),
            "alertas_activas":      len([a for a in alertas if a['estado'] == 'activa']),
            "semanas_con_datos":    len(semanas_unicas),
            "distribucion_niveles": niveles,
            "casos_por_semana":     casos_por_semana,
            "top10_distritos":      top10,
            "heatmap_temporal":     heatmap_list,
            "grupo_etario":         etario,

        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/alertas/guardar")
async def guardar_alertas(semana: int = None, anio: int = None):
    if not semana:
        from datetime import date
        semana = date.today().isocalendar()[1]
    if not anio:
        anio = datetime.now().year

    WEATHER_KEY = os.environ.get("WEATHER_API_KEY")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.weatherapi.com/v1/current.json"
                f"?key={WEATHER_KEY}&q=-12.0464,-77.0428&aqi=no",
                timeout=8
            )
            clima_data = resp.json()
            temp    = clima_data["current"]["temp_c"]
            humedad = clima_data["current"]["humidity"]
            precip  = clima_data["current"]["precip_mm"]
    except:
        temp, humedad, precip = 19.0, 80.0, 0.0

    distritos   = supabase.table("distritos").select("*").execute().data
    hist_result = supabase.table("casos_dengue")\
        .select("distrito_id, casos_confirmados")\
        .eq("semana_epidemiologica", semana).execute()

    casos_por_distrito = {}
    for r in hist_result.data:
        did = r["distrito_id"]
        casos_por_distrito[did] = casos_por_distrito.get(did, 0) + r["casos_confirmados"]

    import math
    import pandas as pd

    alertas_nuevas     = 0
    alertas_duplicadas = 0

    for distrito in distritos:
        try:
            casos_promedio = casos_por_distrito.get(distrito["id"], 0.0)
            dist_cod = int(le_distrito.transform([distrito["nombre"]])[0]) \
                if distrito["nombre"] in le_distrito.classes_ else 0
            prov_cod = int(le_provincia.transform(["LIMA"])[0])

            for horizonte in [1, 2, 3, 4]:
                sem_h      = semana + horizonte
                semana_rad = 2 * math.pi * sem_h / 52

                vector = pd.DataFrame([{
                    'Distrito_cod':            dist_cod,
                    'Provincia_cod':           prov_cod,
                    'Semana_Epidemiologica':   sem_h,
                    'Semana_Sin':              round(math.sin(semana_rad), 4),
                    'Semana_Cos':              round(math.cos(semana_rad), 4),
                    'Temp_Max_C':              temp + 3,
                    'Temp_Min_C':              temp - 3,
                    'Temp_Media_C':            temp,
                    'Rango_Termico_C':         6.0,
                    'Humedad_Pct':             humedad,
                    'Precipitacion_Total_mm':  precip,
                    'Precipitacion_Max_Dia_mm': precip * 0.5,
                    'Casos_Lag1_Semanas':      casos_promedio,
                    'Casos_Lag2_Semanas':      casos_promedio * 0.8,
                    'Casos_Lag3_Semanas':      casos_promedio * 0.6,
                    'Casos_Lag4_Semanas':      casos_promedio * 0.4,
                    'Precip_Lag1_mm':          precip,
                    'Precip_Lag2_mm':          precip,
                    'Precip_Lag3_mm':          precip,
                    'Precip_Lag4_mm':          precip,
                    'Temp_Lag1_C':             temp,
                    'Temp_Lag2_C':             temp,
                    'Temp_Lag3_C':             temp,
                    'Temp_Lag4_C':             temp,
                    'Humedad_Lag1_Pct':        humedad,
                    'Humedad_Lag2_Pct':        humedad,
                    'Humedad_Lag3_Pct':        humedad,
                    'Humedad_Lag4_Pct':        humedad,
                    'Distancia_Estacion_km':   10.0,
                    'Pct_Masculino':           50.0,
                    'Pct_Femenino':            50.0,
                    'Edad_Media':              30.0,
                    'Casos_Menores_1':         0.0,
                    'Casos_1_4':               0.0,
                    'Casos_5_11':              0.0,
                    'Casos_12_17':             0.0,
                    'Casos_18_29':             casos_promedio * 0.3,
                    'Casos_30_59':             casos_promedio * 0.4,
                    'Casos_60_mas':            casos_promedio * 0.2,
                    'tasa_crecimiento':        casos_promedio / (casos_promedio * 0.8 + 1),
                    'acumulado_4sem':          casos_promedio * 2.8,
                    'indice_calor_humedad':    temp * humedad / 100,
                    'tendencia_precip':        0.0,
                }], columns=FEATURES)

                codigo    = int(modelo.predict(vector)[0])
                probas    = modelo.predict_proba(vector)[0]
                nivel     = CLASES[codigo]
                confianza = round(float(probas[codigo]) * 100, 2)

                print(f"{distrito['nombre']} | H+{horizonte} | {nivel} | {confianza}%")

                if codigo >= 2 and confianza >= 75:
                    existe = supabase.table("alertas")\
                        .select("id")\
                        .eq("nombre_distrito", distrito["nombre"])\
                        .eq("semana_alerta",   sem_h)\
                        .eq("horizonte",       horizonte)\
                        .eq("anio",            anio)\
                        .execute()

                    if len(existe.data) == 0:
                        supabase.table("alertas").insert({
                            "nombre_distrito": distrito["nombre"],
                            "nivel":           nivel,
                            "nivel_codigo":    codigo,
                            "horizonte":       horizonte,
                            "semana_alerta":   sem_h,
                            "semana_generada": semana,
                            "anio":            anio,
                            "confianza_pct":   confianza,
                            "estado":          "activa",
                            "fecha_creacion":  datetime.now().isoformat(),
                        }).execute()
                        alertas_nuevas += 1
                    else:
                        alertas_duplicadas += 1

        except Exception as e:
            print(f"Error en {distrito['nombre']}: {e}")

    return {
        "mensaje":            "Proceso completado",
        "alertas_nuevas":     alertas_nuevas,
        "alertas_duplicadas": alertas_duplicadas,
        "semana_generada":    semana,
        "anio":               anio,
    }


@app.get("/alertas/listado")
def listar_alertas(anio: int = None):
    if not anio:
        anio = datetime.now().year
    try:
        result = supabase.table("alertas")\
            .select("*")\
            .eq("anio", anio)\
            .order("nivel_codigo", desc=True)\
            .order("semana_alerta", desc=False)\
            .execute()
        return {"alertas": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))