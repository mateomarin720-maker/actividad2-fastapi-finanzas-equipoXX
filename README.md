# actividad2-fastapi-finanzas-equipo05

API educativa de analisis de señales financieras construida con **FastAPI**, **yfinance** y **Docker**, desarrollada como Actividad Integradora 2 del modulo de Automatizacion, APIs y Despliegue de Modelos.

> Esta API es una herramienta academica de analisis de señales financieras. **No constituye asesoria financiera ni recomendacion de compra o venta de activos.**


## 1. Contexto

El proyecto convierte un modelo predictivo educativo (clasificacion de tendencia: ¿el retorno del dia siguiente sera positivo?) en una API de inferencia reproducible: contratos claros con Pydantic, pruebas automatizadas, y ejecucion en contenedor Docker.

## 2. Stack utilizado

| Herramienta | Proposito |
|---|---|
| **Poetry** | Gestion de dependencias y entorno virtual |
| **yfinance** | Descarga de datos historicos de activos financieros |
| **FastAPI** | Framework de la API |
| **Pydantic** | Contratos de entrada/salida validados |
| **scikit-learn** | Entrenamiento del modelo de clasificacion |
| **pytest + TestClient** | Pruebas automatizadas de la API |
| **Docker** | Empaquetado y ejecucion reproducible |

## 3. Estructura del repositorio

```
actividad2-fastapi-finanzas-equipoXX/
├── data/
│   ├── raw/              # CSV cacheados por simbolo (AAPL.csv, MSFT.csv, GOOGL.csv)
│   └── processed/        # dataset de features combinado (features_dataset.csv)
├── src/
│   └── financial_api/
│       ├── api.py         # endpoints FastAPI
│       ├── schemas.py     # contratos Pydantic
│       ├── data.py        # ingesta con yfinance + cache local
│       ├── features.py    # retornos, medias moviles, volatilidad, rezagos
│       ├── train.py       # entrenamiento y serializacion del modelo
│       └── predict.py     # carga de modelo + inferencia
├── artifacts/
│   ├── model.joblib
│   └── model_metadata.json
├── tests/
│   ├── test_api.py
│   └── test_schemas.py
├── reports/
├── Dockerfile
├── pyproject.toml
├── poetry.lock
├── README.md
├── TEAM.md
└── .gitignore
```

## 4. Tarea predictiva

**Clasificacion de tendencia**: predice si el retorno del cierre del dia siguiente sera positivo (`up`) o negativo/cero (`down`), usando como features: retorno diario, medias moviles (5/10/20 dias), volatilidad (5/10 dias) y retornos rezagados (1, 2 y 3 dias). Modelo: `RandomForestClassifier`.

Simbolos usados por defecto: `AAPL`, `MSFT`, `GOOGL` (configurables).

## 5. Instalacion

```bash
git clone https://github.com/<usuario-u-org>/actividad2-fastapi-finanzas-equipoXX.git
cd actividad2-fastapi-finanzas-equipoXX
poetry install
```

> En Windows, si `poetry` no se reconoce como comando, usar `python -m poetry` en su lugar en todos los comandos de esta guia.

## 6. Reproducir el flujo completo

```bash
# 1. Descargar/cachear datos historicos (usa yfinance si hay internet;
#    si no, reutiliza la copia cacheada en data/raw/)
poetry run python -m financial_api.data

# 2. Entrenar el modelo y generar artifacts/model.joblib + model_metadata.json
poetry run python -m financial_api.train

# 3. Levantar la API en modo desarrollo
poetry run uvicorn financial_api.api:app --reload

# 4. Correr las pruebas automatizadas
poetry run pytest
```

La documentacion interactiva (Swagger) queda disponible en `http://localhost:8000/docs`.

## 7. Reproducibilidad sin internet

El repositorio incluye una copia cacheada de los datos en `data/raw/*.csv`. Si `financial_api.data` no logra conectarse a yfinance durante la evaluacion, reutiliza automaticamente esa copia local — la API y las pruebas funcionan igual sin conexion, tal como exige la actividad. (Si tampoco existe copia cacheada, el modulo genera datos sinteticos de respaldo unicamente para no bloquear el flujo; en ese caso el modelo no tendra señal real, solo sirve como demostracion tecnica.)

## 8. Endpoints

| Metodo | Ruta | Descripcion |
|---|---|---|
| `GET` | `/health` | Verifica que la API este viva y el modelo cargado |
| `GET` | `/market-data/{symbol}` | Datos recientes cacheados de un activo |
| `POST` | `/predict` | Prediccion de tendencia para un simbolo |
| `GET` | `/model/metadata` | Version del modelo, fecha de entrenamiento, simbolos, metrica |

### Ejemplo de request a `/predict`

```json
{
  "symbol": "AAPL",
  "prediction_horizon": 1,
  "use_cached_data": true
}
```

### Ejemplo de respuesta

```json
{
  "symbol": "AAPL",
  "prediction": "up",
  "probability_up": 0.63,
  "model_version": "random_forest_v1",
  "prediction_horizon": "next_day"
}
```

## 9. Ejecucion con Docker

```bash
docker build -t financial-api:local .
docker run --rm -p 8000:8000 financial-api:local
```

La API queda disponible en `http://localhost:8000/docs`. La imagen incluye el modelo ya entrenado (`artifacts/`) y los datos cacheados (`data/raw/`), por lo que funciona sin necesidad de internet ni de correr `data.py`/`train.py` dentro del contenedor.

## 10. Metrica y desempeño esperado

El objetivo de la actividad es la integracion MLOps del servicio, no el mejor rendimiento financiero posible. Con datos reales de mercado es normal obtener un ROC-AUC modesto (cercano a 0.50–0.60): predecir la direccion del retorno diario es un problema con señal debil por naturaleza. Un modelo simple bien empaquetado, probado y documentado es preferible a uno complejo dificil de reproducir.

## 11. Flujo de trabajo en Git/GitHub

- La rama `main` esta protegida: todo cambio se integra mediante **Pull Request**.
- Ramas de trabajo:
  - `feature/data-ingestion`
  - `feature/model-training`
  - `feature/fastapi-service`.

## 12. Roles del equipo

Ver [`TEAM.md`](./TEAM.md).

