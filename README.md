# actividad2-fastapi-finanzas-equipoXX

API educativa de análisis de señales financieras construida con **FastAPI**, **yfinance** y **Docker**, desarrollada como Actividad Integradora 2 del módulo de Automatización, APIs y Despliegue de Modelos.

> ⚠️ Esta API es una herramienta académica de análisis de señales financieras. **No constituye asesoría financiera ni recomendación de compra o venta de activos.**

> Antes de entregar: reemplacen `equipoXX` por su número real de equipo, y completen [`TEAM.md`](./TEAM.md).

## 1. Contexto

El proyecto convierte un modelo predictivo educativo (clasificación de tendencia: ¿el retorno del día siguiente será positivo?) en una API de inferencia reproducible: contratos claros con Pydantic, pruebas automatizadas, y ejecución en contenedor Docker.

## 2. Stack utilizado

| Herramienta | Propósito |
|---|---|
| **Poetry** | Gestión de dependencias y entorno virtual |
| **yfinance** | Descarga de datos históricos de activos financieros |
| **FastAPI** | Framework de la API |
| **Pydantic** | Contratos de entrada/salida validados |
| **scikit-learn** | Entrenamiento del modelo de clasificación |
| **pytest + TestClient** | Pruebas automatizadas de la API |
| **Docker** | Empaquetado y ejecución reproducible |

## 3. Estructura del repositorio

```
actividad2-fastapi-finanzas-equipoXX/
├── data/
│   ├── raw/              # CSV cacheados por símbolo (AAPL.csv, MSFT.csv, GOOGL.csv)
│   └── processed/        # dataset de features combinado (features_dataset.csv)
├── src/
│   └── financial_api/
│       ├── api.py         # endpoints FastAPI
│       ├── schemas.py     # contratos Pydantic
│       ├── data.py        # ingesta con yfinance + cache local
│       ├── features.py    # retornos, medias móviles, volatilidad, rezagos
│       ├── train.py       # entrenamiento y serialización del modelo
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

**Clasificación de tendencia**: predice si el retorno del cierre del día siguiente será positivo (`up`) o negativo/cero (`down`), usando como features: retorno diario, medias móviles (5/10/20 días), volatilidad (5/10 días) y retornos rezagados (1, 2 y 3 días). Modelo: `RandomForestClassifier`.

Símbolos usados por defecto: `AAPL`, `MSFT`, `GOOGL` (configurables).

## 5. Instalación

```bash
git clone https://github.com/<usuario-u-org>/actividad2-fastapi-finanzas-equipoXX.git
cd actividad2-fastapi-finanzas-equipoXX
poetry install
```

> En Windows, si `poetry` no se reconoce como comando, usar `python -m poetry` en su lugar en todos los comandos de esta guía.

## 6. Reproducir el flujo completo

```bash
# 1. Descargar/cachear datos históricos (usa yfinance si hay internet;
#    si no, reutiliza la copia cacheada en data/raw/)
poetry run python -m financial_api.data

# 2. Entrenar el modelo y generar artifacts/model.joblib + model_metadata.json
poetry run python -m financial_api.train

# 3. Levantar la API en modo desarrollo
poetry run uvicorn financial_api.api:app --reload

# 4. Correr las pruebas automatizadas
poetry run pytest
```

La documentación interactiva (Swagger) queda disponible en `http://localhost:8000/docs`.

## 7. Reproducibilidad sin internet

El repositorio incluye una copia cacheada de los datos en `data/raw/*.csv`. Si `financial_api.data` no logra conectarse a yfinance durante la evaluación, reutiliza automáticamente esa copia local — la API y las pruebas funcionan igual sin conexión, tal como exige la actividad. (Si tampoco existe copia cacheada, el módulo genera datos sintéticos de respaldo únicamente para no bloquear el flujo; en ese caso el modelo no tendrá señal real, solo sirve como demostración técnica.)

## 8. Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Verifica que la API esté viva y el modelo cargado |
| `GET` | `/market-data/{symbol}` | Datos recientes cacheados de un activo |
| `POST` | `/predict` | Predicción de tendencia para un símbolo |
| `GET` | `/model/metadata` | Versión del modelo, fecha de entrenamiento, símbolos, métrica |

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

## 9. Ejecución con Docker

```bash
docker build -t financial-api:local .
docker run --rm -p 8000:8000 financial-api:local
```

La API queda disponible en `http://localhost:8000/docs`. La imagen incluye el modelo ya entrenado (`artifacts/`) y los datos cacheados (`data/raw/`), por lo que funciona sin necesidad de internet ni de correr `data.py`/`train.py` dentro del contenedor.

> Si prefieren entrenar dentro del propio contenedor en vez de copiar artefactos ya generados, pueden agregar ese paso al Dockerfile o documentarlo como paso manual adicional.

## 10. Métrica y desempeño esperado

El objetivo de la actividad es la integración MLOps del servicio, no el mejor rendimiento financiero posible. Con datos reales de mercado es normal obtener un ROC-AUC modesto (cercano a 0.50–0.60): predecir la dirección del retorno diario es un problema con señal débil por naturaleza. Un modelo simple bien empaquetado, probado y documentado es preferible a uno complejo difícil de reproducir.

## 11. Flujo de trabajo en Git/GitHub

- La rama `main` está protegida: todo cambio se integra mediante **Pull Request**.
- Ramas de trabajo sugeridas:
  - `feature/data-ingestion`
  - `feature/model-training`
  - `feature/fastapi-service`
- Commits descriptivos, con participación visible de los tres integrantes (evitar un único commit final que concentre todo).

## 12. Roles del equipo

Ver [`TEAM.md`](./TEAM.md).

## 13. Extensiones opcionales (no obligatorias)

- `POST /predict/batch` para múltiples símbolos.
- Variables técnicas adicionales (RSI, MACD, bandas de Bollinger).
- Registro de métricas en MLflow.
- DAG simple de Airflow para actualizar datos y reentrenar.
