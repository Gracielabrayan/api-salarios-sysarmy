import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy import stats

app = FastAPI()

# Configuración de CORS idéntica a tu proyecto del álbum
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RUTA CRON-JOB: Mantiene tu servidor de salarios despierto en Render
@app.get("/")
def home():
    return {"status": "Estadio encendido", "mensaje": "Listo para el análisis de Sysarmy 2026"}

@app.post("/analizar")
async def analizar_salarios(
    file: UploadFile = File(...),
    corte_exp: int = Query(10, description="Años de experiencia para dividir los grupos"),
    nivel_confianza: float = Query(0.95, description="Nivel de confianza para el intervalo (0.90 a 0.99)")
):
    try:
        # 1. Carga de datos idéntica a tu script (header=9 para saltear filas de Sysarmy)
        contenido = await file.read()
        df = pd.read_csv(io.BytesIO(contenido), header=9)
        
        # Limpieza rápida preventiva: eliminar nulos en las variables de tu hipótesis
        df = df.dropna(subset=['anos_de_experiencia', '_sal'])
        
        # 2. Métricas descriptivas generales extraídas de tu informe
        total_muestra = int(df.shape[0])
        mediana_exp = float(df['anos_de_experiencia'].median())
        
        # 3. Segmentación dinámica basada en tu lógica (np.where)
        # Permite usar el corte de 10 años por defecto, o adaptarlo desde la web
        df['grupo_experiencia'] = np.where(
            df['anos_de_experiencia'] < corte_exp,
            f'Menos de {corte_exp} años',
            f'{corte_exp} años o más'
        )
        
        resultados_hipotesis = {}
        
        # 4. Tu bucle exacto de cálculo estadístico (SciPy)
        for grupo in df['grupo_experiencia'].unique():
            subset = df[df['grupo_experiencia'] == grupo]
            salary_data = subset['_sal']
            
            # Tu lógica exacta de Python
            mean = float(salary_data.mean())
            std_error = float(salary_data.std() / np.sqrt(len(salary_data)))
            
            # Valor crítico Z usando SciPy de tu script
            z = float(stats.norm.ppf((1 + nivel_confianza) / 2))
            
            lower_bound = float(mean - z * std_error)
            upper_bound = float(mean + z * std_error)
            
            # Guardamos estadísticas empaquetadas por grupo
            resultados_hipotesis[grupo] = {
                "media": round(mean, 2),
                "ic_inferior": round(max(0, lower_bound), 2),
                "ic_superior": round(upper_bound, 2),
                "n_grupo": int(len(salary_data)),
                "valores_box": salary_data.tolist() # Para que Plotly dibuje el boxplot vertical real
            }
            
        # 5. Respuesta estructurada limpia para tu frontend
        return {
            "total_muestra": total_muestra,
            "mediana_experiencia": mediana_exp,
            "corte_aplicado": corte_exp,
            "confianza_aplicada": nivel_confianza,
            "datos_dispersion": {
                "experiencia": df['anos_de_experiencia'].tolist(),
                "salarios": df['_sal'].tolist()
            },
            "resultados_hipotesis": resultados_hipotesis
        }
        
    except Exception as e:
        return {"error": f"Tarjeta Roja: Falló el procesamiento del CSV. Detalle: {str(e)}"}