import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy import stats

app = FastAPI()

# Configuración de CORS intacta
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Estadio encendido", "mensaje": "Listo para el análisis de Sysarmy 2026"}

@app.post("/analizar")
async def analizar_salarios(
    file: UploadFile = File(...),
    corte_exp: int = Query(10, description="Años de experiencia para dividir los grupos"),
    nivel_confianza: float = Query(0.95, description="Nivel de confianza para el intervalo, ej: 0.95")
):
    try:
        # Leer el contenido del archivo binario
        contenido = await file.read()
        
        # Convertir los bytes a texto plano línea por línea de forma segura
        texto = contenido.decode('utf-8', errors='ignore')
        lineas = texto.splitlines()
        
        # ESCÁNER INTEGRAL: Busca en qué fila arranca la cabecera real
        fila_cabecera_idx = 0
        for idx, linea in enumerate(lineas):
            if 'donde_estas_trabajando' in linea or 'anos_de_experiencia' in linea or 'experiencia' in linea:
                fila_cabecera_idx = idx
                break
                
        # Reconstruimos el CSV únicamente desde la fila real detectada hacia abajo
        texto_limpio_csv = "\n".join(lineas[fila_cabecera_idx:])
        df = pd.read_csv(io.StringIO(texto_limpio_csv))
        
        # Limpieza de espacios en blanco invisibles en nombres de columnas
        df.columns = [str(c).strip() for c in df.columns]
        
        # DETECTOR INTELIGENTE DE COLUMNA DE EXPERIENCIA
        exp_col = None
        for c in df.columns:
            if c == 'anos_de_experiencia':
                exp_col = c
                break
        if not exp_col:
            for c in df.columns:
                if 'experiencia' in c.lower() or 'anos' in c.lower():
                    exp_col = c
                    break
                    
        # DETECTOR INTELIGENTE DE COLUMNA DE SALARIO
        sal_col = None
        if '_sal' in df.columns:
            sal_col = '_sal'
            
        if not sal_col:
            for c in df.columns:
                if 'neto' in c.lower():
                    sal_col = c
                    break
                    
        if not sal_col:
            for c in df.columns:
                if 'bruto' in c.lower() or 'salario' in c.lower() or 'sueldo' in c.lower():
                    sal_col = c
                    break
                    
        if not exp_col or not sal_col:
            return {"error": "Error de matriz: No se localizaron las columnas críticas en esta encuesta."}
            
        # Renombramos las columnas encontradas para unificar la matemática
        df = df.rename(columns={exp_col: 'anos_de_experiencia', sal_col: '_sal'})
        
        # Forzar casteo numérico por si vienen celdas con caracteres extraños
        df['anos_de_experiencia'] = pd.to_numeric(df['anos_de_experiencia'], errors='coerce')
        df['_sal'] = pd.to_numeric(df['_sal'], errors='coerce')
        df = df.dropna(subset=['anos_de_experiencia', '_sal'])
        
        # --- TU LÓGICA DE HIPÓTESIS ORIGINAL DE TU TRABAJO ---
        total_muestra = int(df.shape[0])
        mediana_exp = float(df['anos_de_experiencia'].median())
        
        df['grupo_experiencia'] = np.where(
            df['anos_de_experiencia'] < corte_exp,
            f'Menos de {corte_exp} años',
            f'{corte_exp} años o más'
        )
        
        resultados_hipotesis = {}
        
        for grupo in df['grupo_experiencia'].unique():
            subset = df[df['grupo_experiencia'] == grupo]
            salary_data = subset['_sal']
            
            mean = float(salary_data.mean())
            std_error = float(salary_data.std() / np.sqrt(len(salary_data)))
            
            z = float(stats.norm.ppf((1 + nivel_confianza) / 2))
            lower_bound = float(mean - z * std_error)
            upper_bound = float(mean + z * std_error)
            
            resultados_hipotesis[grupo] = {
                "media": round(mean, 2),
                "ic_inferior": round(max(0, lower_bound), 2),
                "ic_superior": round(upper_bound, 2),
                "n_grupo": int(len(salary_data)),
                "valores_box": salary_data.tolist()
            }
            
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
        return {"error": f"Backend: Error al compilar. Detalle: {str(e)}"}
