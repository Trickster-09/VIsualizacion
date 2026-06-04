# 🗺️ Analizador de Comentarios Turísticos

Script de Python para análisis completo de comentarios turísticos desde línea de comandos. Realiza limpieza de texto, detección de outliers, análisis de sentimientos, modelado de tópicos y visualizaciones interactivas. **100% offline, sin APIs externas.**

---

## 📋 Requisitos

- Python 3.8 o superior
- pip

---

## ⚙️ Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/Trickster-09/VIsualizacion.git
cd analizador-comentarios-turisticos

# 2. Instala las dependencias
pip install -r requirements.txt
```

---

## 🚀 Uso

```bash
python analizar.py <csv> <columna> <idioma> "<titulo>" <paleta> [--optimizar] [--tema "palabras clave"]
```

### Parámetros obligatorios

| # | Parámetro | Descripción | Valores válidos |
|---|-----------|-------------|-----------------|
| 1 | `csv` | Ruta al archivo CSV | cualquier ruta |
| 2 | `columna` | Nombre de la columna con los comentarios | nombre exacto |
| 3 | `idioma` | Idioma del análisis | `es` · `en` · `fr` |
| 4 | `titulo` | Título del reporte | texto libre entre comillas |
| 5 | `paleta` | Paleta de colores | `viridis` · `cividis` · `plasma` · `inferno` |

### Parámetros opcionales

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--optimizar` | Activa grid search de num_topics via Coherence Score (tarda más pero produce mejores tópicos) | desactivado |
| `--tema "palabras"` | Tema personalizado para análisis de similitud semántica | `"precio valor costo barato caro económico tarifa pago cobro"` |

---

## 💡 Ejemplos de ejecución

```bash
# Básico en español con paleta accesible
python analizar.py datos.csv comentario es "Reporte Turístico" cividis

# Con optimización de tópicos via grid search
python analizar.py datos.csv comentario es "Reporte Turístico" cividis --optimizar

# Con tema personalizado
python analizar.py datos.csv comentario es "Reporte Turístico" cividis --tema "limpieza higiene suciedad"

# Con todo junto
python analizar.py datos.csv comentario es "Reporte Turístico" cividis --optimizar --tema "servicio atención personal"

# En inglés
python analizar.py reviews.csv review_text en "Tourism Report" viridis --optimizar

# En francés
python analizar.py avis.csv texte fr "Rapport Touristique" plasma
```

---

## 🔄 Flujo de análisis

```
CSV de entrada
      │
      ▼
1. LIMPIEZA Y PREPROCESAMIENTO
   Elimina URLs, caracteres especiales, stopwords
   Aplica stemming según idioma (es/en/fr)
      │
      ▼
2. DETECCIÓN DE OUTLIERS — Isolation Forest
   Vectoriza con TF-IDF → entrena Isolation Forest
   Etiqueta comentarios como normal / outlier
      │
      ├── OUTLIERS  → análisis de unigramas, bigramas y trigramas
      │               (top 10 y bottom 10)
      │
      └── NORMALES  → continúan al siguiente paso
                │
                ▼
3. N-GRAMAS POR GRUPO
   Corpus general / Outliers / Positivos / Negativos
   Top 10 y Bottom 10 de uni, bi y trigramas
                │
                ▼
4. ANÁLISIS DE SENTIMIENTOS — TextBlob
   Clasifica cada comentario como positivo o negativo
                │
                ├── POSITIVOS ──┐
                │               ▼
                │     5. MODELADO DE TÓPICOS
                │        • Sin --optimizar: LDA con 3 tópicos
                │        • Con --optimizar: Grid Search k=2..12
                │          via Coherence Score → elige mejor k
                │        • Si hay pocos comentarios (<20): WordCloud
                │
                └── NEGATIVOS ──▶ igual que positivos
                                │
                                ▼
                     6. SCATTER PLOTS INTERACTIVOS — Plotly
                        Un punto = un comentario
                        Color + forma = tópico (codificación redundante)
                        Hover = muestra el texto del comentario
                                │
                                ▼
                     7. ANÁLISIS SEMÁNTICO — "Precio / Valor / Costo"
                        Similitud coseno TF-IDF entre comentarios y tema
                        Scatter plot con gradiente de similitud
                        Top 5 comentarios más relacionados
                                │
                                ▼
                     8. REPORTE PDF COMPLETO
                        Portada + estadísticas + todas las gráficas
```

---

## 📁 Archivos generados

Todos se guardan automáticamente en la carpeta `outputs/`:

| Archivo | Descripción |
|---------|-------------|
| `reporte_completo.pdf` | Reporte con portada, n-gramas, curvas de coherence y resumen de tópicos |
| `topicos_sentimientos.html` | Scatter interactivo de tópicos — abrir en navegador |
| `precio_valor_costo.html` | Scatter de similitud semántica con el tema — abrir en navegador |
| `wordcloud_positivos.png` | Nube de palabras positivas (solo si hay pocos comentarios) |
| `wordcloud_negativos.png` | Nube de palabras negativas (solo si hay pocos comentarios) |
| `coherence_positivos.png` | Curva de Coherence Score positivos (solo con `--optimizar`) |
| `coherence_negativos.png` | Curva de Coherence Score negativos (solo con `--optimizar`) |

---

## 🛠️ Solución de problemas comunes

**El CSV tiene comas dentro del texto y se carga mal:**
```bash
python reparar.py
# genera dataset_reparado.csv listo para usar
```

**No sé el nombre exacto de la columna:**
```bash
python -c "import pandas as pd; print(pd.read_csv('mi_archivo.csv').columns.tolist())"
```

**El script tarda mucho con `--optimizar`:**
Es normal — evalúa 11 modelos LDA distintos. Con 6000+ comentarios puede tardar 5-10 minutos. Sin `--optimizar` corre en segundos.

---

## 📦 Estructura del repositorio

```
analizador-comentarios-turisticos/
├── analizar.py          ← script principal
├── reparar.py           ← utilidad para reparar el csv de prueba ¨dataset_instagram_nlp¨
├── requirements.txt     ← dependencias
├── README.md            ← este archivo
├── sample_data/
│   └── dataset_reparado.csv     
│   └── dataset_instagram_nlp.csv
└── outputs/             ← generado automáticamente al ejecutar
```

---

## 🎨 Paletas de colores y accesibilidad

Las paletas `cividis` y `viridis` están diseñadas para ser perceptibles por personas con daltonismo (deuteranopia / protanopia). Se recomienda `cividis` como primera opción para máxima accesibilidad.

| Paleta | Accesible | Descripción |
|--------|-----------|-------------|
| `cividis` | ✅ Sí | Azul a amarillo, optimizada para daltonismo |
| `viridis` | ✅ Sí | Morado a amarillo-verde |
| `plasma` | ⚠️ Parcial | Morado a naranja |
| `inferno` | ⚠️ Parcial | Negro a amarillo |

---

## 📚 Dependencias principales

| Librería | Versión mínima | Uso |
|----------|---------------|-----|
| `pandas` | 1.5.0 | Manejo de datos |
| `numpy` | 1.23.0 | Operaciones numéricas |
| `nltk` | 3.8.0 | Tokenización, stopwords, stemming |
| `textblob` | 0.17.1 | Análisis de sentimientos |
| `scikit-learn` | 1.2.0 | Isolation Forest, TF-IDF, SVD |
| `gensim` | 4.3.0 | LDA y Coherence Score |
| `wordcloud` | 1.9.0 | Nubes de palabras |
| `plotly` | 5.14.0 | Visualizaciones interactivas |
| `matplotlib` | 3.7.0 | Gráficas estáticas y PDF |
