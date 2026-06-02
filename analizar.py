"""
Analizador de Comentarios Turísticos
Proyecto Final - PLN
Uso:
    python analizar.py <csv> <columna> <idioma> <titulo> <paleta>
Ejemplo:
    python analizar.py datos.csv comentario es "Reporte Turístico" viridis
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ─── NLP ───────────────────────────────────────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re

# ─── Outliers ──────────────────────────────────────────────────────────────
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# ─── Sentimientos ──────────────────────────────────────────────────────────
from textblob import TextBlob

# ─── Tópicos ───────────────────────────────────────────────────────────────
import gensim
from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud

# ─── Visualizaciones ───────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


# ════════════════════════════════════════════════════════════════════════════
# 0. DESCARGA DE RECURSOS NLTK
# ════════════════════════════════════════════════════════════════════════════

def descargar_recursos_nltk():
    recursos = ["punkt", "stopwords", "punkt_tab"]
    for r in recursos:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
# 1. CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analizador de Comentarios Turísticos con PLN",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("csv",      help="Ruta al archivo CSV con los datos")
    parser.add_argument("columna",  help="Nombre de la columna con los comentarios")
    parser.add_argument("idioma",   help="Idioma: es | en | fr")
    parser.add_argument("titulo",   help='Título del reporte (entre comillas si tiene espacios)')
    parser.add_argument("paleta",   help="Paleta de colores: viridis | cividis | plasma | inferno")
    parser.add_argument(
        "--tema",
        default="precio valor costo barato caro económico tarifa pago cobro",
        help=(
            "Tema personalizado para análisis por similitud coseno.\n"
            "Escribe palabras clave separadas por espacios.\n"
            "Por defecto analiza el concepto precio/valor/costo.\n"
            "Ejemplo: --tema 'limpieza higiene suciedad basura'"
        )
    )

    args = parser.parse_args()

    paletas_validas = ["viridis", "cividis", "plasma", "inferno", "magma", "turbo"]
    if args.paleta not in paletas_validas:
        print(f"[ADVERTENCIA] Paleta '{args.paleta}' no reconocida. Usando 'cividis' (accesible).")
        args.paleta = "cividis"

    idiomas_validos = {"es": "spanish", "en": "english", "fr": "french"}
    if args.idioma not in idiomas_validos:
        print(f"[ERROR] Idioma '{args.idioma}' no soportado. Usa: es, en, fr")
        sys.exit(1)

    return args


# ════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════

def cargar_datos(ruta_csv, columna):
    print(f"\n[1/7] Cargando datos desde '{ruta_csv}'...")
    if not os.path.exists(ruta_csv):
        print(f"[ERROR] No se encontró el archivo: {ruta_csv}")
        sys.exit(1)

    df = pd.read_csv(ruta_csv)

    if columna not in df.columns:
        print(f"[ERROR] La columna '{columna}' no existe. Columnas disponibles: {list(df.columns)}")
        sys.exit(1)

    df = df[[columna]].rename(columns={columna: "texto_original"})
    df = df.dropna(subset=["texto_original"])
    df["texto_original"] = df["texto_original"].astype(str)

    print(f"    ✓ {len(df)} comentarios cargados.")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 3. LIMPIEZA Y PREPROCESAMIENTO
# ════════════════════════════════════════════════════════════════════════════

IDIOMA_SNOWBALL = {"es": "spanish", "en": "english", "fr": "french"}
IDIOMA_STOPWORDS = {"es": "spanish", "en": "english", "fr": "french"}

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)          # URLs
    texto = re.sub(r"[^a-záéíóúüñàâçèêîôùûœæ\s]", " ", texto)  # chars especiales
    texto = re.sub(r"\s+", " ", texto).strip()              # espacios extra
    return texto

def preprocesar(df, idioma):
    print(f"\n[2/7] Limpieza y preprocesamiento (idioma: {idioma})...")

    idioma_sw  = IDIOMA_STOPWORDS.get(idioma, "english")
    idioma_stem = IDIOMA_SNOWBALL.get(idioma, "english")

    sw = set(stopwords.words(idioma_sw))
    stemmer = SnowballStemmer(idioma_stem)

    textos_limpios  = []
    textos_stemmed  = []

    for texto in df["texto_original"]:
        limpio = limpiar_texto(texto)
        tokens = word_tokenize(limpio)
        tokens_sin_sw = [t for t in tokens if t not in sw and len(t) > 2]
        stemmed = [stemmer.stem(t) for t in tokens_sin_sw]

        textos_limpios.append(" ".join(tokens_sin_sw))
        textos_stemmed.append(" ".join(stemmed))

    df["texto_limpio"]  = textos_limpios
    df["texto_stemmed"] = textos_stemmed

    # Filtrar comentarios vacíos tras limpieza
    df = df[df["texto_limpio"].str.strip() != ""].reset_index(drop=True)
    print(f"    ✓ Preprocesamiento completo. {len(df)} comentarios válidos.")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 4. DETECCIÓN DE OUTLIERS — ISOLATION FOREST
# ════════════════════════════════════════════════════════════════════════════

def detectar_outliers(df, contamination=0.07):
    print(f"\n[3/7] Detectando outliers con Isolation Forest (contamination={contamination})...")

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df["texto_stemmed"])

    modelo = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=contamination,
        max_features=1.0,
        random_state=42
    )
    pred = modelo.fit_predict(X)

    df["es_outlier"] = pred == -1

    n_outliers = df["es_outlier"].sum()
    n_normales = (~df["es_outlier"]).sum()
    print(f"    ✓ Outliers detectados: {n_outliers} | Comentarios normales: {n_normales}")

    return df, vectorizer, X


# ════════════════════════════════════════════════════════════════════════════
# 4b. N-GRAMAS: FUNCIÓN GENERAL + GRÁFICAS TOP/BOTTOM
# ════════════════════════════════════════════════════════════════════════════

def calcular_ngramas(textos, n, top_n=10):
    """Calcula frecuencia de n-gramas y devuelve top y bottom."""
    if not textos:
        return [], []
    vec = TfidfVectorizer(ngram_range=(n, n), max_features=5000)
    try:
        X = vec.fit_transform(textos)
    except ValueError:
        return [], []
    scores = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    pares = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)
    top = pares[:top_n]
    bottom = pares[-top_n:][::-1]
    return top, bottom


def graficar_ngramas(textos, etiqueta, paleta, top_n=10):
    """
    Genera figura matplotlib con top y bottom 10 de unigramas, bigramas y trigramas.
    Devuelve la figura para incluirla en el PDF.
    """
    if not textos:
        return None

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(f"N-gramas — {etiqueta}", fontsize=14, fontweight="bold", y=1.01)

    nombres = ["Unigramas", "Bigramas", "Trigramas"]
    colores_top    = plt.get_cmap(paleta)(0.75)
    colores_bottom = plt.get_cmap(paleta)(0.25)

    for fila, (n, nombre) in enumerate([(1, "Unigramas"), (2, "Bigramas"), (3, "Trigramas")]):
        top, bottom = calcular_ngramas(textos, n, top_n)

        for col, (datos, titulo, color) in enumerate([
            (top,    f"Top {top_n} {nombre}",    colores_top),
            (bottom, f"Bottom {top_n} {nombre}", colores_bottom),
        ]):
            ax = axes[fila][col]
            if not datos:
                ax.text(0.5, 0.5, "Sin datos suficientes", ha="center", va="center")
                ax.set_title(titulo)
                continue

            palabras = [p for p, _ in datos]
            valores  = [v for _, v in datos]

            bars = ax.barh(palabras[::-1], valores[::-1], color=color, edgecolor="white")
            ax.set_title(titulo, fontsize=10, fontweight="bold")
            ax.set_xlabel("TF-IDF score")
            ax.tick_params(axis="y", labelsize=8)
            for bar, val in zip(bars, valores[::-1]):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    return fig


def analizar_ngramas_grupo(textos, etiqueta, paleta, top_n=10):
    """Calcula, imprime y grafica n-gramas de un grupo. Devuelve figura y dict."""
    print(f"\n    → N-gramas [{etiqueta}]:")
    resultados = {}
    for n, nombre in [(1, "Unigramas"), (2, "Bigramas"), (3, "Trigramas")]:
        top, bottom = calcular_ngramas(textos, n, top_n)
        resultados[nombre] = {"top": top, "bottom": bottom}
        top_str    = ", ".join(p for p, _ in top[:5])
        bottom_str = ", ".join(p for p, _ in bottom[:5])
        print(f"       {nombre} top   : {top_str}")
        print(f"       {nombre} bottom: {bottom_str}")

    fig = graficar_ngramas(textos, etiqueta, paleta, top_n)
    return fig, resultados


def analizar_ngramas_outliers(df, paleta, top_n=10):
    outliers = df[df["es_outlier"]]["texto_limpio"].tolist()
    if not outliers:
        print("    (Sin outliers para analizar)")
        return None, {}
    fig, resultados = analizar_ngramas_grupo(outliers, "Outliers", paleta, top_n)
    return fig, resultados


# ════════════════════════════════════════════════════════════════════════════
# 5. ANÁLISIS DE SENTIMIENTOS
# ════════════════════════════════════════════════════════════════════════════

def analizar_sentimientos(df):
    print(f"\n[4/7] Análisis de sentimientos...")

    def polaridad(texto):
        try:
            blob = TextBlob(texto)
            return blob.sentiment.polarity
        except Exception:
            return 0.0

    df_normal = df[~df["es_outlier"]].copy()
    df_normal["polaridad"] = df_normal["texto_original"].apply(polaridad)
    df_normal["sentimiento"] = df_normal["polaridad"].apply(
        lambda p: "positivo" if p >= 0 else "negativo"
    )

    n_pos = (df_normal["sentimiento"] == "positivo").sum()
    n_neg = (df_normal["sentimiento"] == "negativo").sum()
    print(f"    ✓ Positivos: {n_pos} | Negativos: {n_neg}")

    return df_normal


# ════════════════════════════════════════════════════════════════════════════
# 6. MODELADO DE TÓPICOS (LDA) O NUBE DE PALABRAS
# ════════════════════════════════════════════════════════════════════════════

UMBRAL_LDA = 20  # mínimo de documentos para aplicar LDA

def modelar_topicos(textos, etiqueta, paleta, titulo_reporte, n_topics=3):
    textos = [t for t in textos if t.strip()]

    if len(textos) < UMBRAL_LDA:
        print(f"    ⚠ Pocos comentarios {etiqueta} ({len(textos)}). Usando nube de palabras.")
        generar_wordcloud(textos, etiqueta, paleta, titulo_reporte)
        return None, None

    print(f"    → LDA sobre comentarios {etiqueta} ({len(textos)} docs, {n_topics} tópicos)...")

    tokenized = [t.split() for t in textos]
    diccionario = corpora.Dictionary(tokenized)
    diccionario.filter_extremes(no_below=2, no_above=0.9)
    corpus = [diccionario.doc2bow(t) for t in tokenized]

    lda = LdaModel(
        corpus=corpus,
        id2word=diccionario,
        num_topics=n_topics,
        passes=10,
        random_state=42
    )

    print(f"       Tópicos {etiqueta}:")
    for i in range(n_topics):
        palabras = [w for w, _ in lda.show_topic(i, topn=8)]
        print(f"       Tópico {i+1}: {', '.join(palabras)}")

    return lda, diccionario


def generar_wordcloud(textos, etiqueta, paleta, titulo_reporte):
    texto_completo = " ".join(textos).strip()
    if not texto_completo:
        print(f"    (Sin palabras para nube de palabras {etiqueta})")
        return
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap=paleta,
        max_words=50
    ).generate(texto_completo)

    ruta = f"outputs/wordcloud_{etiqueta}.png"
    wc.to_file(ruta)
    print(f"    ✓ Nube de palabras guardada en '{ruta}'")


def comentario_representativo(textos, lda, diccionario):
    """Retorna el comentario más cercano al tópico dominante."""
    if not textos:
        return "(sin comentarios disponibles)"
    mejor_idx = 0
    mejor_score = -1
    for i, t in enumerate(textos):
        if not t.strip():
            continue
        bow = diccionario.doc2bow(t.split())
        dist = lda.get_document_topics(bow)
        if dist:
            score = max(p for _, p in dist)
            if score > mejor_score:
                mejor_score = score
                mejor_idx = i
    if mejor_idx < len(textos):
        return textos[mejor_idx]
    return textos[0]


def comentario_representativo_orig(textos_stem, textos_orig, lda, diccionario):
    """Retorna el comentario ORIGINAL más cercano al tópico dominante."""
    if not textos_stem:
        return "(sin comentarios disponibles)"
    mejor_idx = 0
    mejor_score = -1
    for i, t in enumerate(textos_stem):
        if not t.strip():
            continue
        bow = diccionario.doc2bow(t.split())
        dist = lda.get_document_topics(bow)
        if dist:
            score = max(p for _, p in dist)
            if score > mejor_score:
                mejor_score = score
                mejor_idx = i
    return textos_orig[mejor_idx] if mejor_idx < len(textos_orig) else textos_orig[0]


# ════════════════════════════════════════════════════════════════════════════
# 7. SCATTER PLOTS INTERACTIVOS
# ════════════════════════════════════════════════════════════════════════════

def reducir_dimensiones(vectorizer, textos):
    """TF-IDF + SVD a 2D para scatter plot."""
    X = vectorizer.transform(textos)
    svd = TruncatedSVD(n_components=2, random_state=42)
    X2d = svd.fit_transform(X)
    return X2d


def asignar_topico(texto, lda, diccionario):
    if lda is None:
        return 0
    bow = diccionario.doc2bow(texto.split())
    dist = lda.get_document_topics(bow)
    if not dist:
        return 0
    return max(dist, key=lambda x: x[1])[0]


def generar_scatter(df_sentimiento, vectorizer, lda_pos, dic_pos,
                    lda_neg, dic_neg, paleta, titulo, archivo_salida):
    print(f"\n[5/7] Generando scatter plot interactivo...")

    textos_stem = df_sentimiento["texto_stemmed"].tolist()
    textos_orig = df_sentimiento["texto_original"].tolist()
    sentimientos = df_sentimiento["sentimiento"].tolist()

    X2d = reducir_dimensiones(vectorizer, textos_stem)

    topicos = []
    for i, row in df_sentimiento.iterrows():
        sent = row["sentimiento"]
        stem = row["texto_stemmed"]
        if sent == "positivo" and lda_pos:
            t = asignar_topico(stem, lda_pos, dic_pos)
            topicos.append(f"POS-T{t+1}")
        elif sent == "negativo" and lda_neg:
            t = asignar_topico(stem, lda_neg, dic_neg)
            topicos.append(f"NEG-T{t+1}")
        else:
            topicos.append(sent.upper())

    plot_df = pd.DataFrame({
        "x": X2d[:, 0],
        "y": X2d[:, 1],
        "topico": topicos,
        "sentimiento": sentimientos,
        "texto": [t[:120] + "..." if len(t) > 120 else t for t in textos_orig],
    })

    fig = px.scatter(
        plot_df, x="x", y="y",
        color="topico",
        symbol="sentimiento",         # codificación redundante
        hover_data={"texto": True, "topico": True, "sentimiento": True, "x": False, "y": False},
        title=f"{titulo} — Mapa de Tópicos",
        color_discrete_sequence=px.colors.sample_colorscale(paleta, max(2, len(plot_df["topico"].unique()))),
        labels={"x": "Dimensión 1", "y": "Dimensión 2"}
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(legend_title="Tópico / Sentimiento")

    os.makedirs("outputs", exist_ok=True)
    fig.write_html(f"outputs/{archivo_salida}")
    print(f"    ✓ Scatter guardado en 'outputs/{archivo_salida}'")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 8. ANÁLISIS PRECIO / VALOR / COSTO
# ════════════════════════════════════════════════════════════════════════════

def analisis_precio_valor(df_normal, vectorizer, paleta, titulo, tema):
    tema_label = tema[:40] + "..." if len(tema) > 40 else tema
    print(f"\n[6/7] Análisis de tema: '{tema_label}'...")

    concepto = tema

    textos = df_normal["texto_stemmed"].tolist()
    textos_orig = df_normal["texto_original"].tolist()

    # Vectorizar todo + el concepto
    todos = textos + [concepto]
    X = vectorizer.transform(todos)
    X_norm = normalize(X)

    vec_concepto = X_norm[-1]          # último vector = concepto
    sims = cosine_similarity(X_norm[:-1], vec_concepto).flatten()

    df_normal = df_normal.copy()
    df_normal["sim_precio"] = sims

    # Top 5 más cercanos
    top5_idx = sims.argsort()[-5:][::-1]
    print(f"    Top 5 comentarios relacionados con '{tema_label}':")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"    {rank}. (sim={sims[idx]:.3f}) {textos_orig[idx][:100]}...")

    # Reducir a 2D
    X2d = reducir_dimensiones(vectorizer, textos)

    plot_df = pd.DataFrame({
        "x": X2d[:, 0],
        "y": X2d[:, 1],
        "similitud": sims,
        "texto": [t[:120] + "..." if len(t) > 120 else t for t in textos_orig],
        "es_top5": ["⭐ Top 5" if i in top5_idx else "Resto" for i in range(len(textos))]
    })

    fig = px.scatter(
        plot_df, x="x", y="y",
        color="similitud",
        symbol="es_top5",              # codificación redundante
        color_continuous_scale=paleta,
        hover_data={"texto": True, "similitud": ":.3f", "x": False, "y": False},
        title=f"{titulo} — Relación con '{tema_label}'",
        labels={"x": "Dimensión 1", "y": "Dimensión 2", "similitud": "Similitud"}
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))

    fig.write_html("outputs/precio_valor_costo.html")
    print("    ✓ Scatter de precio/valor/costo guardado en 'outputs/precio_valor_costo.html'")

    return df_normal


# ════════════════════════════════════════════════════════════════════════════
# 9. REPORTE FINAL EN CONSOLA
# ════════════════════════════════════════════════════════════════════════════

def imprimir_reporte(titulo, df_sentimiento, lda_pos, dic_pos, lda_neg, dic_neg,
                     ngramas_outliers, n_outliers):
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  REPORTE: {titulo}")
    print(sep)

    print(f"\n  Total comentarios analizados : {len(df_sentimiento) + n_outliers}")
    print(f"  Outliers detectados          : {n_outliers}")
    print(f"  Comentarios normales         : {len(df_sentimiento)}")
    n_pos = (df_sentimiento["sentimiento"] == "positivo").sum()
    n_neg = (df_sentimiento["sentimiento"] == "negativo").sum()
    print(f"  Positivos                    : {n_pos}")
    print(f"  Negativos                    : {n_neg}")

    if ngramas_outliers:
        print(f"\n  N-gramas en outliers (top 5):")
        for tipo, datos in ngramas_outliers.items():
            top_palabras = [p for p, _ in datos.get('top', [])[:5]]
            print(f"    {tipo}: {', '.join(top_palabras)}")

    for etiqueta, lda, dic in [("POSITIVOS", lda_pos, dic_pos), ("NEGATIVOS", lda_neg, dic_neg)]:
        if lda is None:
            continue
        print(f"\n  Tópicos {etiqueta}:")
        mask = df_sentimiento["sentimiento"] == etiqueta.lower()
        textos_stem = df_sentimiento[mask]["texto_stemmed"].tolist()
        textos_orig = df_sentimiento[mask]["texto_original"].tolist()
        for i in range(lda.num_topics):
            palabras = [w for w, _ in lda.show_topic(i, topn=6)]
            rep = comentario_representativo_orig(textos_stem, textos_orig, lda, dic)
            print(f"    Tópico {i+1}: {', '.join(palabras)}")
            print(f"    Ejemplo  : {rep[:120]}...")

    print(f"\n  Archivos generados en 'outputs/':")
    for f in os.listdir("outputs"):
        print(f"    - {f}")
    print(f"\n{sep}\n")



# ════════════════════════════════════════════════════════════════════════════
# PDF REPORTE COMPLETO
# ════════════════════════════════════════════════════════════════════════════

def generar_pdf(titulo, df, df_normal, ngramas_general, ngramas_outliers,
                ngramas_pos, ngramas_neg, lda_pos, dic_pos, lda_neg, dic_neg,
                fig_ngramas_general, fig_ngramas_outliers,
                fig_ngramas_pos, fig_ngramas_neg, paleta):
    """Genera un PDF con todo el reporte de análisis."""
    os.makedirs("outputs", exist_ok=True)
    ruta = "outputs/reporte_completo.pdf"
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M")

    n_total    = len(df)
    n_outliers = df["es_outlier"].sum()
    n_normales = (~df["es_outlier"]).sum()
    n_pos = (df_normal["sentimiento"] == "positivo").sum()
    n_neg = (df_normal["sentimiento"] == "negativo").sum()

    with PdfPages(ruta) as pdf:

        # ── Portada ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.75, titulo, ha="center", va="center",
                fontsize=24, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.60, "Análisis de Comentarios Turísticos", ha="center",
                fontsize=16, color="gray", transform=ax.transAxes)
        ax.text(0.5, 0.45, f"Generado: {ahora}", ha="center",
                fontsize=12, color="gray", transform=ax.transAxes)

        stats = (
            f"Total comentarios: {n_total}    |    "
            f"Outliers: {n_outliers}    |    "
            f"Normales: {n_normales}\n"
            f"Positivos: {n_pos}    |    Negativos: {n_neg}"
        )
        ax.text(0.5, 0.30, stats, ha="center", fontsize=12,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="gray"))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── N-gramas corpus general ──────────────────────────────────────────
        if fig_ngramas_general:
            fig_ngramas_general.suptitle(f"N-gramas — Corpus General ({n_total} comentarios)",
                                          fontsize=13, fontweight="bold")
            pdf.savefig(fig_ngramas_general, bbox_inches="tight")
            plt.close(fig_ngramas_general)

        # ── N-gramas outliers ───────────────────────────────────────────────
        if fig_ngramas_outliers:
            fig_ngramas_outliers.suptitle(f"N-gramas — Outliers ({n_outliers} comentarios)",
                                           fontsize=13, fontweight="bold")
            pdf.savefig(fig_ngramas_outliers, bbox_inches="tight")
            plt.close(fig_ngramas_outliers)

        # ── N-gramas positivos ──────────────────────────────────────────────
        if fig_ngramas_pos:
            fig_ngramas_pos.suptitle(f"N-gramas — Comentarios Positivos ({n_pos} comentarios)",
                                      fontsize=13, fontweight="bold")
            pdf.savefig(fig_ngramas_pos, bbox_inches="tight")
            plt.close(fig_ngramas_pos)

        # ── N-gramas negativos ──────────────────────────────────────────────
        if fig_ngramas_neg:
            fig_ngramas_neg.suptitle(f"N-gramas — Comentarios Negativos ({n_neg} comentarios)",
                                      fontsize=13, fontweight="bold")
            pdf.savefig(fig_ngramas_neg, bbox_inches="tight")
            plt.close(fig_ngramas_neg)

        # ── Resumen de tópicos ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.97, "Resumen de Tópicos LDA", ha="center", fontsize=16,
                fontweight="bold", transform=ax.transAxes)

        y = 0.88
        for etiqueta, lda, dic, df_mask in [
            ("POSITIVOS", lda_pos, dic_pos, df_normal[df_normal["sentimiento"] == "positivo"]),
            ("NEGATIVOS", lda_neg, dic_neg, df_normal[df_normal["sentimiento"] == "negativo"]),
        ]:
            if lda is None:
                ax.text(0.05, y, f"● {etiqueta}: pocos comentarios → ver nube de palabras",
                        fontsize=11, color="gray", transform=ax.transAxes)
                y -= 0.07
                continue

            ax.text(0.05, y, f"● Tópicos {etiqueta}:", fontsize=12,
                    fontweight="bold", transform=ax.transAxes,
                    color="#2c7bb6" if etiqueta == "POSITIVOS" else "#d7191c")
            y -= 0.05

            textos_stem = df_mask["texto_stemmed"].tolist()
            textos_orig = df_mask["texto_original"].tolist()
            for i in range(lda.num_topics):
                palabras = [w for w, _ in lda.show_topic(i, topn=7)]
                rep = comentario_representativo_orig(textos_stem, textos_orig, lda, dic)
                rep_corto = rep[:90] + "..." if len(rep) > 90 else rep
                ax.text(0.07, y, f"Tópico {i+1}: {', '.join(palabras)}",
                        fontsize=9, fontweight="bold", transform=ax.transAxes)
                y -= 0.04
                ax.text(0.07, y, f'Ejemplo: "{rep_corto}"',
                        fontsize=8, color="gray", style="italic", transform=ax.transAxes)
                y -= 0.05
                if y < 0.05:
                    break
            y -= 0.03

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"    ✓ PDF generado en '{ruta}'")
    return ruta

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    descargar_recursos_nltk()
    args = parse_args()

    # Crear carpeta de salida si no existe
    os.makedirs("outputs", exist_ok=True)

    # 1. Cargar
    df = cargar_datos(args.csv, args.columna)

    # 2. Preprocesar
    df = preprocesar(df, args.idioma)

    # 3a. Outliers
    df, vectorizer, _ = detectar_outliers(df, contamination=0.07)
    n_outliers = df["es_outlier"].sum()

    # 3b. N-gramas corpus general (todos, antes de filtrar)
    print("\n    → N-gramas corpus general:")
    fig_ng_general, ng_general = analizar_ngramas_grupo(
        df["texto_limpio"].tolist(), "Corpus General", args.paleta
    )

    # 3c. N-gramas outliers
    fig_ng_outliers, ng_outliers = analizar_ngramas_outliers(df, args.paleta)

    # 4. Sentimientos
    df_normal = analizar_sentimientos(df)

    # 4b. N-gramas por sentimiento
    pos_textos_limpios = df_normal[df_normal["sentimiento"] == "positivo"]["texto_limpio"].tolist()
    neg_textos_limpios = df_normal[df_normal["sentimiento"] == "negativo"]["texto_limpio"].tolist()

    fig_ng_pos, ng_pos = analizar_ngramas_grupo(pos_textos_limpios, "Positivos", args.paleta)
    fig_ng_neg, ng_neg = analizar_ngramas_grupo(neg_textos_limpios, "Negativos", args.paleta)

    # 5. LDA / Wordcloud
    print(f"\n[5/7] Modelado de tópicos...")
    pos_textos = df_normal[df_normal["sentimiento"] == "positivo"]["texto_stemmed"].tolist()
    neg_textos = df_normal[df_normal["sentimiento"] == "negativo"]["texto_stemmed"].tolist()

    lda_pos, dic_pos = modelar_topicos(pos_textos, "positivos", args.paleta, args.titulo)
    lda_neg, dic_neg = modelar_topicos(neg_textos, "negativos", args.paleta, args.titulo)

    # 6. Scatter tópicos
    generar_scatter(df_normal, vectorizer, lda_pos, dic_pos,
                    lda_neg, dic_neg, args.paleta, args.titulo,
                    "topicos_sentimientos.html")

    # 7. Precio/Valor/Costo
    df_normal = analisis_precio_valor(df_normal, vectorizer, args.paleta, args.titulo, args.tema)

    # 8. Reporte consola + PDF
    print("\n[7/7] Generando reporte y PDF...")
    imprimir_reporte(args.titulo, df_normal, lda_pos, dic_pos,
                     lda_neg, dic_neg, ng_outliers, n_outliers)

    generar_pdf(
        titulo=args.titulo,
        df=df,
        df_normal=df_normal,
        ngramas_general=ng_general,
        ngramas_outliers=ng_outliers,
        ngramas_pos=ng_pos,
        ngramas_neg=ng_neg,
        lda_pos=lda_pos, dic_pos=dic_pos,
        lda_neg=lda_neg, dic_neg=dic_neg,
        fig_ngramas_general=fig_ng_general,
        fig_ngramas_outliers=fig_ng_outliers,
        fig_ngramas_pos=fig_ng_pos,
        fig_ngramas_neg=fig_ng_neg,
        paleta=args.paleta
    )


if __name__ == "__main__":
    main()
