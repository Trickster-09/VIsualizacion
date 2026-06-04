import csv
import pandas as pd

filas = []
with open('dataset_instagram_nlp.csv', encoding='utf-8') as f:
    for linea in f:
        partes = linea.strip().split(',', 11)
        if len(partes) == 12:
            filas.append(partes)

cols = ['lugar','ciudad','categoria','estrellas','texto','fecha','fuente',
        'texto_limpio','idioma','tokens','lemas','pos_tags']

df = pd.DataFrame(filas[1:], columns=cols)
df = df[df['texto'].str.strip() != '']
print(f"Filas recuperadas: {len(df)}")
print(df['texto'].head(5))
df.to_csv('dataset_reparado.csv', index=False, quoting=csv.QUOTE_ALL)
print("Guardado como dataset_reparado.csv")
