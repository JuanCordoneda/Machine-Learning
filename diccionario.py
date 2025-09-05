# 📘 DICCIONARIO DE FUNCIONES USADAS CON EJEMPLOS

df.columns                                  # Muestra los nombres de las columnas del DataFrame

df.sample(5)                                # Devuelve 5 filas aleatorias

df.shape                                    # Devuelve una tupla (n_filas, n_columnas)

df.query("columna > 10")                    # Filtra filas donde 'columna' es mayor a 10

df.loc[df["edad"] > 30, ["nombre", "edad"]] # Filtra por edad y muestra solo ciertas columnas

df["columna"]                               # Selección básica de una columna

df["columna"].max()                         # Valor máximo de una columna

df["columna"].mean()                        # Promedio de los valores

df["columna"].std()                         # Desviación estándar

df["columna"].sum()                         # Suma total de la columna

df.shape[0]                                 # Número de filas del DataFrame

df["TRIMESTRE"].unique()                    # Verificar los valores únicos de la columna que indica el trimestre

# VEMOS SI HAY VALORES NULL O NaN
print(f"¿Tiene datos en null o NaN? {df.isnull().any().any()}")
print(f"¿Cuántos datos son null o NaN? {df.isnull().sum().sum()}")
print("¿Qué columnas poseen null o NaN?")
display(df.isnull().any())

df.pivot_table(index="provincia", values="casos", aggfunc="sum")  # Tabla dinámica

pivot = pd.pivot_table(
    df,
    index="BARRIO",
    columns="CUARTIL_PRECIOUSD",
    values="DIRECCION",  # o cualquier columna no nula por fila
    aggfunc="count",
    fill_value=0
)

casos_tests = df.merge(           # Realizar la fusión 
    df2,
    left_on=["residencia_provincia_id", "fecha"],
    right_on=["codigo_indec_provincia", "fecha"]
)

df.groupby("provincia")["casos"].sum()      # Agrupación por provincia y suma de casos

df.sort_values("salario", ascending=False)  # Ordena los salarios de mayor a menor

df.head(10)                                 # Muestra las primeras 10 filas

df.drop(columns=["col1", "col2"])           # Elimina columnas específicas

df.rename(columns={"sexo": "genero"})       # Cambia el nombre de una columna

df["columna"].notnull().sum()               # Cuenta cuántos valores no nulos hay

df["columna"].str.contains("texto")         # Filtra si una string aparece en la columna

df["columna"].map({"M": "Masculino", "F": "Femenino"})  # Mapea valores para reemplazo

df["id"] = df["id"].astype(str)             # Convierte la columna 'id' a string

np.random.randint(1, 100, 10)               # Genera 10 enteros aleatorios entre 1 y 99

pd.read_csv("archivo.csv")                  # Lee un archivo CSV y lo convierte en un DataFrame

pd.Series([1, 2, 3])                         # Crea una serie unidimensional (como una columna suelta)

alt.Chart(df).mark_bar().encode(            # Gráfico de barras con Altair
    alt.X("mes:O"),
    alt.Y("count():Q")
)

px.line(df, x="fecha", y="casos")           # Gráfico de líneas interactivo con Plotly

plt.gca()                                   # Obtiene el eje actual (gráfico Matplotlib)

plt.gca().spines["top"].set_visible(False)  # Oculta el borde superior del gráfico

pyeph.get(data="eph", year=2021, period=2)  # Descarga datos EPH 2° trimestre 2021

print("Hola mundo")                         # Imprime texto en consola

round(3.14159, 2)                           # Devuelve 3.14

list(range(5))                              # Convierte un rango en una lista: [0, 1, 2, 3, 4]
