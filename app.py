import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import branca.colormap as cm

st.set_page_config(
    page_title="Análisis de proyectos de fotogrametría UAV",
    page_icon="🛰️",
    layout="wide"
)

@st.cache_data
def cargar_datos(ruta: str = "datos.csv") -> pd.DataFrame:
    """Carga los datos del proyecto final. El CSV original usa punto y coma como separador."""
    df = pd.read_csv(ruta, sep=";")
    columnas_numericas = [
        "Altura_Vuelo_m", "Imagenes", "Resolucion_cm_px",
        "Tiempo_Procesamiento_min", "Latitud", "Longitud"
    ]
    for columna in columnas_numericas:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    return df.dropna(subset=["Latitud", "Longitud"])


def interpretar_datos(df_base: pd.DataFrame) -> dict:
    """Calcula indicadores concretos para acompañar la tabla, gráficos y mapas."""
    tabla_software = (
        df_base.groupby("Software")
        .agg(
            Proyectos=("Proyecto", "count"),
            Tiempo_promedio_min=("Tiempo_Procesamiento_min", "mean"),
            Imagenes_promedio=("Imagenes", "mean"),
            Resolucion_promedio_cm_px=("Resolucion_cm_px", "mean"),
            Altura_promedio_m=("Altura_Vuelo_m", "mean")
        )
        .round(2)
        .sort_values("Tiempo_promedio_min")
    )

    correlacion = df_base["Imagenes"].corr(df_base["Tiempo_Procesamiento_min"])
    pendiente, intercepto = np.polyfit(df_base["Imagenes"], df_base["Tiempo_Procesamiento_min"], 1)
    mas_rapido = tabla_software["Tiempo_promedio_min"].idxmin()
    mas_lento = tabla_software["Tiempo_promedio_min"].idxmax()
    proyecto_mayor_tiempo = df_base.loc[df_base["Tiempo_Procesamiento_min"].idxmax()]
    proyecto_menor_tiempo = df_base.loc[df_base["Tiempo_Procesamiento_min"].idxmin()]

    return {
        "tabla_software": tabla_software,
        "correlacion": correlacion,
        "pendiente": pendiente,
        "mas_rapido": mas_rapido,
        "tiempo_rapido": tabla_software.loc[mas_rapido, "Tiempo_promedio_min"],
        "mas_lento": mas_lento,
        "tiempo_lento": tabla_software.loc[mas_lento, "Tiempo_promedio_min"],
        "proyecto_mayor_tiempo": proyecto_mayor_tiempo,
        "proyecto_menor_tiempo": proyecto_menor_tiempo,
    }


def construir_mapa(df_filtrado: pd.DataFrame, variable: str, tipo_mapa: str) -> folium.Map:
    centro = [df_filtrado["Latitud"].mean(), df_filtrado["Longitud"].mean()]
    mapa = folium.Map(location=centro, zoom_start=8, tiles="OpenStreetMap")

    if df_filtrado.empty:
        return mapa

    min_val = float(df_filtrado[variable].min())
    max_val = float(df_filtrado[variable].max())
    if min_val == max_val:
        min_val = min_val - 1
        max_val = max_val + 1

    colormap = cm.linear.YlOrRd_09.scale(min_val, max_val)
    colormap.caption = f"{variable}"
    colormap.add_to(mapa)

    if tipo_mapa == "Mapa de calor del tiempo de procesamiento":
        heat_data = df_filtrado[["Latitud", "Longitud", "Tiempo_Procesamiento_min"]].values.tolist()
        HeatMap(heat_data, radius=28, blur=18, min_opacity=0.35).add_to(mapa)

    cluster = MarkerCluster(name="Proyectos fotogramétricos").add_to(mapa)
    for _, row in df_filtrado.iterrows():
        valor = float(row[variable])
        radio = 6 + (valor - min_val) / (max_val - min_val) * 12
        popup = f"""
        <b>Proyecto:</b> {row['Proyecto']}<br>
        <b>Tipo de terreno:</b> {row['Tipo_Terreno']}<br>
        <b>Software:</b> {row['Software']}<br>
        <b>Altura de vuelo:</b> {row['Altura_Vuelo_m']} m<br>
        <b>Imágenes:</b> {row['Imagenes']}<br>
        <b>Resolución:</b> {row['Resolucion_cm_px']} cm/px<br>
        <b>Tiempo de procesamiento:</b> {row['Tiempo_Procesamiento_min']} min
        """
        folium.CircleMarker(
            location=[row["Latitud"], row["Longitud"]],
            radius=radio,
            popup=folium.Popup(popup, max_width=330),
            tooltip=f"{row['Proyecto']} | {variable}: {valor}",
            color=colormap(valor),
            fill=True,
            fill_color=colormap(valor),
            fill_opacity=0.78,
            weight=2,
        ).add_to(cluster)

    folium.LayerControl().add_to(mapa)
    return mapa


df = cargar_datos()
indicadores = interpretar_datos(df)

st.title("Análisis espacial de proyectos de fotogrametría con drones")

st.markdown("""
Esta aplicación convierte la **Tarea 3 de Fotogrametría** en una aplicación web interactiva desarrollada con Streamlit. El análisis integra procesamiento tabular con **pandas**, visualización estadística con **plotly** y cartografía interactiva con **folium** mediante `streamlit-folium`.

La base de datos contiene diez proyectos fotogramétricos UAV localizados en Costa Rica. Para cada proyecto se registran el tipo de terreno, altura de vuelo, cantidad de imágenes, resolución espacial, tiempo de procesamiento, software empleado y coordenadas geográficas. Las coordenadas provienen del archivo de datos de la tarea; por ello, el mapa debe interpretarse como una visualización académica para integrar análisis tabular, gráfico y espacial, no como un inventario oficial de proyectos reales.
""")

st.info(
    "Mejora incorporada respecto a la revisión: los mapas ahora codifican variables mediante tamaño y color de los marcadores, "
    "y las interpretaciones incluyen valores concretos, como el software más rápido, el más lento y la relación imágenes-tiempo."
)

st.sidebar.header("Filtros interactivos")
software_sel = st.sidebar.multiselect(
    "Software de procesamiento",
    options=sorted(df["Software"].unique()),
    default=sorted(df["Software"].unique())
)
terreno_sel = st.sidebar.multiselect(
    "Tipo de terreno",
    options=sorted(df["Tipo_Terreno"].unique()),
    default=sorted(df["Tipo_Terreno"].unique())
)
min_img, max_img = int(df["Imagenes"].min()), int(df["Imagenes"].max())
rango_imagenes = st.sidebar.slider(
    "Rango de cantidad de imágenes",
    min_value=min_img,
    max_value=max_img,
    value=(min_img, max_img),
    step=10
)
variable_mapa = st.sidebar.radio(
    "Variable para simbolizar el mapa",
    ["Tiempo_Procesamiento_min", "Imagenes", "Resolucion_cm_px", "Altura_Vuelo_m"],
    index=0
)
tipo_mapa = st.sidebar.radio(
    "Tipo de mapa",
    ["Marcadores proporcionales", "Mapa de calor del tiempo de procesamiento"],
    index=0
)

filtro = (
    df["Software"].isin(software_sel)
    & df["Tipo_Terreno"].isin(terreno_sel)
    & df["Imagenes"].between(rango_imagenes[0], rango_imagenes[1])
)
df_filtrado = df.loc[filtro].copy()

if df_filtrado.empty:
    st.warning("No hay registros para la combinación de filtros seleccionada. Ajuste los filtros del panel lateral.")
    st.stop()

st.subheader("Resumen general del conjunto filtrado")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Proyectos", len(df_filtrado))
col2.metric("Tiempo promedio", f"{df_filtrado['Tiempo_Procesamiento_min'].mean():.1f} min")
col3.metric("Imágenes promedio", f"{df_filtrado['Imagenes'].mean():.0f}")
col4.metric("Resolución promedio", f"{df_filtrado['Resolucion_cm_px'].mean():.2f} cm/px")

st.markdown("""
## Descripción del conjunto de datos

El conjunto de datos contiene información de diez proyectos fotogramétricos e incluye las siguientes variables:

- **Proyecto:** nombre del proyecto fotogramétrico.
- **Tipo_Terreno:** categoría temática del área de estudio.
- **Altura_Vuelo_m:** altura de vuelo utilizada durante la captura de imágenes.
- **Imagenes:** cantidad de fotografías obtenidas.
- **Resolucion_cm_px:** resolución espacial alcanzada.
- **Tiempo_Procesamiento_min:** tiempo total requerido para el procesamiento.
- **Software:** plataforma utilizada para el procesamiento fotogramétrico.
- **Latitud y Longitud:** coordenadas utilizadas para la representación cartográfica.
""")

st.markdown("## Tabla de análisis con pandas")
st.markdown(
    "La tabla resume los indicadores promedio por software. Este elemento reutiliza el agrupamiento de la Tarea 3, "
    "pero agrega conteo de proyectos y altura promedio para fortalecer la lectura comparativa."
)
tabla_filtrada = (
    df_filtrado.groupby("Software")
    .agg(
        Proyectos=("Proyecto", "count"),
        Tiempo_promedio_min=("Tiempo_Procesamiento_min", "mean"),
        Imagenes_promedio=("Imagenes", "mean"),
        Resolucion_promedio_cm_px=("Resolucion_cm_px", "mean"),
        Altura_promedio_m=("Altura_Vuelo_m", "mean")
    )
    .round(2)
    .sort_values("Tiempo_promedio_min")
)
st.dataframe(tabla_filtrada, use_container_width=True)

st.markdown(
    f"En el conjunto completo, **{indicadores['mas_rapido']}** es el software con menor tiempo promedio "
    f"(**{indicadores['tiempo_rapido']:.1f} min**), mientras que **{indicadores['mas_lento']}** presenta el mayor tiempo promedio "
    f"(**{indicadores['tiempo_lento']:.1f} min**). Esta comparación corrige la interpretación genérica de la tarea original e incorpora valores concretos."
)

st.markdown("## Gráficos estadísticos")
col_g1, col_g2 = st.columns(2)

with col_g1:
    fig_scatter = px.scatter(
        df_filtrado,
        x="Imagenes",
        y="Tiempo_Procesamiento_min",
        color="Software",
        size="Resolucion_cm_px",
        hover_name="Proyecto",
        trendline="ols" if len(df_filtrado) >= 2 else None,
        title="Relación entre imágenes y tiempo de procesamiento"
    )
    fig_scatter.update_layout(xaxis_title="Cantidad de imágenes", yaxis_title="Tiempo de procesamiento (min)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_g2:
    promedio_filtrado = df_filtrado.groupby("Software", as_index=False)["Tiempo_Procesamiento_min"].mean().round(2)
    fig_bar = px.bar(
        promedio_filtrado,
        x="Software",
        y="Tiempo_Procesamiento_min",
        text="Tiempo_Procesamiento_min",
        title="Tiempo promedio de procesamiento por software"
    )
    fig_bar.update_layout(xaxis_title="Software", yaxis_title="Tiempo promedio (min)")
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown(
    f"Para el conjunto completo, la correlación entre cantidad de imágenes y tiempo de procesamiento es de "
    f"**{indicadores['correlacion']:.2f}**, lo que evidencia una relación positiva fuerte. La recta de tendencia estima "
    f"aproximadamente **{indicadores['pendiente']:.2f} minutos adicionales por cada imagen**; equivalente a cerca de "
    f"**{indicadores['pendiente']*100:.1f} minutos por cada 100 imágenes**. El proyecto con mayor tiempo es "
    f"**{indicadores['proyecto_mayor_tiempo']['Proyecto']}** con **{indicadores['proyecto_mayor_tiempo']['Tiempo_Procesamiento_min']:.0f} min** y "
    f"**{indicadores['proyecto_mayor_tiempo']['Imagenes']:.0f} imágenes**; el menor tiempo corresponde a "
    f"**{indicadores['proyecto_menor_tiempo']['Proyecto']}** con **{indicadores['proyecto_menor_tiempo']['Tiempo_Procesamiento_min']:.0f} min**."
)

st.markdown("## Mapa interactivo")
st.markdown(
    "El mapa representa la distribución espacial de los proyectos. A diferencia del mapa original con marcadores idénticos, "
    "esta versión codifica una variable seleccionada mediante tamaño y color del marcador. Además, puede activarse un mapa de calor "
    "basado en el tiempo de procesamiento."
)
mapa = construir_mapa(df_filtrado, variable_mapa, tipo_mapa)
st_folium(mapa, width=None, height=560)

mayor_filtrado = df_filtrado.loc[df_filtrado["Tiempo_Procesamiento_min"].idxmax()]
st.markdown(
    f"Con los filtros actuales, el mayor tiempo de procesamiento se observa en **{mayor_filtrado['Proyecto']}** "
    f"(**{mayor_filtrado['Tiempo_Procesamiento_min']:.0f} min**, software **{mayor_filtrado['Software']}**). "
    "La lectura espacial debe centrarse en la comparación relativa entre proyectos filtrados, ya que la base corresponde a un ejercicio académico."
)

st.markdown("## Discusión")
st.markdown("""
Los resultados muestran una asociación directa entre el volumen de imágenes capturadas y el tiempo de procesamiento requerido. Este comportamiento es coherente con los flujos fotogramétricos UAV, porque un mayor número de fotografías incrementa las tareas de alineamiento, generación de nube de puntos, construcción de superficies, ortorrectificación y exportación de productos.

También se observan diferencias entre softwares. En el conjunto completo, Drone2Map registra el menor tiempo promedio de procesamiento, con 81.5 minutos, mientras que OpenDroneMap presenta el mayor promedio, con 125 minutos. Esta diferencia no debe interpretarse únicamente como rendimiento del software, porque cada registro también depende del número de imágenes, resolución, altura de vuelo y tipo de terreno.

Desde la perspectiva espacial, el mapa permite explorar de manera integrada la ubicación, el tipo de terreno, el software y la intensidad de variables como tiempo de procesamiento o cantidad de imágenes. La mejora principal respecto a la Tarea 3 consiste en que la simbología ya no muestra todos los proyectos de forma idéntica, sino que asigna tamaño y color de acuerdo con una variable analítica.
""")

st.markdown("## Conclusiones")
st.markdown("""
1. Streamlit permite transformar un notebook de análisis en una aplicación web interactiva, integrando pandas, plotly y folium en una sola interfaz.
2. Pandas facilita el resumen estadístico por software y permite actualizar la tabla de forma dinámica según los filtros definidos por el usuario.
3. Plotly permite visualizar la relación entre cantidad de imágenes y tiempo de procesamiento, así como comparar el tiempo promedio por software.
4. Folium y streamlit-folium permiten representar los proyectos en un mapa interactivo; en esta versión, el mapa mejora su valor analítico mediante marcadores proporcionales y escala de color.
5. La aplicación incorpora filtros por software, tipo de terreno y cantidad de imágenes, por lo que la tabla, los gráficos y el mapa se actualizan simultáneamente.
""")

st.markdown("## Referencias bibliográficas")
st.markdown("""
Colomina, I., & Molina, P. (2014). Unmanned aerial systems for photogrammetry and remote sensing: A review. *ISPRS Journal of Photogrammetry and Remote Sensing, 92*, 79–97. https://doi.org/10.1016/j.isprsjprs.2014.02.013

Goodchild, M. F. (2007). Citizens as sensors: The world of volunteered geography. *GeoJournal, 69*(4), 211–221. https://doi.org/10.1007/s10708-007-9111-y

Lillesand, T. M., Kiefer, R. W., & Chipman, J. W. (2015). *Remote sensing and image interpretation* (7th ed.). Wiley.

McKinney, W. (2022). *Python for data analysis* (3rd ed.). O'Reilly Media.

OpenStreetMap Contributors. (2025). OpenStreetMap. https://www.openstreetmap.org

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Reitz, P., & Biocca, M. (2023). *Folium documentation*. https://python-visualization.github.io/folium/

Van Rossum, G., & Drake, F. (2009). *Python 3 Reference Manual*. CreateSpace.

Westra, E. (2022). *Python geospatial development* (4th ed.). Packt Publishing.

Zar, J. H. (2010). *Biostatistical analysis* (5th ed.). Pearson Education.
""")
