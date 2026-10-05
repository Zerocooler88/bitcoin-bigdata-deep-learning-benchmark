from __future__ import annotations  # Habilita anotaciones modernas de tipos.

from pathlib import Path  # Permite manejar rutas de archivos de forma robusta.
import polars as pl  # Permite leer, filtrar y exportar datos tabulares eficientemente.


# --------------------------------------------------
# RUTAS PRINCIPALES
# --------------------------------------------------
RUTA_GLOBAL = Path(
    "data/1h/global/btcusdt_spot_1h_model_input_2019_2026.parquet"
)  # Define la ruta del dataset global de entrada.

CARPETA_REGIMENES = Path(
    "data/1h/regimenes"
)  # Define la carpeta de salida para escenarios por régimen.
CARPETA_TRIMESTRES = Path(
    "data/1h/trimestres"
)  # Define la carpeta de salida para escenarios trimestrales.
CARPETA_EXTREMOS = Path(
    "data/1h/extremos"
)  # Define la carpeta de salida para trimestres extremos.
CARPETA_RESUMENES = Path(
    "data/1h/resumenes"
)  # Define la carpeta de salida para resúmenes auxiliares.


# --------------------------------------------------
# CREACION DE CARPETAS DE SALIDA
# --------------------------------------------------
for carpeta in [  # Recorre todas las carpetas requeridas para las salidas del script.
    CARPETA_REGIMENES,  # Incluye la carpeta de regímenes.
    CARPETA_TRIMESTRES,  # Incluye la carpeta de trimestres.
    CARPETA_EXTREMOS,  # Incluye la carpeta de extremos.
    CARPETA_RESUMENES,  # Incluye la carpeta de resúmenes.
]:
    carpeta.mkdir(
        parents=True, exist_ok=True
    )  # Crea la carpeta y sus padres si aún no existen.


# --------------------------------------------------
# FUNCION PRINCIPAL
# --------------------------------------------------
def main() -> None:  # Controla el flujo principal de generación de escenarios.
    if (
        not RUTA_GLOBAL.exists()
    ):  # Verifica que el dataset global exista antes de continuar.
        raise FileNotFoundError(
            f"No existe el dataset global: {RUTA_GLOBAL}"
        )  # Detiene la ejecución si falta el archivo base.

    print("=" * 70)  # Imprime un separador visual de inicio.
    print("GENERACION DE ESCENARIOS 1H")  # Informa el proceso que se está ejecutando.
    print(f"Leyendo: {RUTA_GLOBAL}")  # Muestra la ruta del archivo de entrada.

    df = pl.read_parquet(RUTA_GLOBAL).sort(
        "open_datetime_utc"
    )  # Lee el parquet global y lo ordena cronológicamente.

    columnas_obligatorias = (
        [  # Define las columnas mínimas necesarias para construir escenarios.
            "open_datetime_utc",  # Requiere la fecha y hora de apertura.
            "year_quarter",  # Requiere la etiqueta de trimestre calendario.
            "regimen_mercado",  # Requiere la etiqueta de régimen de mercado.
            "close",  # Requiere el precio de cierre para cálculos de resumen.
        ]
    )
    faltantes = [
        c for c in columnas_obligatorias if c not in df.columns
    ]  # Identifica columnas obligatorias ausentes en el dataset.
    if faltantes:  # Verifica si faltan columnas requeridas.
        raise ValueError(
            f"Faltan columnas obligatorias: {faltantes}"
        )  # Detiene la ejecución si la estructura no es válida.

    print(f"Filas leidas: {df.height:,}")  # Informa el número total de filas leídas.
    print(f"Columnas leidas: {df.width}")  # Informa el número total de columnas leídas.

    # --------------------------------------------------
    # 1) REGIMENES
    # --------------------------------------------------
    for regimen in [
        "pre_covid",
        "covid",
        "post_covid",
    ]:  # Recorre los tres regímenes definidos para el estudio.
        df_reg = df.filter(pl.col("regimen_mercado") == regimen).sort(
            "open_datetime_utc"
        )  # Filtra el dataset del régimen actual y lo ordena por fecha.
        ruta_reg = (
            CARPETA_REGIMENES / f"btcusdt_spot_1h_{regimen}.parquet"
        )  # Construye la ruta de salida del régimen actual.
        df_reg.write_parquet(
            ruta_reg, compression="zstd"
        )  # Guarda el escenario del régimen en parquet comprimido.
        print("-" * 70)  # Imprime un separador visual entre resultados.
        print(f"Regimen: {regimen}")  # Informa el régimen procesado.
        print(
            f"Filas: {df_reg.height:,}"
        )  # Informa cuántas filas tiene el escenario del régimen.
        print(f"Archivo: {ruta_reg}")  # Muestra la ruta del archivo generado.

    # --------------------------------------------------
    # 2) TRIMESTRES INDIVIDUALES
    # --------------------------------------------------
    trimestres = (
        df.select("year_quarter").unique().sort("year_quarter").to_series().to_list()
    )  # Extrae la lista ordenada de trimestres únicos presentes en el dataset.

    for trimestre in trimestres:  # Recorre cada trimestre identificado en el dataset.
        df_trim = df.filter(pl.col("year_quarter") == trimestre).sort(
            "open_datetime_utc"
        )  # Filtra el dataset del trimestre actual y lo ordena cronológicamente.
        ruta_trim = (
            CARPETA_TRIMESTRES / f"btcusdt_spot_1h_{trimestre}_model_input.parquet"
        )  # Construye la ruta de salida del trimestre actual.
        df_trim.write_parquet(
            ruta_trim, compression="zstd"
        )  # Guarda el escenario trimestral en parquet comprimido.
        print("-" * 70)  # Imprime un separador visual entre resultados.
        print(f"Trimestre: {trimestre}")  # Informa el trimestre procesado.
        print(
            f"Filas: {df_trim.height:,}"
        )  # Informa cuántas filas tiene el escenario trimestral.
        print(f"Archivo: {ruta_trim}")  # Muestra la ruta del archivo generado.

    # --------------------------------------------------
    # 3) RESUMEN TRIMESTRAL PARA HALLAR EXTREMOS
    # --------------------------------------------------
    resumen_trim = (  # Inicia la construcción del resumen agregado por trimestre.
        df.group_by("year_quarter")  # Agrupa el dataset por trimestre calendario.
        .agg(  # Calcula métricas descriptivas por grupo.
            [
                pl.len().alias("filas"),  # Cuenta cuántas filas tiene cada trimestre.
                pl.col("close")
                .mean()
                .alias("close_promedio"),  # Calcula el cierre promedio por trimestre.
                pl.col("close")
                .min()
                .alias("close_minimo"),  # Calcula el cierre mínimo por trimestre.
                pl.col("close")
                .max()
                .alias("close_maximo"),  # Calcula el cierre máximo por trimestre.
                pl.col("open_datetime_utc")
                .min()
                .alias("fecha_min"),  # Obtiene la fecha mínima del trimestre.
                pl.col("open_datetime_utc")
                .max()
                .alias("fecha_max"),  # Obtiene la fecha máxima del trimestre.
            ]
        )
        .sort("year_quarter")  # Ordena el resumen por trimestre.
    )

    ruta_resumen = (
        CARPETA_RESUMENES / "resumen_trimestres_close.csv"
    )  # Define la ruta del resumen trimestral en CSV.
    resumen_trim.write_csv(ruta_resumen)  # Guarda el resumen trimestral en formato CSV.
    print("-" * 70)  # Imprime un separador visual.
    print(
        f"Resumen trimestral guardado en: {ruta_resumen}"
    )  # Informa la ubicación del resumen exportado.

    # --------------------------------------------------
    # 4) TRIMESTRE MAS BAJO Y TRIMESTRE MAS ALTO
    #    CRITERIO: CLOSE_PROMEDIO
    # --------------------------------------------------
    trimestre_bajo = (  # Inicia la obtención del trimestre con menor cierre promedio.
        resumen_trim.sort(
            "close_promedio"
        )  # Ordena de menor a mayor por cierre promedio.
        .select("year_quarter")  # Conserva solo la columna de trimestre.
        .to_series()  # Convierte el resultado a serie.
        .to_list()[0]  # Extrae el primer trimestre de la lista ordenada.
    )

    trimestre_alto = (  # Inicia la obtención del trimestre con mayor cierre promedio.
        resumen_trim.sort(
            "close_promedio", descending=True
        )  # Ordena de mayor a menor por cierre promedio.
        .select("year_quarter")  # Conserva solo la columna de trimestre.
        .to_series()  # Convierte el resultado a serie.
        .to_list()[0]  # Extrae el primer trimestre de la lista ordenada.
    )

    df_bajo = df.filter(pl.col("year_quarter") == trimestre_bajo).sort(
        "open_datetime_utc"
    )  # Filtra y ordena el trimestre con menor cierre promedio.
    df_alto = df.filter(pl.col("year_quarter") == trimestre_alto).sort(
        "open_datetime_utc"
    )  # Filtra y ordena el trimestre con mayor cierre promedio.

    ruta_bajo = (
        CARPETA_EXTREMOS
        / f"btcusdt_spot_1h_trimestre_mas_bajo_{trimestre_bajo}.parquet"
    )  # Construye la ruta de salida del trimestre más bajo.
    ruta_alto = (
        CARPETA_EXTREMOS
        / f"btcusdt_spot_1h_trimestre_mas_alto_{trimestre_alto}.parquet"
    )  # Construye la ruta de salida del trimestre más alto.

    df_bajo.write_parquet(
        ruta_bajo, compression="zstd"
    )  # Guarda el trimestre más bajo en parquet comprimido.
    df_alto.write_parquet(
        ruta_alto, compression="zstd"
    )  # Guarda el trimestre más alto en parquet comprimido.

    print("-" * 70)  # Imprime un separador visual.
    print(
        f"Trimestre mas bajo por close_promedio: {trimestre_bajo}"
    )  # Informa el trimestre identificado como más bajo.
    print(
        f"Archivo: {ruta_bajo}"
    )  # Muestra la ruta del archivo del trimestre más bajo.
    print(
        f"Trimestre mas alto por close_promedio: {trimestre_alto}"
    )  # Informa el trimestre identificado como más alto.
    print(
        f"Archivo: {ruta_alto}"
    )  # Muestra la ruta del archivo del trimestre más alto.

    print("=" * 70)  # Imprime un separador visual de cierre.
    print(
        "ESCENARIOS GENERADOS CORRECTAMENTE"
    )  # Informa la finalización exitosa del proceso.
    print("=" * 70)  # Imprime un separador visual final.


# --------------------------------------------------
# PUNTO DE ENTRADA
# --------------------------------------------------
if __name__ == "__main__":  # Verifica que el archivo se ejecute directamente.
    main()  # Ejecuta la función principal del script.
