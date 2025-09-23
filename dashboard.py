# ===========================
# DataAqua Dashboard (Streamlit)
# ===========================
# Muestra interactiva de resultados UNISON:
# - Selección de Región y Ciclo
# - KPIs del ciclo
# - Gráficas: Serie diaria, Acumulados, Decádico, Kc–ET0 y Drivers de ET0
#
# Fuente de datos:
# RUTA_SALIDA_UNISON = /lustre/home/mvalenzuela/Ocotillo/DataAqua/Salidas_ETo12/Periodo de Cultivo ETo
# Estructura esperada:
#   <Region>/<Region>-FAO56-<YYYY>[-<YYYY>]-SALIDA.csv
#
# Ejecuta con:
#   streamlit run dashboard.py
# ===========================

from pathlib import Path
import os, re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

# ---------------------------
# CONFIGURACIÓN BÁSICA
# ---------------------------
st.set_page_config(
    page_title="DataAqua — Dashboard",
    page_icon="💧",
    layout="wide"
)

# Rutas (usa tus rutas reales)
RUTA_BASE          = Path("/lustre/home/mvalenzuela/Ocotillo/DataAqua")
RUTA_SALIDA_UNISON = RUTA_BASE / "Salidas_ETo12" / "Periodo de Cultivo ETo"

# Estilos
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------
# MAPEO DE COLUMNAS (mojibake y etiquetas)
# ---------------------------
MAP_UNISON = {
    # mojibake
    "Año_ (YEAR)": "Year", "AÃ±o_ (YEAR)": "Year",
    "Día (DOY)": "DOY",   "DÃ­a (DOY)": "DOY",

    # base NASA renombrada en tu pipeline
    "Tmax (T2M_MAX)": "Tmax", "Tmin (T2M_MIN)": "Tmin",
    "HR (RH2M)": "HR", "Ux (WS2M)": "Ux",
    "Rs (ALLSKY_SFC_SW_DWN)": "Rs",
    "Rl_ (ALLSKY_SFC_LW_DWN)": "Rl",
    "Ptot_ (PRECTOTCORR)": "Ptot",

    # generadas / resultados
    "Pef_": "Pef", "Tmean_": "Tmean", "es_": "es", "ea_": "ea",
    "delta_": "delta", "P_": "P", "gamma_": "gamma",
    "Rns_": "Rns", "Rnl_": "Rnl", "Rn_": "Rn", "Rso_": "Rso",
    "Kc_": "Kc", "decada_": "decada",

    # finales
    "ET0": "ET0", "ETc": "ETc", "ETverde": "ETverde", "ETazul": "ETazul",
    # por si ya vienen así
    "Year": "Year", "DOY": "DOY", "Dia": "Dia"
}

COLUMNAS_MIN = ["Year","DOY","ET0","ETc","ETverde","ETazul","Pef","decada","Rns","Rnl","Rs","Tmean","HR","Ux","Kc"]

# ---------------------------
# PARSER DE NOMBRE DE ARCHIVO
# ---------------------------
def parse_unison_filename(filename: str):
    """
    'Cajeme-FAO56-2014-2015-SALIDA.csv' -> ('Cajeme','2014-2015')
    'Metepec-FAO56-2014-SALIDA.csv'     -> ('Metepec','2014')
    """
    m = re.match(r"([A-Za-z]+)-FAO56-(\d{4})(?:-(\d{4}))?-SALIDA\.csv$", filename, re.I)
    if not m:
        return None, None
    reg, y1, y2 = m.groups()
    if reg == "VillaAllende": reg = "Villa de Allende"
    if reg == "Etchhojoa":    reg = "Etchojoa"
    ciclo = y1 if not y2 else f"{y1}-{y2}"
    return reg, ciclo

# ---------------------------
# CATÁLOGO (con cache)
# ---------------------------
@st.cache_data(show_spinner=False)
def catalogo_unison(base_dir: Path) -> pd.DataFrame:
    regs = []
    if not base_dir.exists():
        return pd.DataFrame(columns=["Region","Ciclo","Ruta"])
    for reg_folder in sorted(os.listdir(base_dir)):
        d = base_dir / reg_folder
        if not d.is_dir():
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".csv"):
                continue
            reg, ciclo = parse_unison_filename(f)
            if reg and ciclo:
                regs.append({"Region": reg, "Ciclo": ciclo, "Ruta": str(d / f)})
    df = pd.DataFrame(regs).sort_values(["Region","Ciclo"]).reset_index(drop=True)
    return df

# ---------------------------
# LECTOR ROBUSTO (con cache)
# ---------------------------
def _year_doy_to_date(y, doy):
    try:
        base = datetime(int(y), 1, 1)
        return base + timedelta(days=int(doy) - 1)
    except Exception:
        return pd.NaT

@st.cache_data(show_spinner=False)
def leer_unison(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    last_err = None
    for enc in ("utf-8","latin-1"):
        try:
            df = pd.read_csv(p, encoding=enc)
            last_err = None
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err is not None:
        df = pd.read_csv(p)

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=lambda c: MAP_UNISON.get(c, c))

    for c in set(COLUMNAS_MIN).intersection(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fecha y Dia_ciclo desde YEAR/DOY
    if {"Year","DOY"}.issubset(df.columns):
        fechas = [_year_doy_to_date(y,d) for y,d in zip(df["Year"], df["DOY"])]
        df["Fecha"] = pd.to_datetime(fechas)
        if df["Fecha"].notna().any():
            f0 = df["Fecha"].dropna().iloc[0]
            df["Dia_ciclo"] = (df["Fecha"] - f0).dt.days.astype("Int64")
        else:
            df["Dia_ciclo"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    else:
        df["Fecha"] = pd.NaT
        df["Dia_ciclo"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # Acumulados y proporciones útiles en panel
    if "ETc" in df:
        df["ETc_acum"] = df["ETc"].cumsum()
    if "ETazul" in df:
        df["ETazul_acum"] = df["ETazul"].cumsum()
    if {"ETc","ETazul"}.issubset(df.columns):
        df["pct_azul"] = np.where(df["ETc"]>0, df["ETazul"]/df["ETc"]*100.0, np.nan)

    return df

# ---------------------------
# KPIs
# ---------------------------
def kpis_ciclo(df: pd.DataFrame) -> dict:
    mask = df["ETc"].notna() if "ETc" in df else pd.Series(False, index=df.index)
    dias = int(mask.sum())

    etc_total = float(df.loc[mask, "ETc"].sum())     if "ETc"     in df else np.nan
    etv_total = float(df.loc[mask, "ETverde"].sum()) if "ETverde" in df else np.nan
    eta_total = float(df.loc[mask, "ETazul"].sum())  if "ETazul"  in df else np.nan
    pef_total = float(df.loc[mask, "Pef"].sum())     if "Pef"     in df else np.nan

    pct_azul = (eta_total/etc_total*100.0) if (etc_total and etc_total>0) else np.nan
    dias_def = int(((df["ETc"] > df.get("Pef", 0)).fillna(False)).sum()) if "ETc" in df else np.nan
    pico_p95 = float(np.nanpercentile(df["ETc"], 95)) if "ETc" in df else np.nan

    return {
        "dias": dias,
        "etc_total": etc_total,
        "etv_total": etv_total,
        "eta_total": eta_total,
        "pct_azul": pct_azul,
        "pef_total": pef_total,
        "dias_def": dias_def,
        "pico_p95": pico_p95
    }

# ---------------------------
# Helpers de ejes y figuras
# ---------------------------
def _xcol(df: pd.DataFrame, prefer="Fecha"):
    if prefer in df.columns and df[prefer].notna().any():
        return prefer
    for alt in ("Fecha","DOY","Dia_ciclo"):
        if alt in df.columns and df[alt].notna().any():
            return alt
    return df.index

def fig_series(df: pd.DataFrame, region: str, ciclo: str, eje="Fecha", mostrar=("ET0","ETc","ETverde","ETazul","Pef")):
    x = _xcol(df, eje)
    fig, ax = plt.subplots(1,1, figsize=(12,4))
    colores = {"ET0":"#4C78A8","ETc":"#F58518","ETverde":"#54A24B","ETazul":"#E45756","Pef":"#9D9D9D"}
    for col in mostrar:
        if col in df:
            ax.plot(df[x], df[col], label=col, lw=1.6, color=colores.get(col, None))
    ax.set_title(f"Serie diaria — {region} ({ciclo})")
    ax.set_xlabel(str(x)); ax.set_ylabel("mm/día")
    ax.legend()
    fig.tight_layout()
    return fig

def fig_acumulados(df: pd.DataFrame, region: str, ciclo: str, eje="Fecha"):
    x = _xcol(df, eje)
    fig, ax = plt.subplots(1,1, figsize=(12,4))
    if "ETc_acum" in df:
        ax.plot(df[x], df["ETc_acum"], label="ETc acumulado", lw=1.8)
    if "ETazul_acum" in df:
        ax.plot(df[x], df["ETazul_acum"], label="ETazul acumulado", lw=1.8)
    ax.set_title(f"Acumulados — {region} ({ciclo})")
    ax.set_xlabel(str(x)); ax.set_ylabel("mm")
    ax.legend()
    fig.tight_layout()
    return fig

def fig_decadico(df: pd.DataFrame, region: str, ciclo: str):
    if "decada" not in df: 
        return None
    g = df.groupby("decada")[["ETc","ETazul"]].sum(min_count=1)
    fig, ax = plt.subplots(1,1, figsize=(10,4))
    g["ETc"].plot(kind="bar", ax=ax, color="#4C78A8", label="ETc")
    if "ETazul" in g:
        ax.plot(np.arange(len(g)), g["ETazul"].values, color="#F58518", lw=2, marker="o", label="ETazul")
    ax.set_title(f"Decádico — {region} ({ciclo})")
    ax.set_xlabel("Década del ciclo"); ax.set_ylabel("mm/decada")
    ax.legend()
    fig.tight_layout()
    return fig

def fig_kc_et0(df: pd.DataFrame, region: str, ciclo: str, eje="Fecha"):
    if "Kc" not in df or "ET0" not in df:
        return None
    x = _xcol(df, eje)
    fig, ax1 = plt.subplots(1,1, figsize=(12,4))
    ax1.plot(df[x], df["ET0"], color="#4C78A8", label="ET0", lw=1.5)
    ax1.set_ylabel("ET0 [mm/día]", color="#4C78A8"); ax1.tick_params(axis='y', labelcolor="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(df[x], df["Kc"], color="#E45756", label="Kc", lw=1.5)
    ax2.set_ylabel("Kc [-]", color="#E45756"); ax2.tick_params(axis='y', labelcolor="#E45756")
    ax1.set_title(f"Kc y ET0 — {region} ({ciclo})"); ax1.set_xlabel(str(x))
    fig.tight_layout()
    return fig

def fig_drivers_et0(df: pd.DataFrame, region: str, ciclo: str):
    if "ET0" not in df: 
        return None
    drivers = [("Rs","Rs [MJ m$^{-2}$ d$^{-1}$]"),
               ("Rnl","Rnl [MJ m$^{-2}$ d$^{-1}$]"),
               ("HR","HR [%]"),
               ("Ux","Viento Ux [m/s]"),
               ("Tmean","Tmean [°C]")]
    cols = [c for c,_ in drivers if c in df.columns]
    if not cols:
        return None
    n = len(cols); ncols, nrows = 3, int(np.ceil(n/3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5*nrows))
    axes = np.atleast_2d(axes).ravel()
    for i,(c,lab) in enumerate([d for d in drivers if d[0] in cols]):
        ax = axes[i]
        ax.scatter(df[c], df["ET0"], alpha=0.6, s=14)
        ax.set_xlabel(lab); ax.set_ylabel("ET0 [mm/día]")
        try:
            r = np.corrcoef(df[c].values, df["ET0"].values)[0,1]
            ax.set_title(f"ET0 vs {c}  (r={r:.2f})")
        except Exception:
            ax.set_title(f"ET0 vs {c}")
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    return fig

# ---------------------------
# UI — SIDEBAR
# ---------------------------
st.sidebar.title("DataAqua — Selección")

# Construir catálogo de archivos
CAT_UNISON = catalogo_unison(RUTA_SALIDA_UNISON)
if CAT_UNISON.empty:
    st.error("No se encontraron archivos en Salidas_ETo12/Periodo de Cultivo ETo.")
    st.stop()

# Selectores
regiones = sorted(CAT_UNISON["Region"].unique())
region_sel = st.sidebar.selectbox("Región:", regiones)

ciclos_reg = CAT_UNISON.loc[CAT_UNISON["Region"]==region_sel, "Ciclo"].unique()
ciclo_sel = st.sidebar.selectbox("Ciclo:", sorted(ciclos_reg))

# Eje temporal preferido
eje_opt = st.sidebar.radio("Eje X:", ["Fecha","DOY","Dia_ciclo"], index=0)

# Selección de gráficas
st.sidebar.markdown("#### Gráficas a mostrar")
show_series     = st.sidebar.checkbox("Serie diaria (ET0, ETc, ETverde, ETazul, Pef)", value=True)
show_acumulados = st.sidebar.checkbox("Acumulados (ETc, ETazul)", value=True)
show_decadico   = st.sidebar.checkbox("Decádico (ETc + línea ETazul)", value=True)
show_kc         = st.sidebar.checkbox("Kc y ET0", value=True)
show_drivers    = st.sidebar.checkbox("Drivers de ET0 (scatter)", value=True)

# ---------------------------
# CARGA DEL CICLO SELECCIONADO
# ---------------------------
ruta_sel = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
if ruta_sel.empty:
    st.error(f"No encontré CSV para {region_sel} / {ciclo_sel}")
    st.stop()

df = leer_unison(ruta_sel.iloc[0])
if df.empty:
    st.error("No fue posible leer el archivo seleccionado.")
    st.stop()

# ---------------------------
# CABECERA
# ---------------------------
st.title("💧 DataAqua — Dashboard de Ciclo de Cultivo")
st.caption("Resultados UNISON (FAO-56). ETc, ETverde y ETazul son magnitudes del cultivo; ET0 es la referencia (césped).")

st.subheader(f"Región: **{region_sel}** — Ciclo: **{ciclo_sel}**")

# ---------------------------
# KPIs
# ---------------------------
k = kpis_ciclo(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Días del ciclo", f"{k['dias']}")
col2.metric("ETc total [mm]", f"{k['etc_total']:.1f}")
col3.metric("ETazul total [mm]", f"{k['eta_total']:.1f}")
col4.metric("% Azul", f"{k['pct_azul']:.1f}%")

col5, col6, col7, col8 = st.columns(4)
col5.metric("ETverde total [mm]", f"{k['etv_total']:.1f}")
col6.metric("Pef total [mm]", f"{k['pef_total']:.1f}")
col7.metric("Días con déficit (ETc>Pef)", f"{k['dias_def']}")
col8.metric("Pico ETc p95 [mm/d]", f"{k['pico_p95']:.2f}")

st.divider()

# ---------------------------
# GRÁFICAS
# ---------------------------
if show_series:
    fig = fig_series(df, region_sel, ciclo_sel, eje=eje_opt)
    st.pyplot(fig, use_container_width=True)

if show_acumulados:
    fig = fig_acumulados(df, region_sel, ciclo_sel, eje=eje_opt)
    st.pyplot(fig, use_container_width=True)

if show_decadico:
    fdec = fig_decadico(df, region_sel, ciclo_sel)
    if fdec is not None:
        st.pyplot(fdec, use_container_width=True)
    else:
        st.info("No hay columna 'decada' en este archivo.")

if show_kc:
    fkc = fig_kc_et0(df, region_sel, ciclo_sel, eje=eje_opt)
    if fkc is not None:
        st.pyplot(fkc, use_container_width=True)
    else:
        st.info("Faltan columnas 'Kc' o 'ET0' para esta gráfica.")

if show_drivers:
    fdrv = fig_drivers_et0(df, region_sel, ciclo_sel)
    if fdrv is not None:
        st.pyplot(fdrv, use_container_width=True)
    else:
        st.info("Faltan columnas para analizar drivers (ET0 y Rs/Rnl/HR/Ux/Tmean).")

st.divider()

# ---------------------------
# DATOS y DESCARGA
# ---------------------------
with st.expander("Ver primeras filas de los datos"):
    st.dataframe(df.head(20), use_container_width=True)

@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Descargar CSV del ciclo filtrado",
    data=to_csv_bytes(df),
    file_name=f"{region_sel}_{ciclo_sel}_DataAqua.csv",
    mime="text/csv"
)
