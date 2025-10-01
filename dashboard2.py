# ===========================
# DataAqua Dashboard 2 (Streamlit)
# ===========================
# Modos:
#  - Ciclo individual
#  - Comparar ciclos (misma región)
#  - Comparar regiones (mismo ciclo)
#
# Secciones (sin pestañas):
#  - KPIs (compactos 5+5 en Individual; A/B en comparaciones)
#  - Serie diaria (ET0, ETc, ETverde, ETazul, Pef)
#  - Temperaturas (Tmin, Tmean, Tmax)
#  - Meteorología (Rs + HR, ejes gemelos)
#  - Viento (Ux) aparte
#
# Ejecuta:
#   streamlit run dashboard2.py
# ===========================

from pathlib import Path
import os, re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="DataAqua — Dashboard 2", page_icon="💧", layout="wide")

# Carpeta de datos relativa al repo
RUTA_SALIDA_UNISON = Path("data") / "Salidas_ETo12_con_uac_y_hh" / "Periodo de Cultivo ETo"

plt.rcParams["figure.dpi"] = 120

# ===========================
# Apariencia global / helpers
# ===========================
PLT_TITLE = 13     # tamaño consistente para títulos
PLT_LABEL = 10     # etiquetas de ejes y ticks
PLT_ANNO  = 8      # anotaciones pequeñas

plt.rcParams.update({
    "axes.titlesize": PLT_TITLE,
    "axes.labelsize": PLT_LABEL,
    "xtick.labelsize": PLT_LABEL,
    "ytick.labelsize": PLT_LABEL,
    "legend.fontsize": 9,
    "figure.autolayout": True,
})

# ---------------------------
# UI helpers
# ---------------------------

def center(fig):
    """Centrar una figura en Streamlit (columna central)."""
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.pyplot(fig, use_container_width=True)

def hr():
    st.markdown(
        "<hr style='margin:0.5rem 0; border:none; border-top:1px solid #DDD;'/>",
        unsafe_allow_html=True
    )

# Column map (tus nombres a nombres limpios)
MAP_UNISON = {
    "Año_ (YEAR)": "Year", "AÃ±o_ (YEAR)": "Year",
    "Día (DOY)": "DOY",   "DÃ­a (DOY)": "DOY",
    "Tmax (T2M_MAX)": "Tmax", "Tmin (T2M_MIN)": "Tmin",
    "HR (RH2M)": "HR", "Ux (WS2M)": "Ux",
    "Rs (ALLSKY_SFC_SW_DWN)": "Rs",
    "Rl_ (ALLSKY_SFC_LW_DWN)": "Rl",
    "Ptot_ (PRECTOTCORR)": "Ptot",
    "Pef_": "Pef", "Tmean_": "Tmean", "es_": "es", "ea_": "ea",
    "delta_": "delta", "P_": "P", "gamma_": "gamma",
    "Rns_": "Rns", "Rnl_": "Rnl", "Rn_": "Rn", "Rso_": "Rso",
    "Kc_": "Kc", "decada_": "decada",
    "ET0": "ET0", "ETc": "ETc", "ETverde": "ETverde", "ETazul": "ETazul",
    "Year": "Year", "DOY": "DOY", "Dia": "Dia",
}

# Columnas numéricas a forzar si aparecen
COLUMNAS_NUM = [
    "Year","DOY","ET0","ETc","ETverde","ETazul","Pef","decada",
    "Rns","Rnl","Rs","Tmean","HR","Ux","Kc","Tmax","Tmin",
    "UACverde_m3_ha","UACazul_m3_ha","HHverde_m3_ton","HHazul_m3_ton"
]

# ---------------------------
# Helpers de archivos
# ---------------------------
def parse_unison_filename(filename: str):
    """
    'Cajeme-FAO56-2014-2015-SALIDA.csv' -> ('Cajeme','2014-2015')
    'Metepec-FAO56-2014-SALIDA.csv'     -> ('Metepec','2014')
    """
    m = re.match(r"([A-Za-zÁÉÍÓÚáéíóúñÑ\s]+)-FAO56-(\d{4})(?:-(\d{4}))?-SALIDA\.csv$", filename, re.I)
    if not m:
        return None, None
    reg, y1, y2 = m.groups()
    if reg == "VillaAllende": reg = "Villa de Allende"
    if reg == "Etchhojoa":    reg = "Etchojoa"
    ciclo = y1 if not y2 else f"{y1}-{y2}"
    return reg.strip(), ciclo

@st.cache_data(show_spinner=False)
def catalogo_unison(base_dir: Path) -> pd.DataFrame:
    rows = []
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
                rows.append({"Region": reg, "Ciclo": ciclo, "Ruta": str(d / f)})
    df = pd.DataFrame(rows).sort_values(["Region","Ciclo"]).reset_index(drop=True)
    return df

# ---------------------------
# Helpers de datos
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

    # Lectura tolerante a encoding
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

    # Campos numéricos
    for c in set(COLUMNAS_NUM).intersection(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fecha y día de ciclo
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

    # Acumulados y proporciones útiles (por si los quieres luego)
    if "ETc" in df:
        df["ETc_acum"] = df["ETc"].cumsum()
    if "ETazul" in df:
        df["ETazul_acum"] = df["ETazul"].cumsum()
    if {"ETc","ETazul"}.issubset(df.columns):
        df["pct_azul"] = np.where(df["ETc"]>0, df["ETazul"]/df["ETc"]*100.0, np.nan)
    return df

def kpis_ciclo(df: pd.DataFrame) -> dict:
    mask = df["ETc"].notna() if "ETc" in df else pd.Series(False, index=df.index)
    dias = int(mask.sum())
    etc_total = float(df.loc[mask, "ETc"].sum())     if "ETc"     in df else np.nan
    etv_total = float(df.loc[mask, "ETverde"].sum()) if "ETverde" in df else np.nan
    eta_total = float(df.loc[mask, "ETazul"].sum())  if "ETazul"  in df else np.nan
    return {"dias": dias, "etc_total": etc_total, "etv_total": etv_total, "eta_total": eta_total}

def fechas_ciclo(df: pd.DataFrame):
    if "Fecha" in df and df["Fecha"].notna().any():
        fmin = pd.to_datetime(df["Fecha"].dropna().iloc[0])
        fmax = pd.to_datetime(df["Fecha"].dropna().iloc[-1])
        return fmin.date(), fmax.date()
    return None, None

def last_valid(df: pd.DataFrame, col: str):
    return float(df[col].dropna().iloc[-1]) if (col in df and df[col].notna().any()) else np.nan

def kpis_ext(df: pd.DataFrame):
    k = kpis_ciclo(df)
    siembra, cosecha = fechas_ciclo(df)

    # UAC y HH desde columnas del CSV (último valor no nulo)
    uacv_ha = last_valid(df, "UACverde_m3_ha")
    uaca_ha = last_valid(df, "UACazul_m3_ha")
    hhv_ton = last_valid(df, "HHverde_m3_ton")
    hha_ton = last_valid(df, "HHazul_m3_ton")

    out = {
        "dias": k["dias"],
        "siembra": siembra, "cosecha": cosecha,
        "etc_total": k["etc_total"],
        "eta_total": k["eta_total"],
        "etv_total": k["etv_total"],
        "tmax": float(df["Tmax"].max()) if "Tmax" in df else np.nan,
        "tmin": float(df["Tmin"].min()) if "Tmin" in df else np.nan,
        "uacv_ha": uacv_ha,  # m³/ha (verde)
        "uaca_ha": uaca_ha,  # m³/ha (azul)
        "hhv_ton": hhv_ton,  # m³/ton (verde)
        "hha_ton": hha_ton,  # m³/ton (azul)
    }
    return out

def _xcol(df: pd.DataFrame, prefer="Fecha"):
    if prefer in df.columns and df[prefer].notna().any():
        return prefer
    for alt in ("Fecha", "Dia_ciclo"):
        if alt in df.columns and df[alt].notna().any():
            return alt
    return df.index

# ---------------------------
# Figuras
# ---------------------------
def fig_series(df: pd.DataFrame, titulo: str, eje="Fecha", mostrar=("ET0","ETc","ETverde","ETazul","Pef")):
    x = _xcol(df, eje)
    fig, ax = plt.subplots(1,1, figsize=(12,4))
    colores = {"ET0":"#4C78A8","ETc":"#F58518","ETverde":"#54A24B","ETazul":"#E45756","Pef":"#9D9D9D"}
    for col in mostrar:
        if col in df:
            ax.plot(df[x], df[col], label=col, lw=1.6, color=colores.get(col, None))
    ax.set_title(titulo); ax.set_xlabel(str(x)); ax.set_ylabel("mm/día"); ax.legend()
    fig.tight_layout(); return fig

def fig_temperaturas(df, titulo, eje="Fecha", mostrar=("Tmin","Tmean","Tmax")):
    x = _xcol(df, eje)
    fig, ax = plt.subplots(figsize=(12,4))
    for c in mostrar:
        if c in df:
            ax.plot(df[x], df[c], lw=1.4, label=c)
    ax.set_title(titulo); ax.set_xlabel(str(x)); ax.set_ylabel("°C"); ax.legend()
    fig.tight_layout(); return fig

def fig_meteo_rs_hr(df, titulo, eje="Fecha", mostrar=("Rs","HR")):
    """Grafica Rs y/o HR con ejes gemelos. Devuelve figura."""
    x = _xcol(df, eje)
    show_rs = "Rs" in mostrar and "Rs" in df
    show_hr = "HR" in mostrar and "HR" in df
    fig, ax1 = plt.subplots(figsize=(12,4))
    lines = []; labels = []

    if show_rs:
        l1, = ax1.plot(df[x], df["Rs"], lw=1.5, label="Rs")
        ax1.set_ylabel("Rs [MJ m$^{-2}$ d$^{-1}$]")
        lines.append(l1); labels.append("Rs")

    if show_hr:
        ax2 = ax1.twinx()
        l2, = ax2.plot(df[x], df["HR"], lw=1.2, label="HR", linestyle="--")
        ax2.set_ylabel("HR [%]")
        lines.append(l2); labels.append("HR")

    ax1.set_title(titulo); ax1.set_xlabel(str(x))
    if lines:
        ax1.legend(lines, labels, loc="upper right")
    fig.tight_layout()
    return fig

def fig_wind(df, titulo, eje="Fecha"):
    x = _xcol(df, eje)
    fig, ax = plt.subplots(figsize=(12,3))
    if "Ux" in df:
        ax.plot(df[x], df["Ux"], lw=1.2, label="Ux")
    ax.set_title(titulo); ax.set_xlabel(str(x)); ax.set_ylabel("m/s")
    ax.legend()
    fig.tight_layout()
    return fig

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("DataAqua — Selección")
CAT_UNISON = catalogo_unison(RUTA_SALIDA_UNISON)
if CAT_UNISON.empty:
    st.error("No se encontraron archivos en la carpeta de datos.")
    st.stop()

modo = st.sidebar.radio("Modo", ["Ciclo individual", "Comparar ciclos", "Comparar regiones"], index=0)

# “Eje X” menos técnico
verpor_label = st.sidebar.radio("Ver por", ["Fecha", "Día del ciclo"], index=0)
eje_opt = "Dia_ciclo" if verpor_label == "Día del ciclo" else "Fecha"

if modo == "Ciclo individual":
    regiones = sorted(CAT_UNISON["Region"].unique())
    region_sel = st.sidebar.selectbox("Región", regiones)
    ciclos_reg = sorted(CAT_UNISON.loc[CAT_UNISON["Region"]==region_sel, "Ciclo"].unique())
    ciclo_sel = st.sidebar.selectbox("Ciclo", ciclos_reg)

elif modo == "Comparar ciclos":
    regiones = sorted(CAT_UNISON["Region"].unique())
    region_sel = st.sidebar.selectbox("Región", regiones)
    ciclos_reg = sorted(CAT_UNISON.loc[CAT_UNISON["Region"]==region_sel, "Ciclo"].unique())
    ciclo_A = st.sidebar.selectbox("Ciclo A", ciclos_reg, key="ciclo_A")
    ciclo_B = st.sidebar.selectbox("Ciclo B", ciclos_reg, index=min(1, len(ciclos_reg)-1), key="ciclo_B")

elif modo == "Comparar regiones":
    ciclos = sorted(CAT_UNISON["Ciclo"].unique())
    ciclo_sel = st.sidebar.selectbox("Ciclo", ciclos)
    regs_ciclo = sorted(CAT_UNISON.loc[CAT_UNISON["Ciclo"]==ciclo_sel, "Region"].unique())
    region_A = st.sidebar.selectbox("Región A", regs_ciclo, key="region_A")
    region_B = st.sidebar.selectbox("Región B", regs_ciclo, index=min(1, len(regs_ciclo)-1), key="region_B")

# ---------------------------
# Layout principal
# ---------------------------
st.title("💧 DataAqua — Dashboard 2")
st.caption("Resultados UNISON (FAO-56). ETc (demanda del cultivo), ETverde (cubierta por Pef) y ETazul (resto). ET0 es referencia (césped).")

tab_vista, tab_modelos = st.tabs(["Vista", "Modelos y Estadística"])

# ===========================
# Pestaña: Vista (todo lo que ya tenías)
# ===========================
with tab_vista:
    # === Modo: Ciclo individual
    if modo == "Ciclo individual":
        ruta_sel = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
        if ruta_sel.empty:
            st.error(f"No encontré CSV para {region_sel} / {ciclo_sel}"); st.stop()
        df = leer_unison(ruta_sel.iloc[0])
        if df.empty:
            st.error("No fue posible leer el archivo seleccionado."); st.stop()

        # ===== KPIs (dos columnas compactas 5+5) =====
        st.subheader(f"KPIs — {region_sel} ({ciclo_sel})")
        k = kpis_ext(df)
        colL, colR = st.columns(2)
        with colL:
            st.metric("Días del ciclo", f"{k['dias']}")
            st.metric("Fecha de siembra", f"{k['siembra'] or '—'}")
            st.metric("Fecha de cosecha", f"{k['cosecha'] or '—'}")
            st.metric("ETc total [mm]", f"{k['etc_total']:.1f}")
            st.metric("ETverde total [mm]", f"{k['etv_total']:.1f}")
        with colR:
            st.metric("% Azul", f"{(k['eta_total']/k['etc_total']*100):.1f}%" if k['etc_total'] else "—")
            st.metric("ETazul total [mm]", f"{k['eta_total']:.1f}")
            st.metric("Tmax / Tmin [°C]", f"{k['tmax']:.1f} / {k['tmin']:.1f}")
            st.metric("UAC verde [m³/ha]", f"{k['uacv_ha']:.0f}" if not np.isnan(k['uacv_ha']) else "—")
            st.metric("UAC azul [m³/ha]",  f"{k['uaca_ha']:.0f}" if not np.isnan(k['uaca_ha']) else "—")
            # Si quieres mostrar HH, descomenta:
            # st.metric("HH verde [m³/ton]", f"{k['hhv_ton']:.0f}" if not np.isnan(k['hhv_ton']) else "—")
            # st.metric("HH azul [m³/ton]",  f"{k['hha_ton']:.0f}" if not np.isnan(k['hha_ton']) else "—")

        hr()

        # Serie diaria (ET) — con multiselect (default todas)
        st.markdown("### Serie diaria (ET)")
        et_opts = [v for v in ["ET0","ETc","ETverde","ETazul","Pef"] if v in df.columns]
        et_sel = st.multiselect("Series a mostrar", et_opts, default=et_opts, key="et_ind")
        fig = fig_series(df, f"Serie diaria (ET) — {ciclo_sel}", eje=eje_opt, mostrar=et_sel or et_opts)
        st.pyplot(fig, use_container_width=True)

        # Temperaturas — con multiselect (default todas)
        st.markdown("### Temperaturas")
        t_opts = [v for v in ["Tmin","Tmean","Tmax"] if v in df.columns]
        t_sel = st.multiselect("Series de temperatura", t_opts, default=t_opts, key="t_ind")
        ftemp = fig_temperaturas(df, f"Temperaturas — {ciclo_sel}", eje=eje_opt, mostrar=t_sel or t_opts)
        st.pyplot(ftemp, use_container_width=True)

        # Meteorología — con multiselect (default todas)
        st.markdown("### Meteorología")
        met_opts = [v for v in ["Rs","HR"] if v in df.columns]
        met_sel = st.multiselect("Variables de meteorología", met_opts, default=met_opts, key="met_ind")
        fmet = fig_meteo_rs_hr(df, f"Meteorología — {ciclo_sel}", eje=eje_opt, mostrar=met_sel or met_opts)
        st.pyplot(fmet, use_container_width=True)

        # Viento (Ux)
        if "Ux" in df.columns:
            st.markdown("### Viento")
            fux = fig_wind(df, f"Viento Ux — {ciclo_sel}", eje=eje_opt)
            st.pyplot(fux, use_container_width=True)

        hr()
        with st.expander("Datos (primeras filas)"):
            st.dataframe(df.head(30), use_container_width=True)

    # === Modo: Comparar ciclos (A arriba, B debajo, por cada sección)
    elif modo == "Comparar ciclos":
        ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_A)]["Ruta"]
        ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_B)]["Ruta"]
        if ruta_A.empty or ruta_B.empty: st.error("No encontré ambos ciclos."); st.stop()
        dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])

        st.subheader(f"{region_sel} — comparación de ciclos")
        st.caption(f"Ciclo A: **{ciclo_A}**  |  Ciclo B: **{ciclo_B}**")

        # KPIs en dos columnas (A | B)
        colA, colB = st.columns(2)
        kA, kB = kpis_ext(dfA), kpis_ext(dfB)
        with colA:
            st.markdown(f"**{ciclo_A}**")
            st.metric("Días del ciclo", f"{kA['dias']}")
            st.metric("% Azul", f"{(kA['eta_total']/kA['etc_total']*100):.1f}%" if kA['etc_total'] else "—")
            st.metric("ETc total [mm]", f"{kA['etc_total']:.1f}")
            st.metric("ETazul total [mm]", f"{kA['eta_total']:.1f}")
            st.metric("UAC verde / azul [m³/ha]",
                      f"{(kA['uacv_ha'] if not np.isnan(kA['uacv_ha']) else 0):.0f} / {(kA['uaca_ha'] if not np.isnan(kA['uaca_ha']) else 0):.0f}")
        with colB:
            st.markdown(f"**{ciclo_B}**")
            st.metric("Días del ciclo", f"{kB['dias']}")
            st.metric("% Azul", f"{(kB['eta_total']/kB['etc_total']*100):.1f}%" if kB['etc_total'] else "—")
            st.metric("ETc total [mm]", f"{kB['etc_total']:.1f}")
            st.metric("ETazul total [mm]", f"{kB['eta_total']:.1f}")
            st.metric("UAC verde / azul [m³/ha]",
                      f"{(kB['uacv_ha'] if not np.isnan(kB['uacv_ha']) else 0):.0f} / {(kB['uaca_ha'] if not np.isnan(kB['uaca_ha']) else 0):.0f}")

        # ===== Serie diaria (ET): A y debajo B =====
        hr()
        st.markdown(f"### Serie diaria (ET) — {ciclo_A}")
        st.pyplot(fig_series(dfA, f"Serie diaria (ET) — {ciclo_A}", eje=eje_opt,
                             mostrar=[c for c in ["ET0","ETc","ETverde","ETazul","Pef"] if c in dfA.columns]),
                  use_container_width=True)
        st.markdown(f"### Serie diaria (ET) — {ciclo_B}")
        st.pyplot(fig_series(dfB, f"Serie diaria (ET) — {ciclo_B}", eje=eje_opt,
                             mostrar=[c for c in ["ET0","ETc","ETverde","ETazul","Pef"] if c in dfB.columns]),
                  use_container_width=True)

        # ===== Temperaturas: A y debajo B =====
        hr()
        st.markdown(f"### Temperaturas — {ciclo_A}")
        st.pyplot(fig_temperaturas(dfA, f"Temperaturas — {ciclo_A}", eje=eje_opt,
                                   mostrar=[c for c in ["Tmin","Tmean","Tmax"] if c in dfA.columns]),
                  use_container_width=True)
        st.markdown(f"### Temperaturas — {ciclo_B}")
        st.pyplot(fig_temperaturas(dfB, f"Temperaturas — {ciclo_B}", eje=eje_opt,
                                   mostrar=[c for c in ["Tmin","Tmean","Tmax"] if c in dfB.columns]),
                  use_container_width=True)

        # ===== Meteorología: A y debajo B =====
        hr()
        st.markdown(f"### Meteorología — {ciclo_A}")
        st.pyplot(fig_meteo_rs_hr(dfA, f"Meteorología — {ciclo_A}", eje=eje_opt,
                                  mostrar=[c for c in ["Rs","HR"] if c in dfA.columns]),
                  use_container_width=True)
        st.markdown(f"### Meteorología — {ciclo_B}")
        st.pyplot(fig_meteo_rs_hr(dfB, f"Meteorología — {ciclo_B}", eje=eje_opt,
                                  mostrar=[c for c in ["Rs","HR"] if c in dfB.columns]),
                  use_container_width=True)

        # ===== Viento: A y debajo B =====
        if "Ux" in dfA.columns or "Ux" in dfB.columns:
            hr()
            if "Ux" in dfA.columns:
                st.markdown(f"### Viento Ux — {ciclo_A}")
                st.pyplot(fig_wind(dfA, f"Viento Ux — {ciclo_A}", eje=eje_opt), use_container_width=True)
            if "Ux" in dfB.columns:
                st.markdown(f"### Viento Ux — {ciclo_B}")
                st.pyplot(fig_wind(dfB, f"Viento Ux — {ciclo_B}", eje=eje_opt), use_container_width=True)

    # === Modo: Comparar regiones (A arriba, B debajo, por cada sección)
    elif modo == "Comparar regiones":
        ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_A) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
        ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_B) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
        if ruta_A.empty or ruta_B.empty: st.error("No encontré ambas regiones."); st.stop()
        dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])

        st.subheader(f"Comparación de regiones — ciclo {ciclo_sel}")
        st.caption(f"Región A: **{region_A}**  |  Región B: **{region_B}**")

        # KPIs en dos columnas (A | B)
        colA, colB = st.columns(2)
        kA, kB = kpis_ext(dfA), kpis_ext(dfB)
        with colA:
            st.markdown(f"**{region_A}**")
            st.metric("Días del ciclo", f"{kA['dias']}")
            st.metric("% Azul", f"{(kA['eta_total']/kA['etc_total']*100):.1f}%" if kA['etc_total'] else "—")
            st.metric("ETc total [mm]", f"{kA['etc_total']:.1f}")
            st.metric("ETazul total [mm]", f"{kA['eta_total']:.1f}")
            st.metric("UAC verde / azul [m³/ha]",
                      f"{(kA['uacv_ha'] if not np.isnan(kA['uacv_ha']) else 0):.0f} / {(kA['uaca_ha'] if not np.isnan(kA['uaca_ha']) else 0):.0f}")
        with colB:
            st.markdown(f"**{region_B}**")
            st.metric("Días del ciclo", f"{kB['dias']}")
            st.metric("% Azul", f"{(kB['eta_total']/kB['etc_total']*100):.1f}%" if kB['etc_total'] else "—")
            st.metric("ETc total [mm]", f"{kB['etc_total']:.1f}")
            st.metric("ETazul total [mm]", f"{kB['eta_total']:.1f}")
            st.metric("UAC verde / azul [m³/ha]",
                      f"{(kB['uacv_ha'] if not np.isnan(kB['uacv_ha']) else 0):.0f} / {(kB['uaca_ha'] if not np.isnan(kB['uaca_ha']) else 0):.0f}")

        # ===== Serie diaria (ET): A y debajo B =====
        hr()
        st.markdown(f"### {region_A} — Serie diaria (ET) — {ciclo_sel}")
        st.pyplot(fig_series(dfA, f"Serie diaria (ET) — {ciclo_sel}", eje=eje_opt,
                             mostrar=[c for c in ["ET0","ETc","ETverde","ETazul","Pef"] if c in dfA.columns]),
                  use_container_width=True)
        st.markdown(f"### {region_B} — Serie diaria (ET) — {ciclo_sel}")
        st.pyplot(fig_series(dfB, f"Serie diaria (ET) — {ciclo_sel}", eje=eje_opt,
                             mostrar=[c for c in ["ET0","ETc","ETverde","ETazul","Pef"] if c in dfB.columns]),
                  use_container_width=True)

        # ===== Temperaturas: A y debajo B =====
        hr()
        st.markdown(f"### {region_A} — Temperaturas — {ciclo_sel}")
        st.pyplot(fig_temperaturas(dfA, f"Temperaturas — {ciclo_sel}", eje=eje_opt,
                                   mostrar=[c for c in ["Tmin","Tmean","Tmax"] if c in dfA.columns]),
                  use_container_width=True)
        st.markdown(f"### {region_B} — Temperaturas — {ciclo_sel}")
        st.pyplot(fig_temperaturas(dfB, f"Temperaturas — {ciclo_sel}", eje=eje_opt,
                                   mostrar=[c for c in ["Tmin","Tmean","Tmax"] if c in dfB.columns]),
                  use_container_width=True)

        # ===== Meteorología: A y debajo B =====
        hr()
        st.markdown(f"### {region_A} — Meteorología — {ciclo_sel}")
        st.pyplot(fig_meteo_rs_hr(dfA, f"Meteorología — {ciclo_sel}", eje=eje_opt,
                                  mostrar=[c for c in ["Rs","HR"] if c in dfA.columns]),
                  use_container_width=True)
        st.markdown(f"### {region_B} — Meteorología — {ciclo_sel}")
        st.pyplot(fig_meteo_rs_hr(dfB, f"Meteorología — {ciclo_sel}", eje=eje_opt,
                                  mostrar=[c for c in ["Rs","HR"] if c in dfB.columns]),
                  use_container_width=True)

        # ===== Viento: A y debajo B =====
        if "Ux" in dfA.columns or "Ux" in dfB.columns:
            hr()
            if "Ux" in dfA.columns:
                st.markdown(f"### {region_A} — Viento Ux — {ciclo_sel}")
                st.pyplot(fig_wind(dfA, f"Viento Ux — {ciclo_sel}", eje=eje_opt), use_container_width=True)
            if "Ux" in dfB.columns:
                st.markdown(f"### {region_B} — Viento Ux — {ciclo_sel}")
                st.pyplot(fig_wind(dfB, f"Viento Ux — {ciclo_sel}", eje=eje_opt), use_container_width=True)

# # ===========================
# # Pestaña: Modelos y Estadística (todo el análisis del cuaderno)
# # ===========================
# with tab_modelos:
#     st.subheader("Modelos y Estadística")

#     # Imports protegidos para no tumbar la app si faltan deps
#     HAVE_SKLEARN = True
#     HAVE_PLOTLY  = True
#     try:
#         import seaborn as sns
#         from sklearn.model_selection import train_test_split
#         from sklearn.linear_model import LinearRegression
#         from sklearn.metrics import mean_squared_error, r2_score
#         from sklearn.ensemble import RandomForestRegressor
#         from sklearn.cluster import KMeans
#         from sklearn.preprocessing import StandardScaler
#     except Exception as e:
#         HAVE_SKLEARN = False
#         err_sklearn = str(e)

#     try:
#         import plotly.express as px
#     except Exception as e:
#         HAVE_PLOTLY = False
#         err_plotly = str(e)

#     if not HAVE_SKLEARN:
#         st.warning(
#             "Faltan dependencias para esta pestaña (scikit-learn). "
#             "El resto del dashboard funciona. Instala las dependencias y recarga."
#         )

#     # Si falta sklearn NO definimos ni usamos render_modelos_para
#     if HAVE_SKLEARN:
#         import warnings
#         warnings.filterwarnings("ignore")

#         def render_modelos_para(df: pd.DataFrame, titulo: str):
#             st.markdown(f"### {titulo}")

#             # ---------- Estadística descriptiva ----------
#             cols_est = [c for c in ["Tmax","Tmin","Tmean","HR","Ux","Rs","ET0","ETc","ETverde","ETazul","Pef"] if c in df.columns]
#             if cols_est:
#                 st.markdown("**Estadística descriptiva**")
#                 desc = df[cols_est].describe(percentiles=[0.25,0.5,0.75]).T
#                 st.dataframe(desc, use_container_width=True)

#             # ---------- Matriz de correlación ----------
#             if cols_est:
#                 st.markdown("**Matriz de correlación (upper)**")
#                 corr = df[cols_est].dropna().corr()
#                 fig, ax = plt.subplots(figsize=(7,5))
#                 mask = np.triu(np.ones_like(corr, dtype=bool))
#                 sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", square=True, mask=mask, ax=ax)
#                 ax.set_title("Correlación")
#                 st.pyplot(fig, use_container_width=True)

#             # ---------- Dispersión (scatter) ----------
#             pares = [("Tmax","ET0"), ("Rs","ET0"), ("HR","ET0"), ("Ux","ET0"), ("ET0","ETc")]
#             pares = [(x,y) for (x,y) in pares if x in df.columns and y in df.columns]
#             if pares:
#                 st.markdown("**Dispersión de variables clave**")
#                 n = len(pares)
#                 ncols = 3
#                 nrows = int(np.ceil(n/ncols))
#                 fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
#                 axes = np.atleast_2d(axes).ravel()
#                 for i,(xv,yv) in enumerate(pares):
#                     sns.scatterplot(x=df[xv], y=df[yv], ax=axes[i])
#                     axes[i].set_title(f"{xv} vs {yv}")
#                 for j in range(i+1, len(axes)):
#                     axes[j].set_visible(False)
#                 st.pyplot(fig, use_container_width=True)

#             # ---------- Regresión lineal para ET0 ----------
#             feats = [c for c in ["Tmax","Tmin","HR","Ux","Rs"] if c in df.columns]
#             if set(feats).issubset(df.columns) and "ET0" in df.columns:
#                 st.markdown("**Regresión lineal (ET0 ~ Tmax + Tmin + HR + Ux + Rs)**")
#                 dmod = df[feats+["ET0"]].dropna()
#                 if len(dmod) > 10:
#                     X = dmod[feats]; y = dmod["ET0"]
#                     Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
#                     lm = LinearRegression().fit(Xtr, ytr)
#                     yhat = lm.predict(Xte)
#                     r2 = r2_score(yte, yhat); mse = mean_squared_error(yte, yhat)
#                     st.write(f"R² = {r2:.4f}  |  MSE = {mse:.4f}")
#                     fig, ax = plt.subplots(figsize=(5,4))
#                     ax.scatter(yte, yhat, alpha=0.7)
#                     m = [min(yte.min(), yhat.min()), max(yte.max(), yhat.max())]
#                     ax.plot(m, m, "r--")
#                     ax.set_xlabel("ET0 real"); ax.set_ylabel("ET0 predicho")
#                     ax.set_title("Predicho vs Real (Regresión lineal)")
#                     st.pyplot(fig, use_container_width=True)

#             # ---------- Random Forest para ET0 ----------
#             if set(feats).issubset(df.columns) and "ET0" in df.columns:
#                 st.markdown("**Random Forest (ET0)**")
#                 dmod = df[feats+["ET0"]].dropna()
#                 if len(dmod) > 10:
#                     X = dmod[feats]; y = dmod["ET0"]
#                     Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
#                     rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
#                     yhat = rf.predict(Xte)
#                     st.write(f"R² = {r2_score(yte, yhat):.4f}  |  MSE = {mean_squared_error(yte, yhat):.4f}")
#                     imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
#                     st.bar_chart(imp)

#             # ---------- Clustering (KMeans) sobre meteo ----------
#             cl_feats = [c for c in ["Tmax","Tmin","HR","Ux","Rs"] if c in df.columns]
#             if len(cl_feats) >= 3:
#                 st.markdown("**Clustering K-Means (método del codo + clusters)**")
#                 X0 = df[cl_feats].dropna()
#                 if len(X0) > 20:
#                     scaler = StandardScaler()
#                     Xs = scaler.fit_transform(X0)

#                     # Método del codo
#                     inertia = []
#                     ks = list(range(2, 9))
#                     for k_ in ks:
#                         km = KMeans(n_clusters=k_, random_state=42, n_init="auto").fit(Xs)
#                         inertia.append(km.inertia_)
#                     fig, ax = plt.subplots(figsize=(5,3))
#                     ax.plot(ks, inertia, marker="o")
#                     ax.set_xlabel("k"); ax.set_ylabel("Inercia"); ax.set_title("Método del codo")
#                     st.pyplot(fig, use_container_width=True)

#                     # Elegimos k=5 por consistencia con el cuaderno
#                     k_final = 5 if len(X0) >= 50 else min(4, len(X0)//5) or 2
#                     km = KMeans(n_clusters=k_final, random_state=42, n_init="auto")
#                     grupos = km.fit_predict(Xs)
#                     dfK = X0.copy()
#                     dfK["Grupo"] = grupos

#                     st.write("**Scatter Tmax vs Rs (coloreado por Grupo)**")
#                     fig, ax = plt.subplots(figsize=(6,4))
#                     sns.scatterplot(data=dfK, x="Tmax", y="Rs", hue="Grupo", palette="Set2", ax=ax)
#                     st.pyplot(fig, use_container_width=True)

#                     # Boxplots por década y grupo (si hay 'decada' y si tenemos plotly)
#                     if "decada" in df.columns and HAVE_PLOTLY:
#                         st.write("**Distribución por década y grupo (Plotly)**")
#                         for var in [c for c in cl_feats if c in df.columns]:
#                             figpx = px.box(
#                                 df.assign(Grupo=grupos),
#                                 x="decada", y=var, color="Grupo",
#                                 title=f"{var} por grupo y década",
#                                 labels={"decada":"Década", var:var, "Grupo":"Grupo"},
#                                 points="all"
#                             )
#                             st.plotly_chart(figpx, use_container_width=True)

#             # ---------- Clustering con ET0 y ETc también ----------
#             cl2 = [c for c in ["Tmax","Tmin","HR","Ux","Rs","ET0","ETc"] if c in df.columns]
#             if len(cl2) >= 4:
#                 st.markdown("**Clustering K-Means (incluye ET0 y ETc)**")
#                 X1 = df[cl2].dropna()
#                 if len(X1) > 20:
#                     scaler = StandardScaler(); Xs = scaler.fit_transform(X1)
#                     kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto").fit(Xs)
#                     g2 = kmeans.predict(Xs)
#                     df2 = X1.copy(); df2["Grupo"] = g2
#                     stats = df2.groupby("Grupo")[cl2].mean()
#                     st.write("Promedios por grupo:")
#                     st.dataframe(stats, use_container_width=True)

#                     # Scatter adicional
#                     if "Tmax" in df2 and "Rs" in df2:
#                         fig, ax = plt.subplots(figsize=(6,4))
#                         sns.scatterplot(data=df2, x="Tmax", y="Rs", hue="Grupo", palette="Set1", ax=ax)
#                         ax.set_title("Clasificación de días climáticos (Tmax vs Rs)")
#                         st.pyplot(fig, use_container_width=True)

#             st.markdown("---")

#         # --- CICLO INDIVIDUAL (un bloque) ---
#         if modo == "Ciclo individual":
#             ruta_sel = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
#             if ruta_sel.empty:
#                 st.warning(f"No encontré CSV para {region_sel} / {ciclo_sel}")
#             else:
#                 dfM = leer_unison(ruta_sel.iloc[0])
#                 if dfM.empty:
#                     st.warning("No fue posible leer el archivo seleccionado.")
#                 else:
#                     render_modelos_para(dfM, f"{region_sel} — {ciclo_sel}")

#         # --- COMPARAR CICLOS (dos bloques: A y B) ---
#         elif modo == "Comparar ciclos":
#             ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_A)]["Ruta"]
#             ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_B)]["Ruta"]
#             if ruta_A.empty or ruta_B.empty:
#                 st.warning("No encontré ambos ciclos para mostrar análisis.")
#             else:
#                 dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])
#                 render_modelos_para(dfA, f"{region_sel} — {ciclo_A}")
#                 render_modelos_para(dfB, f"{region_sel} — {ciclo_B}")

#         # --- COMPARAR REGIONES (dos bloques: A y B) ---
#         elif modo == "Comparar regiones":
#             ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_A) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
#             ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_B) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
#             if ruta_A.empty or ruta_B.empty:
#                 st.warning("No encontré ambas regiones para mostrar análisis.")
#             else:
#                 dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])
#                 render_modelos_para(dfA, f"{region_A} — {ciclo_sel}")
#                 render_modelos_para(dfB, f"{region_B} — {ciclo_sel}")

with tab_modelos:
    # ===========================
    # Modelos y Estadística — “tal como el profesor”, adaptado al dashboard
    # ===========================
    import warnings
    warnings.filterwarnings("ignore")

    # --- Dependencias locales (para no romper la app si falta algo) ---
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.ensemble import RandomForestRegressor
        import plotly.express as px
    except Exception as _e:
        st.error("Faltan dependencias de scikit-learn o plotly. Verifica tu requirements.txt")
        st.stop()

    # --- Estilo compacto y consistente (ligeramente más pequeño) ---
    _TITLE = 12     # títulos un poco más pequeños
    _LABEL = 9      # etiquetas y ticks más pequeños
    _ANNO  = 8

    plt.rcParams.update({
        "axes.titlesize": _TITLE,
        "axes.labelsize": _LABEL,
        "xtick.labelsize": _LABEL,
        "ytick.labelsize": _LABEL,
        "legend.fontsize": 9,
        "figure.autolayout": True,
    })

    def _hr():
        st.markdown("<hr style='margin:0.6rem 0; border:none; border-top:1px solid #DDD;'/>", unsafe_allow_html=True)

    def _presentes(df, cols):
        return [c for c in cols if c in df.columns]

    # ============================================
    # Render “1:1 profesor” para un DataFrame dado
    # ============================================    
    
    def render_modelos_profesor(df_in: pd.DataFrame, region: str, ciclo: str):
        if df_in is None or df_in.empty:
            st.warning(f"No hay datos para **{region} — {ciclo}**.")
            return

        st.markdown(f"### {region} — {ciclo}")

        # ---------------------------
        # 1) Estadística descriptiva (tabla completa)
        # ---------------------------
        _hr()
        st.markdown("#### Estadística descriptiva")
        cols_est = _presentes(df_in, ["Tmax","Tmin","Tmean","HR","Ux","Rs","ET0","ETc","ETverde","ETazul","Pef"])
        if cols_est:
            desc = df_in[cols_est].describe(percentiles=[0.25,0.5,0.75]).T
            st.dataframe(desc, use_container_width=True)
        else:
            st.info("No hay columnas numéricas esperadas para la estadística descriptiva.")

        # ---------------------------
        # 2) Matriz de correlación (triangular “mitad”, paleta viridis)
        # ---------------------------
        if cols_est:
            _hr()
            st.markdown("#### Matriz de correlación (triangular)")
            df_corr = df_in[cols_est].dropna()
            if not df_corr.empty:
                import seaborn as sns
                corr = df_corr.corr()
                fig, ax = plt.subplots(figsize=(6.0, 4.2))  # compacta (más pequeña)
                mask = np.triu(np.ones_like(corr, dtype=bool))  # mostrar la mitad inferior
                sns.heatmap(
                    corr, ax=ax, annot=True, fmt=".2f",
                    cmap="viridis", square=True, mask=mask,
                    annot_kws={"size": _ANNO}
                )
                ax.set_title("Correlación", fontsize=_TITLE)
                ax.tick_params(axis="x", labelsize=_LABEL-1, rotation=45)
                ax.tick_params(axis="y", labelsize=_LABEL-1)
                center(fig)  # centrada

        # ---------------------------
        # 3) Dispersión (scatter) como en el cuaderno
        # ---------------------------
        pares = [("Tmax","ET0"), ("Rs","ET0"), ("HR","ET0"), ("Ux","ET0"), ("ET0","ETc")]
        pares = [(x,y) for (x,y) in pares if x in df_in.columns and y in df_in.columns]
        if pares:
            _hr()
            st.markdown("#### Dispersión de variables clave")
            # Grid 2 x 3, respetando colores del profesor en los pares definidos
            color_map = {
                ("Tmax","ET0"): "blue",
                ("Rs","ET0"):   "green",
                ("HR","ET0"):   "red",
                ("Ux","ET0"):   "purple",
                ("ET0","ETc"):  "orange"
            }
            n = len(pares)
            ncols, nrows = 3, 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
            axes = np.array(axes)
            idx = 0
            for r in range(nrows):
                for c in range(ncols):
                    if idx < n:
                        xv, yv = pares[idx]
                        ax = axes[r, c]
                        import seaborn as sns
                        sns.scatterplot(x=xv, y=yv, data=df_in, ax=ax, color=color_map.get((xv,yv), None))
                        ax.set_title(f"{xv} vs {yv}", fontsize=_TITLE)
                        ax.set_xlabel(xv); ax.set_ylabel(yv)
                        idx += 1
                    else:
                        axes[r, c].set_visible(False)
            st.pyplot(fig, use_container_width=True)

        # ---------------------------
        # 4) Regresión lineal (ET0 ~ Tmax + Tmin + HR + Ux + Rs)
        # ---------------------------
        feats_lin = _presentes(df_in, ["Tmax","Tmin","HR","Ux","Rs"])
        if "ET0" in df_in.columns and len(feats_lin) >= 2:
            dfm = df_in[feats_lin + ["ET0"]].dropna()
            if len(dfm) > 20:
                X = dfm[feats_lin]; y = dfm["ET0"]
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                lm = LinearRegression().fit(Xtr, ytr)
                yhat = lm.predict(Xte)
                r2 = r2_score(yte, yhat); mse = mean_squared_error(yte, yhat)

                st.caption(f"**Regresión lineal — R²:** {r2:.4f}  ·  **MSE:** {mse:.4f}")
                fig, ax = plt.subplots(figsize=(6.0, 4.2))  # compacta + centrada
                ax.scatter(yte, yhat, s=14, alpha=0.8)
                lims = [min(yte.min(), yhat.min()), max(yte.max(), yhat.max())]
                ax.plot(lims, lims, "r--", linewidth=1)
                ax.set_xlabel("ET0 real"); ax.set_ylabel("ET0 predicho")
                ax.set_title("Real vs Predicho (Lineal)", fontsize=_TITLE)
                center(fig)

        # ---------------------------
        # 5) Random Forest (R², MSE + Importancia de variables)
        # ---------------------------
        feats_rf = _presentes(df_in, ["Tmax","Tmin","HR","Ux","Rs"])
        if "ET0" in df_in.columns and len(feats_rf) >= 2:
            dfr = df_in[feats_rf + ["ET0"]].dropna()
            if len(dfr) > 20:
                X = dfr[feats_rf]; y = dfr["ET0"]
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
                yhat = rf.predict(Xte)
                r2rf = r2_score(yte, yhat); mserf = mean_squared_error(yte, yhat)
                st.caption(f"**Random Forest — R²:** {r2rf:.4f}  ·  **MSE:** {mserf:.4f}")

                # Importancias (barras, respetando estilo matplotlib)
                imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
                f_imp, ax_imp = plt.subplots(figsize=(6.0, 4.2))  # compacta + centrada
                ax_imp.barh(imp.index, imp.values)
                ax_imp.set_title("Importancia de variables (RF)", fontsize=_TITLE)
                ax_imp.set_xlabel("Importancia"); ax_imp.set_ylabel("")
                center(f_imp)

        # ---------------------------
        # 6) KMeans SOLO meteorología (como el profe)
        #    - Método del codo (k=2..9)
        #    - k=5 => Grupo
        #    - Scatter Tmax vs Rs (Set2)
        #    - Distribución de días por grupo (scatter Día vs Grupo)
        #    - Boxplots por década y grupo (Plotly) — por variable meteo
        # ---------------------------
        meteo_cols = _presentes(df_in, ["Tmax","Tmin","HR","Ux","Rs"])
        Xmet = df_in[meteo_cols].dropna() if meteo_cols else pd.DataFrame()
        grupos_meteo = None
        if not Xmet.empty and Xmet.shape[1] >= 2:
            _hr()
            st.markdown("#### Clustering (KMeans) — Solo meteorología")

            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xmet)

            # Método del codo
            inertias, ks = [], list(range(2, 10))
            for k_ in ks:
                km = KMeans(n_clusters=k_, random_state=42, n_init=10).fit(Xs)
                inertias.append(km.inertia_)
            f_elb, ax_elb = plt.subplots(figsize=(6.0, 4.2))  # compacta + centrada
            ax_elb.plot(ks, inertias, marker="o")
            ax_elb.set_xlabel("k"); ax_elb.set_ylabel("Inercia")
            ax_elb.set_title("Método del codo", fontsize=_TITLE)
            center(f_elb)

            # k=5 (consistencia con el cuaderno)
            k_final = 5
            km = KMeans(n_clusters=k_final, random_state=42, n_init=10).fit(Xs)
            grupos_meteo = pd.Series(km.labels_, index=Xmet.index, name="Grupo")

            # Scatter Tmax vs Rs (Set2)
            if "Tmax" in df_in.columns and "Rs" in df_in.columns:
                import seaborn as sns
                fig_sc, ax_sc = plt.subplots(figsize=(7, 5))  # como profe
                tmp = df_in.loc[Xmet.index, ["Tmax","Rs"]].copy()
                tmp["Grupo"] = grupos_meteo
                sns.scatterplot(data=tmp, x="Tmax", y="Rs", hue="Grupo", palette="Set2", ax=ax_sc)
                ax_sc.set_title("Clasificación de días climáticos — (Solo meteo)", fontsize=_TITLE)
                st.pyplot(fig_sc, use_container_width=True)

            # Distribución de días por grupo (Día vs Grupo)
            dia_col = None
            for cand in ["Día","Dia","DOY","Dia_ciclo"]:
                if cand in df_in.columns and df_in[cand].notna().any():
                    dia_col = cand; break
            if dia_col is not None:
                df_days = df_in[[dia_col]].copy().loc[Xmet.index]
                df_days["Grupo"] = grupos_meteo
                fig_dg, ax_dg = plt.subplots(figsize=(10, 5))
                import seaborn as sns
                sns.scatterplot(data=df_days, x=dia_col, y="Grupo", hue="Grupo", palette="Set2", ax=ax_dg)
                ax_dg.set_title("Distribución de días por grupo — (Solo meteo)", fontsize=_TITLE)
                st.pyplot(fig_dg, use_container_width=True)

            # Boxplots por década y grupo (Plotly) — por CADA variable meteo (como el cuaderno)
            if "decada" in df_in.columns and grupos_meteo is not None:
                st.markdown("**Distribución por década y grupo (Solo meteo)**")
                for var in meteo_cols:
                    try:
                        join_df = df_in.loc[Xmet.index, ["decada", var]].dropna().copy()
                        join_df["Grupo"] = grupos_meteo.loc[join_df.index]
                        figpx = px.box(
                            join_df, x="decada", y=var, color="Grupo",
                            title=f"{var} por grupo y década — (Solo meteo)",
                            labels={"decada":"Década", var:var, "Grupo":"Grupo"},
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            points="all"
                        )
                        st.plotly_chart(figpx, use_container_width=True)
                    except Exception:
                        pass

        # ---------------------------
        # 7) KMeans con meteorología + ET0 + ETc (como el profe)
        #    - Método del codo
        #    - k=5 => Grupo2
        #    - Boxplots por variable (Seaborn, Set1)
        #    - Scatter Tmax vs Rs (Set1)
        # ---------------------------
        vars_km2 = _presentes(df_in, ["Tmax","Tmin","HR","Ux","Rs","ET0","ETc"])
        if len(vars_km2) >= 3:
            _hr()
            st.markdown("#### Clustering (KMeans) — Meteorología + ET0 + ETc")

            X1 = df_in[vars_km2].dropna()
            if not X1.empty:
                scaler = StandardScaler(); Xs1 = scaler.fit_transform(X1)

                # Método del codo
                inertias2, ks2 = [], list(range(2, 10))
                for k_ in ks2:
                    km2 = KMeans(n_clusters=k_, random_state=42, n_init=10).fit(Xs1)
                    inertias2.append(km2.inertia_)
                f_elb2, ax_elb2 = plt.subplots(figsize=(6.0, 4.2))  # compacta + centrada
                ax_elb2.plot(ks2, inertias2, marker="o")
                ax_elb2.set_xlabel("k"); ax_elb2.set_ylabel("Inercia")
                ax_elb2.set_title("Método del codo (ET0/ETc incluidos)", fontsize=_TITLE)
                center(f_elb2)

                # k=5
                k_final2 = 5
                km2 = KMeans(n_clusters=k_final2, random_state=42, n_init=10).fit(Xs1)
                g2 = pd.Series(km2.labels_, index=X1.index, name="Grupo2")

                # Estadísticos descriptivos por grupo (medias)
                stats_medias = X1.copy()
                stats_medias["Grupo2"] = g2
                st.markdown("**Estadísticas descriptivas por grupo (medias):**")
                st.dataframe(stats_medias.groupby("Grupo2")[vars_km2].mean(), use_container_width=True)

                # Boxplots por variable (Seaborn, paleta Set1), layout 2xN
                import seaborn as sns
                n = len(vars_km2)
                ncols = 2
                nrows = int(np.ceil(n / ncols))
                fig_bx, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4*nrows))
                axes = np.array(axes).reshape(nrows, ncols)
                idx = 0
                for r in range(nrows):
                    for c in range(ncols):
                        if idx < n:
                            var = vars_km2[idx]
                            sns.boxplot(x="Grupo2", y=var, data=stats_medias, ax=axes[r, c], palette="Set1")
                            axes[r, c].set_title(f"Distribución de {var} por grupo", fontsize=_TITLE)
                            axes[r, c].set_xlabel("Grupo"); axes[r, c].set_ylabel(var)
                            idx += 1
                        else:
                            axes[r, c].set_visible(False)
                st.pyplot(fig_bx, use_container_width=True)

                # Scatter Tmax vs Rs (Set1)
                if "Tmax" in stats_medias.columns and "Rs" in stats_medias.columns:
                    fig_sc2, ax_sc2 = plt.subplots(figsize=(7, 5))
                    sns.scatterplot(data=stats_medias, x="Tmax", y="Rs", hue="Grupo2", palette="Set1", ax=ax_sc2)
                    ax_sc2.set_title("Clasificación de días climáticos (ET0/ETc incluidos)", fontsize=_TITLE)
                    st.pyplot(fig_sc2, use_container_width=True)

        # ---------------------------
        # (Opcional) Resumen de métricas como “logs” en expander
        # ---------------------------
        with st.expander("Resultados y métricas (resumen tipo 'logs')"):
            blobs = []
            try:
                blobs.append(f"[Regresión lineal] R²={r2:.4f}, MSE={mse:.4f}")
            except: pass
            try:
                blobs.append(f"[Random Forest] R²={r2rf:.4f}, MSE={mserf:.4f}")
            except: pass
            try:
                if "stats_medias" in locals():
                    blobs.append("Medias por Grupo2 (ET0/ETc incluidos):")
                    blobs.append(stats_medias.groupby("Grupo2")[vars_km2].mean().to_string())
            except: pass
            st.text("\n".join(blobs) if blobs else "—")

    # ===========================
    # Enrutar por modo seleccionado (como en tu app)
    # ===========================
    st.subheader("Modelos y Estadística")

    if modo == "Ciclo individual":
        ruta_sel = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
        if ruta_sel.empty:
            st.warning(f"No encontré CSV para {region_sel} / {ciclo_sel}")
        else:
            dfM = leer_unison(ruta_sel.iloc[0])
            render_modelos_profesor(dfM, region_sel, ciclo_sel)

    elif modo == "Comparar ciclos":
        ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_A)]["Ruta"]
        ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_B)]["Ruta"]
        if ruta_A.empty or ruta_B.empty:
            st.warning("No encontré ambos ciclos para mostrar análisis.")
        else:
            dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])
            render_modelos_profesor(dfA, region_sel, ciclo_A)
            _hr()
            render_modelos_profesor(dfB, region_sel, ciclo_B)

    elif modo == "Comparar regiones":
        ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_A) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
        ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_B) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
        if ruta_A.empty or ruta_B.empty:
            st.warning("No encontré ambas regiones para mostrar análisis.")
        else:
            dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])
            render_modelos_profesor(dfA, region_A, ciclo_sel)
            _hr()
            render_modelos_profesor(dfB, region_B, ciclo_sel)

