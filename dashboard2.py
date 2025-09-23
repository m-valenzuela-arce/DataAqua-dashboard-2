# ===========================
# DataAqua Dashboard 2 (Streamlit)
# ===========================
# Modos:
#  - Individual
#  - Comparar ciclos (misma región)
#  - Comparar regiones (mismo ciclo)
#
# Pestañas:
#  - KPIs
#  - Serie diaria
#  - Acumulados
#  - Decádico
#  - Kc–ET0
#  - Drivers ET0
#
# Ejecuta:
#   streamlit run dashboard2.py
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
# CONFIG
# ---------------------------
st.set_page_config(page_title="DataAqua — Dashboard 2", page_icon="💧", layout="wide")

RUTA_BASE          = Path("/lustre/home/mvalenzuela/Ocotillo/DataAqua")
RUTA_SALIDA_UNISON = RUTA_BASE / "Salidas_ETo12" / "Periodo de Cultivo ETo"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# Column map (tus nombres)
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
COLUMNAS_MIN = [
    "Year","DOY","ET0","ETc","ETverde","ETazul","Pef","decada",
    "Rns","Rnl","Rs","Tmean","HR","Ux","Kc"
]

# ---------------------------
# Helpers
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

    # Acumulados útiles
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
    pef_total = float(df.loc[mask, "Pef"].sum())     if "Pef"     in df else np.nan
    pct_azul = (eta_total/etc_total*100.0) if (etc_total and etc_total>0) else np.nan
    dias_def = int(((df["ETc"] > df.get("Pef", 0)).fillna(False)).sum()) if "ETc" in df else np.nan
    pico_p95 = float(np.nanpercentile(df["ETc"], 95)) if "ETc" in df else np.nan
    return {"dias":dias,"etc_total":etc_total,"etv_total":etv_total,"eta_total":eta_total,
            "pct_azul":pct_azul,"pef_total":pef_total,"dias_def":dias_def,"pico_p95":pico_p95}

def _xcol(df: pd.DataFrame, prefer="Fecha"):
    if prefer in df.columns and df[prefer].notna().any(): return prefer
    for alt in ("Fecha","DOY","Dia_ciclo"):
        if alt in df.columns and df[alt].notna().any():
            return alt
    return df.index

# --- Figuras (devuelven fig) ---
def fig_series(df: pd.DataFrame, titulo: str, eje="Fecha", mostrar=("ET0","ETc","ETverde","ETazul","Pef")):
    x = _xcol(df, eje)
    fig, ax = plt.subplots(1,1, figsize=(12,4))
    colores = {"ET0":"#4C78A8","ETc":"#F58518","ETverde":"#54A24B","ETazul":"#E45756","Pef":"#9D9D9D"}
    for col in mostrar:
        if col in df:
            ax.plot(df[x], df[col], label=col, lw=1.6, color=colores.get(col, None))
    ax.set_title(titulo)
    ax.set_xlabel(str(x)); ax.set_ylabel("mm/día")
    ax.legend()
    fig.tight_layout()
    return fig

def fig_acumulados(df: pd.DataFrame, titulo: str, eje="Fecha"):
    x = _xcol(df, eje)
    fig, ax = plt.subplots(1,1, figsize=(12,4))
    if "ETc_acum" in df:
        ax.plot(df[x], df["ETc_acum"], label="ETc acumulado", lw=1.8)
    if "ETazul_acum" in df:
        ax.plot(df[x], df["ETazul_acum"], label="ETazul acumulado", lw=1.8)
    ax.set_title(titulo)
    ax.set_xlabel(str(x)); ax.set_ylabel("mm")
    ax.legend()
    fig.tight_layout()
    return fig

def fig_decadico(df: pd.DataFrame, titulo: str):
    if "decada" not in df: return None
    g = df.groupby("decada")[["ETc","ETazul"]].sum(min_count=1)
    fig, ax = plt.subplots(1,1, figsize=(10,4))
    g["ETc"].plot(kind="bar", ax=ax, color="#4C78A8", label="ETc")
    if "ETazul" in g:
        ax.plot(np.arange(len(g)), g["ETazul"].values, color="#F58518", lw=2, marker="o", label="ETazul")
    ax.set_title(titulo)
    ax.set_xlabel("Década del ciclo"); ax.set_ylabel("mm/decada")
    ax.legend()
    fig.tight_layout()
    return fig

def fig_kc_et0(df: pd.DataFrame, titulo: str, eje="Fecha"):
    if "Kc" not in df or "ET0" not in df: return None
    x = _xcol(df, eje)
    fig, ax1 = plt.subplots(1,1, figsize=(12,4))
    ax1.plot(df[x], df["ET0"], color="#4C78A8", label="ET0", lw=1.5)
    ax1.set_ylabel("ET0 [mm/día]", color="#4C78A8"); ax1.tick_params(axis='y', labelcolor="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(df[x], df["Kc"], color="#E45756", label="Kc", lw=1.5)
    ax2.set_ylabel("Kc [-]", color="#E45756"); ax2.tick_params(axis='y', labelcolor="#E45756")
    ax1.set_title(titulo); ax1.set_xlabel(str(x))
    fig.tight_layout()
    return fig

def fig_drivers_et0(df: pd.DataFrame, titulo: str):
    if "ET0" not in df: return None
    drivers = [("Rs","Rs [MJ m$^{-2}$ d$^{-1}$]"),
               ("Rnl","Rnl [MJ m$^{-2}$ d$^{-1}$]"),
               ("HR","HR [%]"),
               ("Ux","Viento Ux [m/s]"),
               ("Tmean","Tmean [°C]")]
    cols = [c for c,_ in drivers if c in df.columns]
    if not cols: return None
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
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig

# ---------------------------
# UI — Sidebar
# ---------------------------
st.sidebar.title("DataAqua — Selección")
CAT_UNISON = catalogo_unison(RUTA_SALIDA_UNISON)
if CAT_UNISON.empty:
    st.error("No se encontraron archivos en Salidas_ETo12/Periodo de Cultivo ETo.")
    st.stop()

modo = st.sidebar.radio("Modo", ["Individual", "Comparar ciclos", "Comparar regiones"], index=0)
eje_opt = st.sidebar.radio("Eje X:", ["Fecha","DOY","Dia_ciclo"], index=0)

if modo == "Individual":
    regiones = sorted(CAT_UNISON["Region"].unique())
    region_sel = st.sidebar.selectbox("Región:", regiones)
    ciclos_reg = sorted(CAT_UNISON.loc[CAT_UNISON["Region"]==region_sel, "Ciclo"].unique())
    ciclo_sel = st.sidebar.selectbox("Ciclo:", ciclos_reg)

    st.sidebar.markdown("#### Gráficas")
    show_series     = st.sidebar.checkbox("Serie diaria (ET0, ETc, ETverde, ETazul, Pef)", value=True)
    show_acumulados = st.sidebar.checkbox("Acumulados (ETc, ETazul)", value=True)
    show_decadico   = st.sidebar.checkbox("Decádico (ETc + ETazul)", value=True)
    show_kc         = st.sidebar.checkbox("Kc y ET0", value=True)
    show_drivers    = st.sidebar.checkbox("Drivers de ET0 (scatter)", value=True)

elif modo == "Comparar ciclos":
    regiones = sorted(CAT_UNISON["Region"].unique())
    region_sel = st.sidebar.selectbox("Región:", regiones)
    ciclos_reg = sorted(CAT_UNISON.loc[CAT_UNISON["Region"]==region_sel, "Ciclo"].unique())
    colA, colB = st.sidebar.columns(2)
    ciclo_A = colA.selectbox("Ciclo A", ciclos_reg, key="ciclo_A")
    ciclo_B = colB.selectbox("Ciclo B", ciclos_reg, index=min(1, len(ciclos_reg)-1), key="ciclo_B")

elif modo == "Comparar regiones":
    ciclos = sorted(CAT_UNISON["Ciclo"].unique())
    ciclo_sel = st.sidebar.selectbox("Ciclo:", ciclos)
    regs_ciclo = sorted(CAT_UNISON.loc[CAT_UNISON["Ciclo"]==ciclo_sel, "Region"].unique())
    colA, colB = st.sidebar.columns(2)
    region_A = colA.selectbox("Región A", regs_ciclo, key="region_A")
    region_B = colB.selectbox("Región B", regs_ciclo, index=min(1, len(regs_ciclo)-1), key="region_B")

# ---------------------------
# Layout principal
# ---------------------------
st.title("💧 DataAqua — Dashboard 2")
st.caption("Resultados UNISON (FAO-56). ETc (demanda del cultivo), ETverde (cubierta por Pef) y ETazul (resto). ET0 es referencia (césped).")

tabs_main = st.tabs(["KPIs", "Serie diaria", "Acumulados", "Decádico", "Kc–ET0", "Drivers ET0", "Datos"])

if modo == "Individual":
    ruta_sel = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
    if ruta_sel.empty:
        st.error(f"No encontré CSV para {region_sel} / {ciclo_sel}")
        st.stop()
    df = leer_unison(ruta_sel.iloc[0])
    if df.empty:
        st.error("No fue posible leer el archivo seleccionado.")
        st.stop()

    with tabs_main[0]:
        st.subheader(f"KPIs — {region_sel} ({ciclo_sel})")
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

    with tabs_main[1]:
        if show_series:
            fig = fig_series(df, f"Serie diaria — {region_sel} ({ciclo_sel})", eje=eje_opt)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Activa 'Serie diaria' en el panel izquierdo.")

    with tabs_main[2]:
        if show_acumulados:
            fig = fig_acumulados(df, f"Acumulados — {region_sel} ({ciclo_sel})", eje=eje_opt)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Activa 'Acumulados' en el panel izquierdo.")

    with tabs_main[3]:
        if show_decadico:
            fdec = fig_decadico(df, f"Decádico — {region_sel} ({ciclo_sel})")
            if fdec is not None:
                st.pyplot(fdec, use_container_width=True)
            else:
                st.info("No hay columna 'decada' en este archivo.")
        else:
            st.info("Activa 'Decádico' en el panel izquierdo.")

    with tabs_main[4]:
        if show_kc:
            fkc = fig_kc_et0(df, f"Kc y ET0 — {region_sel} ({ciclo_sel})", eje=eje_opt)
            if fkc is not None:
                st.pyplot(fkc, use_container_width=True)
            else:
                st.info("Faltan columnas 'Kc' o 'ET0'.")
        else:
            st.info("Activa 'Kc y ET0' en el panel izquierdo.")

    with tabs_main[5]:
        if show_drivers:
            fdrv = fig_drivers_et0(df, f"Drivers de ET0 — {region_sel} ({ciclo_sel})")
            if fdrv is not None:
                st.pyplot(fdrv, use_container_width=True)
            else:
                st.info("Faltan columnas para drivers (ET0 y Rs/Rnl/HR/Ux/Tmean).")
        else:
            st.info("Activa 'Drivers de ET0' en el panel izquierdo.")

    with tabs_main[6]:
        st.dataframe(df.head(30), use_container_width=True)
        @st.cache_data(show_spinner=False)
        def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
            return df_in.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar CSV (ciclo seleccionado)",
            data=to_csv_bytes(df),
            file_name=f"{region_sel}_{ciclo_sel}_DataAqua.csv",
            mime="text/csv"
        )

elif modo == "Comparar ciclos":
    ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_A)]["Ruta"]
    ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_B)]["Ruta"]
    if ruta_A.empty or ruta_B.empty:
        st.error("No encontré ambos ciclos para esa región.")
        st.stop()
    dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])

    with tabs_main[0]:
        st.subheader(f"KPIs — {region_sel} | {ciclo_A} vs {ciclo_B}")
        c1, c2 = st.columns(2)
        kA, kB = kpis_ciclo(dfA), kpis_ciclo(dfB)
        with c1:
            st.markdown(f"**Ciclo A:** {ciclo_A}")
            st.metric("ETc total [mm]", f"{kA['etc_total']:.1f}")
            st.metric("ETazul total [mm]", f"{kA['eta_total']:.1f}")
            st.metric("% Azul", f"{kA['pct_azul']:.1f}%")
            st.metric("Días (ETc>Pef)", f"{kA['dias_def']}")
        with c2:
            st.markdown(f"**Ciclo B:** {ciclo_B}")
            st.metric("ETc total [mm]", f"{kB['etc_total']:.1f}")
            st.metric("ETazul total [mm]", f"{kB['eta_total']:.1f}")
            st.metric("% Azul", f"{kB['pct_azul']:.1f}%")
            st.metric("Días (ETc>Pef)", f"{kB['dias_def']}")

    with tabs_main[1]:
        xA = _xcol(dfA, eje_opt); xB = _xcol(dfB, eje_opt)
        fig, ax = plt.subplots(1,1, figsize=(12,4))
        if "ETc" in dfA: ax.plot(dfA[xA], dfA["ETc"], label=f"ETc {ciclo_A}", lw=1.5, color="#1f77b4")
        if "ETc" in dfB: ax.plot(dfB[xB], dfB["ETc"], label=f"ETc {ciclo_B}", lw=1.5, color="#ff7f0e")
        if "ETazul" in dfA: ax.plot(dfA[xA], dfA["ETazul"], label=f"ETazul {ciclo_A}", lw=1.2, color="#1f77b4", ls="--")
        if "ETazul" in dfB: ax.plot(dfB[xB], dfB["ETazul"], label=f"ETazul {ciclo_B}", lw=1.2, color="#ff7f0e", ls="--")
        ax.set_title(f"Serie diaria — {region_sel}")
        ax.set_xlabel(eje_opt); ax.set_ylabel("mm/día"); ax.legend()
        fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    with tabs_main[2]:
        fig, ax = plt.subplots(1,1, figsize=(12,4))
        if "ETc" in dfA: ax.plot(dfA[_xcol(dfA,eje_opt)], dfA["ETc"].cumsum(), label=f"ETc {ciclo_A}", lw=1.8, color="#1f77b4")
        if "ETc" in dfB: ax.plot(dfB[_xcol(dfB,eje_opt)], dfB["ETc"].cumsum(), label=f"ETc {ciclo_B}", lw=1.8, color="#ff7f0e")
        if "ETazul" in dfA: ax.plot(dfA[_xcol(dfA,eje_opt)], dfA["ETazul"].cumsum(), label=f"ETazul {ciclo_A}", lw=1.8, color="#1f77b4", ls="--")
        if "ETazul" in dfB: ax.plot(dfB[_xcol(dfB,eje_opt)], dfB["ETazul"].cumsum(), label=f"ETazul {ciclo_B}", lw=1.8, color="#ff7f0e", ls="--")
        ax.set_title(f"Acumulados — {region_sel}")
        ax.set_xlabel(eje_opt); ax.set_ylabel("mm"); ax.legend()
        fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    with tabs_main[3]:
        fA = fig_decadico(dfA, f"Decádico — {region_sel} ({ciclo_A})")
        fB = fig_decadico(dfB, f"Decádico — {region_sel} ({ciclo_B})")
        cols = st.columns(2)
        if fA: cols[0].pyplot(fA, use_container_width=True)
        if fB: cols[1].pyplot(fB, use_container_width=True)

    with tabs_main[4]:
        fA = fig_kc_et0(dfA, f"Kc–ET0 — {region_sel} ({ciclo_A})", eje=eje_opt)
        fB = fig_kc_et0(dfB, f"Kc–ET0 — {region_sel} ({ciclo_B})", eje=eje_opt)
        cols = st.columns(2)
        if fA: cols[0].pyplot(fA, use_container_width=True)
        if fB: cols[1].pyplot(fB, use_container_width=True)

    with tabs_main[5]:
        fA = fig_drivers_et0(dfA, f"Drivers ET0 — {region_sel} ({ciclo_A})")
        fB = fig_drivers_et0(dfB, f"Drivers ET0 — {region_sel} ({ciclo_B})")
        cols = st.columns(2)
        if fA: cols[0].pyplot(fA, use_container_width=True)
        if fB: cols[1].pyplot(fB, use_container_width=True)

    with tabs_main[6]:
        st.write("**Primeras filas ciclo A**")
        st.dataframe(dfA.head(20), use_container_width=True)
        st.write("**Primeras filas ciclo B**")
        st.dataframe(dfB.head(20), use_container_width=True)

elif modo == "Comparar regiones":
    ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_A) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
    ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_B) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
    if ruta_A.empty or ruta_B.empty:
        st.error("No encontré ambas regiones para ese ciclo.")
        st.stop()
    dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])

    with tabs_main[0]:
        st.subheader(f"KPIs — {ciclo_sel} | {region_A} vs {region_B}")
        c1, c2 = st.columns(2)
        kA, kB = kpis_ciclo(dfA), kpis_ciclo(dfB)
        with c1:
            st.markdown(f"**{region_A}**")
            st.metric("ETc total [mm]", f"{kA['etc_total']:.1f}")
            st.metric("ETazul total [mm]", f"{kA['eta_total']:.1f}")
            st.metric("% Azul", f"{kA['pct_azul']:.1f}%")
            st.metric("Días (ETc>Pef)", f"{kA['dias_def']}")
        with c2:
            st.markdown(f"**{region_B}**")
            st.metric("ETc total [mm]", f"{kB['etc_total']:.1f}")
            st.metric("ETazul total [mm]", f"{kB['eta_total']:.1f}")
            st.metric("% Azul", f"{kB['pct_azul']:.1f}%")
            st.metric("Días (ETc>Pef)", f"{kB['dias_def']}")

    with tabs_main[1]:
        xA = _xcol(dfA, eje_opt); xB = _xcol(dfB, eje_opt)
        fig, ax = plt.subplots(1,1, figsize=(12,4))
        if "ETc" in dfA: ax.plot(dfA[xA], dfA["ETc"], label=f"{region_A} ETc", lw=1.5, color="#1f77b4")
        if "ETc" in dfB: ax.plot(dfB[xB], dfB["ETc"], label=f"{region_B} ETc", lw=1.5, color="#ff7f0e")
        if "ETazul" in dfA: ax.plot(dfA[xA], dfA["ETazul"], label=f"{region_A} ETazul", lw=1.2, color="#1f77b4", ls="--")
        if "ETazul" in dfB: ax.plot(dfB[xB], dfB["ETazul"], label=f"{region_B} ETazul", lw=1.2, color="#ff7f0e", ls="--")
        ax.set_title(f"Serie diaria — {region_A} vs {region_B}")
        ax.set_xlabel(eje_opt); ax.set_ylabel("mm/día"); ax.legend()
        fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    with tabs_main[2]:
        fig, ax = plt.subplots(1,1, figsize=(12,4))
        if "ETc" in dfA: ax.plot(dfA[_xcol(dfA,eje_opt)], dfA["ETc"].cumsum(), label=f"{region_A} ETc", lw=1.8, color="#1f77b4")
        if "ETc" in dfB: ax.plot(dfB[_xcol(dfB,eje_opt)], dfB["ETc"].cumsum(), label=f"{region_B} ETc", lw=1.8, color="#ff7f0e")
        if "ETazul" in dfA: ax.plot(dfA[_xcol(dfA,eje_opt)], dfA["ETazul"].cumsum(), label=f"{region_A} ETazul", lw=1.8, color="#1f77b4", ls="--")
        if "ETazul" in dfB: ax.plot(dfB[_xcol(dfB,eje_opt)], dfB["ETazul"].cumsum(), label=f"{region_B} ETazul", lw=1.8, color="#ff7f0e", ls="--")
        ax.set_title(f"Acumulados — {region_A} vs {region_B}")
        ax.set_xlabel(eje_opt); ax.set_ylabel("mm"); ax.legend()
        fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    with tabs_main[3]:
        fA = fig_decadico(dfA, f"Decádico — {region_A} ({ciclo_sel})")
        fB = fig_decadico(dfB, f"Decádico — {region_B} ({ciclo_sel})")
        cols = st.columns(2)
        if fA: cols[0].pyplot(fA, use_container_width=True)
        if fB: cols[1].pyplot(fB, use_container_width=True)

    with tabs_main[4]:
        fA = fig_kc_et0(dfA, f"Kc–ET0 — {region_A} ({ciclo_sel})", eje=eje_opt)
        fB = fig_kc_et0(dfB, f"Kc–ET0 — {region_B} ({ciclo_sel})", eje=eje_opt)
        cols = st.columns(2)
        if fA: cols[0].pyplot(fA, use_container_width=True)
        if fB: cols[1].pyplot(fB, use_container_width=True)

    with tabs_main[5]:
        fA = fig_drivers_et0(dfA, f"Drivers ET0 — {region_A} ({ciclo_sel})")
        fB = fig_drivers_et0(dfB, f"Drivers ET0 — {region_B} ({ciclo_sel})")
        cols = st.columns(2)
        if fA: cols[0].pyplot(fA, use_container_width=True)
        if fB: cols[1].pyplot(fB, use_container_width=True)

    with tabs_main[6]:
        st.write(f"**Primeras filas {region_A}**")
        st.dataframe(dfA.head(20), use_container_width=True)
        st.write(f"**Primeras filas {region_B}**")
        st.dataframe(dfB.head(20), use_container_width=True)
