# ===========================
# DataAqua Dashboard 3 (Streamlit + Plotly)
# Nuevo layout + paletas conmutables
# ===========================
# Ejecuta:
#   streamlit run dashboard3.py
# ===========================

from pathlib import Path
import os, re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------- CONFIG BÁSICA ----------
st.set_page_config(page_title="DataAqua — Dashboard 3", page_icon="💧", layout="wide")

plt_title = 13
DATA_DIR = Path("data") / "Salidas_ETo12_con_uac_y_hh" / "Periodo de Cultivo ETo"

# --------- ESTILOS / CSS ----------
CSS = """
<style>
/* Header pegajoso + fondo */
.main > div:first-child { padding-top: 0.2rem; }
.block-container { padding-top: 0.8rem; }

/* Cards KPI */
.card {
  border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px;
  background: var(--card-bg,#ffffff); box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.card h4 { margin: 0 0 8px 0; font-size: 0.95rem; color: var(--card-head,#111827); }
.card .metric { font-size: 1.35rem; font-weight: 700; color: var(--card-num,#111827); }
.card .sub { font-size: 0.8rem; color: #6b7280; }

/* Expander más delgado */
.streamlit-expanderHeader { font-size: 0.95rem !important; }

/* Botones "Todo/Nada" compactos */
.small-btn > button { padding: 0.15rem 0.4rem; font-size: 0.75rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------- MAPEO DE COLUMNAS (igual al tuyo) ----------
MAP_UNISON = {
    "Año_ (YEAR)": "Year", "AÃ±o_ (YEAR)": "Year",
    "Día (DOY)": "DOY",   "DÃ­a (DOY)": "DOY",
    "Tmax (T2M_MAX)": "Tmax", "Tmin (T2M_MIN)": "Tmin",
    "HR (RH2M)": "HR", "Ux (WS2M)": "Ux",
    "Rs (ALLSKY_SFC_SW_DWN)": "Rs", "Rl_ (ALLSKY_SFC_LW_DWN)": "Rl",
    "Ptot_ (PRECTOTCORR)": "Ptot", "Pef_": "Pef", "Tmean_": "Tmean",
    "es_": "es", "ea_": "ea", "delta_": "delta", "P_": "P", "gamma_": "gamma",
    "Rns_": "Rns", "Rnl_": "Rnl", "Rn_": "Rn", "Rso_": "Rso",
    "Kc_": "Kc", "decada_": "decada",
    "ET0": "ET0", "ETc": "ETc", "ETverde": "ETverde", "ETazul": "ETazul",
    "Year": "Year", "DOY": "DOY", "Dia": "Dia",
}
NUM_COLS = [
    "Year","DOY","ET0","ETc","ETverde","ETazul","Pef","decada",
    "Rns","Rnl","Rs","Tmean","HR","Ux","Kc","Tmax","Tmin",
    "UACverde_m3_ha","UACazul_m3_ha","HHverde_m3_ton","HHazul_m3_ton"
]

# --------- HELPERS ARCHIVOS ----------
def parse_unison_filename(filename: str):
    m = re.match(r"([\wÁÉÍÓÚáéíóúñÑ\s\-]+)-FAO56-(\d{4})(?:-(\d{4}))?-SALIDA\.csv$", filename, re.I)
    if not m: return None, None
    reg, y1, y2 = m.groups()
    if reg == "VillaAllende": reg = "Villa de Allende"
    if reg == "Etchhojoa":    reg = "Etchojoa"
    ciclo = y1 if not y2 else f"{y1}-{y2}"
    return reg.strip(), ciclo

@st.cache_data(show_spinner=False)
def catalogo(base_dir: Path) -> pd.DataFrame:
    rows = []
    if not base_dir.exists(): return pd.DataFrame(columns=["Region","Ciclo","Ruta"])
    for reg_folder in sorted(os.listdir(base_dir)):
        d = base_dir / reg_folder
        if not d.is_dir(): continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".csv"): continue
            reg, ciclo = parse_unison_filename(f)
            if reg and ciclo: rows.append({"Region": reg, "Ciclo": ciclo, "Ruta": str(d / f)})
    return pd.DataFrame(rows).sort_values(["Region","Ciclo"]).reset_index(drop=True)

# --------- HELPERS DATOS ----------
def _year_doy_to_date(y, doy):
    try:
        base = datetime(int(y), 1, 1)
        return base + timedelta(days=int(doy) - 1)
    except Exception:
        return pd.NaT

@st.cache_data(show_spinner=False, ttl=300)
def leer_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists(): return pd.DataFrame()
    last_err = None
    for enc in ("utf-8","latin-1"):
        try:
            df = pd.read_csv(p, encoding=enc); last_err = None; break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err is not None: df = pd.read_csv(p)

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=lambda c: MAP_UNISON.get(c, c))
    for c in set(NUM_COLS).intersection(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

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

    if "ETc" in df: df["ETc_acum"] = df["ETc"].cumsum()
    if "ETazul" in df: df["ETazul_acum"] = df["ETazul"].cumsum()
    if {"ETc","ETazul"}.issubset(df.columns):
        df["pct_azul"] = np.where(df["ETc"]>0, df["ETazul"]/df["ETc"]*100.0, np.nan)
    return df

def kpis_ext(df: pd.DataFrame):
    def last_valid(df, col): 
        return float(df[col].dropna().iloc[-1]) if (col in df and df[col].notna().any()) else np.nan
    dias = int(df["ETc"].notna().sum()) if "ETc" in df else 0
    etc_total = float(df["ETc"].sum()) if "ETc" in df else np.nan
    etv_total = float(df["ETverde"].sum()) if "ETverde" in df else np.nan
    eta_total = float(df["ETazul"].sum()) if "ETazul" in df else np.nan
    siembra = pd.to_datetime(df["Fecha"].dropna().iloc[0]).date() if "Fecha" in df and df["Fecha"].notna().any() else None
    cosecha = pd.to_datetime(df["Fecha"].dropna().iloc[-1]).date() if "Fecha" in df and df["Fecha"].notna().any() else None
    return {
        "dias": dias, "siembra": siembra, "cosecha": cosecha,
        "etc_total": etc_total, "etv_total": etv_total, "eta_total": eta_total,
        "tmax": float(df["Tmax"].max()) if "Tmax" in df else np.nan,
        "tmin": float(df["Tmin"].min()) if "Tmin" in df else np.nan,
        "uacv_ha": last_valid(df, "UACverde_m3_ha"),
        "uaca_ha": last_valid(df, "UACazul_m3_ha"),
    }

def best_x(df: pd.DataFrame, prefer="Fecha"):
    if prefer in df.columns and df[prefer].notna().any(): return prefer
    for alt in ("Fecha","Dia_ciclo","DOY"):
        if alt in df.columns and df[alt].notna().any(): return alt
    return df.index.name or "index"

# --------- PALETAS ----------
PALETAS = {
    "Teal-Orange": ["#0d9488","#f59e0b","#10b981","#ef4444","#6366f1","#0ea5e9"],
    "Viridis": px.colors.sequential.Viridis,
    "Set2": px.colors.qualitative.Set2,
    "Monocromo oscuro": ["#a3a3a3","#7c7c7c","#5e5e5e","#444","#2f2f2f","#171717"],
}

# --------- SIDEBAR ----------
st.sidebar.title("💧 DataAqua — Dash 3")
CAT = catalogo(DATA_DIR)
if CAT.empty:
    st.sidebar.error("No se encontraron archivos en la carpeta de datos.")
    st.stop()

with st.sidebar:
    paleta = st.selectbox("Paleta de colores", list(PALETAS.keys()), index=0)
    tema_oscuro = st.toggle("Tema oscuro", value=False)
    st.markdown("---")
    modo = st.radio("Modo", ["Ciclo individual","Comparar ciclos","Comparar regiones"], index=0)
    verpor = st.radio("Eje X", ["Fecha","Día del ciclo"], index=0)
    eje_x = "Dia_ciclo" if verpor == "Día del ciclo" else "Fecha"

    if modo == "Ciclo individual":
        regiones = sorted(CAT["Region"].unique())
        region_sel = st.selectbox("Región", regiones)
        ciclos_reg = sorted(CAT.loc[CAT["Region"]==region_sel, "Ciclo"].unique())
        ciclo_sel = st.selectbox("Ciclo", ciclos_reg)

    elif modo == "Comparar ciclos":
        regiones = sorted(CAT["Region"].unique())
        region_sel = st.selectbox("Región", regiones)
        ciclos_reg = sorted(CAT.loc[CAT["Region"]==region_sel, "Ciclo"].unique())
        ciclo_A = st.selectbox("Ciclo A", ciclos_reg, key="ciclo_A3")
        ciclo_B = st.selectbox("Ciclo B", ciclos_reg, index=min(1, len(ciclos_reg)-1), key="ciclo_B3")

    else:  # Comparar regiones
        ciclos = sorted(CAT["Ciclo"].unique())
        ciclo_sel = st.selectbox("Ciclo", ciclos)
        regs_ciclo = sorted(CAT.loc[CAT["Ciclo"]==ciclo_sel, "Region"].unique())
        region_A = st.selectbox("Región A", regs_ciclo, key="region_A3")
        region_B = st.selectbox("Región B", regs_ciclo, index=min(1, len(regs_ciclo)-1), key="region_B3")

    with st.expander("Subir CSV manual (prueba rápida)"):
        up = st.file_uploader("CSV UNISON", type=["csv"], key="up3")
        if up is not None:
            df_up = pd.read_csv(up, encoding="latin-1")
            df_up = df_up.rename(columns=lambda c: MAP_UNISON.get(c.strip(), c.strip()))
            st.session_state["__df_upload3__"] = df_up

# --------- HEADER ----------
title_col1, title_col2 = st.columns([0.75,0.25])
with title_col1:
    st.markdown("## 💧 DataAqua — Dashboard 3")
    st.caption("Layout alternativo con Plotly interactivo, tarjetas KPI y paletas conmutables.")
with title_col2:
    st.markdown(f"**Paleta:** {paleta}")

# Ajuste de tema (colores de tarjetas)
if tema_oscuro:
    st.markdown("<style>:root{--card-bg:#0b0f19; --card-head:#e5e7eb; --card-num:#f8fafc}</style>", unsafe_allow_html=True)

# --------- FUNCIONES DE PLOT ----------
def multiselect_all(label, options, default_all=True, key="ms"):
    cols = st.columns([4,1,1])
    with cols[0]:
        sel = st.multiselect(label, options, default=options if default_all else [], key=key)
    with cols[1]:
        if st.button("Todo", key=f"{key}_all", use_container_width=True): 
            sel = options; st.session_state[key] = sel
    with cols[2]:
        if st.button("Nada", key=f"{key}_none", use_container_width=True): 
            sel = []; st.session_state[key] = sel
    return sel

def px_line(df, x, y_list, title, colors):
    fig = go.Figure()
    for i, col in enumerate(y_list):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=col,
                                     line=dict(color=colors[i % len(colors)], width=2)))
    fig.update_layout(
        title=title, template="plotly_white",
        margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    return fig

def overlay_AB(dfA, dfB, x, cols, labelA="A", labelB="B", colors=None):
    fig = go.Figure()
    cols = [c for c in cols if c in set(dfA.columns) | set(dfB.columns)]
    colors = colors or PALETAS[paleta]
    for j, c in enumerate(cols):
        if c in dfA.columns:
            fig.add_trace(go.Scatter(x=dfA[x], y=dfA[c], mode="lines", name=f"{c} ({labelA})",
                                     line=dict(color=colors[j % len(colors)], width=2, dash="solid")))
        if c in dfB.columns:
            fig.add_trace(go.Scatter(x=dfB[x], y=dfB[c], mode="lines", name=f"{c} ({labelB})",
                                     line=dict(color=colors[j % len(colors)], width=2, dash="dash")))
    fig.update_layout(title="Comparación (Overlay)", template="plotly_white",
                      margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h", y=1.02))
    return fig

def card_metric(title, value, sub=None):
    v = "—" if value is None or (isinstance(value, float) and np.isnan(value)) else value
    sub = sub or ""
    st.markdown(f"""
    <div class="card">
      <h4>{title}</h4>
      <div class="metric">{v}</div>
      <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def fmt_num(x, dec=1, unit=""):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "—"
    return f"{x:.{dec}f}{unit}"

# --------- LOGICA DE MODOS ----------
palette_vals = PALETAS[paleta]

if modo == "Ciclo individual":
    ruta_sel = CAT[(CAT.Region==region_sel) & (CAT.Ciclo==ciclo_sel)]["Ruta"]
    if ruta_sel.empty: st.error(f"No encontré CSV para {region_sel} / {ciclo_sel}"); st.stop()
    df = leer_csv(ruta_sel.iloc[0])
    df = st.session_state.get("__df_upload3__") or df
    if df.empty: st.error("No fue posible leer el archivo seleccionado."); st.stop()

    xcol = best_x(df, prefer=eje_x)
    kp = kpis_ext(df)

    # ---- KPIs (tarjetas en 5 columnas) ----
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: card_metric("Días del ciclo", kp["dias"])
    with c2: card_metric("Siembra", kp["siembra"], "fecha")
    with c3: card_metric("Cosecha", kp["cosecha"], "fecha")
    with c4: card_metric("ETc total", fmt_num(kp["etc_total"],1," mm"))
    with c5: card_metric("ET azul total", fmt_num(kp["eta_total"],1," mm"))
    c6,c7,c8,c9,c10 = st.columns(5)
    with c6: 
        pct_azul = (kp["eta_total"]/kp["etc_total"]*100) if (kp["etc_total"] and kp["etc_total"]>0) else np.nan
        card_metric("% Azul", fmt_num(pct_azul,1,"%"))
    with c7: card_metric("ET verde total", fmt_num(kp["etv_total"],1," mm"))
    with c8: card_metric("Tmax / Tmin", f"{fmt_num(kp['tmax'],1,'°C')} / {fmt_num(kp['tmin'],1,'°C')}")
    with c9: card_metric("UAC verde", fmt_num(kp["uacv_ha"],0," m³/ha"))
    with c10: card_metric("UAC azul", fmt_num(kp["uaca_ha"],0," m³/ha"))

    st.markdown("---")

    # ---- Serie ET (interactiva) ----
    st.subheader("Serie diaria — ET")
    et_opts = [c for c in ["ET0","ETc","ETverde","ETazul","Pef"] if c in df.columns]
    et_sel = multiselect_all("Series ET", et_opts, key="et_ind3")
    if et_sel:
        fig = px_line(df, xcol, et_sel, f"{region_sel} — {ciclo_sel}", colors=palette_vals)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Temperaturas ----
    with st.expander("Temperaturas", expanded=True):
        t_opts = [c for c in ["Tmin","Tmean","Tmax"] if c in df.columns]
        t_sel = multiselect_all("Series T", t_opts, key="t_ind3")
        if t_sel:
            st.plotly_chart(px_line(df, xcol, t_sel, "Temperaturas", colors=palette_vals), use_container_width=True)

    # ---- Meteorología (Rs, HR) + Viento ----
    colM, colW = st.columns([0.65,0.35])
    with colM:
        with st.expander("Radiación & Humedad", expanded=True):
            met_opts = [c for c in ["Rs","HR"] if c in df.columns]
            met_sel = multiselect_all("Variables", met_opts, key="met_ind3")
            if met_sel:
                st.plotly_chart(px_line(df, xcol, met_sel, "Meteo (Rs/HR)", colors=palette_vals), use_container_width=True)
    with colW:
        with st.expander("Viento Ux", expanded=False):
            if "Ux" in df.columns:
                st.plotly_chart(px_line(df, xcol, ["Ux"], "Viento (Ux)", colors=palette_vals), use_container_width=True)

    # ---- Acumulados (extra) ----
    with st.expander("Acumulados (ETc / ETazul)"):
        cols_acc = [c for c in ["ETc_acum","ETazul_acum"] if c in df.columns]
        if cols_acc:
            st.plotly_chart(px_line(df, xcol, cols_acc, "Acumulados", colors=palette_vals), use_container_width=True)

    # ---- Datos ----
    with st.expander("Datos (primeras filas)"):
        st.dataframe(df.head(50), use_container_width=True)

elif modo == "Comparar ciclos":
    ruta_A = CAT[(CAT.Region==region_sel) & (CAT.Ciclo==ciclo_A)]["Ruta"]
    ruta_B = CAT[(CAT.Region==region_sel) & (CAT.Ciclo==ciclo_B)]["Ruta"]
    if ruta_A.empty or ruta_B.empty: st.error("No encontré ambos ciclos."); st.stop()
    dfA = leer_csv(ruta_A.iloc[0]); dfB = leer_csv(ruta_B.iloc[0])
    xA, xB = best_x(dfA, prefer=eje_x), best_x(dfB, prefer=eje_x)

    st.subheader(f"{region_sel} — comparación de ciclos")
    kA, kB = kpis_ext(dfA), kpis_ext(dfB)
    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"#### Ciclo A: {ciclo_A}")
        c1,c2,c3 = st.columns(3)
        with c1: card_metric("Días", kA["dias"])
        with c2: card_metric("ETc", fmt_num(kA["etc_total"],1," mm"))
        with c3: 
            pctA = (kA["eta_total"]/kA["etc_total"]*100) if (kA["etc_total"] and kA["etc_total"]>0) else np.nan
            card_metric("% Azul", fmt_num(pctA,1,"%"))
    with colB:
        st.markdown(f"#### Ciclo B: {ciclo_B}")
        c1,c2,c3 = st.columns(3)
        with c1: card_metric("Días", kB["dias"])
        with c2: card_metric("ETc", fmt_num(kB["etc_total"],1," mm"))
        with c3: 
            pctB = (kB["eta_total"]/kB["etc_total"]*100) if (kB["etc_total"] and kB["etc_total"]>0) else np.nan
            card_metric("% Azul", fmt_num(pctB,1,"%"))

    st.markdown("---")
    cols_show = multiselect_all("Variables a comparar", 
                                [c for c in ["ET0","ETc","ETverde","ETazul","Pef"] if c in set(dfA.columns)|set(dfB.columns)],
                                key="cmp_cic3")
    if cols_show:
        fig_ab = overlay_AB(dfA, dfB, xA, cols_show, labelA=ciclo_A, labelB=ciclo_B, colors=palette_vals)
        st.plotly_chart(fig_ab, use_container_width=True)

    with st.expander("Temperaturas (overlay)"):
        tcols = multiselect_all("Series T", [c for c in ["Tmin","Tmean","Tmax"] if c in set(dfA.columns)|set(dfB.columns)], key="cmp_t3")
        if tcols:
            st.plotly_chart(overlay_AB(dfA, dfB, xA, tcols, ciclo_A, ciclo_B, colors=palette_vals), use_container_width=True)

    with st.expander("Meteo y Viento (overlay)"):
        mcols = multiselect_all("Meteo", [c for c in ["Rs","HR","Ux"] if c in set(dfA.columns)|set(dfB.columns)], key="cmp_m3")
        if mcols:
            st.plotly_chart(overlay_AB(dfA, dfB, xA, mcols, ciclo_A, ciclo_B, colors=palette_vals), use_container_width=True)

elif modo == "Comparar regiones":
    ruta_A = CAT[(CAT.Region==region_A) & (CAT.Ciclo==ciclo_sel)]["Ruta"]
    ruta_B = CAT[(CAT.Region==region_B) & (CAT.Ciclo==ciclo_sel)]["Ruta"]
    if ruta_A.empty or ruta_B.empty: st.error("No encontré ambas regiones."); st.stop()
    dfA = leer_csv(ruta_A.iloc[0]); dfB = leer_csv(ruta_B.iloc[0])
    xA, xB = best_x(dfA, prefer=eje_x), best_x(dfB, prefer=eje_x)

    st.subheader(f"Comparación de regiones — ciclo {ciclo_sel}")
    kA, kB = kpis_ext(dfA), kpis_ext(dfB)
    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"#### {region_A}")
        c1,c2,c3 = st.columns(3)
        with c1: card_metric("Días", kA["dias"])
        with c2: card_metric("ETc", fmt_num(kA["etc_total"],1," mm"))
        with c3:
            pctA = (kA["eta_total"]/kA["etc_total"]*100) if (kA["etc_total"] and kA["etc_total"]>0) else np.nan
            card_metric("% Azul", fmt_num(pctA,1,"%"))
    with colB:
        st.markdown(f"#### {region_B}")
        c1,c2,c3 = st.columns(3)
        with c1: card_metric("Días", kB["dias"])
        with c2: card_metric("ETc", fmt_num(kB["etc_total"],1," mm"))
        with c3:
            pctB = (kB["eta_total"]/kB["etc_total"]*100) if (kB["etc_total"] and kB["etc_total"]>0) else np.nan
            card_metric("% Azul", fmt_num(pctB,1,"%"))

    st.markdown("---")
    cols_show = multiselect_all("Variables a comparar",
                                [c for c in ["ET0","ETc","ETverde","ETazul","Pef"] if c in set(dfA.columns)|set(dfB.columns)],
                                key="cmp_reg3")
    if cols_show:
        st.plotly_chart(overlay_AB(dfA, dfB, xA, cols_show, labelA=region_A, labelB=region_B, colors=palette_vals),
                        use_container_width=True)

    with st.expander("Temperaturas (overlay)"):
        tcols = multiselect_all("Series T", [c for c in ["Tmin","Tmean","Tmax"] if c in set(dfA.columns)|set(dfB.columns)], key="cmp_reg_t3")
        if tcols:
            st.plotly_chart(overlay_AB(dfA, dfB, xA, tcols, region_A, region_B, colors=palette_vals), use_container_width=True)

    with st.expander("Meteo y Viento (overlay)"):
        mcols = multiselect_all("Meteo", [c for c in ["Rs","HR","Ux"] if c in set(dfA.columns)|set(dfB.columns)], key="cmp_reg_m3")
        if mcols:
            st.plotly_chart(overlay_AB(dfA, dfB, xA, mcols, region_A, region_B, colors=palette_vals), use_container_width=True)
