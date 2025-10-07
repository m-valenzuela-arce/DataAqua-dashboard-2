# ==========================================
# DataAqua — Dashboard 4 "Canvas" (sin sidebar)
# Layout tipo landing analítica: hero + chips + panel flotante
# ==========================================
# Requisitos:
#   streamlit>=1.30
#   pandas, numpy, plotly
# Ejecuta:
#   streamlit run dashboard4_canvas.py
# ==========================================

from pathlib import Path
import os, re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(
    page_title="DataAqua — Canvas",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_DIR = Path("data") / "Salidas_ETo12_con_uac_y_hh" / "Periodo de Cultivo ETo"

# ---------- CSS (look fresco) ----------
st.markdown("""
<style>
/* Oculta el menú default superior (mantén el de desplegar si quieres) */
header { visibility: hidden; height: 0; }

/* Hero */
.hero {
  padding: 10px 0 12px 0;
  border-bottom: 1px solid #eceff4;
}
h1.hero-title {
  font-size: 2.0rem; margin: 0 0 6px 0; letter-spacing: -0.02em;
}
.hero-sub { color: #6b7280; }

/* Chips y contenedores */
.chips { display:flex; flex-wrap:wrap; gap:8px; margin: 8px 0 2px 0;}
.chip {
  border:1px solid #e5e7eb; padding:6px 10px; border-radius:999px;
  background:#fff; cursor:pointer; font-size:0.88rem;
}
.chip.active { background:#0ea5e9; color:#fff; border-color:#0ea5e9; }

/* Panel flotante (controles mínimos) */
.floating {
  position: sticky; top: 10px; z-index: 5;
  border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffffcc;
  backdrop-filter: blur(8px);
  padding: 10px 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

/* Tarjetas KPI mini */
.kpi {
  border: 1px dashed #e5e7eb; border-radius: 10px; padding: 10px 12px; background:#fff;
}
.kpi h4 { margin:0; font-size:0.9rem; color:#374151; }
.kpi .v { font-weight:700; font-size:1.25rem; }
.kpi .s { color:#6b7280; font-size:0.78rem; }

/* Secciones */
.section-title { font-weight:700; font-size:1.0rem; margin: 6px 0; }

/* Footer sutil */
.footer { color:#9ca3af; font-size:0.8rem; text-align:center; padding: 8px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ---------- Paletas ----------
PALETTE = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#6366f1", "#14b8a6", "#e11d48"]

# ---------- Mapeo columnas (igual que tu pipeline) ----------
MAP_UNISON = {
    "Año_ (YEAR)": "Year", "AÃ±o_ (YEAR)": "Year",
    "Día (DOY)": "DOY", "DÃ­a (DOY)": "DOY",
    "Tmax (T2M_MAX)": "Tmax", "Tmin (T2M_MIN)": "Tmin",
    "HR (RH2M)": "HR", "Ux (WS2M)": "Ux",
    "Rs (ALLSKY_SFC_SW_DWN)": "Rs", "Rl_ (ALLSKY_SFC_LW_DWN)": "Rl",
    "Ptot_ (PRECTOTCORR)": "Ptot", "Pef_": "Pef", "Tmean_": "Tmean",
    "ET0":"ET0","ETc":"ETc","ETverde":"ETverde","ETazul":"ETazul",
    "Rns_":"Rns","Rnl_":"Rnl","Rn_":"Rn","Rso_":"Rso","Kc_":"Kc","decada_":"decada",
    "Year":"Year","DOY":"DOY","Dia":"Dia",
}
NUM_COLS = [
    "Year","DOY","ET0","ETc","ETverde","ETazul","Pef","decada",
    "Rns","Rnl","Rs","Tmean","HR","Ux","Kc","Tmax","Tmin",
    "UACverde_m3_ha","UACazul_m3_ha","HHverde_m3_ton","HHazul_m3_ton",
]

# ---------- Helpers ----------
def parse_unison_filename(filename: str):
    m = re.match(r"([\wÁÉÍÓÚáéíóúñÑ\s\-]+)-FAO56-(\d{4})(?:-(\d{4}))?-SALIDA\.csv$", filename, re.I)
    if not m: return None, None
    reg, y1, y2 = m.groups()
    if reg == "VillaAllende": reg = "Villa de Allende"
    if reg == "Etchhojoa": reg = "Etchojoa"
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

def _year_doy_to_date(y, d):
    try:
        base = datetime(int(y), 1, 1)
        return base + timedelta(days=int(d) - 1)
    except Exception:
        return pd.NaT

@st.cache_data(show_spinner=False, ttl=300)
def leer_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists(): return pd.DataFrame()
    for enc in ("utf-8","latin-1", None):
        try:
            df = pd.read_csv(p, encoding=enc) if enc else pd.read_csv(p)
            break
        except Exception:
            continue
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
        df["pct_azul"] = np.where(df["ETc"]>0, df["ETazul"]/df["ETc"]*100, np.nan)
    return df

def best_x(df: pd.DataFrame, prefer="Fecha"):
    if prefer in df.columns and df[prefer].notna().any(): return prefer
    for alt in ("Fecha","Dia_ciclo","DOY"):
        if alt in df.columns and df[alt].notna().any(): return alt
    return df.index

def kpis(df: pd.DataFrame):
    def lastv(c): 
        return float(df[c].dropna().iloc[-1]) if (c in df and df[c].notna().any()) else np.nan
    dias = int(df["ETc"].notna().sum()) if "ETc" in df else 0
    etc = float(df["ETc"].sum()) if "ETc" in df else np.nan
    etv = float(df["ETverde"].sum()) if "ETverde" in df else np.nan
    eta = float(df["ETazul"].sum()) if "ETazul" in df else np.nan
    pct = (eta/etc*100) if (etc and etc>0) else np.nan
    siem = pd.to_datetime(df["Fecha"].dropna().iloc[0]).date() if "Fecha" in df and df["Fecha"].notna().any() else None
    cos = pd.to_datetime(df["Fecha"].dropna().iloc[-1]).date() if "Fecha" in df and df["Fecha"].notna().any() else None
    return {
        "dias": dias, "etc": etc, "etv": etv, "eta": eta, "pct": pct,
        "siembra": siem, "cosecha": cos,
        "tmax": float(df["Tmax"].max()) if "Tmax" in df else np.nan,
        "tmin": float(df["Tmin"].min()) if "Tmin" in df else np.nan,
        "uacv": lastv("UACverde_m3_ha"), "uaca": lastv("UACazul_m3_ha")
    }

def fmt(x, dec=1, suf=""):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "—"
    return f"{x:.{dec}f}{suf}"

def line_multi(df, x, ycols, title):
    fig = go.Figure()
    for i, c in enumerate(ycols):
        if c not in df.columns: continue
        fig.add_trace(go.Scatter(
            x=df[x] if isinstance(x,str) and x in df.columns else df.index,
            y=df[c], name=c, mode="lines",
            line=dict(width=2, color=PALETTE[i % len(PALETTE)])
        ))
    fig.update_layout(
        title=title, template="plotly_white",
        margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        height=360
    )
    return fig

def overlay_two(dfA, dfB, x, cols, tagA="A", tagB="B", title="Overlay"):
    fig = go.Figure()
    for i, c in enumerate(cols):
        col = PALETTE[i % len(PALETTE)]
        if c in dfA.columns:
            fig.add_trace(go.Scatter(
                x=dfA[x] if isinstance(x,str) and x in dfA.columns else dfA.index,
                y=dfA[c], name=f"{c} ({tagA})", mode="lines",
                line=dict(width=2, color=col, dash="solid")
            ))
        if c in dfB.columns:
            fig.add_trace(go.Scatter(
                x=dfB[x] if isinstance(x,str) and x in dfB.columns else dfB.index,
                y=dfB[c], name=f"{c} ({tagB})", mode="lines",
                line=dict(width=2, color=col, dash="dash")
            ))
    fig.update_layout(
        title=title, template="plotly_white",
        margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        height=380
    )
    return fig

# ---------- Estado UI (sin sidebar) ----------
if "modo" not in st.session_state: st.session_state["modo"] = "Individual"

# ---------- HERO ----------
st.markdown('<div class="hero">', unsafe_allow_html=True)
c1, c2 = st.columns([0.75,0.25])
with c1:
    st.markdown('<h1 class="hero-title">💧 DataAqua — Canvas</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Panel sin menús: todo en página, con chips y controles mínimos.</div>', unsafe_allow_html=True)
with c2:
    st.markdown("")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Chips de modo ----------
col_chips = st.container()
with col_chips:
    colA, colB, colC = st.columns([0.18,0.18,0.64])
    with colA:
        if st.button("Ciclo individual", use_container_width=True,
                     type=("primary" if st.session_state["modo"]=="Individual" else "secondary")):
            st.session_state["modo"] = "Individual"
    with colB:
        if st.button("Comparar", use_container_width=True,
                     type=("primary" if st.session_state["modo"]=="Comparar" else "secondary")):
            st.session_state["modo"] = "Comparar"
    with colC:
        st.caption("Tip: click en leyendas para ocultar/mostrar series.")

# ---------- Catálogo de archivos ----------
CAT = catalogo(DATA_DIR)
if CAT.empty:
    st.error("No se encontraron archivos en data/Salidas_ETo12_con_uac_y_hh/Periodo de Cultivo ETo")
    st.stop()

# ---------- LAYOUT: portada de datos + panel flotante de controles ----------
left, right = st.columns([0.7, 0.3], gap="large")

with right:  # panel flotante
    st.markdown('<div class="floating">', unsafe_allow_html=True)
    st.markdown("**Controles**")
    eje_x = st.radio("Eje X", ["Fecha","Día del ciclo"], index=0, horizontal=True)
    eje_col = "Dia_ciclo" if eje_x == "Día del ciclo" else "Fecha"

    # Selector según modo
    if st.session_state["modo"] == "Individual":
        regiones = sorted(CAT["Region"].unique())
        region_sel = st.selectbox("Región", regiones)
        ciclos_reg = sorted(CAT.loc[CAT["Region"]==region_sel, "Ciclo"].unique())
        ciclo_sel = st.selectbox("Ciclo", ciclos_reg)

    else:  # Comparar (misma región, 2 ciclos)
        regiones = sorted(CAT["Region"].unique())
        region_sel = st.selectbox("Región", regiones)
        ciclos_reg = sorted(CAT.loc[CAT["Region"]==region_sel, "Ciclo"].unique())
        colX, colY = st.columns(2)
        with colX:  ciclo_A = st.selectbox("Ciclo A", ciclos_reg, key="cA")
        with colY:  ciclo_B = st.selectbox("Ciclo B", ciclos_reg, index=min(1, len(ciclos_reg)-1), key="cB")

    st.markdown("---")
    with st.expander("Subir CSV manual (opcional)"):
        up = st.file_uploader("CSV UNISON", type=["csv"])
        if up is not None:
            df_up = pd.read_csv(up, encoding="latin-1")
            df_up = df_up.rename(columns=lambda c: MAP_UNISON.get(c.strip(), c.strip()))
            st.session_state["__df_canvas__"] = df_up

    st.markdown('</div>', unsafe_allow_html=True)

with left:
    # ====== MODO INDIVIDUAL ======
    if st.session_state["modo"] == "Individual":
        ruta = CAT[(CAT.Region==region_sel) & (CAT.Ciclo==ciclo_sel)]["Ruta"]
        if ruta.empty:
            st.warning("No encontré el archivo seleccionado.")
            st.stop()
        df = leer_csv(ruta.iloc[0])
        df = st.session_state.get("__df_canvas__") or df
        if df.empty:
            st.error("No fue posible leer el CSV.")
            st.stop()

        x = best_x(df, prefer=eje_col)
        k = kpis(df)

        # KPIs (grid compacto 2x5)
        krow1 = st.columns(5)
        for i,(tit,val,sub) in enumerate([
            ("Días", k["dias"], ""),
            ("Siembra", k["siembra"], "fecha"),
            ("Cosecha", k["cosecha"], "fecha"),
            ("ETc total", fmt(k["etc"],1," mm"), ""),
            ("ET azul", fmt(k["eta"],1," mm"), ""),
        ]):
            with krow1[i]:
                st.markdown(f'<div class="kpi"><h4>{tit}</h4><div class="v">{val}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)
        krow2 = st.columns(5)
        vals2 = [
            ("% Azul", fmt(k["pct"],1,"%"), ""),
            ("ET verde", fmt(k["etv"],1," mm"), ""),
            ("Tmax/Tmin", f"{fmt(k['tmax'],1,'°C')} / {fmt(k['tmin'],1,'°C')}", ""),
            ("UAC verde", fmt(k["uacv"],0," m³/ha"), ""),
            ("UAC azul", fmt(k["uaca"],0," m³/ha"), "")
        ]
        for i,(tit,val,sub) in enumerate(vals2):
            with krow2[i]:
                st.markdown(f'<div class="kpi"><h4>{tit}</h4><div class="v">{val}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)

        # Serie ET
        st.markdown('<div class="section-title">Serie diaria — ET</div>', unsafe_allow_html=True)
        et_cols = [c for c in ["ET0","ETc","ETverde","ETazul","Pef"] if c in df.columns]
        fig_et = line_multi(df, x, et_cols, f"{region_sel} — {ciclo_sel}")
        st.plotly_chart(fig_et, use_container_width=True)

        # Temperaturas + Meteo en un row
        colT, colM = st.columns([0.5,0.5])
        with colT:
            st.markdown('<div class="section-title">Temperaturas</div>', unsafe_allow_html=True)
            t_cols = [c for c in ["Tmin","Tmean","Tmax"] if c in df.columns]
            st.plotly_chart(line_multi(df, x, t_cols, "Temperaturas"), use_container_width=True)
        with colM:
            st.markdown('<div class="section-title">Meteo (Rs / HR / Ux)</div>', unsafe_allow_html=True)
            m_cols = [c for c in ["Rs","HR","Ux"] if c in df.columns]
            st.plotly_chart(line_multi(df, x, m_cols, "Meteo"), use_container_width=True)

        # Acumulados
        st.markdown('<div class="section-title">Acumulados</div>', unsafe_allow_html=True)
        acc_cols = [c for c in ["ETc_acum","ETazul_acum"] if c in df.columns]
        st.plotly_chart(line_multi(df, x, acc_cols, "Acumulados (mm)"), use_container_width=True)

        with st.expander("Datos (primeras filas)"):
            st.dataframe(df.head(60), use_container_width=True)

    # ====== MODO COMPARAR ======
    else:
        rutaA = CAT[(CAT.Region==region_sel) & (CAT.Ciclo==ciclo_A)]["Ruta"]
        rutaB = CAT[(CAT.Region==region_sel) & (CAT.Ciclo==ciclo_B)]["Ruta"]
        if rutaA.empty or rutaB.empty:
            st.warning("No encontré ambos ciclos.")
            st.stop()
        dfA = leer_csv(rutaA.iloc[0]); dfB = leer_csv(rutaB.iloc[0])
        xA = best_x(dfA, prefer=eje_col); xB = best_x(dfB, prefer=eje_col)

        kA, kB = kpis(dfA), kpis(dfB)
        st.markdown(f"### {region_sel} — {ciclo_A} vs {ciclo_B}")

        # KPIs comparados (dos filas de 3)
        row1 = st.columns(3)
        with row1[0]:
            st.markdown('<div class="kpi"><h4>ETc (A/B)</h4><div class="v">{}/{} mm</div><div class="s">total</div></div>'.format(fmt(kA["etc"],1), fmt(kB["etc"],1)), unsafe_allow_html=True)
        with row1[1]:
            st.markdown('<div class="kpi"><h4>% Azul (A/B)</h4><div class="v">{}/{}%</div><div class="s">proporción</div></div>'.format(fmt(kA["pct"],1), fmt(kB["pct"],1)), unsafe_allow_html=True)
        with row1[2]:
            st.markdown('<div class="kpi"><h4>Días (A/B)</h4><div class="v">{}/{}</div><div class="s">longitud del ciclo</div></div>'.format(kA["dias"], kB["dias"]), unsafe_allow_html=True)

        # Overlay ET
        st.markdown('<div class="section-title">Overlay — ET</div>', unsafe_allow_html=True)
        cols_et = [c for c in ["ET0","ETc","ETverde","ETazul","Pef"] if c in set(dfA.columns)|set(dfB.columns)]
        st.plotly_chart(overlay_two(dfA, dfB, xA, cols_et, ciclo_A, ciclo_B, "ET (A/B)"), use_container_width=True)

        # Overlay T y Meteo
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">Overlay — Temperaturas</div>', unsafe_allow_html=True)
            tcols = [c for c in ["Tmin","Tmean","Tmax"] if c in set(dfA.columns)|set(dfB.columns)]
            st.plotly_chart(overlay_two(dfA, dfB, xA, tcols, ciclo_A, ciclo_B, "T (A/B)"), use_container_width=True)
        with col2:
            st.markdown('<div class="section-title">Overlay — Meteo</div>', unsafe_allow_html=True)
            mcols = [c for c in ["Rs","HR","Ux"] if c in set(dfA.columns)|set(dfB.columns)]
            st.plotly_chart(overlay_two(dfA, dfB, xA, mcols, ciclo_A, ciclo_B, "Meteo (A/B)"), use_container_width=True)

        # Acumulados comparados
        st.markdown('<div class="section-title">Overlay — Acumulados</div>', unsafe_allow_html=True)
        acc = [c for c in ["ETc_acum","ETazul_acum"] if c in set(dfA.columns)|set(dfB.columns)]
        st.plotly_chart(overlay_two(dfA, dfB, xA, acc, ciclo_A, ciclo_B, "Acumulados (A/B)"), use_container_width=True)

# ---------- Footer ----------
st.markdown('<div class="footer">DataAqua Canvas — layout sin menús. Cambia modo con los botones superiores.</div>', unsafe_allow_html=True)
