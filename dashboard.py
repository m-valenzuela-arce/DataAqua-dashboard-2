# ==========================================
# DataAqua — Dashboard 5 (Dark Analytics)
# Estilo tipo BI: tarjetas KPI + gauges + heatmap + ranking
# ==========================================

from pathlib import Path
import os, re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ------------ CONFIG ------------
st.set_page_config(page_title="DataAqua — Dark Analytics", page_icon="💧", layout="wide")
DATA_DIR = Path("data") / "Salidas_ETo12_con_uac_y_hh" / "Periodo de Cultivo ETo"

DARK_CSS = """
<style>
/* Modo oscuro sobrio */
html, body { background: #0b1220; color: #e5e7eb; }
.block-container { padding-top: 0.6rem; }
section[data-testid="stSidebar"] { display:none !important; }
a { color:#7dd3fc; }

/* Tarjetas KPI columna izquierda */
.kpicard {
  background:#111827; border:1px solid #1f2937; border-radius:12px;
  padding:10px 12px; box-shadow: 0 1px 2px rgba(0,0,0,.25); margin-bottom:10px;
}
.kpicard .title { color:#9ca3af; font-size:.85rem; margin-bottom:6px; }
.kpicard .value { font-size:1.35rem; font-weight:800; color:#f9fafb; }
.kpicard .sub { color:#9ca3af; font-size:.75rem; }

/* Panel info derecha */
.panel {
  background:#0f172a; border:1px solid #1f2937; border-radius:12px;
  padding:12px; margin-bottom:10px;
}

/* Controles top (chips) */
.chiprow { display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 8px 0; }
.chiprow > div { background:#0f172a; padding:6px 10px; border:1px solid #1f2937; border-radius:999px; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ------------ Colores ------------
PALETTE = {
    "accent": "#38bdf8",
    "green":  "#22c55e",
    "orange": "#f59e0b",
    "red":    "#ef4444",
    "violet": "#8b5cf6",
    "gray":   "#9ca3af",
}

# ------------ Mapeo columnas ------------
MAP_UNISON = {
    "Año_ (YEAR)":"Year","AÃ±o_ (YEAR)":"Year",
    "Día (DOY)":"DOY","DÃ­a (DOY)":"DOY",
    "Tmax (T2M_MAX)":"Tmax","Tmin (T2M_MIN)":"Tmin",
    "HR (RH2M)":"HR","Ux (WS2M)":"Ux",
    "Rs (ALLSKY_SFC_SW_DWN)":"Rs","Rl_ (ALLSKY_SFC_LW_DWN)":"Rl",
    "Ptot_ (PRECTOTCORR)":"Ptot","Pef_":"Pef","Tmean_":"Tmean",
    "Rns_":"Rns","Rnl_":"Rnl","Rn_":"Rn","Rso_":"Rso","Kc_":"Kc","decada_":"decada",
    "ET0":"ET0","ETc":"ETc","ETverde":"ETverde","ETazul":"ETazul",
    "Year":"Year","DOY":"DOY","Dia":"Dia",
}
NUM_COLS = ["Year","DOY","ET0","ETc","ETverde","ETazul","Pef","decada",
            "Rns","Rnl","Rs","Tmean","HR","Ux","Kc","Tmax","Tmin",
            "UACverde_m3_ha","UACazul_m3_ha","HHverde_m3_ton","HHazul_m3_ton"]

# ------------ Helpers ------------
def parse_unison_filename(fname:str):
    m = re.match(r"([\wÁÉÍÓÚáéíóúñÑ\s\-]+)-FAO56-(\d{4})(?:-(\d{4}))?-SALIDA\.csv$", fname, re.I)
    if not m: return None, None
    reg, y1, y2 = m.groups()
    if reg == "VillaAllende": reg = "Villa de Allende"
    if reg == "Etchhojoa": reg = "Etchojoa"
    ciclo = y1 if not y2 else f"{y1}-{y2}"
    return reg.strip(), ciclo

@st.cache_data(show_spinner=False)
def catalogo(base: Path) -> pd.DataFrame:
    rows=[]
    if not base.exists(): return pd.DataFrame(columns=["Region","Ciclo","Ruta"])
    for rf in sorted(os.listdir(base)):
        d = base / rf
        if not d.is_dir(): continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".csv"): continue
            reg, ciclo = parse_unison_filename(f)
            if reg and ciclo: rows.append({"Region":reg, "Ciclo":ciclo, "Ruta":str(d/f)})
    return pd.DataFrame(rows).sort_values(["Region","Ciclo"]).reset_index(drop=True)

def _year_doy_to_date(y,d):
    try:
        base = datetime(int(y),1,1); return base + timedelta(days=int(d)-1)
    except: return pd.NaT

@st.cache_data(show_spinner=False, ttl=300)
def leer_csv(path:str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists(): return pd.DataFrame()
    for enc in ("utf-8","latin-1", None):
        try:
            df = pd.read_csv(p, encoding=enc) if enc else pd.read_csv(p); break
        except Exception: continue
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=lambda c: MAP_UNISON.get(c, c))
    for c in set(NUM_COLS).intersection(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"Year","DOY"}.issubset(df.columns):
        fechas = [_year_doy_to_date(y,d) for y,d in zip(df["Year"], df["DOY"])]
        df["Fecha"] = pd.to_datetime(fechas)
        if df["Fecha"].notna().any():
            f0 = df["Fecha"].dropna().iloc[0]
            df["Dia_ciclo"] = (df["Fecha"]-f0).dt.days.astype("Int64")
        else:
            df["Dia_ciclo"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    else:
        df["Fecha"]=pd.NaT; df["Dia_ciclo"]=pd.Series(pd.NA, index=df.index, dtype="Int64")
    if "ETc" in df: df["ETc_acum"]=df["ETc"].cumsum()
    if "ETazul" in df: df["ETazul_acum"]=df["ETazul"].cumsum()
    if {"ETc","ETazul"}.issubset(df.columns):
        df["pct_azul"] = np.where(df["ETc"]>0, df["ETazul"]/df["ETc"]*100, np.nan)
    # extras para heatmap
    if "Fecha" in df and df["Fecha"].notna().any():
        df["Year_"] = df["Fecha"].dt.year
        df["Month_"]= df["Fecha"].dt.month
    return df

def best_x(df, prefer="Fecha"):
    if prefer in df and df[prefer].notna().any(): return prefer
    for c in ("Fecha","Dia_ciclo","DOY"):
        if c in df and df[c].notna().any(): return c
    return df.index

def fmt(x, dec=1, suf=""):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "—"
    return f"{x:.{dec}f}{suf}"

def kpis(df: pd.DataFrame):
    def lastv(c): 
        return float(df[c].dropna().iloc[-1]) if (c in df and df[c].notna().any()) else np.nan
    dias = int(df["ETc"].notna().sum()) if "ETc" in df else 0
    etc = float(df["ETc"].sum()) if "ETc" in df else np.nan
    etv = float(df["ETverde"].sum()) if "ETverde" in df else np.nan
    eta = float(df["ETazul"].sum()) if "ETazul" in df else np.nan
    pct = (eta/etc*100) if (etc and etc>0) else np.nan
    pef = float(df["Pef"].sum()) if "Pef" in df else np.nan
    rain_cover = (pef/etc*100) if (etc and etc>0 and not np.isnan(pef)) else np.nan
    return {
        "dias":dias,"etc":etc,"etv":etv,"eta":eta,"pct":pct,
        "pef":pef,"rain_cover":rain_cover,
        "uacv":lastv("UACverde_m3_ha"), "uaca":lastv("UACazul_m3_ha")
    }

# ------------ Figuras (Plotly dark) ------------
def fig_line(df, x, cols, title):
    fig = go.Figure()
    palette = [PALETTE["accent"], PALETTE["orange"], PALETTE["green"], PALETTE["red"], PALETTE["violet"]]
    for i,c in enumerate(cols):
        if c not in df.columns: continue
        fig.add_trace(go.Scatter(
            x=df[x] if isinstance(x,str) else df.index,
            y=df[c], mode="lines", name=c,
            line=dict(width=2, color=palette[i % len(palette)])
        ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
                      title=title, margin=dict(l=10,r=10,t=40,b=10), height=360,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return fig

def fig_overlay(dfA, dfB, x, cols, tagA="A", tagB="B", title="Overlay"):
    fig = go.Figure()
    palette = [PALETTE["accent"], PALETTE["orange"], PALETTE["green"], PALETTE["red"], PALETTE["violet"]]
    for i,c in enumerate(cols):
        col = palette[i % len(palette)]
        if c in dfA.columns:
            fig.add_trace(go.Scatter(x=dfA[x] if isinstance(x,str) else dfA.index, y=dfA[c],
                                     mode="lines", name=f"{c} ({tagA})",
                                     line=dict(width=2, color=col, dash="solid")))
        if c in dfB.columns:
            fig.add_trace(go.Scatter(x=dfB[x] if isinstance(x,str) else dfB.index, y=dfB[c],
                                     mode="lines", name=f"{c} ({tagB})",
                                     line=dict(width=2, color=col, dash="dash")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
                      title=title, margin=dict(l=10,r=10,t=40,b=10), height=380,
                      legend=dict(orientation="h", y=1.02))
    return fig

def fig_gauge_pct(title, pct, color="#22c55e"):
    # semicircular gauge usando pie
    value = 0 if (pct is None or np.isnan(pct)) else max(0, min(100, pct))
    fig = go.Figure(go.Pie(
        values=[value, 100-value], hole=0.7, sort=False, direction="clockwise",
        marker_colors=[color, "#1f2937"], textinfo="none"
    ))
    fig.update_layout(
        template="plotly_dark", showlegend=False, paper_bgcolor="#111827", plot_bgcolor="#111827",
        margin=dict(l=0,r=0,t=10,b=0), height=140,
        annotations=[
            dict(text=f"{value:.0f}%", x=0.5, y=0.5, font=dict(size=22,color="#f9fafb"), showarrow=False),
            dict(text=title, x=0.5, y=0.1, font=dict(size=12,color="#9ca3af"), showarrow=False)
        ]
    )
    return fig

def fig_heatmap_month(df, value_col="ETc"):
    # Heatmap Año (y) vs Mes (x) del promedio mensual
    if not {"Year_","Month_", value_col}.issubset(df.columns): return go.Figure()
    g = df.groupby(["Year_","Month_"], dropna=True)[value_col].mean().reset_index()
    if g.empty: return go.Figure()
    g_piv = g.pivot(index="Year_", columns="Month_", values=value_col).sort_index(ascending=False)
    fig = px.imshow(g_piv, color_continuous_scale="Blues", aspect="auto",
                    labels=dict(color=f"{value_col} prom."))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
                      margin=dict(l=10,r=10,t=30,b=10), height=260,
                      coloraxis_colorbar=dict(title="mm/día"))
    fig.update_xaxes(title="Mes"); fig.update_yaxes(title="Año")
    return fig

def fig_top_bars(df, col="ETc", topn=10, xlab="ETc total (mm)"):
    if col not in df.columns: return go.Figure()
    d = df[[col,"Fecha"]].copy()
    if "Fecha" in d and d["Fecha"].notna().any():
        d["Dia"] = d["Fecha"].dt.strftime("%Y-%m-%d")
    else:
        d["Dia"] = d.index.astype(str)
    agg = d.groupby("Dia")[col].sum().sort_values(ascending=False).head(topn).iloc[::-1]
    fig = go.Figure(go.Bar(x=agg.values, y=agg.index, orientation="h",
                           marker_color=PALETTE["accent"]))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
                      margin=dict(l=10,r=10,t=30,b=10), height=360, xaxis_title=xlab, yaxis_title="")
    return fig

def kpi_card(title, value, sub=""):
    st.markdown(
        f'<div class="kpicard"><div class="title">{title}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True
    )

# ------------ UI (sin sidebar; layout 3 columnas) ------------
CAT = catalogo(DATA_DIR)
if CAT.empty:
    st.error("No se encontraron archivos en la carpeta de datos.")
    st.stop()

# Controles superiores
colA, colB, colC, colD = st.columns([0.32,0.22,0.22,0.24])
with colA:
    st.markdown("### 💧 DataAqua — Dark Analytics")
    st.caption("KPI • Gauges • Heatmap • Ranking — sin mapas")
with colB:
    modo = st.selectbox("Modo", ["Ciclo individual","Comparar ciclos","Comparar regiones"], index=0)
with colC:
    eje = st.selectbox("Eje X", ["Fecha","Día del ciclo"], index=0)
    eje_col = "Dia_ciclo" if eje == "Día del ciclo" else "Fecha"
with colD:
    st.markdown('<div class="chiprow"><div>Zoom: arrastra</div><div>Click leyenda: oculta/ muestra</div></div>', unsafe_allow_html=True)

# Selección según modo
if modo == "Ciclo individual":
    regiones = sorted(CAT["Region"].unique())
    region_sel = st.selectbox("Región", regiones, key="r1")
    ciclos_reg = sorted(CAT.loc[CAT["Region"]==region_sel,"Ciclo"].unique())
    ciclo_sel = st.selectbox("Ciclo", ciclos_reg, key="c1")

elif modo == "Comparar ciclos":
    regiones = sorted(CAT["Region"].unique())
    region_sel = st.selectbox("Región", regiones, key="r2")
    ciclos_reg = sorted(CAT.loc[CAT["Region"]==region_sel,"Ciclo"].unique())
    ciclo_A = st.selectbox("Ciclo A", ciclos_reg, key="cA2")
    ciclo_B = st.selectbox("Ciclo B", ciclos_reg, index=min(1,len(ciclos_reg)-1), key="cB2")

else:  # Comparar regiones
    ciclos = sorted(CAT["Ciclo"].unique())
    ciclo_sel = st.selectbox("Ciclo", ciclos, key="c3")
    regs_ciclo = sorted(CAT.loc[CAT["Ciclo"]==ciclo_sel,"Region"].unique())
    region_A = st.selectbox("Región A", regs_ciclo, key="rA3")
    region_B = st.selectbox("Región B", regs_ciclo, index=min(1,len(regs_ciclo)-1), key="rB3")

st.markdown("---")

# ====== LAYOUT PRINCIPAL: 3 columnas ======
left, center, right = st.columns([0.18, 0.54, 0.28], gap="large")

# ---------------- INDIVIDUAL ----------------
if modo == "Ciclo individual":
    ruta = CAT[(CAT.Region==region_sel) & (CAT.Ciclo==ciclo_sel)]["Ruta"]
    if ruta.empty: st.error("No encontré CSV."); st.stop()
    df = leer_csv(ruta.iloc[0])
    if df.empty: st.error("CSV vacío o ilegible."); st.stop()
    x = best_x(df, prefer=eje_col)
    kp = kpis(df)

    # LEFT: tarjetas + gauges
    with left:
        kpi_card("ETc total", f"{fmt(kp['etc'],1)} mm")
        kpi_card("ET azul total", f"{fmt(kp['eta'],1)} mm")
        kpi_card("ET verde total", f"{fmt(kp['etv'],1)} mm")
        kpi_card("Días de ciclo", f"{kp['dias']}")
        kpi_card("UAC verde", f"{fmt(kp['uacv'],0)} m³/ha")
        kpi_card("UAC azul", f"{fmt(kp['uaca'],0)} m³/ha")
        st.plotly_chart(fig_gauge_pct("Azul / ETc", kp["pct"], PALETTE["orange"]), use_container_width=True)
        st.plotly_chart(fig_gauge_pct("Pef / ETc", kp["rain_cover"], PALETTE["green"]), use_container_width=True)

    # CENTER: serie + heatmap
    with center:
        st.plotly_chart(
            fig_line(df, x, [c for c in ["ET0","ETc","ETverde","ETazul","Pef"] if c in df.columns],
                     f"{region_sel} — {ciclo_sel}"),
            use_container_width=True
        )
        st.plotly_chart(fig_heatmap_month(df, value_col="ETc"), use_container_width=True)

    # RIGHT: ranking top días + panel info
    with right:
        st.plotly_chart(fig_top_bars(df, col="ETc", topn=12, xlab="ETc total (mm)"), use_container_width=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Acerca de**<br/>• Modo oscuro con Plotly (`plotly_dark`).<br/>• Heatmap = promedio mensual de **ETc** por año.<br/>• Gauges: proporción **ET azul / ETc** y **Pef / ETc**.", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- COMPARAR CICLOS ----------------
elif modo == "Comparar ciclos":
    rutaA = CAT[(CAT.Region==region_sel) & (CAT.Ciclo==ciclo_A)]["Ruta"]
    rutaB = CAT[(CAT.Region==region_sel) & (CAT.Ciclo==ciclo_B)]["Ruta"]
    if rutaA.empty or rutaB.empty: st.error("Faltan archivos."); st.stop()
    dfA, dfB = leer_csv(rutaA.iloc[0]), leer_csv(rutaB.iloc[0])
    xA, xB = best_x(dfA, prefer=eje_col), best_x(dfB, prefer=eje_col)
    kA, kB = kpis(dfA), kpis(dfB)

    with left:
        kpi_card("ETc (A/B)", f"{fmt(kA['etc'],1)} / {fmt(kB['etc'],1)} mm", "Totales")
        kpi_card("% Azul (A/B)", f"{fmt(kA['pct'],1,'%')} / {fmt(kB['pct'],1,'%')}")
        kpi_card("Días (A/B)", f"{kA['dias']} / {kB['dias']}")
        st.plotly_chart(fig_gauge_pct(f"{ciclo_A}: Azul/ETc", kA["pct"], PALETTE["orange"]), use_container_width=True)
        st.plotly_chart(fig_gauge_pct(f"{ciclo_B}: Azul/ETc", kB["pct"], PALETTE["orange"]), use_container_width=True)

    with center:
        st.plotly_chart(
            fig_overlay(dfA, dfB, xA, [c for c in ["ETc","ETazul","ETverde","Pef"] if c in set(dfA.columns)|set(dfB.columns)],
                        tagA=ciclo_A, tagB=ciclo_B, title=f"{region_sel} — ET (A/B)"),
            use_container_width=True
        )
        # Heatmaps lado a lado
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(fig_heatmap_month(dfA, "ETc"), use_container_width=True)
        with c2: st.plotly_chart(fig_heatmap_month(dfB, "ETc"), use_container_width=True)

    with right:
        st.plotly_chart(fig_top_bars(dfA, "ETc", 10, f"Top ETc — {ciclo_A}"), use_container_width=True)
        st.plotly_chart(fig_top_bars(dfB, "ETc", 10, f"Top ETc — {ciclo_B}"), use_container_width=True)

# ---------------- COMPARAR REGIONES ----------------
else:
    rutaA = CAT[(CAT.Region==region_A) & (CAT.Ciclo==ciclo_sel)]["Ruta"]
    rutaB = CAT[(CAT.Region==region_B) & (CAT.Ciclo==ciclo_sel)]["Ruta"]
    if rutaA.empty or rutaB.empty: st.error("Faltan archivos."); st.stop()
    dfA, dfB = leer_csv(rutaA.iloc[0]), leer_csv(rutaB.iloc[0])
    xA, xB = best_x(dfA, prefer=eje_col), best_x(dfB, prefer=eje_col)
    kA, kB = kpis(dfA), kpis(dfB)

    with left:
        kpi_card("ETc (A/B)", f"{fmt(kA['etc'],1)} / {fmt(kB['etc'],1)} mm")
        kpi_card("% Azul (A/B)", f"{fmt(kA['pct'],1,'%')} / {fmt(kB['pct'],1,'%')}")
        kpi_card("Días (A/B)", f"{kA['dias']} / {kB['dias']}")
        st.plotly_chart(fig_gauge_pct(region_A, kA["pct"], PALETTE["orange"]), use_container_width=True)
        st.plotly_chart(fig_gauge_pct(region_B, kB["pct"], PALETTE["orange"]), use_container_width=True)

    with center:
        st.plotly_chart(
            fig_overlay(dfA, dfB, xA, [c for c in ["ETc","ETazul","ETverde","Pef"] if c in set(dfA.columns)|set(dfB.columns)],
                        tagA=region_A, tagB=region_B, title=f"{ciclo_sel} — ET (Regiones)"),
            use_container_width=True
        )
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(fig_heatmap_month(dfA, "ETc"), use_container_width=True)
        with c2: st.plotly_chart(fig_heatmap_month(dfB, "ETc"), use_container_width=True)

    with right:
        st.plotly_chart(fig_top_bars(dfA, "ETc", 10, f"Top ETc — {region_A}"), use_container_width=True)
        st.plotly_chart(fig_top_bars(dfB, "ETc", 10, f"Top ETc — {region_B}"), use_container_width=True)
