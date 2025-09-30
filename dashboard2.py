# # ===========================
# # DataAqua Dashboard 2 (Streamlit)
# # ===========================
# # Modos:
# #  - Ciclo individual
# #  - Comparar ciclos (misma región)
# #  - Comparar regiones (mismo ciclo)
# #
# # Pestañas:
# #  - KPIs
# #  - Serie diaria
# #  - Acumulados
# #  - Decádico
# #  - Kc–ET0
# #  - Drivers ET0
# #
# # Ejecuta:
# #   streamlit run dashboard2.py
# # ===========================

# from pathlib import Path
# import os, re
# from datetime import datetime, timedelta

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import streamlit as st

# # ---------------------------
# # CONFIG
# # ---------------------------
# st.set_page_config(page_title="DataAqua — Dashboard 2", page_icon="💧", layout="wide")

# #RUTA_BASE          = Path("/lustre/home/mvalenzuela/Ocotillo/DataAqua")
# #RUTA_SALIDA_UNISON = RUTA_BASE / "Salidas_ETo12" / "Periodo de Cultivo ETo"
# RUTA_SALIDA_UNISON = Path("data") / "Salidas_ETo12_con_uac_y_hh" / "Periodo de Cultivo ETo"

# sns.set_style("whitegrid")
# plt.rcParams["figure.dpi"] = 120

# # Column map (tus nombres)
# MAP_UNISON = {
#     "Año_ (YEAR)": "Year", "AÃ±o_ (YEAR)": "Year",
#     "Día (DOY)": "DOY",   "DÃ­a (DOY)": "DOY",
#     "Tmax (T2M_MAX)": "Tmax", "Tmin (T2M_MIN)": "Tmin",
#     "HR (RH2M)": "HR", "Ux (WS2M)": "Ux",
#     "Rs (ALLSKY_SFC_SW_DWN)": "Rs",
#     "Rl_ (ALLSKY_SFC_LW_DWN)": "Rl",
#     "Ptot_ (PRECTOTCORR)": "Ptot",
#     "Pef_": "Pef", "Tmean_": "Tmean", "es_": "es", "ea_": "ea",
#     "delta_": "delta", "P_": "P", "gamma_": "gamma",
#     "Rns_": "Rns", "Rnl_": "Rnl", "Rn_": "Rn", "Rso_": "Rso",
#     "Kc_": "Kc", "decada_": "decada",
#     "ET0": "ET0", "ETc": "ETc", "ETverde": "ETverde", "ETazul": "ETazul",
#     "Year": "Year", "DOY": "DOY", "Dia": "Dia",
# }
# COLUMNAS_MIN = [
#     "Year","DOY","ET0","ETc","ETverde","ETazul","Pef","decada",
#     "Rns","Rnl","Rs","Tmean","HR","Ux","Kc"
# ]

# # ---------------------------
# # Helpers
# # ---------------------------
# def parse_unison_filename(filename: str):
#     """
#     'Cajeme-FAO56-2014-2015-SALIDA.csv' -> ('Cajeme','2014-2015')
#     'Metepec-FAO56-2014-SALIDA.csv'     -> ('Metepec','2014')
#     """
#     m = re.match(r"([A-Za-z]+)-FAO56-(\d{4})(?:-(\d{4}))?-SALIDA\.csv$", filename, re.I)
#     if not m:
#         return None, None
#     reg, y1, y2 = m.groups()
#     if reg == "VillaAllende": reg = "Villa de Allende"
#     if reg == "Etchhojoa":    reg = "Etchojoa"
#     ciclo = y1 if not y2 else f"{y1}-{y2}"
#     return reg, ciclo

# @st.cache_data(show_spinner=False)
# def catalogo_unison(base_dir: Path) -> pd.DataFrame:
#     rows = []
#     if not base_dir.exists():
#         return pd.DataFrame(columns=["Region","Ciclo","Ruta"])
#     for reg_folder in sorted(os.listdir(base_dir)):
#         d = base_dir / reg_folder
#         if not d.is_dir():
#             continue
#         for f in sorted(os.listdir(d)):
#             if not f.lower().endswith(".csv"):
#                 continue
#             reg, ciclo = parse_unison_filename(f)
#             if reg and ciclo:
#                 rows.append({"Region": reg, "Ciclo": ciclo, "Ruta": str(d / f)})
#     df = pd.DataFrame(rows).sort_values(["Region","Ciclo"]).reset_index(drop=True)
#     return df

# def _year_doy_to_date(y, doy):
#     try:
#         base = datetime(int(y), 1, 1)
#         return base + timedelta(days=int(doy) - 1)
#     except Exception:
#         return pd.NaT

# @st.cache_data(show_spinner=False)
# def leer_unison(path: str) -> pd.DataFrame:
#     p = Path(path)
#     if not p.exists():
#         return pd.DataFrame()
#     last_err = None
#     for enc in ("utf-8","latin-1"):
#         try:
#             df = pd.read_csv(p, encoding=enc)
#             last_err = None
#             break
#         except UnicodeDecodeError as e:
#             last_err = e
#             continue
#     if last_err is not None:
#         df = pd.read_csv(p)

#     df.columns = [c.strip() for c in df.columns]
#     df = df.rename(columns=lambda c: MAP_UNISON.get(c, c))
#     for c in set(COLUMNAS_MIN).intersection(df.columns):
#         df[c] = pd.to_numeric(df[c], errors="coerce")

#     # Fecha y día de ciclo
#     if {"Year","DOY"}.issubset(df.columns):
#         fechas = [_year_doy_to_date(y,d) for y,d in zip(df["Year"], df["DOY"])]
#         df["Fecha"] = pd.to_datetime(fechas)
#         if df["Fecha"].notna().any():
#             f0 = df["Fecha"].dropna().iloc[0]
#             df["Dia_ciclo"] = (df["Fecha"] - f0).dt.days.astype("Int64")
#         else:
#             df["Dia_ciclo"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
#     else:
#         df["Fecha"] = pd.NaT
#         df["Dia_ciclo"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

#     # Acumulados útiles
#     if "ETc" in df:
#         df["ETc_acum"] = df["ETc"].cumsum()
#     if "ETazul" in df:
#         df["ETazul_acum"] = df["ETazul"].cumsum()
#     if {"ETc","ETazul"}.issubset(df.columns):
#         df["pct_azul"] = np.where(df["ETc"]>0, df["ETazul"]/df["ETc"]*100.0, np.nan)
#     return df

# def kpis_ciclo(df: pd.DataFrame) -> dict:
#     mask = df["ETc"].notna() if "ETc" in df else pd.Series(False, index=df.index)
#     dias = int(mask.sum())
#     etc_total = float(df.loc[mask, "ETc"].sum())     if "ETc"     in df else np.nan
#     etv_total = float(df.loc[mask, "ETverde"].sum()) if "ETverde" in df else np.nan
#     eta_total = float(df.loc[mask, "ETazul"].sum())  if "ETazul"  in df else np.nan
#     pef_total = float(df.loc[mask, "Pef"].sum())     if "Pef"     in df else np.nan
#     pct_azul = (eta_total/etc_total*100.0) if (etc_total and etc_total>0) else np.nan
#     dias_def = int(((df["ETc"] > df.get("Pef", 0)).fillna(False)).sum()) if "ETc" in df else np.nan
#     pico_p95 = float(np.nanpercentile(df["ETc"], 95)) if "ETc" in df else np.nan
#     return {"dias":dias,"etc_total":etc_total,"etv_total":etv_total,"eta_total":eta_total,
#             "pct_azul":pct_azul,"pef_total":pef_total,"dias_def":dias_def,"pico_p95":pico_p95}

# def _xcol(df: pd.DataFrame, prefer="Fecha"):
#     if prefer in df.columns and df[prefer].notna().any(): return prefer
#     #for alt in ("Fecha","DOY","Dia_ciclo"):
#     for alt in ("Fecha", "Dia_ciclo"):
#         if alt in df.columns and df[alt].notna().any():
#             return alt
#     return df.index

# # --- Figuras (devuelven fig) ---
# def fig_series(df: pd.DataFrame, titulo: str, eje="Fecha", mostrar=("ET0","ETc","ETverde","ETazul","Pef")):
#     x = _xcol(df, eje)
#     fig, ax = plt.subplots(1,1, figsize=(12,4))
#     colores = {"ET0":"#4C78A8","ETc":"#F58518","ETverde":"#54A24B","ETazul":"#E45756","Pef":"#9D9D9D"}
#     for col in mostrar:
#         if col in df:
#             ax.plot(df[x], df[col], label=col, lw=1.6, color=colores.get(col, None))
#     ax.set_title(titulo)
#     ax.set_xlabel(str(x)); ax.set_ylabel("mm/día")
#     ax.legend()
#     fig.tight_layout()
#     return fig

# def fig_acumulados(df: pd.DataFrame, titulo: str, eje="Fecha"):
#     x = _xcol(df, eje)
#     fig, ax = plt.subplots(1,1, figsize=(12,4))
#     if "ETc_acum" in df:
#         ax.plot(df[x], df["ETc_acum"], label="ETc acumulado", lw=1.8)
#     if "ETazul_acum" in df:
#         ax.plot(df[x], df["ETazul_acum"], label="ETazul acumulado", lw=1.8)
#     ax.set_title(titulo)
#     ax.set_xlabel(str(x)); ax.set_ylabel("mm")
#     ax.legend()
#     fig.tight_layout()
#     return fig

# def fig_decadico(df: pd.DataFrame, titulo: str):
#     if "decada" not in df: return None
#     g = df.groupby("decada")[["ETc","ETazul"]].sum(min_count=1)
#     fig, ax = plt.subplots(1,1, figsize=(10,4))
#     g["ETc"].plot(kind="bar", ax=ax, color="#4C78A8", label="ETc")
#     if "ETazul" in g:
#         ax.plot(np.arange(len(g)), g["ETazul"].values, color="#F58518", lw=2, marker="o", label="ETazul")
#     ax.set_title(titulo)
#     ax.set_xlabel("Década del ciclo"); ax.set_ylabel("mm/decada")
#     ax.legend()
#     fig.tight_layout()
#     return fig

# def fig_kc_et0(df: pd.DataFrame, titulo: str, eje="Fecha"):
#     if "Kc" not in df or "ET0" not in df: return None
#     x = _xcol(df, eje)
#     fig, ax1 = plt.subplots(1,1, figsize=(12,4))
#     ax1.plot(df[x], df["ET0"], color="#4C78A8", label="ET0", lw=1.5)
#     ax1.set_ylabel("ET0 [mm/día]", color="#4C78A8"); ax1.tick_params(axis='y', labelcolor="#4C78A8")
#     ax2 = ax1.twinx()
#     ax2.plot(df[x], df["Kc"], color="#E45756", label="Kc", lw=1.5)
#     ax2.set_ylabel("Kc [-]", color="#E45756"); ax2.tick_params(axis='y', labelcolor="#E45756")
#     ax1.set_title(titulo); ax1.set_xlabel(str(x))
#     fig.tight_layout()
#     return fig

# def fig_drivers_et0(df: pd.DataFrame, titulo: str):
#     if "ET0" not in df: return None
#     drivers = [("Rs","Rs [MJ m$^{-2}$ d$^{-1}$]"),
#                ("Rnl","Rnl [MJ m$^{-2}$ d$^{-1}$]"),
#                ("HR","HR [%]"),
#                ("Ux","Viento Ux [m/s]"),
#                ("Tmean","Tmean [°C]")]
#     cols = [c for c,_ in drivers if c in df.columns]
#     if not cols: return None
#     n = len(cols); ncols, nrows = 3, int(np.ceil(n/3))
#     fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5*nrows))
#     axes = np.atleast_2d(axes).ravel()
#     for i,(c,lab) in enumerate([d for d in drivers if d[0] in cols]):
#         ax = axes[i]
#         ax.scatter(df[c], df["ET0"], alpha=0.6, s=14)
#         ax.set_xlabel(lab); ax.set_ylabel("ET0 [mm/día]")
#         try:
#             r = np.corrcoef(df[c].values, df["ET0"].values)[0,1]
#             ax.set_title(f"ET0 vs {c}  (r={r:.2f})")
#         except Exception:
#             ax.set_title(f"ET0 vs {c}")
#     for j in range(i+1, len(axes)):
#         axes[j].set_visible(False)
#     fig.suptitle(titulo)
#     fig.tight_layout()
#     return fig

# # ---------------------------
# # UI — Sidebar
# # ---------------------------
# st.sidebar.title("DataAqua — Selección")
# CAT_UNISON = catalogo_unison(RUTA_SALIDA_UNISON)
# if CAT_UNISON.empty:
#     st.error("No se encontraron archivos en Salidas_ETo12/Periodo de Cultivo ETo.")
#     st.stop()

# modo = st.sidebar.radio("Modo", ["Ciclo individual", "Comparar ciclos", "Comparar regiones"], index=0)
# #eje_opt = st.sidebar.radio("Eje X:", ["Fecha","DOY","Dia_ciclo"], index=0)

# eje_label = st.sidebar.radio("Eje X:", ["Fecha", "Día del ciclo"], index=0)
# eje_opt = "Dia_ciclo" if eje_label == "Día del ciclo" else "Fecha"

# # # --- Selector de variables para la Serie diaria ---
# # vars_posibles = ["ET0", "ETc", "ETverde", "ETazul", "Pef"]
# # vars_disponibles = [v for v in vars_posibles if v in df.columns]

# # series_sel = st.sidebar.multiselect(
# #     "Series a mostrar:",
# #     options=vars_disponibles,
# #     default=vars_disponibles
# #)

# if modo == "Ciclo individual":
#     regiones = sorted(CAT_UNISON["Region"].unique())
#     region_sel = st.sidebar.selectbox("Región:", regiones)
#     ciclos_reg = sorted(CAT_UNISON.loc[CAT_UNISON["Region"]==region_sel, "Ciclo"].unique())
#     ciclo_sel = st.sidebar.selectbox("Ciclo:", ciclos_reg)

#     st.sidebar.markdown("#### Gráficas")
#     show_series     = st.sidebar.checkbox("Serie diaria (ET0, ETc, ETverde, ETazul, Pef)", value=True)
#     show_acumulados = st.sidebar.checkbox("Acumulados (ETc, ETazul)", value=True)
#     show_decadico   = st.sidebar.checkbox("Decádico (ETc + ETazul)", value=True)
#     show_kc         = st.sidebar.checkbox("Kc y ET0", value=True)
#     show_drivers    = st.sidebar.checkbox("Drivers de ET0 (scatter)", value=True)

# elif modo == "Comparar ciclos":
#     regiones = sorted(CAT_UNISON["Region"].unique())
#     region_sel = st.sidebar.selectbox("Región:", regiones)
#     ciclos_reg = sorted(CAT_UNISON.loc[CAT_UNISON["Region"]==region_sel, "Ciclo"].unique())
#     colA, colB = st.sidebar.columns(2)
#     ciclo_A = colA.selectbox("Ciclo A", ciclos_reg, key="ciclo_A")
#     ciclo_B = colB.selectbox("Ciclo B", ciclos_reg, index=min(1, len(ciclos_reg)-1), key="ciclo_B")

# elif modo == "Comparar regiones":
#     ciclos = sorted(CAT_UNISON["Ciclo"].unique())
#     ciclo_sel = st.sidebar.selectbox("Ciclo:", ciclos)
#     regs_ciclo = sorted(CAT_UNISON.loc[CAT_UNISON["Ciclo"]==ciclo_sel, "Region"].unique())
#     colA, colB = st.sidebar.columns(2)
#     region_A = colA.selectbox("Región A", regs_ciclo, key="region_A")
#     region_B = colB.selectbox("Región B", regs_ciclo, index=min(1, len(regs_ciclo)-1), key="region_B")

# # ---------------------------
# # Layout principal
# # ---------------------------
# st.title("💧 DataAqua — Dashboard 2")
# st.caption("Resultados UNISON (FAO-56). ETc (demanda del cultivo), ETverde (cubierta por Pef) y ETazul (resto). ET0 es referencia (césped).")

# tabs_main = st.tabs(["KPIs", "Serie diaria", "Acumulados", "Decádico", "Kc–ET0", "Drivers ET0", "Datos"])

# if modo == "Ciclo individual":
#     ruta_sel = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
#     if ruta_sel.empty:
#         st.error(f"No encontré CSV para {region_sel} / {ciclo_sel}")
#         st.stop()
#     df = leer_unison(ruta_sel.iloc[0])
#     if df.empty:
#         st.error("No fue posible leer el archivo seleccionado.")
#         st.stop()

#     # --- Selector de variables para la Serie diaria (ya con df cargado) ---
#     vars_posibles = ["ET0", "ETc", "ETverde", "ETazul", "Pef"]
#     vars_disponibles = [v for v in vars_posibles if v in df.columns]
#     series_sel = st.sidebar.multiselect(
#         "Series a mostrar:",
#         options=vars_disponibles,
#         default=vars_disponibles
#     )


#     with tabs_main[0]:
#         st.subheader(f"KPIs — {region_sel} ({ciclo_sel})")
#         k = kpis_ciclo(df)
#         col1, col2, col3, col4 = st.columns(4)
#         col1.metric("Días del ciclo", f"{k['dias']}")
#         col2.metric("ETc total [mm]", f"{k['etc_total']:.1f}")
#         col3.metric("ETazul total [mm]", f"{k['eta_total']:.1f}")
#         col4.metric("% Azul", f"{k['pct_azul']:.1f}%")
#         col5, col6, col7, col8 = st.columns(4)
#         col5.metric("ETverde total [mm]", f"{k['etv_total']:.1f}")
#         col6.metric("Pef total [mm]", f"{k['pef_total']:.1f}")
#         col7.metric("Días con déficit (ETc>Pef)", f"{k['dias_def']}")
#         col8.metric("Pico ETc p95 [mm/d]", f"{k['pico_p95']:.2f}")

#     with tabs_main[1]:
#         if show_series:
#             #fig = fig_series(df, f"Serie diaria — {region_sel} ({ciclo_sel})", eje=eje_opt)
#             #st.pyplot(fig, use_container_width=True)
#             fig = fig_series(df, f"Serie diaria — {region_sel} ({ciclo_sel})", eje=eje_opt, mostrar=series_sel)
#             st.pyplot(fig, use_container_width=True)            
#         else:
#             st.info("Selecciona al menos una serie para graficar.")
#             #st.info("Activa 'Serie diaria' en el panel izquierdo.")

#     with tabs_main[2]:
#         if show_acumulados:
#             fig = fig_acumulados(df, f"Acumulados — {region_sel} ({ciclo_sel})", eje=eje_opt)
#             st.pyplot(fig, use_container_width=True)
#         else:
#             st.info("Activa 'Acumulados' en el panel izquierdo.")

#     with tabs_main[3]:
#         if show_decadico:
#             fdec = fig_decadico(df, f"Decádico — {region_sel} ({ciclo_sel})")
#             if fdec is not None:
#                 st.pyplot(fdec, use_container_width=True)
#             else:
#                 st.info("No hay columna 'decada' en este archivo.")
#         else:
#             st.info("Activa 'Decádico' en el panel izquierdo.")

#     with tabs_main[4]:
#         if show_kc:
#             fkc = fig_kc_et0(df, f"Kc y ET0 — {region_sel} ({ciclo_sel})", eje=eje_opt)
#             if fkc is not None:
#                 st.pyplot(fkc, use_container_width=True)
#             else:
#                 st.info("Faltan columnas 'Kc' o 'ET0'.")
#         else:
#             st.info("Activa 'Kc y ET0' en el panel izquierdo.")

#     with tabs_main[5]:
#         if show_drivers:
#             fdrv = fig_drivers_et0(df, f"Drivers de ET0 — {region_sel} ({ciclo_sel})")
#             if fdrv is not None:
#                 st.pyplot(fdrv, use_container_width=True)
#             else:
#                 st.info("Faltan columnas para drivers (ET0 y Rs/Rnl/HR/Ux/Tmean).")
#         else:
#             st.info("Activa 'Drivers de ET0' en el panel izquierdo.")

#     with tabs_main[6]:
#         st.dataframe(df.head(30), use_container_width=True)
#         @st.cache_data(show_spinner=False)
#         def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
#             return df_in.to_csv(index=False).encode("utf-8")
#         st.download_button(
#             "Descargar CSV (ciclo seleccionado)",
#             data=to_csv_bytes(df),
#             file_name=f"{region_sel}_{ciclo_sel}_DataAqua.csv",
#             mime="text/csv"
#         )

# elif modo == "Comparar ciclos":
#     ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_A)]["Ruta"]
#     ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_B)]["Ruta"]
#     if ruta_A.empty or ruta_B.empty:
#         st.error("No encontré ambos ciclos para esa región.")
#         st.stop()
#     dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])

#     # Selector de series (intersección de columnas disponibles en ambos)
#     vars_posibles = ["ET0", "ETc", "ETverde", "ETazul", "Pef"]
#     vars_comunes = [v for v in vars_posibles if v in dfA.columns and v in dfB.columns]
#     series_sel = st.sidebar.multiselect(
#         "Series a mostrar:",
#         options=vars_comunes,
#         default=vars_comunes
#     )


#     with tabs_main[0]:
#         st.subheader(f"KPIs — {region_sel} | {ciclo_A} vs {ciclo_B}")
#         c1, c2 = st.columns(2)
#         kA, kB = kpis_ciclo(dfA), kpis_ciclo(dfB)
#         with c1:
#             st.markdown(f"**Ciclo A:** {ciclo_A}")
#             st.metric("ETc total [mm]", f"{kA['etc_total']:.1f}")
#             st.metric("ETazul total [mm]", f"{kA['eta_total']:.1f}")
#             st.metric("% Azul", f"{kA['pct_azul']:.1f}%")
#             st.metric("Días (ETc>Pef)", f"{kA['dias_def']}")
#         with c2:
#             st.markdown(f"**Ciclo B:** {ciclo_B}")
#             st.metric("ETc total [mm]", f"{kB['etc_total']:.1f}")
#             st.metric("ETazul total [mm]", f"{kB['eta_total']:.1f}")
#             st.metric("% Azul", f"{kB['pct_azul']:.1f}%")
#             st.metric("Días (ETc>Pef)", f"{kB['dias_def']}")

#     #with tabs_main[1]:
#         # xA = _xcol(dfA, eje_opt); xB = _xcol(dfB, eje_opt)
#         # fig, ax = plt.subplots(1,1, figsize=(12,4))
#         # if "ETc" in dfA: ax.plot(dfA[xA], dfA["ETc"], label=f"ETc {ciclo_A}", lw=1.5, color="#1f77b4")
#         # if "ETc" in dfB: ax.plot(dfB[xB], dfB["ETc"], label=f"ETc {ciclo_B}", lw=1.5, color="#ff7f0e")
#         # if "ETazul" in dfA: ax.plot(dfA[xA], dfA["ETazul"], label=f"ETazul {ciclo_A}", lw=1.2, color="#1f77b4", ls="--")
#         # if "ETazul" in dfB: ax.plot(dfB[xB], dfB["ETazul"], label=f"ETazul {ciclo_B}", lw=1.2, color="#ff7f0e", ls="--")
#         # ax.set_title(f"Serie diaria — {region_sel}")
#         # ax.set_xlabel(eje_opt); ax.set_ylabel("mm/día"); ax.legend()
#         # fig.tight_layout(); st.pyplot(fig, use_container_width=True)

#     with tabs_main[1]:
#         xA = _xcol(dfA, eje_opt); xB = _xcol(dfB, eje_opt)
#         fig, ax = plt.subplots(1,1, figsize=(12,4))
#         colores = {"ET0":"#4C78A8","ETc":"#F58518","ETverde":"#54A24B","ETazul":"#E45756","Pef":"#9D9D9D"}
#         for v in series_sel:
#             if v in dfA: ax.plot(dfA[xA], dfA[v], label=f"{v} {ciclo_A}", lw=1.5, color=colores.get(v))
#             if v in dfB: ax.plot(dfB[xB], dfB[v], label=f"{v} {ciclo_B}", lw=1.5, linestyle="--", color=colores.get(v))
#         ax.set_title(f"Serie diaria — {region_sel}")
#         ax.set_xlabel(eje_opt); ax.set_ylabel("mm/día"); ax.legend()
#         fig.tight_layout(); st.pyplot(fig, use_container_width=True)
        
#     with tabs_main[2]:
#         fig, ax = plt.subplots(1,1, figsize=(12,4))
#         if "ETc" in dfA: ax.plot(dfA[_xcol(dfA,eje_opt)], dfA["ETc"].cumsum(), label=f"ETc {ciclo_A}", lw=1.8, color="#1f77b4")
#         if "ETc" in dfB: ax.plot(dfB[_xcol(dfB,eje_opt)], dfB["ETc"].cumsum(), label=f"ETc {ciclo_B}", lw=1.8, color="#ff7f0e")
#         if "ETazul" in dfA: ax.plot(dfA[_xcol(dfA,eje_opt)], dfA["ETazul"].cumsum(), label=f"ETazul {ciclo_A}", lw=1.8, color="#1f77b4", ls="--")
#         if "ETazul" in dfB: ax.plot(dfB[_xcol(dfB,eje_opt)], dfB["ETazul"].cumsum(), label=f"ETazul {ciclo_B}", lw=1.8, color="#ff7f0e", ls="--")
#         ax.set_title(f"Acumulados — {region_sel}")
#         ax.set_xlabel(eje_opt); ax.set_ylabel("mm"); ax.legend()
#         fig.tight_layout(); st.pyplot(fig, use_container_width=True)

#     with tabs_main[3]:
#         fA = fig_decadico(dfA, f"Decádico — {region_sel} ({ciclo_A})")
#         fB = fig_decadico(dfB, f"Decádico — {region_sel} ({ciclo_B})")
#         cols = st.columns(2)
#         if fA: cols[0].pyplot(fA, use_container_width=True)
#         if fB: cols[1].pyplot(fB, use_container_width=True)

#     with tabs_main[4]:
#         fA = fig_kc_et0(dfA, f"Kc–ET0 — {region_sel} ({ciclo_A})", eje=eje_opt)
#         fB = fig_kc_et0(dfB, f"Kc–ET0 — {region_sel} ({ciclo_B})", eje=eje_opt)
#         cols = st.columns(2)
#         if fA: cols[0].pyplot(fA, use_container_width=True)
#         if fB: cols[1].pyplot(fB, use_container_width=True)

#     with tabs_main[5]:
#         fA = fig_drivers_et0(dfA, f"Drivers ET0 — {region_sel} ({ciclo_A})")
#         fB = fig_drivers_et0(dfB, f"Drivers ET0 — {region_sel} ({ciclo_B})")
#         cols = st.columns(2)
#         if fA: cols[0].pyplot(fA, use_container_width=True)
#         if fB: cols[1].pyplot(fB, use_container_width=True)

#     with tabs_main[6]:
#         st.write("**Primeras filas ciclo A**")
#         st.dataframe(dfA.head(20), use_container_width=True)
#         st.write("**Primeras filas ciclo B**")
#         st.dataframe(dfB.head(20), use_container_width=True)

# elif modo == "Comparar regiones":
#     ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_A) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
#     ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_B) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
#     if ruta_A.empty or ruta_B.empty:
#         st.error("No encontré ambas regiones para ese ciclo.")
#         st.stop()
#     dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])

#     vars_posibles = ["ET0", "ETc", "ETverde", "ETazul", "Pef"]
#     vars_comunes = [v for v in vars_posibles if v in dfA.columns and v in dfB.columns]
#     series_sel = st.sidebar.multiselect(
#         "Series a mostrar:",
#         options=vars_comunes,
#         default=vars_comunes
#     )

#     with tabs_main[0]:
#         st.subheader(f"KPIs — {ciclo_sel} | {region_A} vs {region_B}")
#         c1, c2 = st.columns(2)
#         kA, kB = kpis_ciclo(dfA), kpis_ciclo(dfB)
#         with c1:
#             st.markdown(f"**{region_A}**")
#             st.metric("ETc total [mm]", f"{kA['etc_total']:.1f}")
#             st.metric("ETazul total [mm]", f"{kA['eta_total']:.1f}")
#             st.metric("% Azul", f"{kA['pct_azul']:.1f}%")
#             st.metric("Días (ETc>Pef)", f"{kA['dias_def']}")
#         with c2:
#             st.markdown(f"**{region_B}**")
#             st.metric("ETc total [mm]", f"{kB['etc_total']:.1f}")
#             st.metric("ETazul total [mm]", f"{kB['eta_total']:.1f}")
#             st.metric("% Azul", f"{kB['pct_azul']:.1f}%")
#             st.metric("Días (ETc>Pef)", f"{kB['dias_def']}")

#     # with tabs_main[1]:
#     #     xA = _xcol(dfA, eje_opt); xB = _xcol(dfB, eje_opt)
#     #     fig, ax = plt.subplots(1,1, figsize=(12,4))
#     #     if "ETc" in dfA: ax.plot(dfA[xA], dfA["ETc"], label=f"{region_A} ETc", lw=1.5, color="#1f77b4")
#     #     if "ETc" in dfB: ax.plot(dfB[xB], dfB["ETc"], label=f"{region_B} ETc", lw=1.5, color="#ff7f0e")
#     #     if "ETazul" in dfA: ax.plot(dfA[xA], dfA["ETazul"], label=f"{region_A} ETazul", lw=1.2, color="#1f77b4", ls="--")
#     #     if "ETazul" in dfB: ax.plot(dfB[xB], dfB["ETazul"], label=f"{region_B} ETazul", lw=1.2, color="#ff7f0e", ls="--")
#     #     ax.set_title(f"Serie diaria — {region_A} vs {region_B}")
#     #     ax.set_xlabel(eje_opt); ax.set_ylabel("mm/día"); ax.legend()
#     #     fig.tight_layout(); st.pyplot(fig, use_container_width=True)

#     with tabs_main[1]:
#         xA = _xcol(dfA, eje_opt); xB = _xcol(dfB, eje_opt)
#         fig, ax = plt.subplots(1,1, figsize=(12,4))
#         colores = {"ET0":"#4C78A8","ETc":"#F58518","ETverde":"#54A24B","ETazul":"#E45756","Pef":"#9D9D9D"}
#         for v in series_sel:
#             if v in dfA: ax.plot(dfA[xA], dfA[v], label=f"{region_A} {v}", lw=1.5, color=colores.get(v))
#             if v in dfB: ax.plot(dfB[xB], dfB[v], label=f"{region_B} {v}", lw=1.5, linestyle="--", color=colores.get(v))
#         ax.set_title(f"Serie diaria — {region_A} vs {region_B}")
#         ax.set_xlabel(eje_opt); ax.set_ylabel("mm/día"); ax.legend()
#         fig.tight_layout(); st.pyplot(fig, use_container_width=True)

#     with tabs_main[2]:
#         fig, ax = plt.subplots(1,1, figsize=(12,4))
#         if "ETc" in dfA: ax.plot(dfA[_xcol(dfA,eje_opt)], dfA["ETc"].cumsum(), label=f"{region_A} ETc", lw=1.8, color="#1f77b4")
#         if "ETc" in dfB: ax.plot(dfB[_xcol(dfB,eje_opt)], dfB["ETc"].cumsum(), label=f"{region_B} ETc", lw=1.8, color="#ff7f0e")
#         if "ETazul" in dfA: ax.plot(dfA[_xcol(dfA,eje_opt)], dfA["ETazul"].cumsum(), label=f"{region_A} ETazul", lw=1.8, color="#1f77b4", ls="--")
#         if "ETazul" in dfB: ax.plot(dfB[_xcol(dfB,eje_opt)], dfB["ETazul"].cumsum(), label=f"{region_B} ETazul", lw=1.8, color="#ff7f0e", ls="--")
#         ax.set_title(f"Acumulados — {region_A} vs {region_B}")
#         ax.set_xlabel(eje_opt); ax.set_ylabel("mm"); ax.legend()
#         fig.tight_layout(); st.pyplot(fig, use_container_width=True)

#     with tabs_main[3]:
#         fA = fig_decadico(dfA, f"Decádico — {region_A} ({ciclo_sel})")
#         fB = fig_decadico(dfB, f"Decádico — {region_B} ({ciclo_sel})")
#         cols = st.columns(2)
#         if fA: cols[0].pyplot(fA, use_container_width=True)
#         if fB: cols[1].pyplot(fB, use_container_width=True)

#     with tabs_main[4]:
#         fA = fig_kc_et0(dfA, f"Kc–ET0 — {region_A} ({ciclo_sel})", eje=eje_opt)
#         fB = fig_kc_et0(dfB, f"Kc–ET0 — {region_B} ({ciclo_sel})", eje=eje_opt)
#         cols = st.columns(2)
#         if fA: cols[0].pyplot(fA, use_container_width=True)
#         if fB: cols[1].pyplot(fB, use_container_width=True)

#     with tabs_main[5]:
#         fA = fig_drivers_et0(dfA, f"Drivers ET0 — {region_A} ({ciclo_sel})")
#         fB = fig_drivers_et0(dfB, f"Drivers ET0 — {region_B} ({ciclo_sel})")
#         cols = st.columns(2)
#         if fA: cols[0].pyplot(fA, use_container_width=True)
#         if fB: cols[1].pyplot(fB, use_container_width=True)

#     with tabs_main[6]:
#         st.write(f"**Primeras filas {region_A}**")
#         st.dataframe(dfA.head(20), use_container_width=True)
#         st.write(f"**Primeras filas {region_B}**")
#         st.dataframe(dfB.head(20), use_container_width=True)

# ===========================
# DataAqua Dashboard 2 (Streamlit)
# ===========================
# Modos:
#  - Ciclo individual
#  - Comparar ciclos (misma región)
#  - Comparar regiones (mismo ciclo)
#
# Todo en una sola página (sin pestañas)
# Secciones:
#  - KPIs (días, siembra, cosecha, ETc, ETazul, ETverde, Tmax, Tmin, UAC/HH)
#  - Serie diaria (ET0, ETc, ETverde, ETazul, Pef)  [multiselect]
#  - Temperaturas (Tmin, Tmean, Tmax)                [multiselect]
#  - Meteorología: Rs + HR (ejes gemelos) + Ux (toggle)
#
# Ejecuta:
#   streamlit run dashboard2.py
# ===========================

# ################################################################################################
# from pathlib import Path
# import os, re
# from datetime import datetime, timedelta

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st

# # ---------------------------
# # CONFIG
# # ---------------------------
# st.set_page_config(page_title="DataAqua — Dashboard 2", page_icon="💧", layout="wide")

# # Carpeta de datos relativa al repo
# RUTA_SALIDA_UNISON = Path("data") / "Salidas_ETo12_con_uac_y_hh" / "Periodo de Cultivo ETo"

# plt.rcParams["figure.dpi"] = 120

# # Column map (tus nombres a nombres limpios)
# MAP_UNISON = {
#     "Año_ (YEAR)": "Year", "AÃ±o_ (YEAR)": "Year",
#     "Día (DOY)": "DOY",   "DÃ­a (DOY)": "DOY",
#     "Tmax (T2M_MAX)": "Tmax", "Tmin (T2M_MIN)": "Tmin",
#     "HR (RH2M)": "HR", "Ux (WS2M)": "Ux",
#     "Rs (ALLSKY_SFC_SW_DWN)": "Rs",
#     "Rl_ (ALLSKY_SFC_LW_DWN)": "Rl",
#     "Ptot_ (PRECTOTCORR)": "Ptot",
#     "Pef_": "Pef", "Tmean_": "Tmean", "es_": "es", "ea_": "ea",
#     "delta_": "delta", "P_": "P", "gamma_": "gamma",
#     "Rns_": "Rns", "Rnl_": "Rnl", "Rn_": "Rn", "Rso_": "Rso",
#     "Kc_": "Kc", "decada_": "decada",
#     "ET0": "ET0", "ETc": "ETc", "ETverde": "ETverde", "ETazul": "ETazul",
#     "Year": "Year", "DOY": "DOY", "Dia": "Dia",
# }

# # Columnas que trataremos como numéricas si aparecen
# COLUMNAS_MIN = [
#     "Year","DOY","ET0","ETc","ETverde","ETazul","Pef","decada",
#     "Rns","Rnl","Rs","Tmean","HR","Ux","Kc","Tmax","Tmin",
#     "UACverde_m3_ha","UACazul_m3_ha","HHverde_m3_ton","HHazul_m3_ton"
# ]

# # ---------------------------
# # Helpers de archivos
# # ---------------------------
# def parse_unison_filename(filename: str):
#     """
#     'Cajeme-FAO56-2014-2015-SALIDA.csv' -> ('Cajeme','2014-2015')
#     'Metepec-FAO56-2014-SALIDA.csv'     -> ('Metepec','2014')
#     """
#     m = re.match(r"([A-Za-zÁÉÍÓÚáéíóúñÑ\s]+)-FAO56-(\d{4})(?:-(\d{4}))?-SALIDA\.csv$", filename, re.I)
#     if not m:
#         return None, None
#     reg, y1, y2 = m.groups()
#     if reg == "VillaAllende": reg = "Villa de Allende"
#     if reg == "Etchhojoa":    reg = "Etchojoa"
#     ciclo = y1 if not y2 else f"{y1}-{y2}"
#     return reg.strip(), ciclo

# @st.cache_data(show_spinner=False)
# def catalogo_unison(base_dir: Path) -> pd.DataFrame:
#     rows = []
#     if not base_dir.exists():
#         return pd.DataFrame(columns=["Region","Ciclo","Ruta"])
#     for reg_folder in sorted(os.listdir(base_dir)):
#         d = base_dir / reg_folder
#         if not d.is_dir():
#             continue
#         for f in sorted(os.listdir(d)):
#             if not f.lower().endswith(".csv"):
#                 continue
#             reg, ciclo = parse_unison_filename(f)
#             if reg and ciclo:
#                 rows.append({"Region": reg, "Ciclo": ciclo, "Ruta": str(d / f)})
#     df = pd.DataFrame(rows).sort_values(["Region","Ciclo"]).reset_index(drop=True)
#     return df

# # ---------------------------
# # Helpers de datos
# # ---------------------------
# def _year_doy_to_date(y, doy):
#     try:
#         base = datetime(int(y), 1, 1)
#         return base + timedelta(days=int(doy) - 1)
#     except Exception:
#         return pd.NaT

# @st.cache_data(show_spinner=False)
# def leer_unison(path: str) -> pd.DataFrame:
#     p = Path(path)
#     if not p.exists():
#         return pd.DataFrame()

#     # Lectura tolerante a encoding
#     last_err = None
#     for enc in ("utf-8","latin-1"):
#         try:
#             df = pd.read_csv(p, encoding=enc)
#             last_err = None
#             break
#         except UnicodeDecodeError as e:
#             last_err = e
#             continue
#     if last_err is not None:
#         df = pd.read_csv(p)

#     df.columns = [c.strip() for c in df.columns]
#     df = df.rename(columns=lambda c: MAP_UNISON.get(c, c))

#     # Campos numéricos
#     for c in set(COLUMNAS_MIN).intersection(df.columns):
#         df[c] = pd.to_numeric(df[c], errors="coerce")

#     # Fecha y día de ciclo
#     if {"Year","DOY"}.issubset(df.columns):
#         fechas = [_year_doy_to_date(y,d) for y,d in zip(df["Year"], df["DOY"])]
#         df["Fecha"] = pd.to_datetime(fechas)
#         if df["Fecha"].notna().any():
#             f0 = df["Fecha"].dropna().iloc[0]
#             df["Dia_ciclo"] = (df["Fecha"] - f0).dt.days.astype("Int64")
#         else:
#             df["Dia_ciclo"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
#     else:
#         df["Fecha"] = pd.NaT
#         df["Dia_ciclo"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

#     # Acumulados y proporciones útiles
#     if "ETc" in df:
#         df["ETc_acum"] = df["ETc"].cumsum()
#     if "ETazul" in df:
#         df["ETazul_acum"] = df["ETazul"].cumsum()
#     if {"ETc","ETazul"}.issubset(df.columns):
#         df["pct_azul"] = np.where(df["ETc"]>0, df["ETazul"]/df["ETc"]*100.0, np.nan)
#     return df

# def kpis_ciclo(df: pd.DataFrame) -> dict:
#     mask = df["ETc"].notna() if "ETc" in df else pd.Series(False, index=df.index)
#     dias = int(mask.sum())
#     etc_total = float(df.loc[mask, "ETc"].sum())     if "ETc"     in df else np.nan
#     etv_total = float(df.loc[mask, "ETverde"].sum()) if "ETverde" in df else np.nan
#     eta_total = float(df.loc[mask, "ETazul"].sum())  if "ETazul"  in df else np.nan
#     return {"dias": dias, "etc_total": etc_total, "etv_total": etv_total, "eta_total": eta_total}

# def fechas_ciclo(df: pd.DataFrame):
#     if "Fecha" in df and df["Fecha"].notna().any():
#         fmin = pd.to_datetime(df["Fecha"].dropna().iloc[0])
#         fmax = pd.to_datetime(df["Fecha"].dropna().iloc[-1])
#         return fmin.date(), fmax.date()
#     return None, None

# def last_valid(df: pd.DataFrame, col: str):
#     return float(df[col].dropna().iloc[-1]) if (col in df and df[col].notna().any()) else np.nan

# def kpis_ext(df: pd.DataFrame):
#     k = kpis_ciclo(df)
#     siembra, cosecha = fechas_ciclo(df)

#     # UAC y HH desde columnas del CSV (último valor no nulo)
#     uacv_ha = last_valid(df, "UACverde_m3_ha")
#     uaca_ha = last_valid(df, "UACazul_m3_ha")
#     hhv_ton = last_valid(df, "HHverde_m3_ton")
#     hha_ton = last_valid(df, "HHazul_m3_ton")

#     out = {
#         "dias": k["dias"],
#         "siembra": siembra, "cosecha": cosecha,
#         "etc_total": k["etc_total"],
#         "eta_total": k["eta_total"],
#         "etv_total": k["etv_total"],
#         "tmax": float(df["Tmax"].max()) if "Tmax" in df else np.nan,
#         "tmin": float(df["Tmin"].min()) if "Tmin" in df else np.nan,
#         "uacv_ha": uacv_ha,  # m³/ha (verde)
#         "uaca_ha": uaca_ha,  # m³/ha (azul)
#         "hhv_ton": hhv_ton,  # m³/ton (verde)
#         "hha_ton": hha_ton,  # m³/ton (azul)
#     }
#     return out

# def _xcol(df: pd.DataFrame, prefer="Fecha"):
#     if prefer in df.columns and df[prefer].notna().any():
#         return prefer
#     for alt in ("Fecha", "Dia_ciclo"):
#         if alt in df.columns and df[alt].notna().any():
#             return alt
#     return df.index

# # ---------------------------
# # Figuras
# # ---------------------------
# def fig_series(df: pd.DataFrame, titulo: str, eje="Fecha", mostrar=("ET0","ETc","ETverde","ETazul","Pef")):
#     x = _xcol(df, eje)
#     fig, ax = plt.subplots(1,1, figsize=(12,4))
#     colores = {"ET0":"#4C78A8","ETc":"#F58518","ETverde":"#54A24B","ETazul":"#E45756","Pef":"#9D9D9D"}
#     for col in mostrar:
#         if col in df:
#             ax.plot(df[x], df[col], label=col, lw=1.6, color=colores.get(col, None))
#     ax.set_title(titulo); ax.set_xlabel(str(x)); ax.set_ylabel("mm/día"); ax.legend()
#     fig.tight_layout(); return fig

# def fig_temperaturas(df, titulo, eje="Fecha", mostrar=("Tmin","Tmean","Tmax")):
#     x = _xcol(df, eje)
#     fig, ax = plt.subplots(figsize=(12,4))
#     for c in mostrar:
#         if c in df:
#             ax.plot(df[x], df[c], lw=1.4, label=c)
#     ax.set_title(titulo); ax.set_xlabel(str(x)); ax.set_ylabel("°C"); ax.legend()
#     fig.tight_layout(); return fig

# def fig_meteo_rs_hr(df, titulo, eje="Fecha", show_ux=True):
#     x = _xcol(df, eje)
#     fig, ax1 = plt.subplots(figsize=(12,4))
#     if "Rs" in df:
#         ax1.plot(df[x], df["Rs"], lw=1.5, label="Rs")
#     ax1.set_ylabel("Rs [MJ m$^{-2}$ d$^{-1}$]")
#     ax2 = ax1.twinx()
#     if "HR" in df:
#         ax2.plot(df[x], df["HR"], lw=1.2, label="HR", linestyle="--")
#     ax2.set_ylabel("HR [%]")
#     ax1.set_title(titulo); ax1.set_xlabel(str(x))
#     fig.tight_layout()

#     fig2 = None
#     if show_ux and "Ux" in df:
#         fig2, ax = plt.subplots(figsize=(12,2.5))
#         ax.plot(df[x], df["Ux"], lw=1.2)
#         ax.set_title("Viento Ux"); ax.set_xlabel(str(x)); ax.set_ylabel("m/s")
#         fig2.tight_layout()
#     return fig, fig2

# # ---------------------------
# # Sidebar
# # ---------------------------
# st.sidebar.title("DataAqua — Selección")
# CAT_UNISON = catalogo_unison(RUTA_SALIDA_UNISON)
# if CAT_UNISON.empty:
#     st.error("No se encontraron archivos en la carpeta de datos.")
#     st.stop()

# modo = st.sidebar.radio("Modo", ["Ciclo individual", "Comparar ciclos", "Comparar regiones"], index=0)

# # “Eje X” menos técnico
# verpor_label = st.sidebar.radio("Ver por", ["Fecha", "Día del ciclo"], index=0)
# eje_opt = "Dia_ciclo" if verpor_label == "Día del ciclo" else "Fecha"

# if modo == "Ciclo individual":
#     regiones = sorted(CAT_UNISON["Region"].unique())
#     region_sel = st.sidebar.selectbox("Región", regiones)
#     ciclos_reg = sorted(CAT_UNISON.loc[CAT_UNISON["Region"]==region_sel, "Ciclo"].unique())
#     ciclo_sel = st.sidebar.selectbox("Ciclo", ciclos_reg)

# elif modo == "Comparar ciclos":
#     regiones = sorted(CAT_UNISON["Region"].unique())
#     region_sel = st.sidebar.selectbox("Región", regiones)
#     ciclos_reg = sorted(CAT_UNISON.loc[CAT_UNISON["Region"]==region_sel, "Ciclo"].unique())
#     colA, colB = st.sidebar.columns(2)
#     ciclo_A = colA.selectbox("Ciclo A", ciclos_reg, key="ciclo_A")
#     ciclo_B = colB.selectbox("Ciclo B", ciclos_reg, index=min(1, len(ciclos_reg)-1), key="ciclo_B")

# elif modo == "Comparar regiones":
#     ciclos = sorted(CAT_UNISON["Ciclo"].unique())
#     ciclo_sel = st.sidebar.selectbox("Ciclo", ciclos)
#     regs_ciclo = sorted(CAT_UNISON.loc[CAT_UNISON["Ciclo"]==ciclo_sel, "Region"].unique())
#     colA, colB = st.sidebar.columns(2)
#     region_A = colA.selectbox("Región A", regs_ciclo, key="region_A")
#     region_B = colB.selectbox("Región B", regs_ciclo, index=min(1, len(regs_ciclo)-1), key="region_B")

# # ---------------------------
# # Layout principal
# # ---------------------------
# st.title("💧 DataAqua — Dashboard 2")
# st.caption("Resultados UNISON (FAO-56). ETc (demanda del cultivo), ETverde (cubierta por Pef) y ETazul (resto). ET0 es referencia (césped).")

# # --- Modo: Ciclo individual
# if modo == "Ciclo individual":
#     ruta_sel = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
#     if ruta_sel.empty:
#         st.error(f"No encontré CSV para {region_sel} / {ciclo_sel}"); st.stop()
#     df = leer_unison(ruta_sel.iloc[0])
#     if df.empty:
#         st.error("No fue posible leer el archivo seleccionado."); st.stop()

#     # KPIs
#     st.subheader(f"KPIs — {region_sel} ({ciclo_sel})")
#     k = kpis_ext(df)
#     c1,c2,c3,c4 = st.columns(4)
#     c1.metric("Días del ciclo", f"{k['dias']}")
#     c2.metric("Fecha de siembra", f"{k['siembra'] or '—'}")
#     c3.metric("Fecha de cosecha", f"{k['cosecha'] or '—'}")
#     c4.metric("% Azul", f"{(k['eta_total']/k['etc_total']*100):.1f}%" if k['etc_total'] else "—")

#     c5,c6,c7,c8 = st.columns(4)
#     c5.metric("ETc total [mm]", f"{k['etc_total']:.1f}")
#     c6.metric("ETazul total [mm]", f"{k['eta_total']:.1f}")
#     c7.metric("ETverde total [mm]", f"{k['etv_total']:.1f}")
#     c8.metric("Tmax / Tmin [°C]", f"{k['tmax']:.1f} / {k['tmin']:.1f}")

#     c9,c10,c11,c12 = st.columns(4)
#     c9.metric("UAC verde [m³/ha]", f"{k['uacv_ha']:.0f}" if not np.isnan(k['uacv_ha']) else "—")
#     c10.metric("UAC azul [m³/ha]",  f"{k['uaca_ha']:.0f}" if not np.isnan(k['uaca_ha']) else "—")
#     c11.metric("HH verde [m³/ton]", f"{k['hhv_ton']:.0f}" if not np.isnan(k['hhv_ton']) else "—")
#     c12.metric("HH azul [m³/ton]",  f"{k['hha_ton']:.0f}" if not np.isnan(k['hha_ton']) else "—")

#     st.divider()

#     # Serie diaria (ET)
#     st.markdown("### Serie diaria (ET)")
#     et_opts = [v for v in ["ET0","ETc","ETverde","ETazul","Pef"] if v in df.columns]
#     et_sel = st.multiselect("Series a mostrar", et_opts, default=et_opts)
#     fig = fig_series(df, f"{region_sel} — {ciclo_sel}", eje=eje_opt, mostrar=et_sel or et_opts)
#     st.pyplot(fig, use_container_width=True)

#     # Temperaturas
#     st.markdown("### Temperaturas")
#     t_opts = [v for v in ["Tmin","Tmean","Tmax"] if v in df.columns]
#     t_sel = st.multiselect("Series de temperatura", t_opts, default=t_opts)
#     ftemp = fig_temperaturas(df, f"{region_sel} — {ciclo_sel}", eje=eje_opt, mostrar=t_sel or t_opts)
#     st.pyplot(ftemp, use_container_width=True)

#     # Meteorología
#     st.markdown("### Meteorología")
#     show_ux = st.toggle("Mostrar Ux (viento)", value=True)
#     frs, fux = fig_meteo_rs_hr(df, f"{region_sel} — {ciclo_sel}", eje=eje_opt, show_ux=show_ux)
#     st.pyplot(frs, use_container_width=True)
#     if fux is not None:
#         st.pyplot(fux, use_container_width=True)

#     st.divider()
#     with st.expander("Datos (primeras filas)"):
#         st.dataframe(df.head(30), use_container_width=True)

# # --- Modo: Comparar ciclos
# elif modo == "Comparar ciclos":
#     ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_A)]["Ruta"]
#     ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_B)]["Ruta"]
#     if ruta_A.empty or ruta_B.empty: st.error("No encontré ambos ciclos."); st.stop()
#     dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])

#     # Para que el multiselect aplique a ambas, usamos intersección
#     et_inter = [v for v in ["ET0","ETc","ETverde","ETazul","Pef"] if v in dfA.columns and v in dfB.columns]
#     t_inter  = [v for v in ["Tmin","Tmean","Tmax"] if v in dfA.columns and v in dfB.columns]

#     colA, colB = st.columns(2, gap="large")
#     with st.container():
#         st.markdown("### Opciones de series (aplican a ambas columnas)")
#         et_sel = st.multiselect("Serie diaria (ET)", et_inter, default=et_inter, key="et_cmp")
#         t_sel  = st.multiselect("Temperaturas", t_inter,  default=t_inter,  key="t_cmp")
#         show_ux = st.toggle("Mostrar Ux (viento)", value=True, key="ux_cmp")

#     for (titulo, df, col) in [(f"{region_sel} — {ciclo_A}", dfA, colA), (f"{region_sel} — {ciclo_B}", dfB, colB)]:
#         with col:
#             st.subheader(titulo)
#             k = kpis_ext(df)
#             c1,c2 = st.columns(2)
#             c1.metric("Días del ciclo", f"{k['dias']}")
#             c2.metric("% Azul", f"{(k['eta_total']/k['etc_total']*100):.1f}%" if k['etc_total'] else "—")
#             c3,c4 = st.columns(2)
#             c3.metric("ETc total [mm]", f"{k['etc_total']:.1f}")
#             c4.metric("ETazul total [mm]", f"{k['eta_total']:.1f}")
#             c5,c6 = st.columns(2)
#             c5.metric("UAC verde [m³/ha]", f"{k['uacv_ha']:.0f}" if not np.isnan(k['uacv_ha']) else "—")
#             c6.metric("UAC azul  [m³/ha]", f"{k['uaca_ha']:.0f}" if not np.isnan(k['uaca_ha']) else "—")

#             st.pyplot(fig_series(df, "Serie diaria (ET)", eje=eje_opt, mostrar=et_sel or et_inter), use_container_width=True)
#             st.pyplot(fig_temperaturas(df, "Temperaturas", eje=eje_opt, mostrar=t_sel or t_inter), use_container_width=True)
#             frs, fux = fig_meteo_rs_hr(df, "Meteorología", eje=eje_opt, show_ux=show_ux)
#             st.pyplot(frs, use_container_width=True)
#             if fux is not None: st.pyplot(fux, use_container_width=True)

# # --- Modo: Comparar regiones
# elif modo == "Comparar regiones":
#     ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_A) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
#     ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_B) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
#     if ruta_A.empty or ruta_B.empty: st.error("No encontré ambas regiones."); st.stop()
#     dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])

#     et_inter = [v for v in ["ET0","ETc","ETverde","ETazul","Pef"] if v in dfA.columns and v in dfB.columns]
#     t_inter  = [v for v in ["Tmin","Tmean","Tmax"] if v in dfA.columns and v in dfB.columns]

#     colA, colB = st.columns(2, gap="large")
#     with st.container():
#         st.markdown("### Opciones de series (aplican a ambas columnas)")
#         et_sel = st.multiselect("Serie diaria (ET)", et_inter, default=et_inter, key="et_cmp_reg")
#         t_sel  = st.multiselect("Temperaturas", t_inter,  default=t_inter,  key="t_cmp_reg")
#         show_ux = st.toggle("Mostrar Ux (viento)", value=True, key="ux_cmp_reg")

#     for (titulo, df, col) in [(f"{region_A} — {ciclo_sel}", dfA, colA), (f"{region_B} — {ciclo_sel}", dfB, colB)]:
#         with col:
#             st.subheader(titulo)
#             k = kpis_ext(df)
#             c1,c2 = st.columns(2)
#             c1.metric("Días del ciclo", f"{k['dias']}")
#             c2.metric("% Azul", f"{(k['eta_total']/k['etc_total']*100):.1f}%" if k['etc_total'] else "—")
#             c3,c4 = st.columns(2)
#             c3.metric("ETc total [mm]", f"{k['etc_total']:.1f}")
#             c4.metric("ETazul total [mm]", f"{k['eta_total']:.1f}")
#             c5,c6 = st.columns(2)
#             c5.metric("UAC verde [m³/ha]", f"{k['uacv_ha']:.0f}" if not np.isnan(k['uacv_ha']) else "—")
#             c6.metric("UAC azul  [m³/ha]", f"{k['uaca_ha']:.0f}" if not np.isnan(k['uaca_ha']) else "—")

#             st.pyplot(fig_series(df, "Serie diaria (ET)", eje=eje_opt, mostrar=et_sel or et_inter), use_container_width=True)
#             st.pyplot(fig_temperaturas(df, "Temperaturas", eje=eje_opt, mostrar=t_sel or t_inter), use_container_width=True)
#             frs, fux = fig_meteo_rs_hr(df, "Meteorología", eje=eje_opt, show_ux=show_ux)
#             st.pyplot(frs, use_container_width=True)
#             if fux is not None: st.pyplot(fux, use_container_width=True)
# #######################################################################################################################################

# ===========================
# DataAqua Dashboard 2 (Streamlit)
# ===========================
# Modos:
#  - Ciclo individual
#  - Comparar ciclos (misma región)
#  - Comparar regiones (mismo ciclo)
#
# Todo en una sola página (sin pestañas)
# Secciones:
#  - KPIs (días, siembra, cosecha, ETc, ETazul, ETverde, Tmax, Tmin, UAC/HH)
#  - Serie diaria (ET0, ETc, ETverde, ETazul, Pef)  [multiselect]
#  - Temperaturas (Tmin, Tmean, Tmax)                [multiselect]
#  - Meteorología: Rs + HR (ejes gemelos)            [multiselect]
#  - Viento: Ux                                      (gráfica aparte)
#
# Ejecuta:
#   streamlit run dashboard2.py
# ===========================

# ===========================
# DataAqua Dashboard 2 (Streamlit)
# ===========================
# Modos:
#  - Ciclo individual
#  - Comparar ciclos (misma región)
#  - Comparar regiones (mismo ciclo)
#
# Todo en una sola página (sin pestañas)
# Secciones:
#  - KPIs (en dos columnas)
#  - Serie diaria (ET0, ETc, ETverde, ETazul, Pef)  [multiselect]
#  - Temperaturas (Tmin, Tmean, Tmax)                [multiselect]
#  - Meteorología: Rs + HR (ejes gemelos)            [multiselect]
#  - Viento: Ux                                      (gráfica aparte)
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

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="DataAqua — Dashboard 2", page_icon="💧", layout="wide")

# Carpeta de datos relativa al repo
RUTA_SALIDA_UNISON = Path("data") / "Salidas_ETo12_con_uac_y_hh" / "Periodo de Cultivo ETo"

plt.rcParams["figure.dpi"] = 120

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

# Columnas que trataremos como numéricas si aparecen
COLUMNAS_MIN = [
    "Year","DOY","ET0","ETc","ETverde","ETazul","Pef","decada",
    "Rns","Rnl","Rs","Tmean","HR","Ux","Kc","Tmax","Tmin",
    "UACverde_m3_ha","UACazul_m3_ha","HHverde_m3_ton","HHazul_m3_ton"
]

# ---------------------------
# Helpers de archivos
# ---------------------------

def hr():
    st.markdown(
        "<hr style='margin:0.5rem 0; border:none; border-top:1px solid #DDD;'/>",
        unsafe_allow_html=True
    )

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

    # Acumulados y proporciones útiles
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

# --- Modo: Ciclo individual
if modo == "Ciclo individual":
    ruta_sel = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
    if ruta_sel.empty:
        st.error(f"No encontré CSV para {region_sel} / {ciclo_sel}"); st.stop()
    df = leer_unison(ruta_sel.iloc[0])
    if df.empty:
        st.error("No fue posible leer el archivo seleccionado."); st.stop()

    # ===== KPIs (dos columnas) =====
    st.subheader(f"KPIs — {region_sel} ({ciclo_sel})")
    colL, colR = st.columns(2)
    k = kpis_ext(df)
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
        # Si quieres también las HH:
        st.metric("HH verde [m³/ton]", f"{k['hhv_ton']:.0f}" if not np.isnan(k['hhv_ton']) else "—")
        st.metric("HH azul [m³/ton]",  f"{k['hha_ton']:.0f}" if not np.isnan(k['hha_ton']) else "—")

    hr()

    # Serie diaria (ET)
    st.markdown("### Serie diaria (ET)")
    et_opts = [v for v in ["ET0","ETc","ETverde","ETazul","Pef"] if v in df.columns]
    et_sel = st.multiselect("Series a mostrar", et_opts, default=et_opts, key="et_ind")
    fig = fig_series(df, f"Serie diaria (ET) — {ciclo_sel}", eje=eje_opt, mostrar=et_sel or et_opts)
    st.pyplot(fig, use_container_width=True)

    # Temperaturas
    st.markdown("### Temperaturas")
    t_opts = [v for v in ["Tmin","Tmean","Tmax"] if v in df.columns]
    t_sel = st.multiselect("Series de temperatura", t_opts, default=t_opts, key="t_ind")
    ftemp = fig_temperaturas(df, f"Temperaturas — {ciclo_sel}", eje=eje_opt, mostrar=t_sel or t_opts)
    st.pyplot(ftemp, use_container_width=True)

    # Meteorología (Rs y HR con multiselect)
    st.markdown("### Meteorología")
    met_opts = [v for v in ["Rs","HR"] if v in df.columns]
    met_sel = st.multiselect("Variables de meteorología", met_opts, default=met_opts, key="met_ind")
    fmet = fig_meteo_rs_hr(df, f"Meteorología — {ciclo_sel}", eje=eje_opt, mostrar=met_sel or met_opts)
    st.pyplot(fmet, use_container_width=True)

    # Viento (Ux) en gráfica aparte
    if "Ux" in df.columns:
        st.markdown("### Viento")
        fux = fig_wind(df, f"Viento Ux — {ciclo_sel}", eje=eje_opt)
        st.pyplot(fux, use_container_width=True)

    hr()
    with st.expander("Datos (primeras filas)"):
        st.dataframe(df.head(30), use_container_width=True)

# --- Modo: Comparar ciclos (vertical, apiladas, separador fino)
elif modo == "Comparar ciclos":
    ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_A)]["Ruta"]
    ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_sel) & (CAT_UNISON.Ciclo==ciclo_B)]["Ruta"]
    if ruta_A.empty or ruta_B.empty: st.error("No encontré ambos ciclos."); st.stop()
    dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])

    # KPIs en dos columnas (A | B)
    st.subheader(f"KPIs — {region_sel} (comparación de ciclos)")
    colA, colB = st.columns(2)
    kA, kB = kpis_ext(dfA), kpis_ext(dfB)
    with colA:
        st.markdown(f"**{ciclo_A}**")
        st.metric("Días del ciclo", f"{kA['dias']}")
        st.metric("% Azul", f"{(kA['eta_total']/kA['etc_total']*100):.1f}%" if kA['etc_total'] else "—")
        st.metric("ETc total [mm]", f"{kA['etc_total']:.1f}")
        st.metric("ETazul total [mm]", f"{kA['eta_total']:.1f}")
        st.metric("UAC verde [m³/ha]", f"{kA['uacv_ha']:.0f}" if not np.isnan(kA['uacv_ha']) else "—")
        st.metric("UAC azul [m³/ha]",  f"{kA['uaca_ha']:.0f}" if not np.isnan(kA['uaca_ha']) else "—")
    with colB:
        st.markdown(f"**{ciclo_B}**")
        st.metric("Días del ciclo", f"{kB['dias']}")
        st.metric("% Azul", f"{(kB['eta_total']/kB['etc_total']*100):.1f}%" if kB['etc_total'] else "—")
        st.metric("ETc total [mm]", f"{kB['etc_total']:.1f}")
        st.metric("ETazul total [mm]", f"{kB['eta_total']:.1f}")
        st.metric("UAC verde [m³/ha]", f"{kB['uacv_ha']:.0f}" if not np.isnan(kB['uacv_ha']) else "—")
        st.metric("UAC azul [m³/ha]",  f"{kB['uaca_ha']:.0f}" if not np.isnan(kB['uaca_ha']) else "—")

    # Opciones comunes para multiselects
    et_inter = [v for v in ["ET0","ETc","ETverde","ETazul","Pef"] if v in dfA.columns and v in dfB.columns]
    t_inter  = [v for v in ["Tmin","Tmean","Tmax"] if v in dfA.columns and v in dfB.columns]
    met_inter= [v for v in ["Rs","HR"] if v in dfA.columns and v in dfB.columns]

    hr()
    st.markdown("#### Opciones de series (aplican a ambos ciclos)")
    et_sel = st.multiselect("Serie diaria (ET)", et_inter, default=et_inter, key="et_cmp")
    t_sel  = st.multiselect("Temperaturas", t_inter,  default=t_inter,  key="t_cmp")
    met_sel= st.multiselect("Meteorología", met_inter, default=met_inter, key="met_cmp")

    # Bloque A (gráficas apiladas)
    hr()
    st.markdown(f"### {region_sel} — {ciclo_A}")
    st.pyplot(fig_series(dfA, f"Serie diaria (ET) — {ciclo_A}", eje=eje_opt, mostrar=et_sel or et_inter), use_container_width=True)
    st.pyplot(fig_temperaturas(dfA, f"Temperaturas — {ciclo_A}", eje=eje_opt, mostrar=t_sel or t_inter), use_container_width=True)
    st.pyplot(fig_meteo_rs_hr(dfA, f"Meteorología — {ciclo_A}", eje=eje_opt, mostrar=met_sel or met_inter), use_container_width=True)
    if "Ux" in dfA.columns: st.pyplot(fig_wind(dfA, f"Viento Ux — {ciclo_A}", eje=eje_opt), use_container_width=True)

    # Bloque B (debajo del A)
    hr()
    st.markdown(f"### {region_sel} — {ciclo_B}")
    st.pyplot(fig_series(dfB, f"Serie diaria (ET) — {ciclo_B}", eje=eje_opt, mostrar=et_sel or et_inter), use_container_width=True)
    st.pyplot(fig_temperaturas(dfB, f"Temperaturas — {ciclo_B}", eje=eje_opt, mostrar=t_sel or t_inter), use_container_width=True)
    st.pyplot(fig_meteo_rs_hr(dfB, f"Meteorología — {ciclo_B}", eje=eje_opt, mostrar=met_sel or met_inter), use_container_width=True)
    if "Ux" in dfB.columns: st.pyplot(fig_wind(dfB, f"Viento Ux — {ciclo_B}", eje=eje_opt), use_container_width=True)

# --- Modo: Comparar regiones (vertical, apiladas, separador fino)
elif modo == "Comparar regiones":
    ruta_A = CAT_UNISON[(CAT_UNISON.Region==region_A) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
    ruta_B = CAT_UNISON[(CAT_UNISON.Region==region_B) & (CAT_UNISON.Ciclo==ciclo_sel)]["Ruta"]
    if ruta_A.empty or ruta_B.empty: st.error("No encontré ambas regiones."); st.stop()
    dfA = leer_unison(ruta_A.iloc[0]); dfB = leer_unison(ruta_B.iloc[0])

    # KPIs en dos columnas (A | B)
    st.subheader(f"KPIs — {ciclo_sel} (comparación de regiones)")
    colA, colB = st.columns(2)
    kA, kB = kpis_ext(dfA), kpis_ext(dfB)
    with colA:
        st.markdown(f"**{region_A}**")
        st.metric("Días del ciclo", f"{kA['dias']}")
        st.metric("% Azul", f"{(kA['eta_total']/kA['etc_total']*100):.1f}%" if kA['etc_total'] else "—")
        st.metric("ETc total [mm]", f"{kA['etc_total']:.1f}")
        st.metric("ETazul total [mm]", f"{kA['eta_total']:.1f}")
        st.metric("UAC verde [m³/ha]", f"{kA['uacv_ha']:.0f}" if not np.isnan(kA['uacv_ha']) else "—")
        st.metric("UAC azul [m³/ha]",  f"{kA['uaca_ha']:.0f}" if not np.isnan(kA['uaca_ha']) else "—")
    with colB:
        st.markdown(f"**{region_B}**")
        st.metric("Días del ciclo", f"{kB['dias']}")
        st.metric("% Azul", f"{(kB['eta_total']/kB['etc_total']*100):.1f}%" if kB['etc_total'] else "—")
        st.metric("ETc total [mm]", f"{kB['etc_total']:.1f}")
        st.metric("ETazul total [mm]", f"{kB['eta_total']:.1f}")
        st.metric("UAC verde [m³/ha]", f"{kB['uacv_ha']:.0f}" if not np.isnan(kB['uacv_ha']) else "—")
        st.metric("UAC azul [m³/ha]",  f"{kB['uaca_ha']:.0f}" if not np.isnan(kB['uaca_ha']) else "—")

    # Opciones comunes
    et_inter = [v for v in ["ET0","ETc","ETverde","ETazul","Pef"] if v in dfA.columns and v in dfB.columns]
    t_inter  = [v for v in ["Tmin","Tmean","Tmax"] if v in dfA.columns and v in dfB.columns]
    met_inter= [v for v in ["Rs","HR"] if v in dfA.columns and v in dfB.columns]

    hr()
    st.markdown("#### Opciones de series (aplican a ambas regiones)")
    et_sel = st.multiselect("Serie diaria (ET)", et_inter, default=et_inter, key="et_cmp_reg")
    t_sel  = st.multiselect("Temperaturas", t_inter,  default=t_inter,  key="t_cmp_reg")
    met_sel= st.multiselect("Meteorología", met_inter, default=met_inter, key="met_cmp_reg")

    # Bloque Región A
    hr()
    st.markdown(f"### {region_A} — {ciclo_sel}")
    st.pyplot(fig_series(dfA, f"Serie diaria (ET) — {ciclo_sel}", eje=eje_opt, mostrar=et_sel or et_inter), use_container_width=True)
    st.pyplot(fig_temperaturas(dfA, f"Temperaturas — {ciclo_sel}", eje=eje_opt, mostrar=t_sel or t_inter), use_container_width=True)
    st.pyplot(fig_meteo_rs_hr(dfA, f"Meteorología — {ciclo_sel}", eje=eje_opt, mostrar=met_sel or met_inter), use_container_width=True)
    if "Ux" in dfA.columns: st.pyplot(fig_wind(dfA, f"Viento Ux — {ciclo_sel}", eje=eje_opt), use_container_width=True)

    # Bloque Región B
    hr()
    st.markdown(f"### {region_B} — {ciclo_sel}")
    st.pyplot(fig_series(dfB, f"Serie diaria (ET) — {ciclo_sel}", eje=eje_opt, mostrar=et_sel or et_inter), use_container_width=True)
    st.pyplot(fig_temperaturas(dfB, f"Temperaturas — {ciclo_sel}", eje=eje_opt, mostrar=t_sel or t_inter), use_container_width=True)
    st.pyplot(fig_meteo_rs_hr(dfB, f"Meteorología — {ciclo_sel}", eje=eje_opt, mostrar=met_sel or met_inter), use_container_width=True)
    if "Ux" in dfB.columns: st.pyplot(fig_wind(dfB, f"Viento Ux — {ciclo_sel}", eje=eje_opt), use_container_width=True)
