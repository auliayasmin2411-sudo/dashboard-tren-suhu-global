import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Suhu Global NOAA",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }
    .metric-card {
        background: linear-gradient(135deg, #1c1f2b, #252836);
        border: 1px solid #2e3247;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-card .label {
        font-size: 12px;
        color: #8b92a5;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        margin-top: 4px;
    }
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 14px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #e85d35;
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("noaa_temperature_data.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_countries():
    try:
        return pd.read_csv("noaa_countries_2024.csv")
    except FileNotFoundError:
        return None

df_raw       = load_data()
df_countries = load_countries()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌡️ NOAA GSOD")
    st.markdown("**Analisis Suhu Global 2022–2024**")
    st.divider()
    year_options   = sorted(df_raw["date"].dt.year.unique().tolist())
    selected_years = st.multiselect("Pilih Tahun", year_options, default=year_options)
    st.divider()
    st.markdown('<p class="section-header">Forecasting</p>', unsafe_allow_html=True)
    n_forecast = st.slider("Periode prediksi ke depan (bulan)", 3, 24, 12)
    st.divider()
    st.caption("Data: NOAA GSOD | BigQuery Public Dataset")

# ─────────────────────────────────────────────
#  FILTER DATA
# ─────────────────────────────────────────────
df_clean   = df_raw[df_raw["date"].dt.year.isin(selected_years)].copy()
df_indexed = df_clean.set_index("date")[["avg_temp","avg_max_temp","avg_min_temp"]]

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("# 🌡️ Dashboard Tren Suhu Global")
st.markdown(
    "Analisis tren temperatur harian global menggunakan data **NOAA GSOD** "
    "dari BigQuery Public Dataset (2022–2024) dengan pemodelan **SARIMA**."
)
st.divider()

# ─────────────────────────────────────────────
#  METRIC CARDS  — "Bulan Terdingin" dihapus
# ─────────────────────────────────────────────
avg_t = df_clean["avg_temp"].mean()
max_t = df_clean["avg_max_temp"].max()

_monthly        = df_clean.copy()
_monthly["_periode"] = _monthly["date"].dt.to_period("M")
_monthly_mean   = _monthly.groupby("_periode")["avg_temp"].mean()
bulan_terpanas  = _monthly_mean.idxmax().strftime("%B %Y")

col1, col2, col3 = st.columns(3)
cards = [
    (col1, "Rata-rata Suhu", f"{avg_t:.1f}°F", "#EF9F27"),
    (col2, "Suhu Tertinggi", f"{max_t:.1f}°F", "#D85A30"),
    (col3, "Bulan Terpanas", bulan_terpanas,    "#D85A30"),
]

for col, label, value, color in cards:
    with col:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="color:{color}">{value}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Tren & Distribusi",
    "Pola Bulanan",
    "Top 10 Negara",
    "Korelasi",
    "Forecasting SARIMA",
])

# ══════════════════════════════════════════════
#  TAB 1 — TREN & DISTRIBUSI
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Tren Temperatur Harian</p>', unsafe_allow_html=True)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_clean["date"], y=df_clean["avg_max_temp"],
        name="Avg Max", line=dict(color="#D85A30", width=1, dash="dot"),
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_clean["date"], y=df_clean["avg_temp"],
        name="Avg Temp", line=dict(color="#EF9F27", width=2),
        fill="tonexty", fillcolor="rgba(239,159,39,0.08)",
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_clean["date"], y=df_clean["avg_min_temp"],
        name="Avg Min", line=dict(color="#378ADD", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(55,138,221,0.06)",
    ))
    fig_trend.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Tanggal", yaxis_title="Suhu (°F)", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Distribusi Temperatur Harian</p>', unsafe_allow_html=True)

    mask_cold = df_clean["avg_temp"] < 41
    mask_hot  = df_clean["avg_temp"] > 68
    mask_norm = ~(mask_cold | mask_hot)

    fig_hist = go.Figure()
    for mask, color, label in [
        (mask_cold, "#378ADD", "Dingin Ekstrem (< 41°F)"),
        (mask_norm, "#EF9F27", "Normal (41–68°F)"),
        (mask_hot,  "#D85A30", "Panas Ekstrem (> 68°F)"),
    ]:
        fig_hist.add_trace(go.Histogram(
            x=df_clean.loc[mask, "avg_temp"], nbinsx=20,
            name=label, marker_color=color, opacity=0.85,
        ))
    fig_hist.add_vline(
        x=avg_t, line_dash="dash", line_color="white",
        annotation_text=f"Rata-rata: {avg_t:.1f}°F", annotation_position="top right",
    )
    fig_hist.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack", xaxis_title="Suhu Rata-rata Harian (°F)", yaxis_title="Jumlah Hari",
        height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 — POLA BULANAN
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Pola Suhu Rata-rata per Bulan</p>', unsafe_allow_html=True)

    monthly_avg = df_indexed.resample("ME").mean().reset_index()

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(x=monthly_avg["date"], y=monthly_avg["avg_max_temp"], name="Avg Max", line=dict(color="#D85A30", width=2)))
    fig_monthly.add_trace(go.Scatter(x=monthly_avg["date"], y=monthly_avg["avg_temp"],     name="Avg Temp",line=dict(color="#EF9F27", width=3)))
    fig_monthly.add_trace(go.Scatter(x=monthly_avg["date"], y=monthly_avg["avg_min_temp"], name="Avg Min", line=dict(color="#378ADD", width=2)))
    fig_monthly.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Bulan", yaxis_title="Suhu (°F)", hovermode="x unified", height=350,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Distribusi Cuaca Ekstrem per Bulan</p>', unsafe_allow_html=True)

    df_tab2 = df_clean.copy()
    df_tab2["bulan_num"]      = df_tab2["date"].dt.month
    df_tab2["panas_ekstrem"]  = df_tab2["avg_temp"] > 68
    df_tab2["dingin_ekstrem"] = df_tab2["avg_temp"] < 41
    df_tab2["normal"]         = ~(df_tab2["panas_ekstrem"] | df_tab2["dingin_ekstrem"])
    extreme_monthly = df_tab2.groupby("bulan_num")[["panas_ekstrem","dingin_ekstrem","normal"]].sum()
    month_labels = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]

    fig_stacked = go.Figure()
    fig_stacked.add_trace(go.Bar(x=month_labels, y=extreme_monthly["dingin_ekstrem"], name="Dingin Ekstrem", marker_color="#378ADD"))
    fig_stacked.add_trace(go.Bar(x=month_labels, y=extreme_monthly["normal"],         name="Normal",        marker_color="#EF9F27"))
    fig_stacked.add_trace(go.Bar(x=month_labels, y=extreme_monthly["panas_ekstrem"],  name="Panas Ekstrem", marker_color="#D85A30"))
    fig_stacked.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack", xaxis_title="Bulan", yaxis_title="Jumlah Hari", height=350,
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Boxplot Suhu per Bulan</p>', unsafe_allow_html=True)
    df_tab2["bulan_label"] = df_tab2["date"].dt.to_period("M").astype(str)
    fig_box = px.box(
        df_tab2, x="bulan_label", y="avg_temp",
        color_discrete_sequence=["#EF9F27"],
        labels={"bulan_label": "Bulan", "avg_temp": "Suhu Rata-rata (°F)"},
    )
    fig_box.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45, height=380,
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 — TOP 10 NEGARA
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Top 10 Negara Terpanas & Terdingin (2024)</p>', unsafe_allow_html=True)

    if df_countries is None:
        st.warning("⚠️ File `noaa_countries_2024.csv` belum ditemukan. Silakan upload ke GitHub.")
        st.code(
            "df_countries.to_csv('noaa_countries_2024.csv', index=False)\n"
            "from google.colab import files\n"
            "files.download('noaa_countries_2024.csv')",
            language="python",
        )
    else:
        df_hot  = df_countries.head(10).copy()
        df_cold = df_countries.tail(10).copy().iloc[::-1]

        col_hot, col_cold = st.columns(2)
        with col_hot:
            st.markdown("#### 🔴 10 Negara Terpanas")
            fig_hot = go.Figure(go.Bar(
                x=df_hot["avg_temp"], y=df_hot["country_code"], orientation="h",
                marker_color="#D85A30",
                text=df_hot["avg_temp"].round(1).astype(str) + "°F",
                textposition="inside", insidetextfont=dict(color="white", size=11),
            ))
            fig_hot.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Rata-rata Suhu (°F)", yaxis=dict(autorange="reversed"),
                height=380, margin=dict(l=10,r=10,t=10,b=40), showlegend=False,
            )
            st.plotly_chart(fig_hot, use_container_width=True)

        with col_cold:
            st.markdown("#### 🔵 10 Negara Terdingin")
            fig_cold = go.Figure(go.Bar(
                x=df_cold["avg_temp"], y=df_cold["country_code"], orientation="h",
                marker_color="#378ADD",
                text=df_cold["avg_temp"].round(1).astype(str) + "°F",
                textposition="outside",
            ))
            fig_cold.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Rata-rata Suhu (°F)", yaxis=dict(autorange="reversed"),
                height=380, margin=dict(l=10,r=10,t=10,b=40), showlegend=False,
            )
            st.plotly_chart(fig_cold, use_container_width=True)

        st.divider()
        st.markdown('<p class="section-header">Semua Negara — Ranking Suhu</p>', unsafe_allow_html=True)
        fig_all = px.bar(
            df_countries, x="country_code", y="avg_temp", color="avg_temp",
            color_continuous_scale=["#378ADD","#EF9F27","#D85A30"],
            labels={"country_code":"Kode Negara","avg_temp":"Suhu Rata-rata (°F)"},
        )
        fig_all.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45, height=400, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_all, use_container_width=True)

        st.divider()
        st.markdown('<p class="section-header">Tabel Data Negara</p>', unsafe_allow_html=True)
        # background_gradient dihapus — membutuhkan matplotlib yang tidak tersedia
        st.dataframe(
            df_countries.rename(columns={"country_code": "Kode Negara", "avg_temp": "Suhu Rata-rata (°F)"}),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════
#  TAB 4 — KORELASI
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Analisis Korelasi Antar Variabel Suhu</p>', unsafe_allow_html=True)

    # Gunakan df_clean agar tidak ada kolom tambahan
    corr_cols = ["avg_temp", "avg_max_temp", "avg_min_temp"]
    corr = df_clean[corr_cols].corr()

    fig_heatmap = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 4),
        texttemplate="%{text}",
        textfont=dict(size=16),
        showscale=True,
    ))
    fig_heatmap.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, title="Matriks Korelasi Pearson",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_sc1 = px.scatter(
            df_clean, x="avg_temp", y="avg_max_temp", opacity=0.4, trendline="ols",
            color_discrete_sequence=["#D85A30"],
            labels={"avg_temp": "Avg Temp (°F)", "avg_max_temp": "Avg Max Temp (°F)"},
            title="avg_temp vs avg_max_temp",
        )
        fig_sc1.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sc1, use_container_width=True)

    with col_b:
        fig_sc2 = px.scatter(
            df_clean, x="avg_temp", y="avg_min_temp", opacity=0.4, trendline="ols",
            color_discrete_sequence=["#378ADD"],
            labels={"avg_temp": "Avg Temp (°F)", "avg_min_temp": "Avg Min Temp (°F)"},
            title="avg_temp vs avg_min_temp",
        )
        fig_sc2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sc2, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 5 — SARIMA FORECASTING
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">Stationarity Check — ADF Test</p>', unsafe_allow_html=True)

    # Selalu pakai df_raw agar data lengkap untuk SARIMA
    ts_full    = df_raw.set_index("date")["avg_temp"]
    monthly_ts = ts_full.resample("ME").mean()

    adf_orig = adfuller(monthly_ts)
    adf_diff = adfuller(monthly_ts.diff().dropna())

    col_adf1, col_adf2 = st.columns(2)
    with col_adf1:
        st.metric("ADF Statistic (original)", f"{adf_orig[0]:.4f}")
        st.metric(
            "p-value (original)", f"{adf_orig[1]:.6f}",
            delta="Stasioner ✅" if adf_orig[1] < 0.05 else "Tidak Stasioner ❌",
            delta_color="normal",
        )
    with col_adf2:
        st.metric("ADF Statistic (setelah differencing)", f"{adf_diff[0]:.4f}")
        st.metric(
            "p-value (setelah differencing)", f"{adf_diff[1]:.6f}",
            delta="Stasioner ✅" if adf_diff[1] < 0.05 else "Tidak Stasioner ❌",
            delta_color="normal",
        )

    st.divider()
    st.markdown('<p class="section-header">Train / Test Split & SARIMA Model</p>', unsafe_allow_html=True)

    train = monthly_ts[:"2023-12"]
    test  = monthly_ts["2024-01":]

    col_tt1, col_tt2 = st.columns(2)
    col_tt1.info(f"🟡 **Train size:** {len(train)} bulan (s.d. Des 2023)")
    col_tt2.info(f"🔵 **Test size:** {len(test)} bulan (Jan–Des 2024)")

    @st.cache_data
    def fit_sarima(train_vals, n_test, n_future):
        mdl = auto_arima(
            train_vals, seasonal=True, m=12, stepwise=True,
            suppress_warnings=True, error_action="ignore",
        )
        fc_test   = mdl.predict(n_periods=n_test)
        fc_future = mdl.predict(n_periods=n_future)
        return mdl, fc_test, fc_future

    with st.spinner("Melatih model SARIMA... ⏳"):
        model, fc_test, fc_future = fit_sarima(train.values, len(test), n_forecast)

    st.success(f"Model terpilih: **{model.order}** x **{model.seasonal_order}**")

    st.divider()
    st.markdown('<p class="section-header">Evaluasi Model</p>', unsafe_allow_html=True)

    mae  = mean_absolute_error(test.values, fc_test)
    rmse = np.sqrt(mean_squared_error(test.values, fc_test))
    mape = mean_absolute_percentage_error(test.values, fc_test) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",  f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("MAPE", f"{mape:.2f}%")

    st.divider()
    st.markdown('<p class="section-header">Visualisasi Forecast</p>', unsafe_allow_html=True)

    future_dates = pd.date_range(
        monthly_ts.index[-1] + pd.DateOffset(months=1), periods=n_forecast, freq="ME"
    )

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=train.index,  y=train.values, name="Train",         line=dict(color="#EF9F27", width=2)))
    fig_fc.add_trace(go.Scatter(x=test.index,   y=test.values,  name="Aktual (Test)", line=dict(color="#ffffff", width=2)))
    fig_fc.add_trace(go.Scatter(x=test.index,   y=fc_test,      name="Forecast Test", line=dict(color="#D85A30", width=2, dash="dash")))
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=fc_future,
        name=f"Forecast {n_forecast} Bulan ke Depan",
        line=dict(color="#4ade80", width=2, dash="dot"),
    ))
    fig_fc.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Bulan", yaxis_title="Suhu Rata-rata (°F)", hovermode="x unified",
        height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Tabel Nilai Forecast</p>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Bulan": future_dates.strftime("%B %Y"),
            "Prediksi Suhu (°F)": np.round(fc_future, 4),
        }),
        use_container_width=True,
        hide_index=True,
    )
