import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    page_title="Dashboard Tren Suhu Global NOAA",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .main { background-color: #0f1117; }

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
        font-size: 32px;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        margin-top: 4px;
    }
    .metric-card .delta {
        font-size: 12px;
        margin-top: 2px;
        color: #8b92a5;
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
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("noaa_temperature_data.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


df_raw = load_data()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌡️ NOAA GSOD")
    st.markdown("**Analisis Tren Suhu Global 2022–2024**")
    st.divider()

    year_options = sorted(df_raw["date"].dt.year.unique().tolist())
    selected_years = st.multiselect(
        "Pilih Tahun", year_options, default=year_options
    )

    st.divider()
    st.markdown(
        '<p class="section-header">Forecasting</p>', unsafe_allow_html=True
    )
    n_forecast = st.slider("Periode prediksi ke depan (bulan)", 3, 24, 12)

    st.divider()
    st.caption("Data: NOAA GSOD | BigQuery Public Dataset")

# ─────────────────────────────────────────────
#  FILTER DATA
# ─────────────────────────────────────────────
df = df_raw[df_raw["date"].dt.year.isin(selected_years)].copy()
df_indexed = df.set_index("date")

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("# 🌡️ Dashboard Suhu Global")
st.markdown(
    "Analisis tren temperatur harian global menggunakan data **NOAA GSOD** "
    "dari BigQuery Public Dataset (2022–2024) dengan pemodelan **SARIMA**."
)
st.divider()

# ─────────────────────────────────────────────
#  METRIC CARDS
# ─────────────────────────────────────────────
avg_t = df["avg_temp"].mean()
max_t = df["avg_max_temp"].max()
min_t = df["avg_min_temp"].min()
total_days = len(df)
panas_pct = (df["avg_temp"] > 68).sum() / total_days * 100
dingin_pct = (df["avg_temp"] < 41).sum() / total_days * 100

col1, col2, col3, col4, col5 = st.columns(5)
cards = [
    (col1, "Rata-rata Suhu", f"{avg_t:.1f}°F", "avg_temp harian"),
    (col2, "Suhu Tertinggi", f"{max_t:.1f}°F", "avg_max_temp"),
    (col3, "Suhu Terendah",  f"{min_t:.1f}°F", "avg_min_temp"),
    (col4, "Hari Panas Ekstrem", f"{panas_pct:.1f}%", "> 68°F"),
    (col5, "Hari Dingin Ekstrem", f"{dingin_pct:.1f}%", "< 41°F"),
]
colors = ["#EF9F27", "#D85A30", "#378ADD", "#D85A30", "#378ADD"]

for (col, label, value, delta), color in zip(cards, colors):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="color:{color}">{value}</div>
                <div class="delta">{delta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TAB LAYOUT
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Tren & Distribusi", "📅 Pola Bulanan", "🔗 Korelasi", "🔮 Forecasting SARIMA"]
)

# ══════════════════════════════════════════════
#  TAB 1 — TREN & DISTRIBUSI
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Tren Temperatur Harian</p>', unsafe_allow_html=True)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df["date"], y=df["avg_max_temp"],
        name="Avg Max", line=dict(color="#D85A30", width=1, dash="dot"),
        fill=None,
    ))
    fig_trend.add_trace(go.Scatter(
        x=df["date"], y=df["avg_temp"],
        name="Avg Temp", line=dict(color="#EF9F27", width=2),
        fill="tonexty", fillcolor="rgba(239,159,39,0.08)",
    ))
    fig_trend.add_trace(go.Scatter(
        x=df["date"], y=df["avg_min_temp"],
        name="Avg Min", line=dict(color="#378ADD", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(55,138,221,0.06)",
    ))
    fig_trend.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Tanggal",
        yaxis_title="Suhu (°F)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Distribusi Temperatur Harian</p>', unsafe_allow_html=True)

    fig_hist = go.Figure()
    # Dingin ekstrem
    mask_cold = df["avg_temp"] < 41
    mask_hot  = df["avg_temp"] > 68
    mask_norm = ~(mask_cold | mask_hot)

    for mask, color, label in [
        (mask_cold, "#378ADD", "Dingin Ekstrem (< 41°F)"),
        (mask_norm, "#EF9F27", "Normal (41–68°F)"),
        (mask_hot,  "#D85A30", "Panas Ekstrem (> 68°F)"),
    ]:
        fig_hist.add_trace(go.Histogram(
            x=df.loc[mask, "avg_temp"],
            nbinsx=20,
            name=label,
            marker_color=color,
            opacity=0.85,
        ))

    fig_hist.add_vline(
        x=avg_t, line_dash="dash", line_color="white",
        annotation_text=f"Rata-rata: {avg_t:.1f}°F",
        annotation_position="top right",
    )
    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack",
        xaxis_title="Suhu Rata-rata Harian (°F)",
        yaxis_title="Jumlah Hari",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 — POLA BULANAN
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Pola Suhu Rata-rata per Bulan</p>', unsafe_allow_html=True)

    df["bulan"] = df["date"].dt.to_period("M").astype(str)
    df["bulan_num"] = df["date"].dt.month
    df["tahun"] = df["date"].dt.year

    monthly_avg = (
        df_indexed.resample("ME")[["avg_temp", "avg_max_temp", "avg_min_temp"]]
        .mean()
        .reset_index()
    )

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(
        x=monthly_avg["date"], y=monthly_avg["avg_max_temp"],
        name="Avg Max", line=dict(color="#D85A30", width=2),
    ))
    fig_monthly.add_trace(go.Scatter(
        x=monthly_avg["date"], y=monthly_avg["avg_temp"],
        name="Avg Temp", line=dict(color="#EF9F27", width=3),
    ))
    fig_monthly.add_trace(go.Scatter(
        x=monthly_avg["date"], y=monthly_avg["avg_min_temp"],
        name="Avg Min", line=dict(color="#378ADD", width=2),
    ))
    fig_monthly.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Bulan",
        yaxis_title="Suhu (°F)",
        hovermode="x unified",
        height=350,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.divider()

    # Cuaca Ekstrem per Bulan
    st.markdown('<p class="section-header">Distribusi Cuaca Ekstrem per Bulan (Semua Tahun)</p>', unsafe_allow_html=True)

    df["panas_ekstrem"] = df["avg_temp"] > 68
    df["dingin_ekstrem"] = df["avg_temp"] < 41
    df["normal"] = ~(df["panas_ekstrem"] | df["dingin_ekstrem"])

    extreme_monthly = df.groupby("bulan_num")[["panas_ekstrem", "dingin_ekstrem", "normal"]].sum()
    month_labels = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]

    fig_stacked = go.Figure()
    fig_stacked.add_trace(go.Bar(
        x=month_labels, y=extreme_monthly["dingin_ekstrem"],
        name="Dingin Ekstrem", marker_color="#378ADD",
    ))
    fig_stacked.add_trace(go.Bar(
        x=month_labels, y=extreme_monthly["normal"],
        name="Normal", marker_color="#EF9F27",
    ))
    fig_stacked.add_trace(go.Bar(
        x=month_labels, y=extreme_monthly["panas_ekstrem"],
        name="Panas Ekstrem", marker_color="#D85A30",
    ))
    fig_stacked.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack",
        xaxis_title="Bulan",
        yaxis_title="Jumlah Hari",
        height=350,
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    # Boxplot per bulan
    st.divider()
    st.markdown('<p class="section-header">Boxplot Suhu per Bulan</p>', unsafe_allow_html=True)
    df["bulan_label"] = df["date"].dt.strftime("%b %Y")

    fig_box = px.box(
        df, x="bulan", y="avg_temp",
        color_discrete_sequence=["#EF9F27"],
        labels={"bulan": "Bulan", "avg_temp": "Suhu Rata-rata (°F)"},
    )
    fig_box.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45,
        height=380,
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 — KORELASI
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Analisis Korelasi Antar Variabel Suhu</p>', unsafe_allow_html=True)

    corr = df[["avg_temp", "avg_max_temp", "avg_min_temp"]].corr()

    fig_heatmap = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.round(4).values,
        texttemplate="%{text}",
        textfont=dict(size=16),
        showscale=True,
    ))
    fig_heatmap.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        title="Matriks Korelasi Pearson",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("**Interpretasi:**")
    st.markdown(
        "- `avg_temp` vs `avg_max_temp`: **{:.4f}** — korelasi {}  \n"
        "- `avg_temp` vs `avg_min_temp`: **{:.4f}** — korelasi {}  \n"
        "- `avg_max_temp` vs `avg_min_temp`: **{:.4f}** — korelasi {}".format(
            corr.loc["avg_temp","avg_max_temp"],
            "sangat kuat" if abs(corr.loc["avg_temp","avg_max_temp"]) > 0.8 else "sedang",
            corr.loc["avg_temp","avg_min_temp"],
            "sangat kuat" if abs(corr.loc["avg_temp","avg_min_temp"]) > 0.8 else "sedang",
            corr.loc["avg_max_temp","avg_min_temp"],
            "sangat kuat" if abs(corr.loc["avg_max_temp","avg_min_temp"]) > 0.8 else "sedang",
        )
    )

    st.divider()
    st.markdown('<p class="section-header">Scatter Plot</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_sc1 = px.scatter(
            df, x="avg_temp", y="avg_max_temp",
            opacity=0.4, trendline="ols",
            color_discrete_sequence=["#D85A30"],
            labels={"avg_temp": "Avg Temp (°F)", "avg_max_temp": "Avg Max Temp (°F)"},
            title="avg_temp vs avg_max_temp",
        )
        fig_sc1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc1, use_container_width=True)

    with col_b:
        fig_sc2 = px.scatter(
            df, x="avg_temp", y="avg_min_temp",
            opacity=0.4, trendline="ols",
            color_discrete_sequence=["#378ADD"],
            labels={"avg_temp": "Avg Temp (°F)", "avg_min_temp": "Avg Min Temp (°F)"},
            title="avg_temp vs avg_min_temp",
        )
        fig_sc2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc2, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 — SARIMA FORECASTING
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Stationarity Check — ADF Test</p>', unsafe_allow_html=True)

    ts_full = df_raw.set_index("date")["avg_temp"]
    monthly_ts = ts_full.resample("ME").mean()

    adf_result = adfuller(monthly_ts)
    adf_diff   = adfuller(monthly_ts.diff().dropna())

    col_adf1, col_adf2 = st.columns(2)
    with col_adf1:
        st.metric("ADF Statistic (original)", f"{adf_result[0]:.4f}")
        st.metric("p-value (original)", f"{adf_result[1]:.6f}",
                  delta="Stasioner ✅" if adf_result[1] < 0.05 else "Tidak Stasioner ❌",
                  delta_color="normal")
    with col_adf2:
        st.metric("ADF Statistic (setelah differencing)", f"{adf_diff[0]:.4f}")
        st.metric("p-value (setelah differencing)", f"{adf_diff[1]:.6f}",
                  delta="Stasioner ✅" if adf_diff[1] < 0.05 else "Tidak Stasioner ❌",
                  delta_color="normal")

    st.divider()
    st.markdown('<p class="section-header">Train / Test Split & SARIMA Model</p>', unsafe_allow_html=True)

    train = monthly_ts[:"2023-12"]
    test  = monthly_ts["2024-01":]

    col_tt1, col_tt2 = st.columns(2)
    col_tt1.info(f"🟡 **Train size:** {len(train)} bulan (s.d. Des 2023)")
    col_tt2.info(f"🔵 **Test size:** {len(test)} bulan (Jan–Des 2024)")

    @st.cache_data
    def fit_sarima(train_values, n_periods_forecast):
        model = auto_arima(
            train_values,
            seasonal=True,
            m=12,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        forecast_test   = model.predict(n_periods=len(test))
        forecast_future = model.predict(n_periods=n_periods_forecast)
        return model, forecast_test, forecast_future

    with st.spinner("Melatih model SARIMA... ⏳"):
        model, forecast_test, forecast_future = fit_sarima(
            train.values, n_forecast
        )

    st.success(f"Model terpilih: **{model.order}** x **{model.seasonal_order}**")

    # ── Evaluasi ──
    st.divider()
    st.markdown('<p class="section-header">Evaluasi Model</p>', unsafe_allow_html=True)

    mae  = mean_absolute_error(test.values, forecast_test)
    rmse = np.sqrt(mean_squared_error(test.values, forecast_test))
    mape = mean_absolute_percentage_error(test.values, forecast_test) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",  f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("MAPE", f"{mape:.2f}%")

    # ── Plot Train/Test/Forecast ──
    st.divider()
    st.markdown('<p class="section-header">Visualisasi Forecast</p>', unsafe_allow_html=True)

    last_date    = monthly_ts.index[-1]
    future_dates = pd.date_range(
        last_date + pd.DateOffset(months=1), periods=n_forecast, freq="ME"
    )

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=train.index, y=train.values,
        name="Train", line=dict(color="#EF9F27", width=2),
    ))
    fig_fc.add_trace(go.Scatter(
        x=test.index, y=test.values,
        name="Aktual (Test)", line=dict(color="#ffffff", width=2),
    ))
    fig_fc.add_trace(go.Scatter(
        x=test.index, y=forecast_test,
        name="Forecast Test", line=dict(color="#D85A30", width=2, dash="dash"),
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=forecast_future,
        name=f"Forecast {n_forecast} Bulan ke Depan",
        line=dict(color="#4ade80", width=2, dash="dot"),
    ))
    fig_fc.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Bulan",
        yaxis_title="Suhu Rata-rata (°F)",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Tabel forecast ──
    st.divider()
    st.markdown('<p class="section-header">Tabel Nilai Forecast</p>', unsafe_allow_html=True)
    df_forecast_table = pd.DataFrame({
        "Bulan": future_dates.strftime("%B %Y"),
        "Prediksi Suhu (°F)": forecast_future.round(4),
    })
    st.dataframe(df_forecast_table, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "📡 Data: NOAA GSOD via BigQuery Public Dataset &nbsp;|&nbsp; "
    "🤖 Model: SARIMA (Auto-ARIMA, seasonal m=12) &nbsp;|&nbsp; "
    "🛠️ Stack: Streamlit · Plotly · pmdarima · statsmodels"
)
