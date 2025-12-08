# app.py — Superstore Sales Dashboard + Time Series & Forecasting
# ---------------------------------------------------------------
# Columns expected:
# ['Row ID','Order ID','Order Date','Ship Date','Ship Mode','Customer ID','Customer Name',
#  'Segment','Country','City','State','Region','Product ID','Category','Sub-Category',
#  'Product Name','Sales','Quantity','Discount','Profit']

import io
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# Optional (installed with statsmodels)
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf as sm_acf, pacf as sm_pacf
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SM = True
except Exception:
    HAS_SM = False

# ---------------------------
# Page config & styles
# ---------------------------
st.set_page_config(page_title="Sales Dashboard + Forecasting", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
      .metric-row div[data-testid="stMetric"]{background:#f7f7fb;border-radius:14px;padding:10px;}
      .muted{color:#666;font-size:.9rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Loaders
# ---------------------------
@st.cache_data(show_spinner=False)
def read_any(upload, delimiter=",", sheet: Optional[str] = None) -> pd.DataFrame:
    name = getattr(upload, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload, sheet_name=sheet) if sheet else pd.read_excel(upload)
    return pd.read_csv(upload, sep=delimiter, encoding="ISO-8859-1")

@st.cache_data(show_spinner=False)
def try_default_file() -> Optional[pd.DataFrame]:
    for fname in ["superstore.csv", "sales_analysis.csv", "sales.csv"]:
        try:
            df = pd.read_csv(fname, encoding="ISO-8859-1")
            return df
        except Exception:
            continue
    return None

# ---------------------------
# Sidebar: upload + options
# ---------------------------
st.sidebar.title("⚙️ Controls")

with st.sidebar.expander("1) Upload data", expanded=True):
    up = st.file_uploader("CSV/Excel", type=["csv", "xlsx", "xls"])
    delimiter = st.selectbox("Delimiter (CSV)", [",", ";", "\t", "|"], index=0)
    sheet = st.text_input("Sheet name (Excel, optional)", "")

with st.sidebar.expander("2) Options", expanded=False):
    lower = st.toggle("Lowercase column names", value=False)
    drop_dupes = st.toggle("Drop duplicate rows", value=True)
    parse_dates = st.toggle("Parse dates", value=True)

# Read data
if up is not None:
    df = read_any(up, delimiter=delimiter, sheet=sheet or None)
else:
    df = try_default_file()
    if df is None:
        st.title("📈 Sales Analysis & Forecasting Dashboard")
        st.info("Upload a CSV/Excel from the sidebar, or place `superstore.csv` / `sales_analysis.csv` next to this file.")
        st.stop()

# Normalize columns if user wants
if lower:
    df.columns = [str(c).strip().lower() for c in df.columns]
    # map expected names to lowercase aliases
    COL = {
        "order_date": "order date", "ship_date": "ship date", "ship_mode": "ship mode",
        "region": "region", "state": "state", "city": "city", "segment": "segment",
        "category": "category", "sub_category": "sub-category", "sales": "sales",
        "profit": "profit", "order_id": "order id", "quantity": "quantity",
        "product_name": "product name", "customer_name": "customer name"
    }
else:
    COL = {
        "order_date": "Order Date", "ship_date": "Ship Date", "ship_mode": "Ship Mode",
        "region": "Region", "state": "State", "city": "City", "segment": "Segment",
        "category": "Category", "sub_category": "Sub-Category", "sales": "Sales",
        "profit": "Profit", "order_id": "Order ID", "quantity": "Quantity",
        "product_name": "Product Name", "customer_name": "Customer Name"
    }

# Basic cleaning
if drop_dupes:
    df = df.drop_duplicates().reset_index(drop=True)

# Date parsing
if parse_dates and COL["order_date"] in df.columns:
    df[COL["order_date"]] = pd.to_datetime(df[COL["order_date"]], errors="coerce")
if parse_dates and COL["ship_date"] in df.columns:
    df[COL["ship_date"]] = pd.to_datetime(df[COL["ship_date"]], errors="coerce")

# ---------------------------
# Global Filters
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

filtered = df.copy()

# Date filter
if COL["order_date"] in filtered.columns and np.issubdtype(filtered[COL["order_date"]].dtype, np.datetime64):
    mind, maxd = filtered[COL["order_date"]].min(), filtered[COL["order_date"]].max()
    if pd.notna(mind) and pd.notna(maxd):
        d1, d2 = st.sidebar.date_input("Order Date range", (mind.date(), maxd.date()))
        d1, d2 = pd.to_datetime(d1), pd.to_datetime(d2)
        filtered = filtered[(filtered[COL["order_date"]] >= d1) & (filtered[COL["order_date"]] <= d2)]

# Helper to add simple multiselects
def add_categorical_filter(label_key):
    colname = COL[label_key]
    if colname in filtered.columns:
        opts = sorted(filtered[colname].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect(colname, opts, default=opts)
        if sel and len(sel) < len(opts):
            return filtered[filtered[colname].astype(str).isin(sel)]
    return filtered

for key in ["region", "state", "segment", "category", "sub_category", "ship_mode"]:
    filtered = add_categorical_filter(key)

# ---------------------------
# KPIs
# ---------------------------
st.title("📊 Sales Overview")

sales_col = COL["sales"] if COL["sales"] in filtered.columns else None
profit_col = COL["profit"] if COL["profit"] in filtered.columns else None
order_col = COL["order_id"] if COL["order_id"] in filtered.columns else None
qty_col = COL["quantity"] if COL["quantity"] in filtered.columns else None

total_sales = float(filtered[sales_col].sum()) if sales_col else 0.0
total_profit = float(filtered[profit_col].sum()) if profit_col else 0.0
orders = int(filtered[order_col].nunique()) if order_col else len(filtered)
qty = int(filtered[qty_col].sum()) if qty_col else 0
aov = (total_sales / orders) if orders else 0.0

st.markdown('<div class="metric-row">', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("💰 Total Sales", f"₹{total_sales:,.2f}")
k2.metric("📈 Total Profit", f"₹{total_profit:,.2f}", delta_color="normal" if total_profit >= 0 else "inverse")
k3.metric("🧾 Orders", f"{orders:,}")
k4.metric("📦 Units Sold", f"{qty:,}")
st.markdown("</div>", unsafe_allow_html=True)
st.caption("AOV (Avg Order Value): ₹{:,.2f}".format(aov))

st.markdown("### Data Preview")
st.dataframe(filtered.head(50), use_container_width=True)
st.download_button("⬇️ Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_sales.csv", mime="text/csv")

st.markdown("---")

# ---------------------------
# Descriptive Visuals
# ---------------------------
def vc_bar(df_in: pd.DataFrame, col: str, title: str):
    # value_counts -> unique column names (avoids 'count' duplicate)
    vc = df_in[col].value_counts(dropna=False).reset_index()
    vc.columns = [col, "Count"]
    fig = px.bar(vc, x=col, y="Count", text="Count", title=title)
    fig.update_layout(xaxis_title=col, yaxis_title="Count")
    return fig

# Sales trend
if COL["order_date"] in filtered.columns and sales_col:
    st.subheader("📅 Sales Trend Over Time")
    trend = (filtered.groupby(COL["order_date"])[sales_col]
             .sum().reset_index().sort_values(COL["order_date"]))
    st.plotly_chart(px.line(trend, x=COL["order_date"], y=sales_col, title="Sales Over Time"),
                    use_container_width=True)

# Region Sales
if COL["region"] in filtered.columns and sales_col:
    st.subheader("🌍 Sales by Region")
    reg = filtered.groupby(COL["region"])[sales_col].sum().reset_index()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(reg, x=COL["region"], y=sales_col, text=sales_col, title="Total Sales by Region"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(reg, names=COL["region"], values=sales_col, title="Sales Contribution by Region"),
                        use_container_width=True)

# State Sales (Top 15)
if COL["state"] in filtered.columns and sales_col:
    st.subheader("🗺️ Top 15 States by Sales")
    st_df = (filtered.groupby(COL["state"])[sales_col]
             .sum().sort_values(ascending=False).head(15).reset_index())
    st.plotly_chart(px.bar(st_df, x=sales_col, y=COL["state"], orientation="h", text=sales_col),
                    use_container_width=True)

# Category & Sub-Category
if COL["category"] in filtered.columns and COL["sub_category"] in filtered.columns and sales_col:
    st.subheader("📦 Category & Sub-Category Performance")
    cat = filtered.groupby([COL["category"], COL["sub_category"]])[sales_col].sum().reset_index()
    st.plotly_chart(px.treemap(cat, path=[COL["category"], COL["sub_category"]], values=sales_col),
                    use_container_width=True)

# Ship Mode analysis (no 'count' duplication)
if COL["ship_mode"] in filtered.columns:
    st.subheader("🚚 Ship Mode — Usage & Sales")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(vc_bar(filtered, COL["ship_mode"], "Orders by Ship Mode"), use_container_width=True)
    if sales_col:
        sm_sales = filtered.groupby(COL["ship_mode"])[sales_col].sum().reset_index()
        with c2:
            st.plotly_chart(px.pie(sm_sales, names=COL["ship_mode"], values=sales_col, title="Sales by Ship Mode"),
                            use_container_width=True)

# Segment
if COL["segment"] in filtered.columns and sales_col:
    st.subheader("👥 Segment-wise Sales")
    seg = filtered.groupby(COL["segment"])[sales_col].sum().reset_index()
    st.plotly_chart(px.pie(seg, names=COL["segment"], values=sales_col, hole=.35), use_container_width=True)

# Top Products
if COL["product_name"] in filtered.columns and sales_col:
    st.subheader("🏆 Top 10 Products by Sales")
    top_products = (filtered.groupby(COL["product_name"])[sales_col]
                    .sum().reset_index().sort_values(sales_col, ascending=False).head(10))
    st.plotly_chart(px.bar(top_products, x=sales_col, y=COL["product_name"], orientation="h", text=sales_col),
                    use_container_width=True)

st.markdown("---")

# ---------------------------
# Time Series Analysis & Forecasting
# ---------------------------
st.header("⏱️ Time Series & Forecasting")

if COL["order_date"] in filtered.columns and sales_col and np.issubdtype(filtered[COL["order_date"]].dtype, np.datetime64):

    freq = st.selectbox("Resample frequency", ["D", "W", "M", "Q", "Y"], index=2)
    ma_win = st.slider("Moving Average window", 1, 24, 3)

    ts = (filtered[[COL["order_date"], sales_col]]
          .dropna()
          .sort_values(COL["order_date"])
          .set_index(COL["order_date"])[sales_col]
          .resample(freq).sum()
          .to_frame("sales"))

    ts[f"MA_{ma_win}"] = ts["sales"].rolling(ma_win, min_periods=1).mean()
    st.plotly_chart(px.line(ts.reset_index(), x=COL["order_date"], y=["sales", f"MA_{ma_win}"],
                            title=f"Resampled ({freq}) Sales with MA({ma_win})"), use_container_width=True)

    # Decomposition (monthly preferred)
    if HAS_SM:
        st.subheader("🔎 Seasonal Decomposition (Monthly)")
        ts_m = (filtered.set_index(COL["order_date"])[sales_col]
                .resample("M").sum().asfreq("M"))
        if ts_m.dropna().shape[0] >= 24:
            model_type = st.selectbox("Decompose model", ["additive", "multiplicative"], index=0)
            decomp = seasonal_decompose(ts_m, model=model_type, period=12, extrapolate_trend='freq')
            dfig = go.Figure()
            dfig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name="Trend"))
            dfig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name="Seasonal"))
            dfig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, name="Residual"))
            dfig.update_layout(title="Decomposition (M=12)", legend_orientation="h")
            st.plotly_chart(dfig, use_container_width=True)
        else:
            st.info("Decomposition ke liye ~24 monthly points chahiye.")

        # Stationarity (ADF)
        st.subheader("🧪 ADF Stationarity Test (Monthly Sales)")
        try:
            ts_adf = ts_m.dropna()
            adf_stat, pval, *_ = adfuller(ts_adf.values)
            st.write({"ADF Statistic": round(adf_stat, 4), "p-value": round(pval, 6)})
        except Exception as e:
            st.warning(f"ADF failed: {e}")

        # ACF / PACF
        st.subheader("📊 ACF & PACF")
        try:
            nlags = st.slider("Lags", 10, 60, 24)
            acf_vals = sm_acf(ts_m.dropna(), nlags=nlags)
            pacf_vals = sm_pacf(ts_m.dropna(), nlags=nlags, method="ywm")
            acf_df = pd.DataFrame({"lag": range(len(acf_vals)), "acf": acf_vals})
            pacf_df = pd.DataFrame({"lag": range(len(pacf_vals)), "pacf": pacf_vals})
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.bar(acf_df, x="lag", y="acf", title="ACF"), use_container_width=True)
            with c2:
                st.plotly_chart(px.bar(pacf_df, x="lag", y="pacf", title="PACF"), use_container_width=True)
        except Exception as e:
            st.warning(f"ACF/PACF failed: {e}")

        # ------- Forecasting -------
        st.subheader("🔮 Forecasting (ARIMA / SARIMA)")
        horizon = st.slider("Forecast horizon (periods)", 3, 24, 12, help="Monthly periods for forecast")
        model_kind = st.selectbox("Model", ["ARIMA", "SARIMA"], index=1)
        c1, c2, c3 = st.columns(3)
        with c1:
            p = st.number_input("p", 0, 5, 1)
        with c2:
            d = st.number_input("d", 0, 2, 1)
        with c3:
            q = st.number_input("q", 0, 5, 1)
        P = D = Q = s = 0
        if model_kind == "SARIMA":
            c1, c2, c3, c4 = st.columns(4)
            with c1: P = st.number_input("P (seasonal)", 0, 5, 1)
            with c2: D = st.number_input("D (seasonal)", 0, 2, 1)
            with c3: Q = st.number_input("Q (seasonal)", 0, 5, 1)
            with c4: s = st.number_input("s (seasonal period)", 0, 24, 12)

        if st.button("Run Forecast", use_container_width=True):
            try:
                y = ts_m.dropna().astype(float)
                if y.empty:
                    st.error("Not enough data to fit model.")
                else:
                    if model_kind == "ARIMA":
                        model = SARIMAX(y, order=(int(p), int(d), int(q)),
                                        enforce_stationarity=False, enforce_invertibility=False)
                    else:
                        model = SARIMAX(y, order=(int(p), int(d), int(q)),
                                        seasonal_order=(int(P), int(D), int(Q), int(s)),
                                        enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    pred = res.get_forecast(steps=int(horizon))
                    pred_mean = pred.predicted_mean
                    ci = pred.conf_int()

                    # future index (monthly)
                    fidx = pd.date_range(start=y.index[-1] + pd.offsets.MonthEnd(1),
                                         periods=int(horizon), freq="M")
                    figf = go.Figure()
                    figf.add_trace(go.Scatter(x=y.index, y=y.values, name="Actual"))
                    figf.add_trace(go.Scatter(x=fidx, y=pred_mean.values, name="Forecast"))
                    if ci is not None and ci.shape[1] == 2:
                        figf.add_trace(go.Scatter(x=fidx, y=ci.iloc[:, 0].values, name="Lower CI",
                                                  line=dict(dash="dash")))
                        figf.add_trace(go.Scatter(x=fidx, y=ci.iloc[:, 1].values, name="Upper CI",
                                                  line=dict(dash="dash")))
                    st.plotly_chart(figf, use_container_width=True)

                    out = pd.DataFrame({"date": fidx, "forecast": pred_mean.values})
                    st.download_button("⬇️ Download forecast CSV",
                                       out.to_csv(index=False).encode("utf-8"),
                                       file_name="forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Forecast failed: {e}")
    else:
        st.info("statsmodels install nahi hai — `pip install statsmodels` karke forecasting/univariate tools enable kar sakte ho.")
else:
    st.info("Time series features ke liye valid 'Order Date' (datetime) aur 'Sales' columns required hain.")

st.success("✅ Dashboard ready. Run:  streamlit run app.py")
