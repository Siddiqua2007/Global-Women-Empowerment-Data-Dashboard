import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.lines import Line2D

# ============================
# PAGE CONFIG & STYLING
# ============================
st.set_page_config(page_title="Women Empowerment Dashboard", layout="wide")

# Optional: Load Custom CSS
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ============================
# FUNCTION: FEMALE POPULATION MAP
# ============================
def female_population_map():
    st.markdown("## 👩‍🗺 Female Population Percentage by Country (2024)")

    # --- Load Dataset ---
    df = pd.read_csv(r"C:\Users\admin\Desktop\Pyhton\Female_population.csv")

    val_col = "Population, female (% of total population)"

    # --- Create Choropleth ---
    fig = px.choropleth(
        df,
        locations="Economy Code",
        color=val_col,
        hover_name="Economy",
        hover_data={val_col: ':.3f'},
        color_continuous_scale="RdPu",
        range_color=[47.5, 52.5],
        color_continuous_midpoint=50,
        title="Female Population % by Country (2024)",
        projection="natural earth"
    )

    # --- Customize Layout ---
    fig.update_layout(
        height=600,
        width=1200,
        geo=dict(
            showframe=True,
            framecolor="black",
            showcoastlines=True,
            coastlinecolor="gray",
            showland=True,
            landcolor="white",
            bgcolor="white"
        ),
        coloraxis_colorbar=dict(
            title="Female Population (%)",
            ticksuffix="%",
            tickvals=[48, 49, 50, 51, 52],
            ticktext=["48%", "49%", "50%", "51%", "52%"]
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Source: World Bank Data | Visualization: Women Empowerment Dashboard")

# ============================
# FUNCTION: Literacy vs Employment
# ============================
def literacy_vs_employment():
    st.markdown("## 📊 Literacy vs Employment Analysis")

    # --- Load Data ---
    literacy = pd.read_csv(r"C:\Users\admin\Desktop\Pyhton\women_literacy_rate.csv", skiprows=4)
    employment = pd.read_csv(r"C:\Users\admin\Desktop\Pyhton\Employement_rate_data.csv", skiprows=4)

    # --- Transform Data ---
    def reshape(df):
        df_long = df.melt(
            id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
            var_name="Year",
            value_name="Value"
        )
        df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
        df_long = df_long.dropna(subset=["Year", "Value"])
        return df_long

    lit_long = reshape(literacy)
    emp_long = reshape(employment)

    lit_long = lit_long.rename(columns={"Value": "Literacy"})
    emp_long = emp_long.rename(columns={"Value": "Employment"})

    # --- Merge Datasets ---
    df = pd.merge(
        lit_long[["Country Name", "Country Code", "Year", "Literacy"]],
        emp_long[["Country Name", "Country Code", "Year", "Employment"]],
        on=["Country Name", "Country Code", "Year"],
        how="inner"
    )

    df = df[df["Year"] >= 2015]

    # --- Country Groups ---
    south_asia = ["India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Bhutan", "Afghanistan", "Maldives"]
    brics = ["India", "Brazil", "Russia", "China", "South Africa"]
    nordic = ["Norway", "Sweden", "Finland", "Iceland", "Denmark"]

    df_world = df.groupby("Year")[["Literacy", "Employment"]].mean().reset_index()
    df_world["Country Name"] = "World Average"

    country = "India"
    df_country = df[df["Country Name"] == country]

    df_sa = df[df["Country Name"].isin(south_asia)]
    df_brics = df[df["Country Name"].isin(brics)]
    df_nordic = df[df["Country Name"].isin(nordic)]

    # ============================
    # PLOTS
    # ============================

    # 1️⃣ India Trend — Literacy vs Employment
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(df_country["Year"], df_country["Literacy"], marker="o", label="India Literacy", linewidth=1.6)
    ax.plot(df_country["Year"], df_country["Employment"], marker="s", label="India Employment", linewidth=1.6)
    ax.set_title("India — Female Literacy & Employment Trend", fontsize=9)
    ax.set_xlabel("Year", fontsize=7)
    ax.set_ylabel("Percentage", fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)

    # Helper function for group comparison
    def plot_group_dual(df_group, title):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Literacy
        for c in df_group["Country Name"].unique():
            d = df_group[df_group["Country Name"] == c]
            axes[0].plot(d["Year"], d["Literacy"], marker="o", label=f"{c}")
        axes[0].plot(df_country["Year"], df_country["Literacy"], marker="s", linewidth=3, color="black", label="India (Highlight)")
        axes[0].set_title(f"Female Literacy — {title}")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("%")
        axes[0].legend(fontsize=8)
        axes[0].grid(True)

        # Employment
        for c in df_group["Country Name"].unique():
            d = df_group[df_group["Country Name"] == c]
            axes[1].plot(d["Year"], d["Employment"], marker="o", label=f"{c}")
        axes[1].plot(df_country["Year"], df_country["Employment"], marker="s", linewidth=3, color="black", label="India (Highlight)")
        axes[1].set_title(f"Female Employment — {title}")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("%")
        axes[1].legend(fontsize=8)
        axes[1].grid(True)

        st.pyplot(fig)

    # Regional plots
    st.markdown("### 🌏 India vs Other South Asian Countries")
    plot_group_dual(df_sa, "South Asia")

    st.markdown("### 🌐 India vs BRICS Countries")
    plot_group_dual(df_brics, "BRICS")

    st.markdown("### ❄ India vs Nordic Countries")
    plot_group_dual(df_nordic, "Nordic")

    # World average comparison
    st.markdown("### 🌍 India vs World Average")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_country["Year"], df_country["Literacy"], marker="o", label="India")
    axes[0].plot(df_world["Year"], df_world["Literacy"], marker="s", label="World Avg")
    axes[0].set_title("Female Literacy — India vs World Average")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("%")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df_country["Year"], df_country["Employment"], marker="o", label="India")
    axes[1].plot(df_world["Year"], df_world["Employment"], marker="s", label="World Avg")
    axes[1].set_title("Female Employment — India vs World Average")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("%")
    axes[1].legend()
    axes[1].grid(True)

    st.pyplot(fig)

# ============================
# FUNCTION: Employment & Safety Indices (Global + India highlight)
# ============================
def employment_and_safety_indices():
    st.markdown("## 🛡 Employment & Safety Indices")

    # ---- File paths (edit if needed)
    EMPLOY_PATH = r"C:\Users\admin\Desktop\Pyhton\Employement_rate_data.csv"
    EMPOWER_PATH = r"C:\Users\admin\Desktop\Pyhton\women_empowerement_index_data.csv"
    SAFETY_PATH  = r"C:\Users\admin\Desktop\Pyhton\safety_index.csv"

    try:
        # --- Employment (World Bank wide format, skip 4 metadata rows)
        emp = pd.read_csv(EMPLOY_PATH, skiprows=4)
        emp = emp[emp["Indicator Code"].eq("SL.TLF.CACT.FE.ZS")].copy()

        year_cols = sorted([c for c in emp.columns if c.isdigit()], key=int)

        def latest_non_null(row):
            vals = row[year_cols]
            not_null = vals.dropna()
            return not_null.iloc[-1] if len(not_null) else np.nan

        emp["EmploymentRate_Female"] = emp.apply(latest_non_null, axis=1)
        emp_slim = emp.rename(columns={"Country Name": "Country"})[["Country", "EmploymentRate_Female"]]

        # --- Empowerment (WEI 2022 + Region)
        empower = pd.read_csv(EMPOWER_PATH).rename(columns={
            "Women's Empowerment Index (WEI) - 2022": "WEI_2022",
            "Sustainable Development Goal regions": "Region"
        })[["Country", "WEI_2022", "Region"]]

        # --- Safety (2024 only)
        safety = pd.read_csv(SAFETY_PATH)
        safety_2024 = safety[safety["year"].eq(2024)].rename(columns={
            "country": "Country",
            "score": "Safety_2024"
        })[["Country", "Safety_2024"]]

        # --- Country name fixes (if needed)
        country_fix = {
            # "United States of America": "United States",
            # "Russian Federation": "Russia",
        }
        emp_slim["Country"]    = emp_slim["Country"].replace(country_fix)
        empower["Country"]     = empower["Country"].replace(country_fix)
        safety_2024["Country"] = safety_2024["Country"].replace(country_fix)

        # --- Merge
        df = emp_slim.merge(empower, on="Country", how="inner") \
                     .merge(safety_2024, on="Country", how="inner")
        df = df.dropna(subset=["EmploymentRate_Female", "WEI_2022", "Safety_2024"]).reset_index(drop=True)

        if df.empty:
            st.warning("No rows after merging the datasets. Please check file paths and column names.")
            return

        # --- Common prep
        size_factor = 8
        df["bubble_size"] = df["EmploymentRate_Female"] * size_factor

        regions = df["Region"].fillna("Unknown")
        unique_regions = regions.unique()
        region_to_num = {r: i for i, r in enumerate(unique_regions)}
        df["color_num"] = df["Region"].fillna("Unknown").map(region_to_num)

        # Axis limits
        x_min, x_max = float(df["WEI_2022"].min()), float(df["WEI_2022"].max())
        y_min, y_max = float(df["Safety_2024"].min()), float(df["Safety_2024"].max())
        pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 0.05
        pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 0.05

        # ===== Chart 1: Global (all countries)
        fig1, ax1 = plt.subplots(figsize=(9, 5))
        sc = ax1.scatter(
            df["WEI_2022"], df["Safety_2024"],
            s=df["bubble_size"], c=df["color_num"],
            alpha=0.65, edgecolors="white", linewidths=0.7
        )
        ax1.set_xlabel("Women's Empowerment Index (2022)")
        ax1.set_ylabel("Women's Safety Index (2024)")
        ax1.set_title("Women’s Empowerment vs Safety • Bubble size = Female Employment Rate")
        ax1.set_xlim(x_min - pad_x, x_max + pad_x)
        ax1.set_ylim(y_min - pad_y, y_max + pad_y)
        ax1.grid(True, linestyle="--", alpha=0.3)

        handles = [Line2D([0],[0], marker='o', linestyle='',
                          markersize=8,
                          markerfacecolor=sc.cmap(region_to_num[r]/max(1, len(unique_regions)-1)),
                          markeredgecolor="white", markeredgewidth=0.7,
                          label=r) for r in unique_regions]
        ax1.legend(handles=handles, title="Region", loc="lower right", frameon=True)

        st.pyplot(fig1)

        # ===== Chart 2: Global with India highlighted
        india = df[df["Country"].str.strip().eq("India")].copy()
        others = df[~df.index.isin(india.index)].copy()

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        sc2 = ax2.scatter(
            others["WEI_2022"], others["Safety_2024"],
            s=others["bubble_size"], c=others["color_num"],
            alpha=0.55, edgecolors="white", linewidths=0.7
        )
        if not india.empty:
            ax2.scatter(
                india["WEI_2022"], india["Safety_2024"],
                s=india["bubble_size"] * 1.6,
                facecolors="none", edgecolors="black", linewidths=2.0, label="India"
            )
            for _, r in india.iterrows():
                ax2.annotate("India", (r["WEI_2022"], r["Safety_2024"]),
                             xytext=(8,8), textcoords="offset points", fontsize=9, weight="bold")

        ax2.set_xlabel("Women's Empowerment Index (2022)")
        ax2.set_ylabel("Women's Safety Index (2024)")
        ax2.set_title("Global • Empowerment vs Safety (India Highlighted)")
        ax2.set_xlim(x_min - pad_x, x_max + pad_x)
        ax2.set_ylim(y_min - pad_y, y_max + pad_y)
        ax2.grid(True, linestyle="--", alpha=0.3)

        handles2 = [Line2D([0],[0], marker='o', linestyle='',
                           markersize=8,
                           markerfacecolor=sc2.cmap(region_to_num[r]/max(1, len(unique_regions)-1)),
                           markeredgecolor="white", markeredgewidth=0.7,
                           label=r) for r in unique_regions]
        if not india.empty:
            handles2.append(Line2D([0],[0], marker='o', linestyle='',
                                   markersize=8, markerfacecolor="none",
                                   markeredgecolor="black", markeredgewidth=2.0,
                                   label="India"))
        ax2.legend(handles=handles2, title="Region / Highlight", loc="lower right", frameon=True)

        st.pyplot(fig2)

        # Optional mini table for India stats
        if not india.empty:
            st.markdown("#### 🇮🇳 India — quick stats")
            ist = india[["Country", "WEI_2022", "Safety_2024", "EmploymentRate_Female"]].copy()
            ist.rename(columns={
                "WEI_2022": "WEI (2022)",
                "Safety_2024": "Safety (2024)",
                "EmploymentRate_Female": "Female Employment (%)"
            }, inplace=True)
            st.dataframe(ist, use_container_width=True)

    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except KeyError as e:
        st.error(f"Missing expected column: {e}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# ============================
# SIDEBAR NAVIGATION
# ============================
with st.sidebar:
    st.markdown("## Women Empowerment Data Dashboard")

    selected = option_menu(
    menu_title=None,
    options=[
        "Dashboard",
        "Female Population",
        "Literacy vs Employement",
        "Employement and Safety Indices",
        "ML Prediction"
    ],
        icons=["bar-chart", "globe", "book", "shield-lock"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#0C0F1A"},
            "icon": {"color": "#BD00FF"},
            "nav-link": {"color": "#CFCFCF", "font-size": "15px"},
            "nav-link-selected": {"background-color": "#3B0E70"},
        }
    )

def dashboard_overview():
    st.markdown("## 🌍 Women Empowerment Data Dashboard — Overview")
    st.markdown(
        """
        This dashboard unites four key pillars of women's progress — 
        *Literacy, **Employment, **Empowerment (WEI 2022), and **Safety Index (2024)* — 
        offering a clear, data-driven view of how education, opportunity, and protection 
        combine to shape equality worldwide.
        """
    )

    # --- Styles (cleaner, taller cards)
    st.markdown("""
    <style>
      .cards {display:grid;grid-template-columns:repeat(4, minmax(200px,1fr));gap:18px;margin-top:15px;}
      @media (max-width:1100px){.cards{grid-template-columns:repeat(2, minmax(200px,1fr));}}
      @media (max-width:700px){.cards{grid-template-columns:1fr;}}
      .card{
        background:#111827; border:1px solid #2a2f3a; border-radius:20px;
        padding:22px 22px; height:190px;
        transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
      }
      .card:hover{ transform: translateY(-4px) scale(1.015); border-color:#6d28d9; box-shadow:0 10px 18px rgba(0,0,0,.25);}
      .card:active{ transform: scale(.98); }
      .kpi-title{ font-size:15px; color:#c7c7c7; margin-bottom:8px; }
      .kpi-value{ font-size:30px; font-weight:700; line-height:1.1; color:#f5f5f5;}
      .kpi-sub{ font-size:13px; color:#9aa0a6; margin-top:10px;}
    </style>
    """, unsafe_allow_html=True)

    # --- KPI placeholders
    kpis = {
        "LIT": {"title":"Literacy", "value":"—", "sub":"Latest female literacy (World avg)"},
        "EMP": {"title":"Employment", "value":"—", "sub":"Latest female employment rate (World avg)"},
        "WEI": {"title":"Empowerment", "value":"—", "sub":"WEI 2022 (World avg)"},
        "SAFE":{"title":"Safety", "value":"—", "sub":"Safety Index 2024 (World avg)"}
    }

    details_text = {
        "LIT": "Higher literacy accelerates agency, income, and health outcomes across generations.",
        "EMP": "Employment reflects access to jobs, fair conditions, and supportive policy environments.",
        "WEI": "WEI captures multidimensional empowerment — from decision-making power to resources.",
        "SAFE":"Safety captures mobility freedom and violence risk — foundational for any progress."
    }

    # --- Card display
    st.markdown('<div class="cards">', unsafe_allow_html=True)
    cols = st.columns(4)
    keys = ["LIT","EMP","WEI","SAFE"]
    emojis = {"LIT":"📚", "EMP":"💼", "WEI":"🟣", "SAFE":"🛡"}

    clicked_card = None  # track which one was clicked

    for i, k in enumerate(keys):
        with cols[i]:
            clicked = st.button(
                f"{emojis[k]} {kpis[k]['title']}",
                key=f"btn_{k}",
                use_container_width=True
            )
            st.markdown(
                f"""
                <div class="card">
                  <div class="kpi-title">{kpis[k]['title']}</div>
                  <div class="kpi-value">{kpis[k]['value']}</div>
                  <div class="kpi-sub">{kpis[k]['sub']}</div>
                </div>
                """, unsafe_allow_html=True
            )
            if clicked:
                clicked_card = k

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Show detail section when a card is clicked
    if clicked_card:
        st.markdown("---")
        st.markdown(f"### 🔍 {kpis[clicked_card]['title']} — Parameter")
        st.info(f"{details_text[clicked_card]}")
        

    # ============================
# FUNCTION: Employment Prediction (ML)
# ============================
EMPLOY_PATH = r"C:\Users\admin\Desktop\Pyhton\Employement_rate_data.csv"
LIT_PATH = r"C:\Users\admin\Desktop\Pyhton\women_literacy_rate.csv"
POP_PATH = r"C:\Users\admin\Desktop\Pyhton\Female_population.csv"
EMPOWER_PATH = r"C:\Users\admin\Desktop\Pyhton\women_empowerement_index_data.csv"
SAFETY_PATH = r"C:\Users\admin\Desktop\Pyhton\safety_index.csv"

import joblib

def employment_prediction_ml():

    st.markdown("## 🔮 Machine Learning Prediction — Female Employment Rate")

    # ---------------------
    # Load datasets
    # ---------------------
    emp = pd.read_csv(EMPLOY_PATH, skiprows=4)
    lit = pd.read_csv(LIT_PATH, skiprows=4)
    pop = pd.read_csv(POP_PATH)
    wei = pd.read_csv(EMPOWER_PATH)
    safety = pd.read_csv(SAFETY_PATH)

    # ---------------------
    # Clean & prepare data
    # ---------------------
    # Fix column names from empowerment dataset
    wei = wei.rename(columns={
        "Women's Empowerment Index (WEI) - 2022": "WEI_2022",
        "Sustainable Development Goal regions": "Region"
    })

    # Fix safety dataset
    safety_2024 = safety[safety["year"] == 2024].rename(columns={
        "country": "Country",
        "score": "Safety_2024"
    })

    # Select only female employment indicator
    emp = emp[emp["Indicator Code"] == "SL.TLF.CACT.FE.ZS"]

    # Extract last valid employment value per country
    year_cols = [c for c in emp.columns if c.isdigit()]

    def latest_valid(row):
        vals = row[year_cols].dropna()
        return vals.iloc[-1] if len(vals) else None

    emp["Employment"] = emp.apply(latest_valid, axis=1)
    emp = emp.rename(columns={"Country Name": "Country"})[["Country", "Employment"]]

    # Literacy reshape
    lit = lit.rename(columns={"Country Name": "Country"})
    lit_cols = [c for c in lit.columns if c.isdigit()]
    lit["Literacy"] = lit[lit_cols].ffill(axis=1).iloc[:, -1]

    lit_slim = lit[["Country", "Literacy"]]

    # Population female %
    pop = pop.rename(columns={
        "Economy": "Country",
        "Population, female (% of total population)": "Population_female_pct"
    })[["Country", "Population_female_pct"]]

    # Empowerment (WEI)
    wei = wei[["Country", "WEI_2022"]]

    # Safety (2024)
    safety_slim = safety_2024[["Country", "Safety_2024"]]

    # ---------------------
    # Merge all indicators
    # ---------------------
    df = emp.merge(lit_slim, on="Country", how="inner") \
            .merge(pop, on="Country", how="left") \
            .merge(wei, on="Country", how="left") \
            .merge(safety_slim, on="Country", how="left")

    df = df.dropna()

    # ---------------------
    # User selects country
    # ---------------------
    st.markdown("### 🌍 Select a Country")

    country_list = sorted(df["Country"].unique())
    country = st.selectbox("Choose Country:", country_list, index=country_list.index("India"))

    row = df[df["Country"] == country].iloc[0]

    st.markdown("### 📌 Latest Available Indicators")
    st.write(f"**Employment:** {row['Employment']:.2f}%")
    st.write(f"**Literacy:** {row['Literacy']:.2f}%")
    st.write(f"**Female Population %:** {row['Population_female_pct']:.2f}%")
    st.write(f"**Empowerment Index (WEI 2022):** {row['WEI_2022']:.2f}")
    st.write(f"**Safety Index (2024):** {row['Safety_2024']:.2f}")

    # ---------------------
    # Load trained ML model
    # ---------------------
    model = joblib.load("rf_employment_model.joblib")

    # Make prediction
    X_pred = np.array([[
        row["Employment"],         # lag-1
        row["Literacy"],
        row["Population_female_pct"],
        row["WEI_2022"],
        row["Safety_2024"]
    ]])

    pred = model.predict(X_pred)[0]

    # ---------------------
    # Display prediction
    # ---------------------
    st.markdown("### 🔮 Predicted Female Employment — **Next Year**")
    st.success(f"### {pred:.2f}%")

    # ---------------------
    # Optional trend + prediction chart
    # ---------------------
    st.markdown("### 📈 Past Trend + Prediction")

    fig, ax = plt.subplots(figsize=(6, 3))

    # Past values (last 5 years)
    emp_country = pd.read_csv(EMPLOY_PATH, skiprows=4)
    emp_country = emp_country[emp_country["Country Name"] == country]

    years = [c for c in emp_country.columns if c.isdigit()]
    vals = emp_country[years].iloc[0].dropna()

    last_years = vals.index[-5:].astype(int)
    last_vals = vals.values[-5:]

    ax.plot(last_years, last_vals, marker="o", label="Historical Employment")
    ax.scatter(last_years[-1] + 1, pred, color="red", label="Predicted Next Year", s=80)

    ax.set_xlabel("Year")
    ax.set_ylabel("Employment (%)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# ============================
# PAGE ROUTING
# ============================
if selected == "Dashboard":
    dashboard_overview()
elif selected == "Female Population":
    female_population_map()
elif selected == "Literacy vs Employement":
    literacy_vs_employment()
elif selected == "Employement and Safety Indices":
    employment_and_safety_indices()
elif selected == "ML Prediction":
    employment_prediction_ml()
elif selected == "Health":
    st.info("Health indicators visualizations coming soon.")


