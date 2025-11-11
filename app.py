import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, export_text

import folium
from streamlit_folium import st_folium

# ---------------------------------------------------
# OpenAI import (handles both new & old SDK)
# ---------------------------------------------------
try:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else None
    NEW_SDK = True
except ImportError:  # legacy SDK
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    client = None
    NEW_SDK = False

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Irish Agri-Food AI Dashboard",
    layout="wide",
    page_icon="🌾",
)

DATA_DIR = Path("data")


@st.cache_data
def load_csv(path_str: str):
    """Robust CSV loader.

    - Accepts either a relative path (under DATA_DIR) or an absolute path.
    - Fixes the common case where the file has been saved with the entire row
      as a single quoted column like 'year,index_type,value'.
    - Normalises column names (lowercase, strip spaces, remove BOM).
    - Heuristically restores a 'year' column when possible.
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = DATA_DIR / path_str

    if not path.exists():
        st.warning(f"CSV not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Failed to load {path}: {e}")
        return None

    if df is None:
        return None

    # Case 1: single-column CSV like "year,index_type,value"
    if df.shape[1] == 1:
        colname = df.columns[0]
        if isinstance(colname, str) and "," in colname:
            try:
                header_cols = [c.strip() for c in colname.split(",")]
                split_df = df[colname].astype(str).str.split(",", expand=True)
                n_split = split_df.shape[1]
                # Align header length with actual split columns
                if n_split == len(header_cols):
                    split_df.columns = header_cols
                elif n_split > len(header_cols):
                    extra = [f"extra_{i+1}" for i in range(n_split - len(header_cols))]
                    split_df.columns = header_cols + extra
                else:  # n_split < len(header_cols)
                    split_df.columns = header_cols[:n_split]
                df = split_df
            except Exception as e:
                st.warning(f"Could not normalise single-column CSV {path}: {e}")
                return df  # fall back to whatever we have

    # Normalise column names: strip, lower, remove BOM
    clean_map = {}
    for c in df.columns:
        if not isinstance(c, str):
            clean_map[c] = c
            continue
        clean = c.replace("\ufeff", "").strip().lower()
        clean_map[c] = clean
    df = df.rename(columns=clean_map)

    # If 'year' not present but a column name contains 'year', force-rename it
    if "year" not in df.columns:
        for c in list(df.columns):
            if isinstance(c, str) and "year" in c:
                df = df.rename(columns={c: "year"})
                break

    # Heuristic: if we still don't have 'year', try to infer from a numeric column
    if "year" not in df.columns:
        for c in list(df.columns):
            s_num = pd.to_numeric(df[c], errors="ignore")
            if not hasattr(s_num, "dtype"):
                continue
            if not np.issubdtype(getattr(s_num, "dtype", object), np.number):
                continue
            non_null = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(non_null) == 0:
                continue
            vmin, vmax = non_null.min(), non_null.max()
            if 1900 <= vmin <= 2100 and 1900 <= vmax <= 2100:
                try:
                    df["year"] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
                    break
                except Exception:
                    continue

    return df


def load_data():
    return {
        "prices": load_csv("cso_agri_price_indices.csv"),
        "ghg": load_csv("agri_ghg_emissions.csv"),
        "cap": load_csv("cap_climate_spend.csv"),
        "digital": load_csv("digital_adoption_survey.csv"),
        "copernicus": load_csv("ie_copernicus_agri_econ_panel_2016_2024.csv"),
    }


data = load_data()

# ---------------------------------------------------
# Policy corpus: EU AI Act and related EU instruments
# (short summaries used as context for all NLP modules)
# ---------------------------------------------------

POLICY_CORPUS = {
    "EU_AI_ACT": {
        "label": "EU AI Act (Reg. 2024/1689)",
        "summary": (
            "The EU AI Act establishes a risk-based framework for AI (prohibited, high-risk, "
            "limited- and minimal-risk). High-risk systems must comply with requirements on "
            "risk management, data governance, technical documentation, transparency, human "
            "oversight, robustness, cybersecurity and post-market monitoring. Sectoral rules, "
            "including for agriculture and financial services, must be interpreted consistently "
            "with these horizontal obligations."
        ),
    },
    "EU_GDPR": {
        "label": "GDPR (Reg. 2016/679)",
        "summary": (
            "The GDPR protects personal data and fundamental rights. Any AI system processing "
            "personal data must comply with lawfulness, fairness, transparency, purpose "
            "limitation, data minimisation, accuracy, storage limitation, integrity and "
            "confidentiality, plus accountability and data subject rights."
        ),
    },
    "EU_DATA_ACT": {
        "label": "EU Data Act (Reg. 2023/2854)",
        "summary": (
            "The Data Act sets rules for fair access to and use of data, including business-to-"
            "business and business-to-government data sharing, with obligations around "
            "contractual fairness, FRAND access conditions and safeguards for trade secrets."
        ),
    },
    "EU_DATA_GOV_ACT": {
        "label": "EU Data Governance Act (Reg. 2022/868)",
        "summary": (
            "The Data Governance Act creates trusted mechanisms for data sharing and reuse, "
            "including data intermediaries and data altruism. It underpins safe use of public "
            "and private data assets as inputs to AI systems."
        ),
    },
    "EU_CYBERSEC_ACT": {
        "label": "EU Cybersecurity Act (Reg. 2019/881)",
        "summary": (
            "The Cybersecurity Act establishes an EU-wide cybersecurity certification "
            "framework. High-risk AI systems and connected agri-tech infrastructure should be "
            "aligned with appropriate assurance levels and security-by-design principles."
        ),
    },
}


def build_policy_context(selected_keys: list[str]) -> str:
    """Build a text block summarising the selected EU instruments."""
    if not selected_keys:
        return ""
    chunks = [
        "You must ground your answer in the following EU regulatory context (summarised):",
    ]
    for key in selected_keys:
        meta = POLICY_CORPUS.get(key)
        if not meta:
            continue
        chunks.append(f"- {meta['label']}: {meta['summary']}")
    chunks.append("")
    return "\n".join(chunks)


# ---------------------------------------------------
# Helper: NLP explanation via GPT-4o-mini
# ---------------------------------------------------


def gpt_chat(prompt: str, max_tokens: int = 800, temperature: float = 0.5) -> str:
    if not api_key:
        return "⚠️ OpenAI API key not set. Configure OPENAI_API_KEY in your environment."

    try:
        if NEW_SDK and client is not None:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        else:
            import openai  # type: ignore

            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ NLP explanation failed: {e}"


# ---------------------------------------------------
# RL Simulation (global, reused in dashboard)
# ---------------------------------------------------

np.random.seed(42)
actions = ["Train Farmers", "Subsidise Tech", "Regulate Emissions", "Do Nothing"]
cost = {"Train Farmers": 40, "Subsidise Tech": 70, "Regulate Emissions": 50, "Do Nothing": 0}
state = dict(adoption=0.55, ghg=23.1)
records: list[dict] = []

for ep in range(60):
    a = np.random.choice(actions)
    ns = state.copy()
    if a == "Train Farmers":
        ns["adoption"] += np.random.uniform(0.03, 0.07)
        ns["ghg"] -= np.random.uniform(0.05, 0.15)
    elif a == "Subsidise Tech":
        ns["adoption"] += np.random.uniform(0.05, 0.10)
        ns["ghg"] -= np.random.uniform(0.00, 0.05)
    elif a == "Regulate Emissions":
        ns["adoption"] += np.random.uniform(-0.02, 0.03)
        ns["ghg"] -= np.random.uniform(0.15, 0.30)
    else:  # Do Nothing – small noise
        ns["adoption"] += np.random.uniform(-0.01, 0.01)
        ns["ghg"] += np.random.uniform(-0.05, 0.05)

    ns["adoption"] = float(np.clip(ns["adoption"], 0, 1))
    ns["ghg"] = float(np.clip(ns["ghg"], 15, 25))

    reward = (ns["adoption"] - state["adoption"]) * 10 - (ns["ghg"] - state["ghg"]) * 5 - cost[a] / 100
    records.append({**ns, "episode": ep + 1, "action": a, "reward": reward})
    state = ns

rl_df = pd.DataFrame(records)


def policy_label(row: pd.Series) -> str:
    if row.adoption > 0.7 and row.ghg < 20:
        return "🟢 Sustainable"
    elif 0.6 <= row.adoption <= 0.75 and 20 <= row.ghg <= 22:
        return "🟡 Balanced"
    else:
        return "🔴 Unsustainable"


rl_df["policy"] = rl_df.apply(policy_label, axis=1)

# Train decision tree once
X = rl_df[["adoption", "ghg", "reward"]]
y = rl_df["policy"].replace(
    {"🟢 Sustainable": "Sustainable", "🟡 Balanced": "Balanced", "🔴 Unsustainable": "Unsustainable"}
)
clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X, y)

# ---------------------------------------------------
# Sidebar – choose regulatory lenses
# ---------------------------------------------------

st.sidebar.title("Regulatory Lenses")
policy_labels = [meta["label"] for meta in POLICY_CORPUS.values()]
default_labels = [POLICY_CORPUS["EU_AI_ACT"]["label"], POLICY_CORPUS["EU_GDPR"]["label"]]
selected_labels = st.sidebar.multiselect(
    "Apply EU frameworks to all AI/NLP reasoning:",
    policy_labels,
    default=default_labels,
)

selected_keys: list[str] = []
for key, meta in POLICY_CORPUS.items():
    if meta["label"] in selected_labels:
        selected_keys.append(key)

policy_context = build_policy_context(selected_keys)

# ---------------------------------------------------
# Layout – Single-page dashboard grid
# ---------------------------------------------------


st.title("Agri-Policy-Twin: A Self-Learning, Reasoning & Governing Digital Twin for Ireland’s Agri-Food Sector")

st.markdown(
    """
   System Architecture, Design and Engineering: Shubhojit Bagchi ©️ 2025
   """
)

# ---------------------------------------------------
# Row 0 – GeoSpatial Copernicus Layer (Top)
# ---------------------------------------------------

st.markdown("---")
st.subheader("GeoSpatial Mapping")

cop = data.get("copernicus")
if cop is None:
    st.info(
        "Place `ie_copernicus_agri_econ_panel_2016_2024.csv` into the `data/` folder or update the path in load_data() to enable the geospatial layer."
    )
else:
    year_min = int(cop["year"].min())
    year_max = int(cop["year"].max())
    year_sel = st.slider(
        "Select Copernicus year", year_min, year_max, year_max, key="cop_year_slider"
    )

    cop_year = cop[cop["year"] == year_sel]

    # Approximate centroids for Irish counties (for Leaflet markers)
    county_coords = {
        "Carlow": (52.84, -6.93),
        "Cavan": (53.99, -7.36),
        "Clare": (52.84, -8.98),
        "Cork": (51.90, -8.47),
        "Donegal": (54.95, -7.73),
        "Dublin": (53.35, -6.26),
        "Galway": (53.27, -9.05),
        "Kerry": (52.27, -9.70),
        "Kildare": (53.22, -6.66),
        "Kilkenny": (52.65, -7.25),
        "Laois": (53.03, -7.30),
        "Leitrim": (53.95, -8.09),
        "Limerick": (52.66, -8.63),
        "Longford": (53.73, -7.80),
        "Louth": (54.00, -6.40),
        "Mayo": (53.86, -9.30),
        "Meath": (53.65, -6.68),
        "Monaghan": (54.25, -6.97),
        "Offaly": (53.27, -7.49),
        "Roscommon": (53.63, -8.18),
        "Sligo": (54.27, -8.47),
        "Tipperary": (52.68, -7.82),
        "Waterford": (52.26, -7.11),
        "Westmeath": (53.53, -7.34),
        "Wexford": (52.34, -6.46),
        "Wicklow": (52.98, -6.04),
    }

    # Create Leaflet map
    m = folium.Map(location=[53.5, -8.0], zoom_start=6, tiles="CartoDB positron")

    for _, row in cop_year.iterrows():
        coords = county_coords.get(row["county"])
        if not coords:
            continue

        lat, lon = coords

        # Add "whiskers" (small crosshair-style lines) around each county location
        # to visually emphasise the point on the map.
        whisk = 0.08  # degrees; small enough not to clutter Ireland map view
        folium.PolyLine(
            locations=[[lat - whisk, lon], [lat + whisk, lon]],
            color="lightblue",
            weight=2,
            opacity=0.9,
        ).add_to(m)
        folium.PolyLine(
            locations=[[lat, lon - whisk], [lat, lon + whisk]],
            color="lightblue",
            weight=2,
            opacity=0.9,
        ).add_to(m)

        tooltip = (
            f"{row['county']} ({year_sel})\n"
            f"NDVI: {row['ndvi_mean']:.3f} | Soil moisture: {row['s1_soil_moisture']:.3f}\n"
            f"GHG: {row['ghg_kgco2e_per_ha']} kgCO₂e/ha | CAP support: {row['cap_support_eur_per_ha']:.0f} €/ha"
        )

        # Blue arrow marker at the centre of the whiskers for each county
        folium.Marker(
            location=[lat, lon],
            tooltip=tooltip,
            icon=folium.Icon(color="blue", icon="arrow-up"),
        ).add_to(m)

    map_state = st_folium(m, height=550, width="100%")

    st.markdown("**NLP explanation for clicked location**")

    if map_state and map_state.get("last_clicked"):
        lat = map_state["last_clicked"]["lat"]
        lon = map_state["last_clicked"]["lng"]

        # Find nearest county centroid to the click
        nearest_county = None
        best_d2 = None
        for c_name, (clat, clon) in county_coords.items():
            d2 = (lat - clat) ** 2 + (lon - clon) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                nearest_county = c_name

        if nearest_county and nearest_county in cop_year["county"].values:
            row_sel = cop_year[cop_year["county"] == nearest_county].iloc[0]
            metrics = {
                "ndvi_mean": float(row_sel["ndvi_mean"]),
                "ndwi_mean": float(row_sel["ndwi_mean"]),
                "ndbi_mean": float(row_sel["ndbi_mean"]),
                "s1_soil_moisture": float(row_sel["s1_soil_moisture"]),
                "rainfall_total_mm": float(row_sel["rainfall_total_mm"]),
                "t2m_mean_c": float(row_sel["t2m_mean_c"]),
                "ghg_kgco2e_per_ha": int(row_sel["ghg_kgco2e_per_ha"]),
                "cap_support_eur_per_ha": float(row_sel["cap_support_eur_per_ha"]),
                "stocking_rate_lu_per_ha": float(row_sel["stocking_rate_lu_per_ha"]),
                "farm_count": int(row_sel["farm_count"]),
                "cattle_head": int(row_sel["cattle_head"]),
                "sheep_head": int(row_sel["sheep_head"]),
            }
            st.write(f"**Selected county:** {nearest_county} ({year_sel})")

            if st.button("Explain this location", key="explain_cop_county"):
                prompt = (
                    policy_context
                    + f"""You are analysing a Copernicus-based agro-economic panel for Irish counties.\n
County: {nearest_county}\nYear: {year_sel}\nMetrics:\n{json.dumps(metrics, indent=2)}\n\n
Provide an integrated explanation of vegetation condition, climate stress, production intensity and policy levers under the selected EU frameworks.\n"""
                )
                st.write(gpt_chat(prompt, max_tokens=800, temperature=0.5))
        else:
            st.info("Click on one of the blue arrow markers to get an explanation.")
    else:
        st.info("Click near a blue arrow marker to activate an explanation.")

# ---------------------------------------------------
# Row 1 – Overview & Key Metrics
# ---------------------------------------------------

row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    st.subheader("1️⃣ System Overview")
    st.markdown(
        """
- Agri-food underpins Irish employment and exports.
- Food Vision 2030 & CAP SP 2023–2027 embed **sustainability** and **digitalisation**.
- Risk: a widening gap between **digitally advanced** and **lagging** farms & processors.
        """
    )

with row1_col2:
    st.subheader("Key RL Indicators")
    latest_adoption = rl_df["adoption"].iloc[-1]
    latest_ghg = rl_df["ghg"].iloc[-1]
    sustainable_share = (rl_df["policy"] == "🟢 Sustainable").mean() * 100

    st.metric("RL Final Adoption", f"{latest_adoption:.2f}")
    st.metric("RL Final GHG (Mt CO₂e)", f"{latest_ghg:.2f}")
    st.metric("Sustainable Episodes", f"{sustainable_share:.1f}%")


# ---------------------------------------------------
# Row 2 – Key Forces + NLP explanation
# ---------------------------------------------------

st.markdown("---")
row2_col1, row2_col2 = st.columns([2, 1])

with row2_col1:
    st.subheader("2️⃣ Disruption Drivers")
    forces_df = pd.DataFrame(
        {
            "Force": [
                "Policy & Sustainability",
                "Technology Maturity",
                "New Entrants & Capital",
                "Farmer Capability Gaps",
            ],
            "Impact": [0.85, 0.9, 0.8, 0.6],
        }
    )
    fig_forces = px.bar(
        forces_df,
        x="Force",
        y="Impact",
        text="Impact",
        range_y=[0, 1],
        title="Impact of Disruption Drivers",
    )
    fig_forces.update_traces(texttemplate="%{text:.0%}", textposition="outside")
    st.plotly_chart(fig_forces, use_container_width=True)

with row2_col2:
    st.subheader("NLP View – Disruption Drivers")
    if st.button("Explain Disruption Drivers", key="explain_forces"):
        prompt = (
            policy_context
            + f"""
You are explaining a bar chart showing disruption drivers for the Irish agri-food sector.
Data:
{forces_df.to_markdown(index=False)}

Provide a short, executive-style explanation of what this chart implies for strategy and compliance under the selected EU frameworks.
            """
        )
        st.write(gpt_chat(prompt))


# ---------------------------------------------------
# Row 3 – Value Chain & Winners/Losers
# ---------------------------------------------------

st.markdown("---")
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.subheader("3️⃣ Value Chain Digital Maturity")
    chain_df = pd.DataFrame(
        {
            "Segment": ["On-Farm", "Processing", "Logistics", "Finance & Advisory"],
            "Digital Maturity": [0.7, 0.8, 0.75, 0.65],
            "Adoption Speed": [0.6, 0.7, 0.8, 0.55],
        }
    )
    fig_chain = px.scatter(
        chain_df,
        x="Digital Maturity",
        y="Adoption Speed",
        size="Digital Maturity",
        color="Segment",
        text="Segment",
        title="Digital Maturity vs Adoption Speed",
    )
    st.plotly_chart(fig_chain, use_container_width=True)
    if st.button("Explain Value Chain Chart"):
        prompt = (
            policy_context
            + f"""
You are explaining a scatter plot with segments of the Irish agri-food value chain.
Data:
{chain_df.to_markdown(index=False)}

Identify which segments are leading and lagging, and outline policy levers or AI-governance requirements relevant to each segment under the selected EU frameworks.
            """
        )
        st.write(gpt_chat(prompt))

with row3_col2:
    st.subheader("4️⃣ Winners & Losers")
    wl_df = pd.DataFrame(
        {
            "Group": [
                "Digital Farmers",
                "Processors / Co-ops",
                "AgTech Start-ups",
                "Traditional Intermediaries",
            ],
            "Digital Intensity": [0.9, 0.85, 0.95, 0.4],
            "Value-Chain Control": [0.8, 0.9, 0.7, 0.3],
        }
    )
    fig_wl = px.scatter(
        wl_df,
        x="Digital Intensity",
        y="Value-Chain Control",
        size="Digital Intensity",
        color="Group",
        text="Group",
        title="Digital Positioning Matrix",
    )
    st.plotly_chart(fig_wl, use_container_width=True)
    if st.button("Explain Winners & Losers"):
        prompt = (
            policy_context
            + f"""
You are explaining a digital positioning matrix for groups in the Irish agri-food value chain.
Data:
{wl_df.to_markdown(index=False)}

Explain who is likely to gain power, who is at risk, and how EU AI and data regulations should shape platform design, data contracts and oversight for each group.
            """
        )
        st.write(gpt_chat(prompt))


# ---------------------------------------------------
# Row 4 – Risks & Directions
# ---------------------------------------------------

st.markdown("---")
row4_col1, row4_col2 = st.columns([1.2, 1.8])

with row4_col1:
    st.subheader("5️⃣ Systemic Risk Gauge")
    risks_df = pd.DataFrame(
        {
            "Risk": [
                "Fragmented Strategy",
                "Platform Lock-in",
                "Skills Gap",
                "Two-Speed Rural Economy",
            ],
            "Severity": [0.85, 0.8, 0.9, 0.75],
        }
    )
    fig_risk = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risks_df["Severity"].mean() * 100,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkgreen"},
                "steps": [
                    {"range": [0, 50], "color": "lightgreen"},
                    {"range": [50, 80], "color": "gold"},
                    {"range": [80, 100], "color": "red"},
                ],
            },
            title={"text": "Average Systemic Risk (%)"},
        )
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with row4_col2:
    st.subheader("6️⃣ Systemic Risk Interpretation & Strategic Directions")

    st.markdown(
        """
**Risk breakdown**

- **Fragmented Strategy** (0.85): Weak coordination across agencies / data silos  
- **Platform Lock-in** (0.80): Over-reliance on proprietary AgTech ecosystems  
- **Skills Gap** (0.90): Severe shortage of digital & AI literacy in rural workforce  
- **Two-Speed Rural Economy** (0.75): Uneven access to broadband, finance, innovation
        """
    )

    st.success(
        """
**Strategic directions**

1. Build a **coherent national digital-agri architecture** (interoperable platforms, open APIs).
2. Invest in **skills, advisory, and explainable AI**, not just hardware.
3. Guarantee **data ownership and portability** for farmers and SMEs, aligning with the Data Act & GDPR.
4. Use traceability and AEIs to support **premium export positioning** and demonstrable AI compliance.
        """
    )


# ---------------------------------------------------
# Row 5 – Empirical Charts with NLP
# ---------------------------------------------------

st.markdown("---")
st.subheader("7️⃣ Empirical Indicators")
row5_col1, row5_col2, row5_col3 = st.columns(3)

# Prices
with row5_col1:
    st.markdown("**Price Indices**")
    prices = data["prices"]
    if prices is not None:
        fig_prices = px.line(
            prices,
            x="year",
            y="value",
            color="index_type",
            markers=True,
            title="Agri Price Indices",
        )
        st.plotly_chart(fig_prices, use_container_width=True)
        if st.button("Explain Price Trends"):
            prompt = (
                policy_context
                + f"""
You are explaining agricultural input and output price indices for Ireland.
Sample of the data:
{prices.head().to_markdown(index=False)}

Describe general trends and potential implications for farmer margins, investment in AI, and any distributional or fairness concerns that might arise under EU law.
                """
            )
            st.write(gpt_chat(prompt))
    else:
        st.info("Upload `cso_agri_price_indices.csv` in the data folder.")

# GHG vs CAP
with row5_col2:
    st.markdown("**GHG vs CAP Climate Spend**")
    ghg = data["ghg"]
    cap = data["cap"]
    if ghg is not None and cap is not None and "year" in ghg.columns and "year" in cap.columns:
        try:
            merged = pd.merge(ghg, cap, on="year", how="inner")
        except Exception as e:
            st.warning(f"Failed to merge GHG and CAP data on 'year': {e}")
            merged = None
        if merged is not None and not merged.empty:
            fig_ghg = go.Figure()
            if "ghg_mtco2e" in merged.columns:
                fig_ghg.add_trace(
                    go.Scatter(
                        x=merged["year"],
                        y=merged["ghg_mtco2e"],
                        mode="lines+markers",
                        name="GHG (Mt CO₂e)",
                    )
                )
            if "climate_spend_million" in merged.columns:
                fig_ghg.add_trace(
                    go.Bar(
                        x=merged["year"],
                        y=merged["climate_spend_million"],
                        name="CAP Climate Spend (€m)",
                        opacity=0.5,
                    )
                )
            fig_ghg.update_layout(barmode="overlay", title="GHG vs CAP Climate Spend")
            st.plotly_chart(fig_ghg, use_container_width=True)
            if st.button("Explain GHG vs CAP"):
                prompt = (
                    policy_context
                    + f"""
You are explaining a chart of agricultural GHG emissions versus CAP climate/environment spend for Ireland.
Data:
{merged.to_markdown(index=False)}

Discuss whether increased spend appears to correlate with stabilisation or reduction in emissions, and how this interacts with AI-based MRV systems and high-risk AI obligations.
                    """
                )
                st.write(gpt_chat(prompt))
        else:
            st.info("GHG and CAP data loaded but no overlapping years after merge.")
            try:
                ghg_years = sorted(ghg["year"].dropna().unique().tolist())
                cap_years = sorted(cap["year"].dropna().unique().tolist())
                st.markdown("**Debug – year coverage**")
                st.write({"ghg_years": ghg_years, "cap_years": cap_years})
            except Exception as e:
                st.warning(f"Could not display year coverage for GHG/CAP data: {e}")
    elif ghg is None or cap is None:
        st.info("Upload `agri_ghg_emissions.csv` and `cap_climate_spend.csv` in the data folder.")
    else:
        st.warning(f"GHG/CAP CSVs loaded but 'year' column is missing. Columns in ghg: {list(ghg.columns)}, columns in cap: {list(cap.columns)}")

# Digital Adoption
with row5_col3:
    st.markdown("**Digital Adoption**")
    digital = data["digital"]
    if digital is not None:
        fig_dig = px.bar(
            digital,
            x="tech_category",
            y="adoption_rate",
            color="year",
            barmode="group",
            text="adoption_rate",
            title="Digital Adoption Rates",
        )
        fig_dig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
        st.plotly_chart(fig_dig, use_container_width=True)
        if st.button("Explain Digital Adoption"):
            prompt = (
                policy_context
                + f"""
You are explaining digital technology adoption among Irish farmers.
Data:
{digital.to_markdown(index=False)}

Summarise which technologies are most/least adopted, and describe how EU AI, data and cybersecurity rules should shape incentives, risk management and support schemes for each.
                """
            )
            st.write(gpt_chat(prompt))
    else:
        st.info("Upload `digital_adoption_survey.csv` in the data folder.")


# ---------------------------------------------------
# Row 6 – RL + Policy Tree + NLP
# ---------------------------------------------------

st.markdown("---")
st.subheader("8️⃣ RL Policy Simulation & Explainable Policy Layer")

row6_col1, row6_col2 = st.columns([1.6, 1.4])

with row6_col1:
    st.markdown("**RL Episodes – Policy Classification**")
    fig_rl = px.scatter(
        rl_df,
        x="ghg",
        y="adoption",
        color="policy",
        hover_data=["episode", "action", "reward"],
        title="RL Episodes – Sustainable vs Balanced vs Unsustainable",
    )
    st.plotly_chart(fig_rl, use_container_width=True)
    if st.button("Explain RL Scatter"):
        summary = rl_df.groupby("policy")["reward"].mean().reset_index()
        prompt = (
            policy_context
            + f"""
You are explaining a scatter of RL episodes coloured by policy class for an Irish agri-food AI scenario.
Episode summary by policy class:
{summary.to_markdown(index=False)}

Explain how often the system achieves sustainable outcomes, where it struggles, and how the EU AI Act's high-risk obligations and governance mechanisms should influence reward design and allowed actions.
            """
        )
        st.write(gpt_chat(prompt))

with row6_col2:
    st.markdown("**Decision Tree Policy Rules**")
    rules_text = export_text(clf, feature_names=list(X.columns))
    st.code(rules_text)

    st.markdown("**Ask RL Teacher (GPT-4o-mini)**")
    snapshot = {
        "episodes": int(len(rl_df)),
        "avg_reward": float(rl_df["reward"].mean()),
        "sustainable_share": float((rl_df["policy"] == "🟢 Sustainable").mean()),
    }
    if st.button("Generate RL Strategy Advice"):
        prompt = (
            policy_context
            + f"""
You are an Agentic RL Teacher advising Ireland's agri-food policymakers.
Here is an RL simulation snapshot:
{json.dumps(snapshot, indent=2)}

Provide 3 concrete policy moves to improve sustainable outcomes while staying compliant with the selected EU frameworks, and one 2-3 sentence insight.
            """
        )
        st.write(gpt_chat(prompt, max_tokens=800, temperature=0.4))

    st.markdown("**RAG-Style Policy Question (GPT-4o-mini)**")
    user_q = st.text_area(
        "Ask a question about agri-food digital policy (eco-schemes, smart farming, etc.)",
        key="rag_q",
    )
    if st.button("Answer Policy Question") and user_q.strip():
        prompt = (
            policy_context
            + f"""
You are an Irish agri-food policy analyst.
Use the selected EU frameworks as your primary legal reference.
Question: {user_q}

Respond with 4-6 concise bullet points, grounded in realistic policy logic and compliance obligations.
            """
        )
        st.write(gpt_chat(prompt, max_tokens=800, temperature=0.5))

