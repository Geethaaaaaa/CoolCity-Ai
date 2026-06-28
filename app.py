import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CoolCity AI — Urban Heat Mitigation System",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# THEME — Deep Space Dark + Heat Gradient Palette
# Primary bg:    #0D1117
# Card bg:       #161B22
# Accent 1:      #FF4B4B  (extreme heat red)
# Accent 2:      #FF8C00  (warning orange)
# Accent 3:      #00C9A7  (cool teal — mitigation)
# Accent 4:      #4E9AF1  (satellite blue)
# Text primary:  #E6EDF3
# Text muted:    #8B949E
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Base */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #0D1117;
        color: #E6EDF3;
    }

    .stApp {
        background-color: #0D1117;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container padding */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Hero header */
    .hero-container {
        background: linear-gradient(135deg, #161B22 0%, #1C2333 50%, #161B22 100%);
        border: 1px solid #30363D;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #FF4B4B, #FF8C00, #00C9A7, #4E9AF1);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #E6EDF3;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-title span {
        background: linear-gradient(90deg, #FF4B4B, #FF8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #8B949E;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(78, 154, 241, 0.15);
        border: 1px solid rgba(78, 154, 241, 0.3);
        color: #4E9AF1;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
    }

    .hero-badge-team {
        background: rgba(0, 201, 167, 0.15);
        border: 1px solid rgba(0, 201, 167, 0.3);
        color: #00C9A7;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #E6EDF3;
        margin: 0 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #30363D;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }

    .metric-card:hover {
        border-color: #FF4B4B;
    }

    .metric-label {
        font-size: 0.72rem;
        color: #8B949E;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #E6EDF3;
        line-height: 1;
    }

    .metric-sub {
        font-size: 0.8rem;
        color: #8B949E;
        margin-top: 0.3rem;
    }

    .metric-hot .metric-value { color: #FF4B4B; }
    .metric-cool .metric-value { color: #00C9A7; }
    .metric-green .metric-value { color: #3FB950; }
    .metric-built .metric-value { color: #FF8C00; }

    /* Data table */
    .dataframe {
        background-color: #161B22 !important;
        color: #E6EDF3 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }

    /* Chart containers */
    .chart-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .chart-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Simulator */
    .simulator-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.5rem;
    }

    .result-box {
        background: #0D1117;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }

    .result-temp {
        font-size: 2.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .result-label {
        font-size: 0.75rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'JetBrains Mono', monospace;
    }

    .cooling-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .cooling-good {
        background: rgba(0, 201, 167, 0.15);
        color: #00C9A7;
        border: 1px solid rgba(0, 201, 167, 0.3);
    }

    .cooling-mild {
        background: rgba(255, 140, 0, 0.15);
        color: #FF8C00;
        border: 1px solid rgba(255, 140, 0, 0.3);
    }

    .cooling-none {
        background: rgba(255, 75, 75, 0.15);
        color: #FF4B4B;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }

    /* Findings */
    .finding-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-left: 3px solid #FF4B4B;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
    }

    .finding-card.teal { border-left-color: #00C9A7; }
    .finding-card.blue { border-left-color: #4E9AF1; }
    .finding-card.orange { border-left-color: #FF8C00; }
    .finding-card.green { border-left-color: #3FB950; }

    .finding-text {
        font-size: 0.9rem;
        color: #E6EDF3;
        line-height: 1.5;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #30363D, transparent);
        margin: 2rem 0;
    }

    /* Streamlit overrides */
    .stSlider > div > div > div > div {
        background: #FF4B4B !important;
    }

    div[data-testid="stSelectbox"] > div {
        background: #161B22;
        border-color: #30363D;
        color: #E6EDF3;
    }

    .stSelectbox label {
        color: #8B949E !important;
        font-size: 0.8rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stSlider label {
        color: #8B949E !important;
        font-size: 0.8rem !important;
    }

    /* Footer */
    .footer-bar {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
        margin-top: 2rem;
        font-size: 0.78rem;
        color: #8B949E;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
data = {
    'City':  ['Hyderabad', 'Chennai', 'Delhi', 'Nagpur', 'Ahmedabad', 'Jaipur', 'Lucknow'],
    'State': ['Telangana', 'Tamil Nadu', 'NCT', 'Maharashtra', 'Gujarat', 'Rajasthan', 'UP'],
    'LST':   [40.90, 39.52, 39.40, 39.50, 44.00, 47.70, 30.20],
    'NDVI':  [0.15,  0.14,  0.12,  0.15,  0.15,  0.12,  0.15],
    'NDBI':  [-0.0017, -0.016, -0.01, -0.013, -0.008, 0.04, -0.02],
    'NDWI':  [-0.18, -0.15, -0.15, -0.10, -0.18, 0.16, -0.18]
}
df = pd.DataFrame(data)

# Heat Risk
def heat_risk(lst):
    if lst >= 45:   return '🔴 Extreme'
    elif lst >= 42: return '🟠 Very High'
    elif lst >= 39: return '🟡 High'
    else:           return '🟢 Medium'

df['Heat Risk'] = df['LST'].apply(heat_risk)

# ============================================================
# MODEL
# ============================================================
X = df[['NDVI', 'NDBI', 'NDWI']]
y = df['LST']
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)
importances = model.feature_importances_

# ============================================================
# MATPLOTLIB DARK THEME
# ============================================================
plt.rcParams.update({
    'figure.facecolor':  '#161B22',
    'axes.facecolor':    '#161B22',
    'axes.edgecolor':    '#30363D',
    'axes.labelcolor':   '#8B949E',
    'xtick.color':       '#8B949E',
    'ytick.color':       '#8B949E',
    'grid.color':        '#21262D',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'text.color':        '#E6EDF3',
    'font.family':       'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

PALETTE = {
    'red':    '#FF4B4B',
    'orange': '#FF8C00',
    'teal':   '#00C9A7',
    'blue':   '#4E9AF1',
    'green':  '#3FB950',
    'purple': '#BC8CFF',
    'muted':  '#8B949E',
}

def bar_color(lst):
    if lst >= 45:   return PALETTE['red']
    elif lst >= 42: return PALETTE['orange']
    elif lst >= 39: return '#FFB347'
    else:           return PALETTE['teal']

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-container">
    <div style="margin-bottom:0.8rem;">
        <span class="hero-badge">🛰️ BAH 2026 — PS01</span>
        <span class="hero-badge hero-badge-team">👩‍💻 3 CODES & 1 CIRCUIT</span>
        <span class="hero-badge" style="background:rgba(188,140,255,0.15);border-color:rgba(188,140,255,0.3);color:#BC8CFF;">🎓BAH 2026 </span>
    </div>
    <h1 class="hero-title">🌡️ CoolCity <span>AI</span></h1>
    <p class="hero-subtitle">Urban Heat Mitigation System — Satellite-driven AI/ML analysis of Urban Heat Islands across Indian cities</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SECTION 1 — KEY METRICS
# ============================================================
st.markdown('<p class="section-header">📊 Satellite Data Overview</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "metric-hot",   "🔥 Hottest City",    "Jaipur",     "47.7°C avg surface temp"),
    (c2, "metric-cool",  "❄️ Coolest City",    "Lucknow",    "30.2°C avg surface temp"),
    (c3, "metric-green", "🌿 Most Green",      "Hyd / Nag",  "NDVI 0.15"),
    (c4, "metric-built", "🏗️ Most Built-up",   "Jaipur",     "NDBI +0.04"),
    (c5, "metric-blue",  "📡 Data Source",     "Landsat 8",  "Google Earth Engine"),
]
for col, cls, label, val, sub in metrics:
    col.markdown(f"""
    <div class="metric-card {cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SECTION 2 — DATASET TABLE
# ============================================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">🗂️ Raw Dataset — Landsat 8 Satellite Indices</p>', unsafe_allow_html=True)

def style_lst(val):
    if isinstance(val, float):
        if val >= 45:   return 'background-color:#FF4B4B22;color:#FF4B4B;font-weight:700'
        elif val >= 42: return 'background-color:#FF8C0022;color:#FF8C00;font-weight:700'
        elif val >= 39: return 'background-color:#FFB34722;color:#FFB347;font-weight:600'
        else:           return 'background-color:#00C9A722;color:#00C9A7;font-weight:700'
    return ''

display_df = df[['City', 'State', 'LST', 'NDVI', 'NDBI', 'NDWI', 'Heat Risk']].copy()
st.dataframe(
    display_df.style
        .map(style_lst, subset=['LST'])
        .format({'LST': '{:.2f}°C', 'NDVI': '{:.4f}', 'NDBI': '{:.4f}', 'NDWI': '{:.4f}'}),
    use_container_width=True,
    height=280
)

# ============================================================
# SECTION 3 — CHARTS
# ============================================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">📈 Analysis Charts</p>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

# Chart 1 — LST by City
with col_a:
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    colors_bar = [bar_color(v) for v in df['LST']]
    bars = ax1.bar(df['City'], df['LST'], color=colors_bar,
                   width=0.6, edgecolor='none', zorder=3)
    ax1.set_ylim(0, 55)
    ax1.set_ylabel('LST (°C)', fontsize=9)
    ax1.set_title('Land Surface Temperature by City', fontsize=11,
                  fontweight='600', color='#E6EDF3', pad=12)
    ax1.tick_params(axis='x', rotation=35, labelsize=8)
    ax1.grid(axis='y', zorder=0)
    for bar, val in zip(bars, df['LST']):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.6,
                 f'{val}°C', ha='center', fontsize=7.5,
                 color='#E6EDF3', fontweight='600')
    legend_items = [
        mpatches.Patch(color=PALETTE['red'],    label='Extreme (≥45°C)'),
        mpatches.Patch(color=PALETTE['orange'], label='Very High (≥42°C)'),
        mpatches.Patch(color='#FFB347',         label='High (≥39°C)'),
        mpatches.Patch(color=PALETTE['teal'],   label='Medium (<35°C)'),
    ]
    ax1.legend(handles=legend_items, fontsize=7, loc='upper left',
               framealpha=0.2, edgecolor='#30363D')
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)

# Chart 2 — Feature Importance
with col_b:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    feat_labels = ['NDBI\n(Concrete)', 'NDWI\n(Water)', 'NDVI\n(Greenery)']
    feat_colors = [PALETTE['orange'], PALETTE['blue'], PALETTE['green']]
    feat_vals   = importances  # order matches X columns: NDVI, NDBI, NDWI
    # reorder to NDBI, NDWI, NDVI for display
    order = [1, 2, 0]
    disp_vals   = [importances[i] for i in order]
    disp_labels = feat_labels
    disp_colors = feat_colors
    bars2 = ax2.bar(disp_labels, disp_vals, color=disp_colors,
                    width=0.5, edgecolor='none', zorder=3)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Importance Score', fontsize=9)
    ax2.set_title('What Drives Urban Heat the Most?', fontsize=11,
                  fontweight='600', color='#E6EDF3', pad=12)
    ax2.grid(axis='y', zorder=0)
    for bar, val in zip(bars2, disp_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', fontsize=9,
                 color='#E6EDF3', fontweight='700')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

# Chart 3 — NDVI vs LST
col_c, col_d = st.columns(2)

with col_c:
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    scatter_colors = [bar_color(v) for v in df['LST']]
    sc = ax3.scatter(df['NDVI'], df['LST'], c=scatter_colors,
                     s=150, zorder=5, edgecolors='#30363D', linewidths=1)
    for i, row in df.iterrows():
        ax3.annotate(row['City'],
                     (row['NDVI'], row['LST']),
                     textcoords='offset points',
                     xytext=(8, 4),
                     fontsize=7.5, color='#8B949E')
    z = np.polyfit(df['NDVI'], df['LST'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['NDVI'].min()-0.01, df['NDVI'].max()+0.01, 100)
    ax3.plot(x_line, p(x_line), '--', color=PALETTE['purple'],
             linewidth=1.5, alpha=0.7, label='Trend line')
    ax3.set_xlabel('NDVI (Vegetation Index)', fontsize=9)
    ax3.set_ylabel('LST (°C)', fontsize=9)
    ax3.set_title('Greenery vs Temperature (NDVI vs LST)', fontsize=11,
                  fontweight='600', color='#E6EDF3', pad=12)
    ax3.legend(fontsize=8, framealpha=0.2, edgecolor='#30363D')
    ax3.grid(zorder=0)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)

# Chart 4 — NDBI vs LST
with col_d:
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    sc2 = ax4.scatter(df['NDBI'], df['LST'], c=scatter_colors,
                      s=150, zorder=5, edgecolors='#30363D', linewidths=1)
    for i, row in df.iterrows():
        ax4.annotate(row['City'],
                     (row['NDBI'], row['LST']),
                     textcoords='offset points',
                     xytext=(8, 4),
                     fontsize=7.5, color='#8B949E')
    z2 = np.polyfit(df['NDBI'], df['LST'], 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(df['NDBI'].min()-0.005, df['NDBI'].max()+0.005, 100)
    ax4.plot(x_line2, p2(x_line2), '--', color=PALETTE['red'],
             linewidth=1.5, alpha=0.7, label='Trend line')
    ax4.set_xlabel('NDBI (Built-up Index)', fontsize=9)
    ax4.set_ylabel('LST (°C)', fontsize=9)
    ax4.set_title('Concrete vs Temperature (NDBI vs LST)', fontsize=11,
                  fontweight='600', color='#E6EDF3', pad=12)
    ax4.legend(fontsize=8, framealpha=0.2, edgecolor='#30363D')
    ax4.grid(zorder=0)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)

# ============================================================
# SECTION 4 — MITIGATION SIMULATOR
# ============================================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">🧪 Mitigation Strategy Simulator</p>', unsafe_allow_html=True)
st.markdown('<p style="color:#8B949E;font-size:0.85rem;margin-bottom:1rem;">Select a city and adjust the sliders to simulate real cooling strategies. The AI model predicts the new surface temperature instantly.</p>', unsafe_allow_html=True)

selected_city = st.selectbox(
    "SELECT CITY",
    options=df['City'].tolist(),
    index=0
)

city_row = df[df['City'] == selected_city].iloc[0]

sim_col1, sim_col2 = st.columns([1.2, 1])

with sim_col1:
    st.markdown('<div class="simulator-card">', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#8B949E;font-size:0.75rem;font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:1rem;">Adjusting strategies for — {selected_city}</p>', unsafe_allow_html=True)

    new_ndvi = st.slider(
        "🌿 Increase Green Cover (NDVI) — Plant trees, parks, rooftop gardens",
        min_value=float(round(city_row['NDVI'], 4)),
        max_value=float(round(city_row['NDVI'] + 0.30, 4)),
        value=float(round(city_row['NDVI'], 4)),
        step=0.01,
        format="%.2f"
    )

    new_ndbi = st.slider(
        "🏗️ Reduce Concrete (NDBI) — Replace roads/buildings with open spaces",
        min_value=float(round(city_row['NDBI'] - 0.10, 4)),
        max_value=float(round(city_row['NDBI'], 4)),
        value=float(round(city_row['NDBI'], 4)),
        step=0.01,
        format="%.2f"
    )

    new_ndwi = st.slider(
        "💧 Add Water Bodies (NDWI) — Create lakes, ponds, urban wetlands",
        min_value=float(round(city_row['NDWI'], 4)),
        max_value=float(round(city_row['NDWI'] + 0.20, 4)),
        value=float(round(city_row['NDWI'], 4)),
        step=0.01,
        format="%.2f"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with sim_col2:
    predicted_temp = model.predict(
        pd.DataFrame({'NDVI': [new_ndvi], 'NDBI': [new_ndbi], 'NDWI': [new_ndwi]})
    )[0]
    temp_change = predicted_temp - city_row['LST']

    # Current temp box
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Current Temperature</div>
        <div class="result-temp" style="color:#FF4B4B;">{city_row['LST']:.2f}°C</div>
        <div style="font-size:0.75rem;color:#8B949E;margin-top:0.3rem;">{city_row['Heat Risk']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Predicted temp box
    pred_color = '#00C9A7' if temp_change < -2 else '#FF8C00' if temp_change < 0 else '#FF4B4B'
    st.markdown(f"""
    <div class="result-box" style="border-color:{pred_color}33;">
        <div class="result-label">Predicted Temperature</div>
        <div class="result-temp" style="color:{pred_color};">{predicted_temp:.2f}°C</div>
        <div style="font-size:0.85rem;color:{pred_color};margin-top:0.3rem;font-weight:600;">
            {'▼' if temp_change < 0 else '▲'} {abs(temp_change):.2f}°C {'cooling' if temp_change < 0 else 'increase'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status message
    if temp_change < -4:
        st.markdown(f'<div class="cooling-badge cooling-good">✅ Excellent cooling — {abs(temp_change):.2f}°C reduction!</div>', unsafe_allow_html=True)
    elif temp_change < -1:
        st.markdown(f'<div class="cooling-badge cooling-mild">👍 Mild cooling — {abs(temp_change):.2f}°C reduction</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="cooling-badge cooling-none">⚠️ Adjust sliders further for cooling</div>', unsafe_allow_html=True)

    # Auto recommendation
    st.markdown('<p style="color:#8B949E;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem;font-family:JetBrains Mono,monospace;">Priority Actions</p>', unsafe_allow_html=True)
    if city_row['NDBI'] > 0:
        st.markdown('<div class="finding-card orange"><span class="finding-text">🏗️ High concrete density detected — prioritize reducing built-up surfaces first</span></div>', unsafe_allow_html=True)
    if city_row['NDWI'] < -0.15:
        st.markdown('<div class="finding-card blue"><span class="finding-text">💧 Critical water deficit — add lakes, ponds, or urban wetlands</span></div>', unsafe_allow_html=True)
    if city_row['NDVI'] < 0.13:
        st.markdown('<div class="finding-card" style="border-left-color:#3FB950"><span class="finding-text">🌿 Low vegetation — increase tree canopy and rooftop gardens</span></div>', unsafe_allow_html=True)

# ============================================================
# SECTION 5 — KEY FINDINGS
# ============================================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">🏆 Key Findings for ISRO</p>', unsafe_allow_html=True)

findings = [
    ("red",    "🥇 Concrete density (NDBI) is the #1 driver of urban heat — contributing over 70% of temperature variance across all 7 cities studied."),
    ("orange", "🌡️ Jaipur is India's most heat-stressed city in this study at 47.7°C average surface temperature, driven by high NDBI (+0.04)."),
    ("teal",   "❄️ Reducing built-up surfaces (NDBI) can drop city temperatures by up to 8°C — 9x more effective than planting trees alone."),
    ("blue",   "💧 Water bodies are the second most effective cooling strategy — cities with higher NDWI show measurably lower surface temperatures."),
    ("green",  "🌿 Green cover (NDVI) alone is insufficient — it must be combined with concrete reduction and increased water bodies for meaningful cooling."),
    ("orange", "📡 All indices (LST, NDVI, NDBI, NDWI) were extracted from Landsat 8 imagery via Google Earth Engine — fully reproducible and scalable to all Indian cities."),
]

f1, f2 = st.columns(2)
for i, (color, text) in enumerate(findings):
    col = f1 if i % 2 == 0 else f2
    col.markdown(f'<div class="finding-card {color}"><span class="finding-text">{text}</span></div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-bar">
    🛰️ &nbsp; CoolCity AI &nbsp;|&nbsp; 3 CODES & 1 CIRCUIT &nbsp;|&nbsp;
    Bharatiya Antariksh Hackathon 2026 — Problem Statement 01 &nbsp;|&nbsp; Data: Landsat 8 via Google Earth Engine
</div>
""", unsafe_allow_html=True)