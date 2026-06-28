import folium
import pandas as pd
from folium.plugins import HeatMap

# City data with coordinates
data = {
    'City':  ['Hyderabad', 'Chennai', 'Delhi', 'Nagpur', 'Ahmedabad', 'Jaipur', 'Lucknow'],
    'State': ['Telangana', 'Tamil Nadu', 'NCT', 'Maharashtra', 'Gujarat', 'Rajasthan', 'UP'],
    'LST':   [40.90, 39.52, 39.40, 39.50, 44.00, 47.70, 30.20],
    'NDVI':  [0.15, 0.14, 0.12, 0.15, 0.15, 0.12, 0.15],
    'NDBI':  [-0.0017, -0.016, -0.01, -0.013, -0.008, 0.04, -0.02],
    'NDWI':  [-0.18, -0.15, -0.15, -0.10, -0.18, 0.16, -0.18],
    'Lat':   [17.385, 13.082, 28.613, 21.145, 23.022, 26.912, 26.846],
    'Lon':   [78.486, 80.270, 77.209, 79.088, 72.571, 75.787, 80.946]
}

df = pd.DataFrame(data)

# ============================================================
# CREATE BASE MAP — centered on India
# ============================================================
m = folium.Map(
    location=[22.5, 78.9],
    zoom_start=5,
    tiles='CartoDB dark_matter'
)

# ============================================================
# HEATMAP LAYER — intensity based on LST
# ============================================================
heat_data = [[row['Lat'], row['Lon'], row['LST']/50]
             for _, row in df.iterrows()]

HeatMap(
    heat_data,
    min_opacity=0.4,
    radius=60,
    blur=40,
    gradient={
        '0.3': '#00C9A7',
        '0.5': '#FFB347',
        '0.7': '#FF8C00',
        '1.0': '#FF4B4B'
    }
).add_to(m)

# ============================================================
# CITY MARKERS with popup info
# ============================================================
def get_color(lst):
    if lst >= 45:   return '#FF4B4B'
    elif lst >= 42: return '#FF8C00'
    elif lst >= 39: return '#FFB347'
    else:           return '#00C9A7'

def get_risk(lst):
    if lst >= 45:   return '🔴 EXTREME'
    elif lst >= 42: return '🟠 VERY HIGH'
    elif lst >= 39: return '🟡 HIGH'
    else:           return '🟢 MEDIUM'

for _, row in df.iterrows():
    color = get_color(row['LST'])
    risk  = get_risk(row['LST'])

    popup_html = f"""
    <div style="
        font-family: 'Segoe UI', sans-serif;
        background: #1C2333;
        color: #E6EDF3;
        padding: 12px 16px;
        border-radius: 10px;
        border-left: 4px solid {color};
        min-width: 200px;
    ">
        <h3 style="margin:0 0 8px;color:{color};">🏙️ {row['City']}</h3>
        <p style="margin:2px 0;font-size:12px;color:#8B949E;">{row['State']}</p>
        <hr style="border-color:#30363D;margin:8px 0;">
        <table style="width:100%;font-size:12px;">
            <tr><td>🌡️ Temperature</td><td style="color:{color};font-weight:700;">{row['LST']}°C</td></tr>
            <tr><td>⚠️ Heat Risk</td><td style="color:{color};font-weight:700;">{risk}</td></tr>
            <tr><td>🌿 NDVI</td><td style="color:#3FB950;">{row['NDVI']}</td></tr>
            <tr><td>🏗️ NDBI</td><td style="color:#FF8C00;">{row['NDBI']}</td></tr>
            <tr><td>💧 NDWI</td><td style="color:#4E9AF1;">{row['NDWI']}</td></tr>
        </table>
    </div>
    """

    folium.CircleMarker(
        location=[row['Lat'], row['Lon']],
        radius=18,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{row['City']} — {row['LST']}°C"
    ).add_to(m)

    # City name label
    folium.Marker(
        location=[row['Lat'] + 0.4, row['Lon']],
        icon=folium.DivIcon(
            html=f'<div style="font-size:11px;font-weight:700;color:{color};text-shadow:1px 1px 2px #000;">{row["City"]}</div>',
            icon_size=(100, 20)
        )
    ).add_to(m)

# ============================================================
# LEGEND
# ============================================================
legend_html = """
<div style="
    position: fixed;
    bottom: 30px; right: 30px;
    background: #1C2333;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 12px 16px;
    font-family: 'Segoe UI', sans-serif;
    color: #E6EDF3;
    z-index: 1000;
    font-size: 12px;
">
    <b style="color:#E6EDF3;">🌡️ Heat Risk Level</b><br><br>
    <span style="color:#FF4B4B;">●</span> Extreme  (≥45°C)<br>
    <span style="color:#FF8C00;">●</span> Very High (≥42°C)<br>
    <span style="color:#FFB347;">●</span> High      (≥39°C)<br>
    <span style="color:#00C9A7;">●</span> Medium    (&lt;35°C)<br><br>
    <small style="color:#8B949E;">Data: Landsat 8 via GEE</small>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ============================================================
# SAVE MAP
# ============================================================
m.save('heat_map.html')
print("✅ heat_map.html saved successfully!")