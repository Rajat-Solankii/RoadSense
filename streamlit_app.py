"""
RoadSense AI: Streamlit Frontend
================================
Interactive Streamlit application providing 1-click preset scenario buttons
and real-time collision severity & disaster risk assessment.

Run via:
    streamlit run streamlit_app.py
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath("."))
from src.features import (
    ROAD_CATEGORIES,
    ROAD_TYPES,
    WEATHER_CONDITIONS,
    ROAD_SURFACES,
    LIGHT_CONDITIONS,
    JUNCTION_TYPES,
    VEHICLE_TYPES,
    COLLISION_TYPES,
    SPEED_LIMITS_KMH,
)
from predict import load_models, get_preset_scenarios, predict_scenario

# Page configuration
st.set_page_config(
    page_title="RoadSense AI — Road Safety & Disaster Prevention",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-fatal {
        border-left: 5px solid #ef4444;
    }
    .kpi-grievous {
        border-left: 5px solid #f59e0b;
    }
    .kpi-minor {
        border-left: 5px solid #10b981;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_models():
    return load_models()


models = get_cached_models()
presets = get_preset_scenarios()

st.markdown('<div class="main-title">🛡️ RoadSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Indian Road Safety & Disaster Prevention System • MoRTH & iRAD Calibrated</div>', unsafe_allow_html=True)

# Preset Buttons Section
st.subheader("⚡ Clickable Preset Scenarios")
cols = st.columns(4)

if "active_scenario" not in st.session_state:
    st.session_state["active_scenario"] = presets[0]["scenario"].copy()
    st.session_state["active_name"] = presets[0]["name"]

preset_icons = ["🌫️", "🌧️", "🚜", "🛺"]
for idx, p in enumerate(presets):
    with cols[idx]:
        if st.button(f"{preset_icons[idx]} Preset {idx+1}\n\n{p['name'][:32]}...", key=f"btn_p_{idx}", use_container_width=True):
            st.session_state["active_scenario"] = p["scenario"].copy()
            st.session_state["active_name"] = p["name"]

cur_scenario = st.session_state["active_scenario"]
st.info(f"📍 Active Scenario: **{st.session_state.get('active_name', 'Custom Scenario')}**")

# Form inputs layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("🛠️ Scenario Parameters")
    
    with st.expander("📍 Road & Environmental Context", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            rc_idx = ROAD_CATEGORIES.index(cur_scenario.get("road_category", ROAD_CATEGORIES[0])) if cur_scenario.get("road_category") in ROAD_CATEGORIES else 0
            road_cat = st.selectbox("Road Category", ROAD_CATEGORIES, index=rc_idx)
            
            wc_idx = WEATHER_CONDITIONS.index(cur_scenario.get("weather_condition", WEATHER_CONDITIONS[0])) if cur_scenario.get("weather_condition") in WEATHER_CONDITIONS else 0
            weather = st.selectbox("Weather Condition", WEATHER_CONDITIONS, index=wc_idx)
            
            lc_idx = LIGHT_CONDITIONS.index(cur_scenario.get("light_condition", LIGHT_CONDITIONS[0])) if cur_scenario.get("light_condition") in LIGHT_CONDITIONS else 0
            light = st.selectbox("Lighting Condition", LIGHT_CONDITIONS, index=lc_idx)
        with c2:
            rt_idx = ROAD_TYPES.index(cur_scenario.get("road_type", ROAD_TYPES[0])) if cur_scenario.get("road_type") in ROAD_TYPES else 0
            road_type = st.selectbox("Road Geometry", ROAD_TYPES, index=rt_idx)
            
            rs_idx = ROAD_SURFACES.index(cur_scenario.get("road_surface", ROAD_SURFACES[0])) if cur_scenario.get("road_surface") in ROAD_SURFACES else 0
            surface = st.selectbox("Road Surface", ROAD_SURFACES, index=rs_idx)
            
            jt_idx = JUNCTION_TYPES.index(cur_scenario.get("junction_type", JUNCTION_TYPES[0])) if cur_scenario.get("junction_type") in JUNCTION_TYPES else 0
            junction = st.selectbox("Junction Type", JUNCTION_TYPES, index=jt_idx)

    with st.expander("🚗 Vehicles & Traffic Dynamics", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            pv_idx = VEHICLE_TYPES.index(cur_scenario.get("primary_vehicle_type", VEHICLE_TYPES[0])) if cur_scenario.get("primary_vehicle_type") in VEHICLE_TYPES else 0
            primary_veh = st.selectbox("Primary Vehicle", VEHICLE_TYPES, index=pv_idx)
            
            col_list = COLLISION_TYPES
            ct_idx = col_list.index(cur_scenario.get("collision_type", col_list[0])) if cur_scenario.get("collision_type") in col_list else 0
            collision = st.selectbox("Collision Mechanism", col_list, index=ct_idx)
            
            sl_idx = SPEED_LIMITS_KMH.index(cur_scenario.get("speed_limit_kmh", 80)) if cur_scenario.get("speed_limit_kmh") in SPEED_LIMITS_KMH else 5
            speed_limit = st.selectbox("Speed Limit (km/h)", SPEED_LIMITS_KMH, index=sl_idx)
        with c2:
            sec_options = ["None"] + VEHICLE_TYPES
            sv_idx = sec_options.index(cur_scenario.get("secondary_vehicle_type", "None")) if cur_scenario.get("secondary_vehicle_type") in sec_options else 0
            secondary_veh = st.selectbox("Secondary Vehicle / Obstacle", sec_options, index=sv_idx)
            
            num_veh = st.number_input("Vehicles Involved", min_value=1, max_value=25, value=int(cur_scenario.get("number_of_vehicles", 2)))
            occupants = st.number_input("Total Occupants Involved", min_value=1, max_value=80, value=int(cur_scenario.get("occupants_count", 4)))

        speed_delta = st.slider("Speed Delta vs Limit (km/h)", min_value=-40, max_value=80, value=int(cur_scenario.get("speed_delta_kmh", 0)))

    with st.expander("⚠️ Behavioral Risk Factors", expanded=True):
        b1, b2 = st.columns(2)
        with b1:
            wrong_side = st.checkbox("⛔ Wrong-Side Driving", value=bool(cur_scenario.get("is_wrong_side", 0)))
            overloaded = st.checkbox("📦 Overloaded / Triple Riding", value=bool(cur_scenario.get("is_overloaded", 0)))
            gear_violation = st.checkbox("🪖 Safety Gear (No Helmet/Seatbelt)", value=bool(cur_scenario.get("safety_gear_violation", 0)))
            high_beam = st.checkbox("💡 Blinding High-Beam Glare", value=bool(cur_scenario.get("high_beam_hazard", 0)))
        with b2:
            stray_animal = st.checkbox("🐄 Stray Cattle / Animal Hazard", value=bool(cur_scenario.get("stray_animal_hazard", 0)))
            drowsy = st.checkbox("🥱 Drowsy Driving Window (02-05 AM)", value=bool(cur_scenario.get("drowsy_window", 0)))
            pothole = st.checkbox("🕳️ Deep Pothole / Speed-Breaker", value=bool(cur_scenario.get("pothole_or_speedbreaker", 0)))

    predict_btn = st.button("⚡ Assess Collision & Disaster Risk", type="primary", use_container_width=True)

# Build current payload
payload = {
    "road_category": road_cat,
    "road_type": road_type,
    "weather_condition": weather,
    "road_surface": surface,
    "light_condition": light,
    "junction_type": junction,
    "primary_vehicle_type": primary_veh,
    "secondary_vehicle_type": secondary_veh,
    "collision_type": collision,
    "speed_limit_kmh": speed_limit,
    "speed_delta_kmh": speed_delta,
    "number_of_vehicles": num_veh,
    "occupants_count": occupants,
    "is_wrong_side": int(wrong_side),
    "is_overloaded": int(overloaded),
    "safety_gear_violation": int(gear_violation),
    "high_beam_hazard": int(high_beam),
    "stray_animal_hazard": int(stray_animal),
    "drowsy_window": int(drowsy),
    "pothole_or_speedbreaker": int(pothole),
}

with col_right:
    st.subheader("🎯 Risk Assessment Output")
    
    # Run prediction
    results = predict_scenario(models, payload)
    
    task_b = results.get("task_b", {})
    task_c = results.get("task_c", {})
    disaster = results.get("disaster", {})
    
    severity = task_b.get("prediction", "FATAL")
    dri = disaster.get("disaster_risk_index", 0.0)
    casualties = task_c.get("predicted_casualties", 1.0)
    
    # Hero KPI Cards
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    with kpi_col1:
        st.metric(label="Predicted Severity", value=severity, delta="Critical" if severity=="FATAL" else "Elevated")
        
    with kpi_col2:
        st.metric(label="Disaster Risk Index", value=f"{dri:.1f}%", delta=disaster.get("risk_band", "RISK"))
        
    with kpi_col3:
        st.metric(label="Expected Casualties", value=f"{casualties:.1f}", delta=f"from {occupants} occupants")
        
    # Granular Probabilities
    st.markdown("#### 📊 Granular Severity Probabilities")
    probs = task_b.get("probabilities", {"Fatal": 0.33, "Grievous": 0.33, "Minor": 0.34})
    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.progress(probs.get("Fatal", 0.0), text=f"Fatal: {probs.get('Fatal', 0.0)*100:.1f}%")
    p_col2.progress(probs.get("Grievous", 0.0), text=f"Grievous: {probs.get('Grievous', 0.0)*100:.1f}%")
    p_col3.progress(probs.get("Minor", 0.0), text=f"Minor: {probs.get('Minor', 0.0)*100:.1f}%")
    
    # Urgency & Escalation Threats
    st.markdown("#### 🚨 Urgent Threat Escalation Assessment")
    st.warning(f"**{disaster.get('risk_band', 'CODE RED')}**: {disaster.get('urgency_assessment', '')}")
    
    threats = disaster.get("escalation_threats", [])
    if threats:
        for t in threats:
            st.markdown(f"- ⚠️ {t}")
            
    # Directives & Emergency
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("#### 🛡️ In-Transit Directives")
        directives = disaster.get("mitigation_directives", [])
        for d in directives:
            st.markdown(f"- 👉 {d}")
            
    with d_col2:
        st.markdown("#### 🚑 Emergency & Containment")
        st.info("📞 **NHAI Expressway Helpline:** 1033 | 🚨 **Police / Ambulance:** 112")
        protocols = disaster.get("secondary_prevention_protocols", [])
        for p in protocols:
            st.markdown(f"- ⚡ {p}")
