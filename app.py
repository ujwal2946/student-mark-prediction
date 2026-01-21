import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Student Exam Score Predictor", layout="wide", page_icon="📊")

# ----------------- PERSISTENT STORAGE FUNCTIONS ----------------- #
def save_history_to_file():
    """Save history to a local JSON file"""
    try:
        history_data = {
            "history": st.session_state["history"],
            "favorites": st.session_state["favorites"],
            "last_saved": datetime.now().isoformat()
        }
        with open("student_predictions_data.json", "w") as f:
            json.dump(history_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving history to file: {e}")
        return False

def load_history_from_file():
    """Load history from local JSON file"""
    try:
        if os.path.exists("student_predictions_data.json"):
            with open("student_predictions_data.json", "r") as f:
                history_data = json.load(f)
            st.session_state["history"] = history_data.get("history", [])
            st.session_state["favorites"] = history_data.get("favorites", [])
            return True
    except Exception as e:
        st.error(f"Error loading history from file: {e}")
    return False

def save_history():
    """Save history to persistent storage"""
    return save_history_to_file()

def load_history():
    """Load history from persistent storage"""
    return load_history_from_file()

# ----------------- MODEL LOADING ----------------- #
model_path = "best_model.pkl"
if not os.path.exists(model_path):
    st.warning("⚠️ Model file 'best_model.pkl' not found. Using enhanced fallback prediction model.")
    class FallbackModel:
        def predict(self, X):
            study_hours, attendance, mental_health, sleep_hours, part_time_job = X[0]
            
            # Base score calculation with realistic factors
            base_score = 40  # Base score for average student
            
            # Study hours impact (realistic scaling)
            study_impact = 0
            if study_hours <= 1:
                study_impact = -15
            elif study_hours <= 2:
                study_impact = -5
            elif study_hours <= 4:
                study_impact = study_hours * 3
            elif study_hours <= 6:
                study_impact = 12 + (study_hours - 4) * 2
            else:
                study_impact = 16 + (study_hours - 6) * 1
                
            # Attendance impact
            attendance_impact = 0
            if attendance < 60:
                attendance_impact = -20
            elif attendance < 75:
                attendance_impact = (attendance - 60) * 0.8
            elif attendance < 90:
                attendance_impact = 12 + (attendance - 75) * 1.2
            else:
                attendance_impact = 30 + (attendance - 90) * 0.5
                
            # Mental health impact
            mental_impact = 0
            if mental_health <= 3:
                mental_impact = -15
            elif mental_health <= 5:
                mental_impact = (mental_health - 3) * 2.5
            elif mental_health <= 8:
                mental_impact = 5 + (mental_health - 5) * 3
            else:
                mental_impact = 14 + (mental_health - 8) * 2
                
            # Sleep hours impact
            sleep_impact = 0
            if sleep_hours < 5:
                sleep_impact = -20
            elif sleep_hours < 6:
                sleep_impact = -10
            elif sleep_hours < 7:
                sleep_impact = -5
            elif sleep_hours <= 8:
                sleep_impact = 10
            elif sleep_hours <= 9:
                sleep_impact = 5
            else:
                sleep_impact = -5
                
            # Part-time job impact
            job_impact = -12 if part_time_job == 1 else 5
            
            # Calculate final score
            final_score = (base_score + study_impact + attendance_impact + 
                         mental_impact + sleep_impact + job_impact)
            
            return max(0, min(100, final_score))
    
    model = FallbackModel()
    model_name = "Enhanced Fallback Model"
else:
    try:
        model = joblib.load(model_path)
        model_name = type(model).__name__
    except:
        st.warning("⚠️ Error loading model. Using enhanced fallback prediction.")
        model = FallbackModel()
        model_name = "Enhanced Fallback Model"

# ----------------- SESSION STATE ----------------- #
if "history" not in st.session_state:
    st.session_state["history"] = []
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "show_history" not in st.session_state:
    st.session_state["show_history"] = False
if "prediction_made" not in st.session_state:
    st.session_state["prediction_made"] = False
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []
if "analyze_config" not in st.session_state:
    st.session_state["analyze_config"] = None
if "displayed_prediction" not in st.session_state:
    st.session_state["displayed_prediction"] = False
if "persistent_loaded" not in st.session_state:
    st.session_state["persistent_loaded"] = False

# Load persistent data on first run
if not st.session_state["persistent_loaded"]:
    if load_history():
        st.success("📁 Loaded previous session data!")
    st.session_state["persistent_loaded"] = True

# ----------------- DEFAULT VALUES ----------------- #
defaults = {
    "study_hours": 2.0,
    "attendance": 80,
    "mental_health": 5,
    "sleep_hours": 7.0,
    "part_time_job": "No"
}

# ----------------- HELPER FUNCTIONS ----------------- #
def glow_color(score):
    if score < 50: return "#ff4b4b"
    elif score < 75: return "#ffa500"
    elif score < 85: return "#4caf50"
    elif score < 95: return "#2196f3"
    else: return "#9c27b0"

def feedback_text(score):
    if score < 50: return "Needs Improvement"
    elif score < 75: return "Moderate"
    elif score < 85: return "Good"
    elif score < 95: return "Excellent"
    else: return "Outstanding"

def get_study_tips(score, study_hours, attendance, mental_health, sleep_hours, part_time_job):
    tips = []
    if study_hours < 2:
        tips.append("📚 **Increase study hours** to at least 2-3 hours daily")
    elif study_hours > 6:
        tips.append("📚 **Balance study time** - avoid burnout")
    if attendance < 75:
        tips.append("🏫 **Improve attendance** - aim for at least 75%")
    if mental_health < 5:
        tips.append("🧠 **Focus on mental wellbeing** - take breaks and manage stress")
    if sleep_hours < 6:
        tips.append("💤 **Get more sleep** - aim for 7-8 hours")
    elif sleep_hours > 9:
        tips.append("💤 **Maintain consistent sleep** - 7-8 hours is optimal")
    if part_time_job == "Yes" and study_hours > 4:
        tips.append("💼 **Balance work and study** - consider reducing hours during exams")
    if not tips:
        tips.append("🎯 **Maintain your current habits** - you're on the right track!")
    return tips

def get_study_profile(study_hours, attendance, mental_health, sleep_hours, part_time_job):
    if (study_hours >= 4 and attendance >= 90 and mental_health >= 8 and 
        7 <= sleep_hours <= 8 and part_time_job == "No"):
        return "🎯 High Performer"
    elif (study_hours >= 3 and attendance >= 80 and mental_health >= 6):
        return "⚡ Balanced Student"
    elif (study_hours < 2 or attendance < 70 or mental_health < 4):
        return "💪 Needs Support"
    else:
        return "📊 Average Performer"

def display_countup_score(prediction):
    """Display score with countup animation"""
    color = glow_color(prediction)
    feedback = feedback_text(prediction)
    placeholder = st.empty()
    step = max(1, int(prediction / 50))
    
    for i in range(0, int(prediction) + 1, step):
        html_content = f"""
        <div style='text-align:center; padding:30px; margin-top:10px; border-radius:15px;
                    background: linear-gradient(135deg, #1f1f2e, #2e2e3e);
                    box-shadow: 0 0 20px {color}, 0 0 40px {color};
                    color:white; position:relative; min-height:120px'>
            <h1 style='color:{color}; text-shadow: 0 0 15px {color}, 0 0 30px {color}; font-size:48px; font-weight:bold'>
                {i:.0f}%
            </h1>
            <h3 style='color:{color}; margin-top:10px;'>{feedback}</h3>
        </div>
        """
        placeholder.markdown(html_content, unsafe_allow_html=True)
        time.sleep(0.02)
    
    # Mark as displayed
    st.session_state["displayed_prediction"] = True

def display_static_score(prediction):
    """Display static score (for already shown predictions)"""
    color = glow_color(prediction)
    feedback = feedback_text(prediction)
    st.markdown(f"""
    <div style='text-align:center; padding:30px; margin-top:10px; border-radius:15px;
                background: linear-gradient(135deg, #1f1f2e, #2e2e3e);
                box-shadow: 0 0 20px {color}, 0 0 40px {color};
                color:white; position:relative; min-height:120px'>
        <h1 style='color:{color}; text-shadow: 0 0 15px {color}, 0 0 30px {color}; font-size:48px; font-weight:bold'>
            {prediction:.1f}%
        </h1>
        <h3 style='color:{color}; margin-top:10px;'>{feedback}</h3>
    </div>
    """, unsafe_allow_html=True)

def create_progress_chart(score):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = score,
        number = {'suffix': "%", 'font': {'size': 40}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': glow_color(score), 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 75, 75, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(255, 165, 0, 0.2)'},
                {'range': [75, 85], 'color': 'rgba(76, 175, 80, 0.2)'},
                {'range': [85, 95], 'color': 'rgba(33, 150, 243, 0.2)'},
                {'range': [95, 100], 'color': 'rgba(156, 39, 176, 0.2)'}],
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"}
    )
    return fig

def add_timestamp_to_history():
    if st.session_state["history"]:
        st.session_state["history"][-1]["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["history"][-1]["Study Profile"] = get_study_profile(
            st.session_state["history"][-1]["Study Hours"],
            st.session_state["history"][-1]["Attendance"],
            st.session_state["history"][-1]["Mental Health"],
            st.session_state["history"][-1]["Sleep Hours"],
            st.session_state["history"][-1]["Part-time Job"]
        )
        # Save to persistent storage after adding new entry
        save_history()

def show_simple_analysis(idx):
    """Show simple and understandable analysis for a specific history entry"""
    if idx is None or idx >= len(st.session_state["history"]):
        return
    
    row = st.session_state["history"][idx]
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
        <h2 style='color: white; text-align: center; margin: 0;'>🔍 Quick Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Score Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Your Score Breakdown")
        
        # Simple factor analysis
        factors = [
            ("📚 Study Hours", row['Study Hours'], "3-5h optimal"),
            ("🏫 Attendance", row['Attendance'], "85%+ excellent"),
            ("🧠 Mental Health", row['Mental Health'], "7+ strong"),
            ("💤 Sleep Hours", row['Sleep Hours'], "7-8h best"),
            ("💼 Part-time Job", row['Part-time Job'], "No preferred")
        ]
        
        for factor, value, optimal in factors:
            if factor == "💼 Part-time Job":
                status = "🟢 Good" if value == "No" else "🟡 Okay"
            elif factor == "📚 Study Hours":
                status = "🟢 Good" if 3 <= value <= 5 else "🟡 Okay" if 2 <= value < 3 else "🔴 Improve"
            elif factor == "🏫 Attendance":
                status = "🟢 Good" if value >= 85 else "🟡 Okay" if value >= 75 else "🔴 Improve"
            elif factor == "🧠 Mental Health":
                status = "🟢 Good" if value >= 7 else "🟡 Okay" if value >= 5 else "🔴 Improve"
            elif factor == "💤 Sleep Hours":
                status = "🟢 Good" if 7 <= value <= 8 else "🟡 Okay" if 6 <= value <= 9 else "🔴 Improve"
            
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin: 5px 0;'>
                <div style='display: flex; justify-content: between; align-items: center;'>
                    <span style='color: white; font-weight: bold;'>{factor}</span>
                    <span style='color: white;'>{value}</span>
                    <span>{status}</span>
                </div>
                <div style='color: #888; font-size: 12px;'>{optimal}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Score Display
        score_color = glow_color(row['Predicted Score'])
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px;'>
            <div style='color: {score_color}; font-size: 48px; font-weight: bold;'>
                {row['Predicted Score']:.1f}%
            </div>
            <div style='color: {score_color}; font-size: 16px;'>
                {feedback_text(row['Predicted Score'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Study Profile
        profile = get_study_profile(row['Study Hours'], row['Attendance'], row['Mental Health'], row['Sleep Hours'], row['Part-time Job'])
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 10px; margin-top: 10px;'>
            <div style='color: #00b4d8; font-weight: bold;'>{profile}</div>
        </div>
        """, unsafe_allow_html=True)

    # Quick Tips
    st.markdown("### 💡 Quick Tips to Improve")
    tips = get_study_tips(row['Predicted Score'], row['Study Hours'], row['Attendance'], 
                         row['Mental Health'], row['Sleep Hours'], row['Part-time Job'])
    
    for tip in tips[:3]:  # Show only top 3 tips
        st.markdown(f"• {tip}")

    # What's Working Well
    st.markdown("### ✅ What's Working Well")
    good_points = []
    if row['Study Hours'] >= 3:
        good_points.append(f"Good study routine ({row['Study Hours']}h daily)")
    if row['Attendance'] >= 80:
        good_points.append(f"Solid attendance ({row['Attendance']}%)")
    if row['Mental Health'] >= 6:
        good_points.append(f"Decent mental health ({row['Mental Health']}/10)")
    if 7 <= row['Sleep Hours'] <= 8:
        good_points.append(f"Good sleep habits ({row['Sleep Hours']}h)")
    if row['Part-time Job'] == "No":
        good_points.append("No part-time job distraction")
    
    if good_points:
        for point in good_points[:3]:  # Show only top 3 good points
            st.markdown(f"• {point}")
    else:
        st.markdown("• Keep working on building good habits!")

    # Close Analysis Button
    if st.button("🔙 Back to History", use_container_width=True):
        st.session_state["analyze_config"] = None
        st.rerun()

def show_predictions_table():
    """Show predictions with delete option"""
    for idx, row in enumerate(st.session_state["history"]):
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
            
            with col1:
                profile = row.get('Study Profile', get_study_profile(
                    row['Study Hours'], row['Attendance'], row['Mental Health'], 
                    row['Sleep Hours'], row['Part-time Job']
                ))
                st.markdown(f"""
                <div style='padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px; margin: 5px 0;'>
                    <div style='color: #00b4d8; font-weight: bold;'>{profile}</div>
                    <div style='color: white; font-size: 12px;'>
                        📚 {row['Study Hours']}h • 🏫 {row['Attendance']}% • 🧠 {row['Mental Health']}/10
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                score_color = glow_color(row['Predicted Score'])
                st.markdown(f"""
                <div style='text-align: center; padding: 10px;'>
                    <div style='color: {score_color}; font-weight: bold; font-size: 18px;'>
                        {row['Predicted Score']:.1f}%
                    </div>
                    <div style='color: #888; font-size: 12px;'>{feedback_text(row['Predicted Score'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                is_favorite = idx in st.session_state["favorites"]
                button_text = "🌟" if is_favorite else "⭐"
                if st.button(button_text, key=f"fav_{idx}", help="Add to favorites"):
                    if is_favorite:
                        st.session_state["favorites"].remove(idx)
                    else:
                        st.session_state["favorites"].append(idx)
                    save_history()
                    st.rerun()
            
            with col4:
                if st.button("📊", key=f"analyze_{idx}", help="View detailed analysis"):
                    st.session_state["analyze_config"] = idx
                    st.rerun()
            
            with col5:
                if st.button("🗑️", key=f"delete_{idx}", help="Delete this entry"):
                    # Remove from history
                    st.session_state["history"].pop(idx)
                    # Remove from favorites if present
                    if idx in st.session_state["favorites"]:
                        st.session_state["favorites"].remove(idx)
                    # Adjust other favorites indices
                    st.session_state["favorites"] = [
                        fav_idx if fav_idx < idx else fav_idx - 1 
                        for fav_idx in st.session_state["favorites"]
                    ]
                    save_history()
                    st.rerun()

def show_favorites_section():
    if st.session_state["favorites"]:
        st.markdown("### ⭐ Favorite Study Profiles")
        for fav_idx, idx in enumerate(st.session_state["favorites"]):
            if idx < len(st.session_state["history"]):
                row = st.session_state["history"][idx]
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.markdown(f"""
                        <div style='padding: 15px; background: rgba(255,215,0,0.1); border-radius: 10px; margin: 10px 0; border: 2px solid #FFD700;'>
                            <div style='color: #FFD700; font-weight: bold;'>{get_study_profile(row['Study Hours'], row['Attendance'], row['Mental Health'], row['Sleep Hours'], row['Part-time Job'])}</div>
                            <div style='color: white; font-size: 14px;'>
                                📚 {row['Study Hours']}h • 🏫 {row['Attendance']}% • 🧠 {row['Mental Health']}/10
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        score_color = glow_color(row['Predicted Score'])
                        st.markdown(f"""
                        <div style='text-align: center; padding: 15px;'>
                            <div style='color: {score_color}; font-weight: bold; font-size: 20px;'>
                                {row['Predicted Score']:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        if st.button("🔄 Use", key=f"use_fav_{fav_idx}", help="Load this profile"):
                            st.session_state["study_hours"] = row['Study Hours']
                            st.session_state["attendance"] = row['Attendance']
                            st.session_state["mental_health"] = row['Mental Health']
                            st.session_state["sleep_hours"] = row['Sleep Hours']
                            st.session_state["part_time_job"] = row['Part-time Job']
                            st.session_state["prediction_made"] = False
                            st.session_state["prediction"] = None
                            st.session_state["displayed_prediction"] = False
                            st.rerun()
    else:
        st.info("⭐ No favorites yet. Click the star icon to add profiles to favorites.")

def show_comparison_tool():
    """Enhanced comparison tool with better insights"""
    if len(st.session_state["history"]) >= 2:
        st.markdown("### 🔍 Compare Study Profiles")
        
        # Get profile options
        profile_options = []
        for idx, row in enumerate(st.session_state["history"]):
            profile = get_study_profile(row['Study Hours'], row['Attendance'], row['Mental Health'], row['Sleep Hours'], row['Part-time Job'])
            profile_options.append(f"Profile {idx+1}: {row['Predicted Score']:.1f}% ({profile})")
        
        col1, col2 = st.columns(2)
        with col1:
            profile1_idx = st.selectbox("Select first profile", range(len(profile_options)), format_func=lambda x: profile_options[x])
        with col2:
            profile2_idx = st.selectbox("Select second profile", range(len(profile_options)), format_func=lambda x: profile_options[x], index=min(1, len(profile_options)-1))
        
        if profile1_idx != profile2_idx:
            row1 = st.session_state["history"][profile1_idx]
            row2 = st.session_state["history"][profile2_idx]
            
            # Simple comparison table
            comparison_data = {
                'Factor': ['Study Hours', 'Attendance', 'Mental Health', 'Sleep', 'Part-time Job', 'Score'],
                f'Profile {profile1_idx+1}': [
                    f"{row1['Study Hours']}h", 
                    f"{row1['Attendance']}%", 
                    f"{row1['Mental Health']}/10", 
                    f"{row1['Sleep Hours']}h", 
                    row1['Part-time Job'], 
                    f"{row1['Predicted Score']:.1f}%"
                ],
                f'Profile {profile2_idx+1}': [
                    f"{row2['Study Hours']}h", 
                    f"{row2['Attendance']}%", 
                    f"{row2['Mental Health']}/10", 
                    f"{row2['Sleep Hours']}h", 
                    row2['Part-time Job'], 
                    f"{row2['Predicted Score']:.1f}%"
                ]
            }
            
            # Create DataFrame and display
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Simple Comparison Summary
            st.markdown("### 📊 Quick Comparison")
            
            score_diff = row2['Predicted Score'] - row1['Predicted Score']
            
            if score_diff > 5:
                st.success(f"**Profile {profile2_idx+1} scores {score_diff:.1f}% higher**")
                st.info(f"**Key differences:** Profile {profile2_idx+1} has better study habits or attendance")
            elif score_diff < -5:
                st.warning(f"**Profile {profile2_idx+1} scores {abs(score_diff):.1f}% lower**")
                st.info(f"**Key differences:** Profile {profile1_idx+1} has better factors")
            else:
                st.info("**Similar performance** - minor differences in study habits")
    else:
        st.warning("📋 Need at least 2 study profiles to compare. Make more predictions to enable comparison.")

def show_empty_history():
    st.info("📭 No predictions yet. Make your first prediction to see history here!")

def show_dashboard(study_hours, attendance, mental_health, sleep_hours, part_time_job):
    dashboard_html = ""
    inputs = [
        ("📚 Study Hours", f"{study_hours}h"),
        ("🏫 Attendance", f"{attendance}%"),
        ("🧠 Mental Health", f"{mental_health}/10"),
        ("💤 Sleep Hours", f"{sleep_hours}h"),
        ("💼 Part-time Job", part_time_job)
    ]
    for label, value in inputs:
        dashboard_html += f"""
        <div style='background:#2e2e3e; border-radius:12px; padding:10px; margin-bottom:8px;
                    box-shadow: 0 0 10px #00b4d8, 0 0 15px #00b4d8; text-align:center;'>
            <h5 style='color:#00b4d8; margin:0'>{label}</h5>
            <p style='font-size:18px; color:white; font-weight:bold; margin:0'>{value}</p>
        </div>
        """
    st.markdown(dashboard_html, unsafe_allow_html=True)

def show_enhanced_history_section():
    if st.session_state["history"]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: white; text-align: center; margin: 0;'>📜 Prediction History</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Show data info
        if "last_saved" in st.session_state.get("history", [{}])[-1]:
            last_entry = st.session_state["history"][-1]
            if "Timestamp" in last_entry:
                st.caption(f"📅 Last prediction: {last_entry['Timestamp']} • Total predictions: {len(st.session_state['history'])}")
        
        tab1, tab2, tab3 = st.tabs(["📋 All Predictions", "⭐ Favorites", "🔍 Compare"])
        
        with tab1:
            show_predictions_table()
        with tab2:
            show_favorites_section()
        with tab3:
            show_comparison_tool()
    else:
        show_empty_history()

# ----------------- MAIN APP ----------------- #
st.markdown("<h1 style='text-align:center; color:#00b4d8;'>📊 Student Exam Score Predictor</h1>", unsafe_allow_html=True)

# Check if we should show analysis
if st.session_state["analyze_config"] is not None:
    show_simple_analysis(st.session_state["analyze_config"])
else:
    col_inputs, col_dashboard = st.columns([2,1])

    with col_inputs:
        st.markdown("<h3 style='color:#00b4d8;'>📝 Student Profile</h3>", unsafe_allow_html=True)
        
        # Initialize session state with defaults if not exists
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
        
        # Get current values from session state
        study_hours = st.slider("📚 Study Hours per day", 0.0, 12.0, float(st.session_state["study_hours"]), 0.5)
        attendance = st.slider("🏫 Attendance (%)", 0, 100, int(st.session_state["attendance"]), 1)
        mental_health = st.slider("🧠 Mental Health (1-10)", 1, 10, int(st.session_state["mental_health"]), 1)
        sleep_hours = st.slider("💤 Sleep Hours per night", 0.0, 12.0, float(st.session_state["sleep_hours"]), 0.5)
        
        part_time_index = 0 if st.session_state["part_time_job"] == "No" else 1
        part_time_job = st.selectbox("💼 Do you have a Part-time Job?", ["No","Yes"], index=part_time_index)
        
        # Update session state
        st.session_state["study_hours"] = study_hours
        st.session_state["attendance"] = attendance
        st.session_state["mental_health"] = mental_health
        st.session_state["sleep_hours"] = sleep_hours
        st.session_state["part_time_job"] = part_time_job
        
        part_time_binary = 1 if part_time_job=="Yes" else 0

        # Buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not st.session_state["prediction_made"]:
                if st.button("🎯 Predict Score", use_container_width=True, type="primary"):
                    with st.spinner("🔮 Predicting your score..."):
                        time.sleep(1)
                        input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, part_time_binary]])
                        prediction = model.predict(input_data)[0]
                        prediction = max(0, min(100, round(prediction, 1)))
                        st.session_state["prediction"] = prediction
                        st.session_state["prediction_made"] = True
                        st.session_state["displayed_prediction"] = False
                        
                        # Add to history
                        st.session_state["history"].append({
                            "Study Hours": study_hours,
                            "Attendance": attendance,
                            "Mental Health": mental_health,
                            "Sleep Hours": sleep_hours,
                            "Part-time Job": part_time_job,
                            "Predicted Score": prediction
                        })
                        add_timestamp_to_history()
                        
                        # Trigger rerun to show prediction
                        st.rerun()
            else:
                if st.button("🔄 Predict Again", use_container_width=True):
                    # Reset to defaults
                    for k, v in defaults.items():
                        st.session_state[k] = v
                    st.session_state["prediction"] = None
                    st.session_state["prediction_made"] = False
                    st.session_state["show_history"] = False
                    st.session_state["analyze_config"] = None
                    st.session_state["displayed_prediction"] = False
                    st.rerun()
        
        with col2:
            history_text = "📜 Hide History" if st.session_state["show_history"] else "📜 View History"
            if st.button(history_text, use_container_width=True):
                st.session_state["show_history"] = not st.session_state["show_history"]
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state["history"] = []
                st.session_state["favorites"] = []
                st.session_state["displayed_prediction"] = False
                save_history()
                st.rerun()

    with col_dashboard:
        st.markdown("<h3 style='color:#00b4d8;'>📊 Dashboard</h3>", unsafe_allow_html=True)
        show_dashboard(study_hours, attendance, mental_health, sleep_hours, part_time_job)

    # ----------------- PREDICTION DISPLAY ----------------- #
    if st.session_state["prediction_made"] and st.session_state["prediction"] is not None:
        prediction = st.session_state["prediction"]
        
        # Show animation only on first display
        if not st.session_state["displayed_prediction"]:
            display_countup_score(prediction)
            st.balloons()
        else:
            display_static_score(prediction)
        
        st.markdown("---")
        st.markdown("<h3 style='color:#00b4d8;'>💡 Quick Tips</h3>", unsafe_allow_html=True)
        tips = get_study_tips(prediction, study_hours, attendance, mental_health, sleep_hours, part_time_job)
        for tip in tips:
            st.markdown(f"- {tip}")

    # ----------------- HISTORY DISPLAY ----------------- #
    if st.session_state["show_history"]:
        show_enhanced_history_section()

# ----------------- FOOTER ----------------- #
st.divider()
st.markdown("""
<div style='text-align:center; color:#888;'>
    <p>👨‍💻 Developed by Ujwal | 📊 Student Performance Predictor</p>
    <p style='font-size: 12px;'>💾 History is automatically saved and will persist between sessions</p>
</div>
""", unsafe_allow_html=True)
