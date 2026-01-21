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
import base64

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

def save_history_to_query_params():
    """Save history to URL query parameters (limited storage)"""
    try:
        if st.session_state["history"]:
            # Convert history to JSON and then to base64 for URL safety
            history_data = {
                "h": st.session_state["history"],
                "f": st.session_state["favorites"],
                "t": datetime.now().isoformat()
            }
            history_json = json.dumps(history_data)
            history_b64 = base64.b64encode(history_json.encode()).decode()
            
            # Set query parameters
            st.query_params["data"] = history_b64
        return True
    except Exception as e:
        # Silently fail for query params - it's just a backup
        return False

def load_history_from_query_params():
    """Load history from URL query parameters"""
    try:
        if "data" in st.query_params:
            history_b64 = st.query_params["data"]
            history_json = base64.b64decode(history_b64).decode()
            history_data = json.loads(history_json)
            
            st.session_state["history"] = history_data.get("h", [])
            st.session_state["favorites"] = history_data.get("f", [])
            return True
    except Exception as e:
        # Silently fail for query params - it's just a backup
        pass
    return False

def save_history():
    """Save history using multiple methods for persistence"""
    # Save to file (primary method)
    file_success = save_history_to_file()
    
    # Also save to query params as backup
    param_success = save_history_to_query_params()
    
    return file_success or param_success

def load_history():
    """Load history using multiple methods"""
    # Try file first
    if load_history_from_file():
        return True
    
    # If file doesn't exist, try query params
    if load_history_from_query_params():
        # If loaded from params, save to file for future
        save_history_to_file()
        return True
    
    return False

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

# ----------------- SESSION STATE INITIALIZATION ----------------- #
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
if "persistent_loaded" not in st.session_state:
    st.session_state["persistent_loaded"] = False
if "displayed_prediction" not in st.session_state:
    st.session_state["displayed_prediction"] = False
if "show_balloons" not in st.session_state:
    st.session_state["show_balloons"] = False

# Load persistent data on first run
if not st.session_state["persistent_loaded"]:
    if load_history():
        st.success("📁 Loaded previous session data!")
    st.session_state["persistent_loaded"] = True

# ----------------- DEFAULT VALUES ----------------- #
defaults = {
    "student_name": "Student",
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

def input_glow_color(value, input_type):
    """Return glow colors based on input quality"""
    if input_type == "study_hours":
        if value >= 4: return "#4caf50"  # Green - optimal
        elif value >= 2: return "#ffa500"  # Orange - moderate
        else: return "#ff4b4b"  # Red - low
    
    elif input_type == "attendance":
        if value >= 90: return "#4caf50"  # Green - excellent
        elif value >= 75: return "#ffa500"  # Orange - good
        else: return "#ff4b4b"  # Red - poor
    
    elif input_type == "mental_health":
        if value >= 8: return "#4caf50"  # Green - strong
        elif value >= 6: return "#ffa500"  # Orange - moderate
        else: return "#ff4b4b"  # Red - weak
    
    elif input_type == "sleep_hours":
        if 7 <= value <= 8: return "#4caf50"  # Green - optimal
        elif 6 <= value <= 9: return "#ffa500"  # Orange - moderate
        else: return "#ff4b4b"  # Red - poor
    
    elif input_type == "part_time_job":
        return "#4caf50" if value == "No" else "#ffa500"  # Green for No, Orange for Yes
    
    return "#00b4d8"  # Default blue

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

def display_countup_score(prediction, student_name):
    """Display score with countup animation only once"""
    if st.session_state["displayed_prediction"]:
        # If already displayed, show static version
        color = glow_color(prediction)
        feedback = feedback_text(prediction)
        st.markdown(f"""
        <div style='text-align:center; padding:30px; margin-top:10px; border-radius:15px;
                    background: linear-gradient(135deg, #1f1f2e, #2e2e3e);
                    box-shadow: 0 0 20px {color}, 0 0 40px {color};
                    color:white; position:relative; min-height:120px'>
            <h3 style='color:#00b4d8; margin-bottom:10px;'>🎓 {student_name}'s Predicted Score</h3>
            <h1 style='color:{color}; text-shadow: 0 0 15px {color}, 0 0 30px {color}; font-size:48px; font-weight:bold'>
                {prediction:.1f}%
            </h1>
            <h3 style='color:{color}; margin-top:10px;'>{feedback}</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        # First time display - show animation
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
                <h3 style='color:#00b4d8; margin-bottom:10px;'>🎓 {student_name}'s Predicted Score</h3>
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

def create_performance_chart(row):
    """Create a radar chart for student performance"""
    categories = ['Study Hours', 'Attendance', 'Mental Health', 'Sleep Quality']
    
    # Normalize values for radar chart (0-10 scale)
    study_norm = min(row['Study Hours'] * 2, 10)  # 5h = 10 points
    attendance_norm = row['Attendance'] / 10       # 100% = 10 points
    mental_norm = row['Mental Health']             # Already 1-10 scale
    sleep_norm = 10 if 7 <= row['Sleep Hours'] <= 8 else (
                 8 if 6 <= row['Sleep Hours'] <= 9 else 5)  # Optimal sleep gets 10
    
    values = [study_norm, attendance_norm, mental_norm, sleep_norm]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Performance',
        line=dict(color='#00b4d8'),
        fillcolor='rgba(0, 180, 216, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_factor_impact_chart(row):
    """Create a bar chart showing impact of each factor"""
    factors = ['Study Hours', 'Attendance', 'Mental Health', 'Sleep Hours', 'Part-time Job']
    
    # Calculate individual impacts (simplified version)
    study_impact = min(row['Study Hours'] * 8, 40)  # Max 40 points
    attendance_impact = row['Attendance'] * 0.4     # Max 40 points
    mental_impact = row['Mental Health'] * 4        # Max 40 points
    sleep_impact = 20 if 7 <= row['Sleep Hours'] <= 8 else (
                  10 if 6 <= row['Sleep Hours'] <= 9 else 0)  # Max 20 points
    job_impact = 10 if row['Part-time Job'] == 'No' else -10  # ±10 points
    
    impacts = [study_impact, attendance_impact, mental_impact, sleep_impact, job_impact]
    colors = ['#4caf50' if x > 0 else '#ff4b4b' for x in impacts]
    
    fig = go.Figure(data=[
        go.Bar(
            x=factors,
            y=impacts,
            marker_color=colors,
            text=[f"+{x}" if x > 0 else str(x) for x in impacts],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Factor Impact on Score",
        xaxis_title="Factors",
        yaxis_title="Impact Points",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(tickangle=-45)
    )
    
    return fig

def show_simple_analysis(idx):
    """Show simple and understandable analysis for a specific history entry"""
    if idx is None or idx >= len(st.session_state["history"]):
        return
    
    row = st.session_state["history"][idx]
    student_name = row.get('Student Name', 'Student')
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
        <h2 style='color: white; text-align: center; margin: 0;'>🔍 Detailed Analysis: {student_name}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Student Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Score Display
        score_color = glow_color(row['Predicted Score'])
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #00b4d8; margin-bottom: 10px;'>🎓 {student_name}</h3>
            <div style='color: {score_color}; font-size: 48px; font-weight: bold;'>
                {row['Predicted Score']:.1f}%
            </div>
            <div style='color: {score_color}; font-size: 16px;'>
                {feedback_text(row['Predicted Score'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance Radar Chart
        st.markdown("### 📊 Performance Radar")
        radar_fig = create_performance_chart(row)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Factor Impact Chart
        st.markdown("### 📈 Factor Impact Analysis")
        impact_fig = create_factor_impact_chart(row)
        st.plotly_chart(impact_fig, use_container_width=True)
    
    with col2:
        # Study Profile
        profile = get_study_profile(row['Study Hours'], row['Attendance'], row['Mental Health'], row['Sleep Hours'], row['Part-time Job'])
        profile_color = "#4caf50" if "High" in profile else "#ffa500" if "Balanced" in profile else "#ff4b4b" if "Support" in profile else "#2196f3"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; margin-bottom: 20px; border: 2px solid {profile_color}'>
            <h4 style='color: {profile_color}; margin-bottom: 10px;'>Study Profile</h4>
            <div style='color: {profile_color}; font-weight: bold; font-size: 20px;'>{profile}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Stats
        st.markdown("### 📋 Quick Stats")
        stats = [
            ("Overall Score", f"{row['Predicted Score']:.1f}%"),
            ("Study Hours", f"{row['Study Hours']}h"),
            ("Attendance", f"{row['Attendance']}%"),
            ("Mental Health", f"{row['Mental Health']}/10"),
            ("Sleep Hours", f"{row['Sleep Hours']}h"),
            ("Part-time Job", row['Part-time Job'])
        ]
        
        for stat, value in stats:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px; margin: 5px 0; text-align: center;'>
                <div style='color: #00b4d8; font-size: 12px;'>{stat}</div>
                <div style='color: white; font-weight: bold; font-size: 16px;'>{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # Improvement Recommendations
    st.markdown("### 💡 Improvement Recommendations")
    tips = get_study_tips(row['Predicted Score'], row['Study Hours'], row['Attendance'], 
                         row['Mental Health'], row['Sleep Hours'], row['Part-time Job'])
    
    for i, tip in enumerate(tips[:4], 1):
        st.markdown(f"{i}. {tip}")

    # Performance Trends (if multiple entries for same student)
    same_student_entries = [h for h in st.session_state["history"] if h.get('Student Name') == student_name]
    if len(same_student_entries) > 1:
        st.markdown("### 📈 Performance Trend")
        
        # Create trend data
        trend_data = []
        for entry in same_student_entries:
            if 'Timestamp' in entry:
                trend_data.append({
                    'Date': entry['Timestamp'],
                    'Score': entry['Predicted Score'],
                    'Study Hours': entry['Study Hours']
                })
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            trend_df['Date'] = pd.to_datetime(trend_df['Date'])
            trend_df = trend_df.sort_values('Date')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df['Date'], 
                y=trend_df['Score'],
                mode='lines+markers',
                name='Score Trend',
                line=dict(color='#00b4d8', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Score Progress Over Time",
                xaxis_title="Date",
                yaxis_title="Predicted Score (%)",
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # Close Analysis Button
    if st.button("🔙 Back to History", use_container_width=True):
        st.session_state["analyze_config"] = None
        st.session_state["displayed_prediction"] = False
        st.rerun()

def show_predictions_table():
    """Show predictions with delete option"""
    for idx, row in enumerate(st.session_state["history"]):
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
            
            with col1:
                student_name = row.get('Student Name', 'Student')
                profile = row.get('Study Profile', get_study_profile(
                    row['Study Hours'], row['Attendance'], row['Mental Health'], 
                    row['Sleep Hours'], row['Part-time Job']
                ))
                st.markdown(f"""
                <div style='padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; margin: 5px 0;'>
                    <div style='color: #00b4d8; font-weight: bold; font-size: 16px;'>🎓 {student_name}</div>
                    <div style='color: #00b4d8; font-weight: bold;'>{profile}</div>
                    <div style='color: white; font-size: 12px;'>
                        📚 {row['Study Hours']}h • 🏫 {row['Attendance']}% • 🧠 {row['Mental Health']}/10
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                score_color = glow_color(row['Predicted Score'])
                st.markdown(f"""
                <div style='text-align: center; padding: 12px;'>
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
                    st.session_state["displayed_prediction"] = False
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
        st.markdown("### ⭐ Favorite Student Profiles")
        for fav_idx, idx in enumerate(st.session_state["favorites"]):
            if idx < len(st.session_state["history"]):
                row = st.session_state["history"][idx]
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        student_name = row.get('Student Name', 'Student')
                        st.markdown(f"""
                        <div style='padding: 15px; background: rgba(255,215,0,0.1); border-radius: 10px; margin: 10px 0; border: 2px solid #FFD700;'>
                            <div style='color: #FFD700; font-weight: bold; font-size: 16px;'>🎓 {student_name}</div>
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
                        if st.button("🗑️", key=f"delete_fav_{fav_idx}", help="Remove from favorites"):
                            st.session_state["favorites"].remove(idx)
                            save_history()
                            st.rerun()
    else:
        st.info("⭐ No favorites yet. Click the star icon to add student profiles to favorites.")

def show_comparison_tool():
    """Enhanced comparison tool with better insights"""
    if len(st.session_state["history"]) >= 2:
        st.markdown("### 🔍 Compare Student Performance")
        
        # Get student names for selection
        student_options = []
        for idx, row in enumerate(st.session_state["history"]):
            student_name = row.get('Student Name', f'Student {idx+1}')
            score = row['Predicted Score']
            profile = get_study_profile(row['Study Hours'], row['Attendance'], row['Mental Health'], row['Sleep Hours'], row['Part-time Job'])
            student_options.append((idx, f"{student_name} - {score:.1f}% ({profile})"))
        
        col1, col2 = st.columns(2)
        with col1:
            profile1_idx = st.selectbox(
                "Select first student", 
                range(len(student_options)),
                format_func=lambda x: student_options[x][1],
                key="comp1"
            )
        with col2:
            profile2_idx = st.selectbox(
                "Select second student", 
                range(len(student_options)),
                format_func=lambda x: student_options[x][1],
                index=min(1, len(student_options)-1),
                key="comp2"
            )
        
        if profile1_idx != profile2_idx:
            row1 = st.session_state["history"][profile1_idx]
            row2 = st.session_state["history"][profile2_idx]
            student1_name = row1.get('Student Name', 'Student 1')
            student2_name = row2.get('Student Name', 'Student 2')
            
            # Comparison Visualization
            st.markdown("### 📊 Comparison Chart")
            
            # Create comparison bar chart
            categories = ['Study Hours', 'Attendance', 'Mental Health', 'Sleep Hours', 'Predicted Score']
            student1_values = [row1['Study Hours'], row1['Attendance'], row1['Mental Health'], row1['Sleep Hours'], row1['Predicted Score']]
            student2_values = [row2['Study Hours'], row2['Attendance'], row2['Mental Health'], row2['Sleep Hours'], row2['Predicted Score']]
            
            fig = go.Figure(data=[
                go.Bar(name=student1_name, x=categories, y=student1_values, marker_color='#00b4d8'),
                go.Bar(name=student2_name, x=categories, y=student2_values, marker_color='#ffa500')
            ])
            
            fig.update_layout(
                barmode='group',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced Comparison Insights
            st.markdown("### 💡 Performance Insights")
            score_diff = row2['Predicted Score'] - row1['Predicted Score']
            
            if abs(score_diff) > 10:
                if score_diff > 0:
                    st.success(f"**Significant Advantage**\n\n{student2_name} performs **{score_diff:.1f}% better** than {student1_name}")
                else:
                    st.error(f"**Significant Gap**\n\n{student2_name} scores **{abs(score_diff):.1f}% lower** than {student1_name}")
            elif abs(score_diff) > 5:
                if score_diff > 0:
                    st.info(f"**Noticeable Difference**\n\n{student2_name} is **{score_diff:.1f}% ahead** of {student1_name}")
                else:
                    st.warning(f"**Moderate Difference**\n\n{student2_name} is **{abs(score_diff):.1f}% behind** {student1_name}")
            else:
                st.success(f"**Close Competition**\n\nBoth students show similar performance levels")
                    
    else:
        st.warning("📋 Need at least 2 student profiles to compare. Make more predictions to enable comparison.")

def show_empty_history():
    st.info("📭 No predictions yet. Make your first prediction to see history here!")

def show_dashboard(student_name, study_hours, attendance, mental_health, sleep_hours, part_time_job):
    """Display the current student inputs with glowing effects in medium size"""
    
    # Create a container for the dashboard
    with st.container():
        # Student Info Card with glow - Medium size
        student_glow = input_glow_color(study_hours, "study_hours")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1f1f2e, #2e2e3e); 
                    padding: 18px; border-radius: 15px; margin-bottom: 15px; 
                    box-shadow: 0 0 18px {student_glow}, 0 0 35px {student_glow};
                    text-align: center; border: 2px solid {student_glow};'>
            <h3 style='color: {student_glow}; margin: 0; text-shadow: 0 0 10px {student_glow}; font-size: 20px;'>🎓 Student Dashboard</h3>
            <h2 style='color: white; margin: 8px 0; font-size: 22px;'>{student_name}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Medium grid layout - 2 columns, comfortable spacing
        col1, col2 = st.columns(2)
        
        with col1:
            # Study Hours - Medium size
            study_glow = input_glow_color(study_hours, "study_hours")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1f1f2e, #2e2e3e); 
                        padding: 16px; border-radius: 12px; margin-bottom: 12px;
                        box-shadow: 0 0 15px {study_glow}, 0 0 30px {study_glow};
                        border: 2px solid {study_glow}; text-align: center; height: 110px; display: flex; flex-direction: column; justify-content: center;'>
                <div style='color: {study_glow}; margin: 0; text-shadow: 0 0 8px {study_glow}; font-size: 16px; font-weight: bold;'>📚 Study Hours</div>
                <div style='color: white; margin: 5px 0; font-size: 24px; font-weight: bold;'>{study_hours}h</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Attendance - Medium size
            attendance_glow = input_glow_color(attendance, "attendance")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1f1f2e, #2e2e3e); 
                        padding: 16px; border-radius: 12px; margin-bottom: 12px;
                        box-shadow: 0 0 15px {attendance_glow}, 0 0 30px {attendance_glow};
                        border: 2px solid {attendance_glow}; text-align: center; height: 110px; display: flex; flex-direction: column; justify-content: center;'>
                <div style='color: {attendance_glow}; margin: 0; text-shadow: 0 0 8px {attendance_glow}; font-size: 16px; font-weight: bold;'>🏫 Attendance</div>
                <div style='color: white; margin: 5px 0; font-size: 24px; font-weight: bold;'>{attendance}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Mental Health - Medium size
            mental_glow = input_glow_color(mental_health, "mental_health")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1f1f2e, #2e2e3e); 
                        padding: 16px; border-radius: 12px; margin-bottom: 12px;
                        box-shadow: 0 0 15px {mental_glow}, 0 0 30px {mental_glow};
                        border: 2px solid {mental_glow}; text-align: center; height: 110px; display: flex; flex-direction: column; justify-content: center;'>
                <div style='color: {mental_glow}; margin: 0; text-shadow: 0 0 8px {mental_glow}; font-size: 16px; font-weight: bold;'>🧠 Mental Health</div>
                <div style='color: white; margin: 5px 0; font-size: 24px; font-weight: bold;'>{mental_health}/10</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sleep Hours - Medium size
            sleep_glow = input_glow_color(sleep_hours, "sleep_hours")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1f1f2e, #2e2e3e); 
                        padding: 16px; border-radius: 12px; margin-bottom: 12px;
                        box-shadow: 0 0 15px {sleep_glow}, 0 0 30px {sleep_glow};
                        border: 2px solid {sleep_glow}; text-align: center; height: 110px; display: flex; flex-direction: column; justify-content: center;'>
                <div style='color: {sleep_glow}; margin: 0; text-shadow: 0 0 8px {sleep_glow}; font-size: 16px; font-weight: bold;'>💤 Sleep Hours</div>
                <div style='color: white; margin: 5px 0; font-size: 24px; font-weight: bold;'>{sleep_hours}h</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Part-time Job - Medium size single row
        job_glow = input_glow_color(part_time_job, "part_time_job")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1f1f2e, #2e2e3e); 
                    padding: 16px; border-radius: 12px; margin-top: 8px;
                    box-shadow: 0 0 15px {job_glow}, 0 0 30px {job_glow};
                    border: 2px solid {job_glow}; text-align: center;'>
            <div style='color: {job_glow}; margin: 0; text-shadow: 0 0 8px {job_glow}; font-size: 16px; font-weight: bold;'>💼 Part-time Job</div>
            <div style='color: white; margin: 5px 0; font-size: 22px; font-weight: bold;'>{part_time_job}</div>
        </div>
        """, unsafe_allow_html=True)

def show_enhanced_history_section():
    if st.session_state["history"]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: white; text-align: center; margin: 0;'>📜 Student Prediction History</h2>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📋 All Students", "⭐ Favorites", "🔍 Compare Students"])
        
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
    # Balanced column ratios for medium dashboard
    col_inputs, col_dashboard = st.columns([2, 1])

    with col_inputs:
        st.markdown("<h3 style='color:#00b4d8;'>📝 Student Profile</h3>", unsafe_allow_html=True)
        
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
        
        # Student Name Input
        student_name = st.text_input("🎓 Student Name", value=st.session_state["student_name"], 
                                   placeholder="Enter student name")
        
        study_hours = st.slider("📚 Study Hours per day", 0.0, 12.0, float(st.session_state["study_hours"]), 0.5)
        attendance = st.slider("🏫 Attendance (%)", 0, 100, int(st.session_state["attendance"]), 1)
        mental_health = st.slider("🧠 Mental Health (1-10)", 1, 10, int(st.session_state["mental_health"]), 1)
        sleep_hours = st.slider("💤 Sleep Hours per night", 0.0, 12.0, float(st.session_state["sleep_hours"]), 0.5)
        
        part_time_index = 0 if st.session_state["part_time_job"] == "No" else 1
        part_time_job = st.selectbox("💼 Do you have a Part-time Job?", ["No","Yes"], index=part_time_index)
        
        st.session_state["student_name"] = student_name
        st.session_state["study_hours"] = study_hours
        st.session_state["attendance"] = attendance
        st.session_state["mental_health"] = mental_health
        st.session_state["sleep_hours"] = sleep_hours
        st.session_state["part_time_job"] = part_time_job
        
        part_time_binary = 1 if part_time_job=="Yes" else 0

        col1, col2, col3 = st.columns(3)
        
        # Prediction button - always show when no prediction is made
        if not st.session_state["prediction_made"]:
            if col1.button("🎯 Predict Score", use_container_width=True, type="primary"):
                with st.spinner("🔮 Predicting score..."):
                    time.sleep(1)
                    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, part_time_binary]])
                    prediction = model.predict(input_data)[0]
                    prediction = max(0, min(100, round(prediction, 1)))
                    st.session_state["prediction"] = prediction
                    st.session_state["prediction_made"] = True
                    st.session_state["displayed_prediction"] = False
                    
                    # Add to history
                    st.session_state["history"].append({
                        "Student Name": student_name,
                        "Study Hours": study_hours,
                        "Attendance": attendance,
                        "Mental Health": mental_health,
                        "Sleep Hours": sleep_hours,
                        "Part-time Job": part_time_job,
                        "Predicted Score": prediction
                    })
                    add_timestamp_to_history()
                    
                    # Trigger balloons for this prediction
                    st.session_state["show_balloons"] = True
                    st.rerun()
        else:
            if col1.button("🔄 Predict Again", use_container_width=True):
                for k, v in defaults.items():
                    st.session_state[k] = v
                st.session_state["prediction"] = None
                st.session_state["prediction_made"] = False
                st.session_state["show_history"] = False
                st.session_state["analyze_config"] = None
                st.session_state["displayed_prediction"] = False
                st.session_state["show_balloons"] = False
                st.rerun()
        
        # Always show history button
        history_text = "📜 Hide History" if st.session_state["show_history"] else "📜 View History"
        if col2.button(history_text, use_container_width=True):
            st.session_state["show_history"] = not st.session_state["show_history"]
            st.rerun()
        
        # Clear history button
        if col3.button("🗑️ Clear All History", use_container_width=True):
            st.session_state["history"] = []
            st.session_state["favorites"] = []
            st.session_state["displayed_prediction"] = False
            st.session_state["show_balloons"] = False
            save_history()
            st.rerun()

    with col_dashboard:
        show_dashboard(student_name, study_hours, attendance, mental_health, sleep_hours, part_time_job)

    # ----------------- PREDICTION DISPLAY ----------------- #
    if st.session_state["prediction_made"] and st.session_state["prediction"] is not None:
        prediction = st.session_state["prediction"]
        display_countup_score(prediction, student_name)
        
        # Show balloons only when triggered by prediction button
        if st.session_state["show_balloons"]:
            st.balloons()
            st.session_state["show_balloons"] = False  # Reset after showing
        
        st.markdown("---")
        st.markdown(f"<h3 style='color:#00b4d8;'>💡 Personalized Tips for {student_name}</h3>", unsafe_allow_html=True)
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
    <p>👨‍💻 Developed by Karthikeya PV | 📊 Student Performance Predictor</p>
    <p><b>Disclaimer:</b> This tool provides estimated exam scores based on input factors and is for educational purposes only. Actual scores may vary.</p>
    <p> made using linear regression model.</p>
    
</div>
""", unsafe_allow_html=True)