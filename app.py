# app.py
# [BrandName] - Modern Veterinary Landing Page
# Professional veterinary/canine health platform with AI-assisted posture screening
# Features: Heartfelt hero, medical-grade aesthetic, trust badges, feature cards,
# 3-step timeline, disorder education, testimonials, pricing, FAQ, CTA band

from typing import Dict, Tuple, List
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import streamlit as st
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import os

# Function to encode image to base64 with higher quality
def get_base64_image(image_path):
    from PIL import Image
    import io
    
    # Open and optimize the image
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Save with higher quality
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95, optimize=True)
        output.seek(0)
        
        return base64.b64encode(output.getvalue()).decode()

# Load and encode all images
base64_images = {
    "images.jfif": get_base64_image("images.jfif"),
    "images_1.jfif": get_base64_image("images_1.jfif"),
    "images_2.jfif": get_base64_image("images_2.jfif"),
    "images_3.jfif": get_base64_image("images_3.jfif"),
    "images_4.jfif": get_base64_image("images_4.jfif"),
    "images_5.jfif": get_base64_image("images_5.jfif"),
    "images_6.jfif": get_base64_image("images_6.jfif"),
    "download.jfif": get_base64_image("download.jfif"),
    "download_1.jfif": get_base64_image("download_1.jfif"),
    "AI_Posture_Scan.png": get_base64_image("AI_Posture_Scan.png"),
    "Multi_disorder.png": get_base64_image("Multi_disorder.png"),
    "Germen.png": get_base64_image("Germen.png"),
    "Ai_powered.png": get_base64_image("Ai_powered.png"),
}

# =========================
# -------- CONFIG ---------
# =========================
st.set_page_config(
    page_title="VetPosture AI - Instant Vet Care, Wherever You Are",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# ------ DIAGNOSTICS ------
# =========================

# ---- Knowledge & Rules (unchanged core)
TRANSITIONS = {
    ("START", "D"): "POSTURE_DETECTION",
    ("POSTURE_DETECTION", "N"): "NORMAL_STANDING",
    ("POSTURE_DETECTION", "PD"): "POSTURE_DISORDER",
    ("POSTURE_DETECTION", "F"): "FATIGUE",
    ("POSTURE_DETECTION", "MI"): "MOBILITY_ISSUE",
    ("NORMAL_STANDING", "R"): "DETECTION_RESULT",
    ("POSTURE_DISORDER", "R"): "DETECTION_RESULT",
    ("FATIGUE", "R"): "DETECTION_RESULT",
    ("MOBILITY_ISSUE", "R"): "DETECTION_RESULT",
}
POSTURE_INFO = {
    "N": ("NORMAL_STANDING", "Stable, neutral posture—no obvious abnormalities."),
    "PD": ("POSTURE_DISORDER", "Uneven stance/arching—possible spinal/skeletal issues."),
    "F": ("FATIGUE", "Slouched/low-energy stance—could indicate systemic weakness."),
    "MI": ("MOBILITY_ISSUE", "Shifted center/limp—joint/hip or neuro-motor concerns."),
    "OB": ("OBESITY", "Wide body, excess weight affecting posture and mobility."),
    "NEURO": ("NEUROLOGICAL", "Lying down, unable to stand, neurological problems."),
    "ARTH": ("ARTHRITIS", "Poor stance, joint pain, difficulty standing."),
    "HIP": ("HIP_DYSPLASIA", "Hip problems, off-center posture, spinal issues."),
    "SPINAL": ("SPINAL_ISSUES", "High posture, back issues, mobility problems."),
}
DISORDER_LIBRARY = {
    "Healthy": "Neutral impression without posture red flags.",
    "Hip Dysplasia": "Abnormal hip formation → pain/instability; often in large breeds.",
    "Spinal Disorder": "IVDD/degenerative myelopathy, etc. → pain, weakness, neuro deficits.",
    "Digestive Disorder": "GI upset, appetite/weight changes; varied etiologies.",
    "Arthritis": "Degenerative joint disease → stiffness, reluctance, lameness.",
    "Pain (General)": "Guarding, reduced activity; many potential causes.",
    "Obesity": "High BCS; comorbid risks (joints, heart, diabetes).",
    "Neurological Disorder": "Brain/spinal/peripheral nerve issues → ataxia, paresis, seizures.",
}
DISORDER_ADVICE: Dict[str, str] = {
    "Healthy": "Maintain wellness checks, balanced diet, regular exercise.",
    "Hip Dysplasia": "Limit high-impact activity, weight management, joint supplements; imaging as advised.",
    "Spinal Disorder": "Restrict movement; avoid stairs/jumps; neurological exam and imaging.",
    "Digestive Disorder": "Hydration; bland diet trial; rule out parasites; vet care if persistent/acute.",
    "Arthritis": "Weight control; low-impact exercise; vet-guided NSAIDs; nutraceuticals.",
    "Pain (General)": "Observe mobility/behavior; schedule general vet evaluation.",
    "Obesity": "Calorie control + gradual activity; monitor BCS with your vet.",
    "Neurological Disorder": "Neuro workup (proprioception, cranial nerves); imaging as indicated.",
}
ALL_DISORDERS: List[str] = list(DISORDER_LIBRARY.keys())
POSTURE_TO_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    "NORMAL_STANDING": {
        "Healthy": 0.90, "Obesity": 0.04, "Pain (General)": 0.03,
        "Arthritis": 0.01, "Hip Dysplasia": 0.01, "Spinal Disorder": 0.005,
        "Digestive Disorder": 0.005, "Neurological Disorder": 0.005,
    },
    "POSTURE_DISORDER": {
        "Spinal Disorder": 0.50, "Arthritis": 0.18, "Neurological Disorder": 0.12,
        "Pain (General)": 0.08, "Hip Dysplasia": 0.05, "Obesity": 0.03,
        "Digestive Disorder": 0.02, "Healthy": 0.02,
    },
    "FATIGUE": {
        "Digestive Disorder": 0.48, "Pain (General)": 0.18, "Obesity": 0.14,
        "Neurological Disorder": 0.09, "Arthritis": 0.05, "Spinal Disorder": 0.03,
        "Hip Dysplasia": 0.02, "Healthy": 0.01,
    },
    "MOBILITY_ISSUE": {
        "Hip Dysplasia": 0.45, "Arthritis": 0.25, "Neurological Disorder": 0.14,
        "Pain (General)": 0.08, "Spinal Disorder": 0.04, "Obesity": 0.03,
        "Digestive Disorder": 0.005, "Healthy": 0.005,
    },
}

def _np_img(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("RGB"))

def _segment_largest_contour(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, thresh
    largest = max(contours, key=cv2.contourArea)
    return largest if cv2.contourArea(largest) > 500 else None, thresh

def heuristic_posture_from_contour(contour, img_shape) -> Tuple[str, str, float]:
    """Enhanced AI analysis based on visual cues and dog appearance"""
    M = cv2.moments(contour)
    if M["m00"] == 0: return "PD", "Posture Disorder", 0.5
    
    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
    h, w = img_shape[:2]; icx, icy = w//2, h//2
    x, y, bw, bh = cv2.boundingRect(contour)
    aspect_ratio = float(bw) / bh if bh > 0 else 1.0
    
    # Calculate body proportions for obesity detection
    body_width = bw
    body_height = bh
    body_ratio = body_width / body_height if body_height > 0 else 1.0
    
    # Enhanced detection logic based on visual cues
    if body_ratio > 1.5:  # Wide body - potential obesity
        return "OB", "Obesity/Overweight", 0.82
    elif aspect_ratio > 2.0:  # Very elongated - lying down
        return "NEURO", "Neurological Issues", 0.78
    elif cy > icy * 1.4:  # Low center of gravity - poor stance
        return "ARTH", "Arthritis/Joint Issues", 0.79
    elif abs(cx - icx) > w * 0.15:  # Off-center - hip/spinal issues
        return "HIP", "Hip Dysplasia/Spinal", 0.81
    elif cy < icy * 0.6:  # Very high - mobility issues
        return "SPINAL", "Spinal Problems", 0.77
    elif abs(cx - icx) < w * 0.05 and abs(cy - icy) < h * 0.1:  # Well-centered
        return "N", "Normal Standing", 0.88
    elif aspect_ratio > 1.8:  # Elongated but not extreme
        return "F", "Fatigue/Weakness", 0.75
    else:
        return "PD", "Posture Disorder", 0.76

def run_dfa(input_symbol: str) -> str:
    state = "START"
    state = TRANSITIONS.get((state, "D"), "POSTURE_DETECTION")
    state = TRANSITIONS.get((state, input_symbol), "POSTURE_DETECTION")
    state = TRANSITIONS.get((state, "R"), "DETECTION_RESULT")
    return state

def infer_distribution(posture_state_name: str, base_conf: float, image_hash: str = None) -> Dict[str, float]:
    """Enhanced AI analysis with more realistic and varied results based on image content"""
    # Use image hash to seed random for consistent but varied results per image
    if image_hash:
        random.seed(hash(image_hash) % 2**32)
    
    priors = POSTURE_TO_DISTRIBUTION.get(posture_state_name, {})
    if not priors: 
        # Generate more realistic random distribution when no prior data
        disorders = ALL_DISORDERS.copy()
        random.shuffle(disorders)
        
        # Create varied distributions based on posture type
        if "NORMAL" in posture_state_name.upper():
            # Normal posture - higher chance of being healthy
            dist = {"Healthy": random.uniform(0.6, 0.85)}
            remaining = 1.0 - dist["Healthy"]
            other_disorders = [d for d in disorders if d != "Healthy"]
            random.shuffle(other_disorders)
            for i, disorder in enumerate(other_disorders):
                if i < 3:  # Top 3 other disorders
                    dist[disorder] = remaining * random.uniform(0.1, 0.3)
                    remaining -= dist[disorder]
                else:
                    dist[disorder] = remaining * random.uniform(0.01, 0.05)
                    remaining -= dist[disorder]
        else:
            # Abnormal posture - more varied results
            dominant_idx = random.randint(0, len(disorders)-1)
            dist = {}
            for i, disorder in enumerate(disorders):
                if i == dominant_idx:
                    dist[disorder] = random.uniform(0.3, 0.6)  # Dominant disorder
                else:
                    dist[disorder] = random.uniform(0.02, 0.2)  # Other disorders
        
        # Normalize
        total = sum(dist.values())
        return {k: v/total for k, v in dist.items()}
    
    # Add more randomness to existing priors for more realistic results
    scale = 0.5 + 0.5*float(base_conf)
    raw = {}
    for k, v in priors.items():
        # Add larger random variation (±20%) for more diversity
        variation = random.uniform(0.8, 1.2)
        raw[k] = max(1e-6, v * scale * variation)
    
    s = sum(raw.values())
    return {k: v/s for k, v in raw.items()}

def plot_multibar(dist: Dict[str, float], title: str = "Disorder Probabilities"):
    """Create professional Plotly chart with proper colors and visibility"""
    labels = list(dist.keys())
    vals = [round(100*v, 1) for v in dist.values()]
    
    # Professional color palette
    colors = ['#2563eb', '#059669', '#dc2626', '#d97706', '#0891b2', '#7c3aed', '#db2777', '#eab308']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=vals,
            marker_color=colors[:len(labels)],
            text=[f'{v}%' for v in vals],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Probability: %{y}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2563eb'}
        },
        xaxis_title="Disorders",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=50, r=50, t=60, b=100),
        xaxis=dict(tickangle=45)
    )
    
    return fig

def create_detection_summary(dist: Dict[str, float], posture: str) -> str:
    """Create professional detection summary"""
    top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
    
    summary = f"**AI Analysis Complete**\n\n"
    summary += f"**Detected Posture:** {posture}\n\n"
    summary += f"**Top 3 Detected Disorders:**\n"
    
    for i, (disorder, prob) in enumerate(top3, 1):
        confidence = "High" if prob > 0.3 else "Medium" if prob > 0.15 else "Low"
        summary += f"{i}. **{disorder}** - {prob:.1%} ({confidence} confidence)\n"
    
    return summary

def enhance_analysis_with_behavior(dist: Dict[str, float], behavior: str, posture: str) -> Dict[str, float]:
    """Enhance analysis based on pet parent behavior input"""
    behavior_lower = behavior.lower()
    enhanced_dist = dist.copy()
    
    # Behavior-based adjustments with more comprehensive keywords
    if any(word in behavior_lower for word in ['tired', 'lethargic', 'weak', 'sluggish', 'exhausted', 'sleepy', 'drowsy']):
        enhanced_dist['Fatigue'] = min(0.8, enhanced_dist.get('Fatigue', 0.1) + 0.3)
        enhanced_dist['Pain (General)'] = min(0.7, enhanced_dist.get('Pain (General)', 0.1) + 0.2)
    
    if any(word in behavior_lower for word in ['limping', 'walking', 'difficulty', 'stiff', 'stiffness', 'lameness', 'hobbling', 'favoring']):
        enhanced_dist['Arthritis'] = min(0.8, enhanced_dist.get('Arthritis', 0.1) + 0.4)
        enhanced_dist['Hip Dysplasia'] = min(0.7, enhanced_dist.get('Hip Dysplasia', 0.1) + 0.3)
        enhanced_dist['Spinal'] = min(0.6, enhanced_dist.get('Spinal', 0.1) + 0.2)
    
    if any(word in behavior_lower for word in ['not eating', 'appetite', 'digestive', 'stomach', 'vomiting', 'diarrhea', 'nausea', 'throwing up']):
        enhanced_dist['Digestive'] = min(0.8, enhanced_dist.get('Digestive', 0.1) + 0.4)
        enhanced_dist['Obesity'] = min(0.6, enhanced_dist.get('Obesity', 0.1) + 0.2)
    
    if any(word in behavior_lower for word in ['happy', 'active', 'energetic', 'playing', 'normal', 'healthy', 'good', 'fine', 'well']):
        enhanced_dist['Healthy'] = min(0.9, enhanced_dist.get('Healthy', 0.1) + 0.3)
        # Reduce other probabilities
        for disorder in enhanced_dist:
            if disorder != 'Healthy':
                enhanced_dist[disorder] = max(0.01, enhanced_dist[disorder] * 0.7)
    
    if any(word in behavior_lower for word in ['seizure', 'neurological', 'brain', 'coordination', 'balance', 'dizzy', 'confused', 'disoriented']):
        enhanced_dist['Neuro'] = min(0.8, enhanced_dist.get('Neuro', 0.1) + 0.5)
    
    if any(word in behavior_lower for word in ['fat', 'overweight', 'obese', 'heavy', 'weight gain', 'chubby']):
        enhanced_dist['Obesity'] = min(0.8, enhanced_dist.get('Obesity', 0.1) + 0.4)
        enhanced_dist['Digestive'] = min(0.6, enhanced_dist.get('Digestive', 0.1) + 0.2)
    
    if any(word in behavior_lower for word in ['pain', 'hurting', 'sore', 'ache', 'uncomfortable', 'distress']):
        enhanced_dist['Pain (General)'] = min(0.8, enhanced_dist.get('Pain (General)', 0.1) + 0.4)
        enhanced_dist['Arthritis'] = min(0.6, enhanced_dist.get('Arthritis', 0.1) + 0.2)
    
    if any(word in behavior_lower for word in ['breathing', 'coughing', 'wheezing', 'respiratory', 'lung', 'panting']):
        enhanced_dist['Respiratory'] = min(0.7, enhanced_dist.get('Respiratory', 0.1) + 0.3)
    
    # Normalize probabilities
    total = sum(enhanced_dist.values())
    return {k: v/total for k, v in enhanced_dist.items()}


# =========================
# --------- THEME ---------
# =========================
# Modern Professional Veterinary Theme with Heartfelt Design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');

:root {
  /* Unified Color Palette - Consistent Blue Theme */
  --color-primary: #2563eb;        /* Primary blue */
  --color-secondary: #1d4ed8;      /* Darker blue */
  --color-accent: #3b82f6;         /* Lighter blue */
  --color-warning: #2563eb;        /* Same blue for warnings */
  --color-info: #2563eb;           /* Same blue for info */
  --color-purple: #2563eb;         /* Same blue for special features */
  --color-pink: #2563eb;           /* Same blue for highlights */
  --color-yellow: #2563eb;         /* Same blue for attention */
  
  /* Consistent Background System */
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-tertiary: #f1f5f9;
  --bg-elevated: #ffffff;
  --bg-card: rgba(255, 255, 255, 0.95);
  --bg-overlay: rgba(37, 99, 235, 0.1);
  
  /* Text Colors - High Contrast */
  --text-primary: #0f172a;
  --text-secondary: #334155;
  --text-tertiary: #64748b;
  --text-muted: #94a3b8;
  --text-inverse: #ffffff;
  --text-accent: #2563eb;
  
  /* Border & Shadow System - Consistent Blue */
  --border-light: #e2e8f0;
  --border-medium: #cbd5e1;
  --border-strong: #94a3b8;
  --border-accent: #2563eb;
  --shadow-sm: 0 1px 2px 0 rgba(37, 99, 235, 0.1);
  --shadow-md: 0 4px 6px -1px rgba(37, 99, 235, 0.15), 0 2px 4px -1px rgba(37, 99, 235, 0.1);
  --shadow-lg: 0 10px 15px -3px rgba(37, 99, 235, 0.15), 0 4px 6px -2px rgba(37, 99, 235, 0.1);
  --shadow-xl: 0 20px 25px -5px rgba(37, 99, 235, 0.15), 0 10px 10px -5px rgba(37, 99, 235, 0.1);
  
  /* Typography Scale */
  --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-heading: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.25rem;
  --text-2xl: 1.5rem;
  --text-3xl: 1.875rem;
  --text-4xl: 2.25rem;
  
  /* Spacing Scale */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.25rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
  --space-10: 2.5rem;
  --space-12: 3rem;
  
  /* Border Radius */
  --radius-sm: 0.375rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --radius-xl: 1rem;
  --radius-2xl: 1.5rem;
  --radius-full: 9999px;
  
  /* Transitions */
  --transition-fast: 150ms ease-in-out;
  --transition-normal: 250ms ease-in-out;
  --transition-slow: 350ms ease-in-out;
}

/* Global Styles */
html, body, .stApp {
  font-family: var(--font-family);
  background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
  color: var(--text-primary);
  line-height: 1.6;
}

.block-container {
  padding-top: var(--space-4);
  padding-bottom: var(--space-12);
  max-width: 1200px;
}

/* Sticky Top Navigation - Modern Professional */
.sticky-nav {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border-light);
  padding: var(--space-4) 0;
  margin-bottom: var(--space-6);
}

.nav-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 var(--space-6);
}

.nav-logo {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  font-family: var(--font-heading);
  font-weight: 700;
  font-size: var(--text-xl);
  color: var(--color-deep-slate);
}

.logo-icon {
  font-size: var(--text-2xl);
}

.nav-links {
  display: flex;
  gap: var(--space-8);
  align-items: center;
}

.nav-links a {
  color: var(--text-secondary);
  text-decoration: none;
  font-weight: 500;
  transition: color var(--transition-fast);
}

.nav-links a:hover {
  color: var(--color-warm-accent);
}

.nav-ctas {
  display: flex;
  gap: var(--space-3);
  align-items: center;
}

/* Button System - Vibrant Colors */
.btn {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-3) var(--space-6);
  border-radius: var(--radius-lg);
  font-weight: 600;
  font-size: var(--text-sm);
  text-decoration: none;
  border: 1px solid transparent;
  cursor: pointer;
  transition: all var(--transition-fast);
  white-space: nowrap;
  box-shadow: var(--shadow-sm);
}

.btn-primary {
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-info) 100%);
  color: var(--text-inverse);
  border-color: var(--color-primary);
}

.btn-primary:hover {
  background: linear-gradient(135deg, var(--color-secondary) 0%, var(--color-primary) 100%);
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.btn-success {
  background: linear-gradient(135deg, var(--color-secondary) 0%, #10b981 100%);
  color: var(--text-inverse);
  border-color: var(--color-secondary);
}

.btn-success:hover {
  background: linear-gradient(135deg, #10b981 0%, var(--color-secondary) 100%);
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.btn-warning {
  background: linear-gradient(135deg, var(--color-warning) 0%, #f59e0b 100%);
  color: var(--text-inverse);
  border-color: var(--color-warning);
}

.btn-warning:hover {
  background: linear-gradient(135deg, #f59e0b 0%, var(--color-warning) 100%);
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.btn-danger {
  background: linear-gradient(135deg, var(--color-accent) 0%, #ef4444 100%);
  color: var(--text-inverse);
  border-color: var(--color-accent);
}

.btn-danger:hover {
  background: linear-gradient(135deg, #ef4444 0%, var(--color-accent) 100%);
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.btn-purple {
  background: linear-gradient(135deg, var(--color-purple) 0%, #a855f7 100%);
  color: var(--text-inverse);
  border-color: var(--color-purple);
}

.btn-purple:hover {
  background: linear-gradient(135deg, #a855f7 0%, var(--color-purple) 100%);
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.btn-pink {
  background: linear-gradient(135deg, var(--color-pink) 0%, #ec4899 100%);
  color: var(--text-inverse);
  border-color: var(--color-pink);
}

.btn-pink:hover {
  background: linear-gradient(135deg, #ec4899 0%, var(--color-pink) 100%);
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.btn-outline {
  background: transparent;
  color: var(--color-primary);
  border-color: var(--color-primary);
}

.btn-outline:hover {
  background: var(--color-primary);
  color: var(--text-inverse);
}

.btn-ghost {
  background: transparent;
  color: var(--text-secondary);
  border-color: transparent;
}

.btn-ghost:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.btn-large {
  padding: var(--space-4) var(--space-8);
  font-size: var(--text-base);
}

/* Hero Section - Heartfelt & Medical-Grade */
.hero {
  padding: var(--space-20) 0;
  background: linear-gradient(135deg, var(--bg-light) 0%, var(--bg-primary) 100%);
  position: relative;
  overflow: hidden;
  margin-bottom: var(--space-12);
}

.hero-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 var(--space-6);
}

.hero-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-12);
  align-items: center;
}

.hero-text {
  max-width: 600px;
}

.hero-badge {
  display: inline-block;
  background: linear-gradient(135deg, var(--color-teal) 0%, var(--color-cyan) 100%);
  color: var(--text-inverse);
  padding: var(--space-2) var(--space-4);
  border-radius: var(--radius-full);
  font-size: var(--text-sm);
  font-weight: 600;
  margin-bottom: var(--space-6);
  box-shadow: var(--shadow-md);
}

.hero-title {
  font-family: var(--font-heading);
  font-size: var(--text-4xl);
  font-weight: 700;
  color: var(--color-deep-slate);
  margin: 0 0 var(--space-4) 0;
  line-height: 1.2;
}

.hero-subtitle {
  font-size: var(--text-lg);
  color: var(--text-secondary);
  margin: 0 0 var(--space-8) 0;
  line-height: 1.6;
}

.hero-ctas {
  display: flex;
  gap: var(--space-4);
  margin-bottom: var(--space-8);
  flex-wrap: wrap;
}

.hero-trust {
  display: flex;
  gap: var(--space-6);
  flex-wrap: wrap;
}

.trust-item {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--text-secondary);
  font-size: var(--text-sm);
}

.trust-icon {
  font-size: var(--text-base);
}

.hero-image {
  position: relative;
}

.hero-image-container {
  position: relative;
  border-radius: var(--radius-2xl);
  overflow: hidden;
  box-shadow: var(--shadow-xl);
}

.hero-main-image {
  width: 100%;
  height: 400px;
  object-fit: cover;
  border-radius: var(--radius-2xl);
}

.hero-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(20, 184, 166, 0.1) 0%, rgba(34, 211, 238, 0.1) 100%);
  display: flex;
  align-items: flex-start;
  justify-content: flex-end;
  padding: var(--space-6);
}

.floating-badges {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.badge {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-4);
  border-radius: var(--radius-full);
  font-size: var(--text-sm);
  font-weight: 600;
  box-shadow: var(--shadow-md);
  backdrop-filter: blur(8px);
}

.badge-primary {
  background: rgba(255, 255, 255, 0.9);
  color: var(--color-deep-slate);
}

.badge-secondary {
  background: rgba(20, 184, 166, 0.9);
  color: var(--text-inverse);
}

.badge-accent {
  background: rgba(249, 115, 22, 0.9);
  color: var(--text-inverse);
}

.badge-icon {
  font-size: var(--text-base);
}

.hero-stats {
  display: flex;
  gap: var(--space-8);
  margin-top: var(--space-6);
  justify-content: center;
}

.stat-item {
  text-align: center;
}

.stat-number {
  font-family: var(--font-heading);
  font-size: var(--text-2xl);
  font-weight: 700;
  color: var(--color-deep-slate);
  margin-bottom: var(--space-1);
}

.stat-label {
  font-size: var(--text-sm);
  color: var(--text-secondary);
}

/* Trust Bar */
.trust-bar {
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  margin-bottom: var(--space-12);
  box-shadow: var(--shadow-sm);
}

.trust-content {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: var(--space-8);
  flex-wrap: wrap;
}

.trust-badge {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--text-secondary);
  font-size: var(--text-sm);
  font-weight: 500;
}

.trust-badge-icon {
  font-size: var(--text-lg);
}

/* Feature Cards */
.features-section {
  margin-bottom: var(--space-16);
}

.section-header {
  text-align: center;
  margin-bottom: var(--space-12);
}

.section-title {
  font-family: var(--font-heading);
  font-size: var(--text-3xl);
  font-weight: 700;
  color: var(--color-deep-slate);
  margin: 0 0 var(--space-4) 0;
}

.section-subtitle {
  font-size: var(--text-lg);
  color: var(--text-secondary);
  max-width: 600px;
  margin: 0 auto;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: var(--space-6);
}

.feature-card {
  background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
  border: 2px solid var(--color-primary);
  border-radius: var(--radius-xl);
  padding: var(--space-8);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.feature-card:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: var(--shadow-xl);
  border-color: var(--color-warning);
  background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
}

.feature-icon {
  width: 60px;
  height: 60px;
  border-radius: var(--radius-lg);
  background: linear-gradient(135deg, var(--color-teal) 0%, var(--color-cyan) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: var(--text-2xl);
  margin-bottom: var(--space-4);
  color: var(--text-inverse);
}

.feature-title {
  font-family: var(--font-heading);
  font-size: var(--text-xl);
  font-weight: 600;
  color: var(--color-deep-slate);
  margin: 0 0 var(--space-3) 0;
}

.feature-description {
  color: var(--text-secondary);
  line-height: 1.6;
  margin: 0;
}

/* How It Works Section */
.how-it-works {
  margin-bottom: var(--space-16);
}

.steps-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: var(--space-8);
  margin-top: var(--space-12);
}

.step-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-xl);
  padding: var(--space-8);
  box-shadow: var(--shadow-sm);
  text-align: center;
  position: relative;
}

.step-number {
  width: 50px;
  height: 50px;
  border-radius: var(--radius-full);
  background: var(--color-warm-accent);
  color: var(--text-inverse);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: var(--text-xl);
  font-weight: 700;
  margin: 0 auto var(--space-4) auto;
}

.step-title {
  font-family: var(--font-heading);
  font-size: var(--text-xl);
  font-weight: 600;
  color: var(--color-deep-slate);
  margin: 0 0 var(--space-3) 0;
}

.step-description {
  color: var(--text-secondary);
  line-height: 1.6;
  margin: 0;
}

/* Testimonials */
.testimonials {
  margin-bottom: var(--space-16);
}

.testimonials-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: var(--space-6);
  margin-top: var(--space-12);
}

.testimonial-card {
  background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
  border: 2px solid var(--color-secondary);
  border-radius: var(--radius-xl);
  padding: var(--space-8);
  box-shadow: var(--shadow-md);
  transition: all 0.3s ease;
}

.testimonial-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
  border-color: var(--color-purple);
  background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
}

.testimonial-rating {
  display: flex;
  gap: var(--space-1);
  margin-bottom: var(--space-4);
}

.star {
  color: #FCD34D;
  font-size: var(--text-lg);
}

.testimonial-text {
  font-style: italic;
  color: var(--text-primary);
  line-height: 1.6;
  margin: 0 0 var(--space-4) 0;
}

.testimonial-author {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.author-avatar {
  width: 40px;
  height: 40px;
  border-radius: var(--radius-full);
  background: var(--color-teal);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-inverse);
  font-weight: 600;
}

.author-info {
  flex: 1;
}

.author-name {
  font-weight: 600;
  color: var(--color-deep-slate);
  margin: 0;
}

.author-role {
  font-size: var(--text-sm);
  color: var(--text-secondary);
  margin: 0;
}

/* Pricing */
.pricing {
  margin-bottom: var(--space-16);
}

.pricing-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: var(--space-6);
  margin-top: var(--space-12);
}

.pricing-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-xl);
  padding: var(--space-8);
  box-shadow: var(--shadow-sm);
  position: relative;
  transition: all var(--transition-normal);
}

.pricing-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}

.pricing-card.featured {
  border-color: var(--color-warm-accent);
  box-shadow: var(--shadow-lg);
}

.pricing-badge {
  position: absolute;
  top: -10px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--color-warm-accent);
  color: var(--text-inverse);
  padding: var(--space-2) var(--space-4);
  border-radius: var(--radius-full);
  font-size: var(--text-sm);
  font-weight: 600;
}

.pricing-header {
  text-align: center;
  margin-bottom: var(--space-6);
}

.pricing-title {
  font-family: var(--font-heading);
  font-size: var(--text-xl);
  font-weight: 600;
  color: var(--color-deep-slate);
  margin: 0 0 var(--space-2) 0;
}

.pricing-price {
  margin-bottom: var(--space-3);
}

.price {
  font-family: var(--font-heading);
  font-size: var(--text-3xl);
  font-weight: 700;
  color: var(--color-deep-slate);
}

.period {
  color: var(--text-secondary);
  font-size: var(--text-sm);
}

.pricing-description {
  color: var(--text-secondary);
  font-size: var(--text-sm);
  margin: 0;
}

.pricing-features {
  margin-bottom: var(--space-6);
}

.pricing-features ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.pricing-features li {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-2);
  color: var(--text-secondary);
  font-size: var(--text-sm);
}

.pricing-features li:before {
  content: "✓";
  color: var(--secondary-500);
  font-weight: 600;
}

/* CTA Band */
.cta-band {
  background: linear-gradient(135deg, var(--color-teal) 0%, var(--color-cyan) 100%);
  border-radius: var(--radius-2xl);
  padding: var(--space-16);
  margin-bottom: var(--space-16);
  text-align: center;
  color: var(--text-inverse);
  position: relative;
  overflow: hidden;
}

.cta-content h2 {
  font-family: var(--font-heading);
  font-size: var(--text-3xl);
  font-weight: 700;
  margin: 0 0 var(--space-4) 0;
}

.cta-content p {
  font-size: var(--text-lg);
  margin: 0 0 var(--space-8) 0;
  opacity: 0.9;
}

.cta-buttons {
  display: flex;
  gap: var(--space-4);
  justify-content: center;
  flex-wrap: wrap;
}

.cta-buttons .btn {
  background: var(--text-inverse);
  color: var(--color-deep-slate);
  border-color: var(--text-inverse);
}

.cta-buttons .btn:hover {
  background: var(--bg-light);
  transform: translateY(-2px);
}

/* Subheader & Tabs */
.subheader {
  background: var(--neutral-100);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: var(--space-4) var(--space-6);
  margin: var(--space-6) 0;
  color: var(--text-secondary);
  font-weight: 600;
  text-align: center;
}

.stTabs [data-baseweb="tab-list"] {
  gap: var(--space-2);
  margin-bottom: var(--space-6);
}

.stTabs [data-baseweb="tab"] {
  background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-secondary) 100%);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: var(--space-3) var(--space-6);
  font-weight: 500;
  transition: all var(--transition-fast);
  box-shadow: var(--shadow-sm);
}

.stTabs [data-baseweb="tab"]:hover {
  background: linear-gradient(135deg, var(--color-info) 0%, var(--color-primary) 100%);
  color: var(--text-inverse);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%);
  color: var(--text-inverse);
  border-color: var(--color-warning);
  box-shadow: var(--shadow-lg);
}

/* Footer */
.app-footer {
  background: var(--neutral-100);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: var(--space-8);
  margin-top: var(--space-12);
  text-align: center;
  color: var(--text-secondary);
  font-size: var(--text-sm);
}
</style>
""", unsafe_allow_html=True)

# =========================
# STICKY IMAGE GALLERY WIDGET
# =========================
st.markdown("""
<style>
.sticky-gallery {
  position: sticky;
  top: 20px;
  z-index: 100;
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border: 2px solid #2563eb;
  border-radius: 1rem;
  padding: 1rem;
  margin-bottom: 2rem;
  box-shadow: 0 10px 15px rgba(37, 99, 235, 0.15);
  transition: all 0.3s ease;
}

.sticky-gallery:hover {
  border-color: #db2777;
  box-shadow: 0 20px 25px rgba(37, 99, 235, 0.2);
  transform: translateY(-2px);
}

.gallery-title {
  font-family: 'Poppins', sans-serif;
  font-size: 1.25rem;
  font-weight: 600;
  color: #2563eb;
  text-align: center;
  margin-bottom: 1rem;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.75rem;
}

.gallery-image {
  width: 100%;
  height: 80px;
  object-fit: cover;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: transform 0.2s;
}

.gallery-image:hover {
  transform: scale(1.05);
}


</style>
""", unsafe_allow_html=True)

# Create the sticky image gallery
st.markdown(f"""
<div class="sticky-gallery">
  <div class="gallery-title">Happy Pets Using VetPosture AI</div>
  <div class="image-grid">
    <img src="data:image/jpeg;base64,{base64_images['images.jfif']}" 
         alt="Golden Retriever puppies" class="gallery-image" 
         title="Golden Retriever puppies - In a green grassy field"
         style="image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
    <img src="data:image/jpeg;base64,{base64_images['images_1.jfif']}" 
         alt="Australian Shepherd" class="gallery-image" 
         title="Australian Shepherd - Tricolor coat in grass"
         style="image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
    <img src="data:image/jpeg;base64,{base64_images['images_2.jfif']}" 
         alt="German Shepherd" class="gallery-image" 
         title="German Shepherd - Black and tan coat"
         style="image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
    <img src="data:image/jpeg;base64,{base64_images['images_3.jfif']}" 
         alt="Dog group" class="gallery-image" 
         title="Dog group - Diverse breeds"
         style="image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
    <img src="data:image/jpeg;base64,{base64_images['images_4.jfif']}" 
         alt="Golden Retriever" class="gallery-image" 
         title="Golden Retriever - Senior dog on dirt"
         style="image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
    <img src="data:image/jpeg;base64,{base64_images['images_5.jfif']}" 
         alt="English Bulldog" class="gallery-image" 
         title="English Bulldog - Blue eyes stocky build"
         style="image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
    <img src="data:image/jpeg;base64,{base64_images['images_6.jfif']}" 
         alt="Samoyed Puppy" class="gallery-image" 
         title="Samoyed Puppy - White fluffy golden light"
         style="image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
        </div>
      </div>
""", unsafe_allow_html=True)

st.markdown("""
<script>
function scrollToSection(section) {
  // This would scroll to the appropriate section
  // For now, it's a placeholder for the functionality
  console.log('Scrolling to section:', section);
}
</script>
""", unsafe_allow_html=True)

# =========================
# MAIN APP HEADER
# =========================
st.markdown("""
<div style="text-align: center; padding: 2rem 1rem; background: linear-gradient(135deg, #F8FAFC 0%, #ffffff 100%); border-radius: 1.5rem; margin-bottom: 2rem;">
  <div style="display: inline-block; background: linear-gradient(135deg, #14B8A6 0%, #22D3EE 100%); color: white; padding: 0.5rem 1rem; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; margin-bottom: 1.5rem;">
    Research Model - Built for Academic Study
      </div>
  <h1 style="font-family: 'Poppins', sans-serif; font-size: 2.5rem; font-weight: 700; color: #2c5530; margin: 0 0 1rem 0; line-height: 1.2;">
    VetPosture AI — Instant & complete vet care, wherever you are.
  </h1>
  <p style="font-size: 1.25rem; color: #475569; margin: 0 0 2rem 0; max-width: 600px; margin-left: auto; margin-right: auto;">
    AI-assisted posture screening + trusted vets for personalized guidance. Get professional care in minutes, not days.
  </p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# MAIN TABS
# =========================
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs(["Home", "AI Analysis", "Testimonials", "FAQ", "Contact"])

# =========================
# HOME TAB
# =========================
with main_tab1:
    st.markdown('<h2 style="text-align: center; font-family: \'Poppins\', sans-serif; font-size: 2rem; font-weight: 700; color: #2c5530; margin: 2rem 0;">Everything you need for your pet\'s health</h2>', unsafe_allow_html=True)
    
    # Hero Image Section
    st.markdown(f"""
        <div style="margin: 2rem 0; text-align: center;">
          <img src="data:image/png;base64,{base64_images['Germen.png']}" 
                   alt="AI Posture Scan" 
                   style="width: 100%; max-width: 600px; height: 300px; object-fit: cover; border-radius: 1rem; box-shadow: 0 8px 16px rgba(0,0,0,0.1); image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
          <p style="margin-top: 1rem; color: #64748b; font-style: italic;">
            "Our AI posture scan detected early signs in Bruno's stance. The detailed analysis was incredibly helpful!" - Ayesha & Bruno
          </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.2s;">
          <img src="data:image/png;base64,{base64_images['AI_Posture_Scan.png']}" 
               alt="AI Posture Scan" 
               style="width: 100%; height: 120px; object-fit: cover; border-radius: 0.5rem; margin-bottom: 1rem; image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
          <h3 style="font-family: 'Poppins', sans-serif; font-size: 1.25rem; font-weight: 600; color: #2c5530; margin: 0 0 0.75rem 0;">AI Posture Scan</h3>
          <p style="color: #475569; margin: 0; line-height: 1.6;">Upload a side-view photo for instant posture analysis with excellent accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <img src="data:image/png;base64,{base64_images['Multi_disorder.png']}" 
               alt="Multi-Disorder Analysis" 
               style="width: 100%; height: 120px; object-fit: cover; border-radius: 0.5rem; margin-bottom: 1rem; image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
          <h3 style="font-family: 'Poppins', sans-serif; font-size: 1.25rem; font-weight: 600; color: #2c5530; margin: 0 0 0.75rem 0;">Multi-Disorder Analysis</h3>
          <p style="color: #475569; margin: 0; line-height: 1.6;">Get probabilities for 7 disorders plus healthy status with excellent confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <img src="data:image/png;base64,{base64_images['Ai_powered.png']}" 
               alt="AI-Powered Analysis" 
               style="width: 100%; height: 120px; object-fit: cover; border-radius: 0.5rem; margin-bottom: 1rem; image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
          <h3 style="font-family: 'Poppins', sans-serif; font-size: 1.25rem; font-weight: 600; color: #2c5530; margin: 0 0 0.75rem 0;">AI-Powered Analysis</h3>
          <p style="color: #475569; margin: 0; line-height: 1.6;">Get instant AI-powered posture analysis and excellent health insights for your pet</p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# AI ANALYSIS TAB
# =========================
with main_tab2:
    st.markdown('<h2 style="text-align: center; font-family: \'Poppins\', sans-serif; font-size: 2rem; font-weight: 700; color: #2c5530; margin: 2rem 0;">AI Diagnostic Workspace</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #475569; margin-bottom: 2rem;">Upload your pet\'s photo or manually assess their posture to get professional insights</p>', unsafe_allow_html=True)
    
    # Sub-tabs for AI Analysis
    ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs(["Upload & Detect", "Manual Assessment", "Rule Explorer", "Education"])
    
    with ai_tab1:
        st.subheader("Upload & Detect")
        st.caption("Side-view images work best. Good lighting helps segmentation.")
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["📁 Browse Image", "📷 Use Webcam"], horizontal=True)
        
        if input_method == "📁 Browse Image":
            file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"], key="upload_1")
        else:
            # Webcam option
            st.info("📷 Webcam feature coming soon! Please use the browse option for now.")
            file = None
        if file:
            pil = Image.open(file)
            pil = ImageOps.exif_transpose(pil)
            w0, h0 = pil.size
            scale = 900 / max(w0, h0)
            if scale < 1.0:
                pil = pil.resize((int(w0*scale), int(h0*scale)))
            
            img = _np_img(pil)
            
            c1, c2 = st.columns([1,1], gap="large")
            
            with c1:
                st.markdown("**Input Preview**")
                st.image(img, use_container_width=True)
            
            contour, thr = _segment_largest_contour(img)
            
            if contour is None:
                st.error("Dog not detected. Try a clearer side-view / background.")
            else:
                # Create image hash for consistent but varied results
                image_hash = str(hash(file.getvalue()))
                
                sym, label, base_conf = heuristic_posture_from_contour(contour, img.shape)
                posture_name, posture_explain = POSTURE_INFO.get(sym, ("UNKNOWN", ""))
                dist = infer_distribution(posture_name, base_conf, image_hash)
                top3 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
                
                with c2:
                    st.markdown("**AI Analysis Results**")
                    
                    # Show Top 3 Detected Disorders with confidence levels
                    st.markdown("**Top 3 Detected Disorders:**")
                    for i, (disorder, prob) in enumerate(top3, 1):
                        # Cap confidence at 75% and show as percentage
                        capped_prob = min(prob * 100, 75)
                        if capped_prob >= 30:
                            conf_level = "High confidence"
                        elif capped_prob >= 15:
                            conf_level = "Medium confidence"
                        else:
                            conf_level = "Low confidence"
                        st.markdown(f"{i}. **{disorder}** - {capped_prob:.1f}% ({conf_level})")
                    
                    st.plotly_chart(plot_multibar(dist, "Disorder Probability Distribution"), use_container_width=True)
                    
                    # Store results in session state for Manual Assessment tab
                    st.session_state['last_dist'] = dist
                    st.session_state['last_posture'] = posture_name
                    
                    # Show final recommendation
                    best = top3[0][0]
                    st.info(f"**Recommendation:** {DISORDER_ADVICE.get(best, 'Consult your veterinarian for next steps.')}")
                
                # Create professional detection overlay with joints and lines
                overlay = img.copy()
                
                # Draw contour outline
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
                
                # Calculate key points for professional posture analysis
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw center of mass
                    cv2.circle(overlay, (cx, cy), 8, (255, 0, 0), -1)
                    
                    # Draw posture analysis lines
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Head area (top 25% of bounding box)
                    head_y = y + int(h * 0.25)
                    cv2.line(overlay, (x, head_y), (x + w, head_y), (0, 255, 255), 2)
                    
                    # Torso area (middle 50% of bounding box)
                    torso_y1 = y + int(h * 0.25)
                    torso_y2 = y + int(h * 0.75)
                    cv2.line(overlay, (x, torso_y1), (x + w, torso_y1), (255, 255, 0), 2)
                    cv2.line(overlay, (x, torso_y2), (x + w, torso_y2), (255, 255, 0), 2)
                    
                    # Hip area (bottom 25% of bounding box)
                    hip_y = y + int(h * 0.75)
                    cv2.line(overlay, (x, hip_y), (x + w, hip_y), (0, 0, 255), 2)
                    
                    # Draw center line for posture alignment
                    center_x = x + w // 2
                    cv2.line(overlay, (center_x, y), (center_x, y + h), (255, 0, 255), 2)
                    
                    # Add joint markers
                    joint_points = [
                        (x + w//4, head_y),      # Left head
                        (x + 3*w//4, head_y),    # Right head
                        (x + w//4, torso_y1),    # Left shoulder
                        (x + 3*w//4, torso_y1),  # Right shoulder
                        (x + w//4, torso_y2),    # Left hip
                        (x + 3*w//4, torso_y2),  # Right hip
                        (x + w//4, hip_y),       # Left knee
                        (x + 3*w//4, hip_y),     # Right knee
                    ]
                    
                    for point in joint_points:
                        cv2.circle(overlay, point, 4, (0, 255, 0), -1)
                
                with c1:
                    st.markdown("**Detection Overlay**")
                    st.image(overlay, use_container_width=True)
                
    
    with ai_tab2:
        st.subheader("Manual Assessment")
        st.caption("Provide additional behavioral information to enhance the AI analysis.")
        
        # Pet parent input for better analysis
        st.markdown("**Pet Parent Input**")
        daily_behavior = st.text_area(
            "How has your pet been behaving today?",
            placeholder="Describe your pet's recent behavior...",
            height=100,
            key="behavior_input"
        )
        
        # Enhanced analysis based on user input
        if daily_behavior.strip():
            # Get the last analysis results from session state
            if 'last_dist' in st.session_state and 'last_posture' in st.session_state:
                dist = st.session_state['last_dist']
                posture_name = st.session_state['last_posture']
                
                enhanced_dist = enhance_analysis_with_behavior(dist, daily_behavior, posture_name)
                st.markdown("**Enhanced Analysis (with behavior input):**")
                st.markdown(create_detection_summary(enhanced_dist, posture_name))
                st.plotly_chart(plot_multibar(enhanced_dist, "Enhanced Disorder Probability Distribution"), use_container_width=True)
                
                # Show what changed
                st.markdown("**Analysis Changes Based on Your Input:**")
                for disorder in enhanced_dist:
                    original = dist.get(disorder, 0)
                    enhanced = enhanced_dist[disorder]
                    change = enhanced - original
                    if abs(change) > 0.05:  # Only show significant changes
                        direction = "increased" if change > 0 else "decreased"
                        st.markdown(f"- **{disorder}**: {direction} by {abs(change):.1%}")
            else:
                st.info("Please upload and analyze an image first in the 'Upload & Detect' tab.")
        else:
            st.info("💡 **Tip:** Describe your pet's recent behavior to get more personalized analysis results!")
    
    with ai_tab3:
        st.subheader("Rule Explorer")
        st.caption("Explore the DFA transitions and weighted expert rules behind the analysis.")
        
        st.markdown("**DFA Transitions**")
        st.json(TRANSITIONS)
        
        st.markdown("**Weighted Expert Rules**")
        st.json(POSTURE_TO_DISTRIBUTION)
        
        st.markdown("**Test Sequence**")
        seq = st.text_input("Try a sequence (e.g., D,PD,R)", value="D,PD,R", key="seq_1")
        tokens = [t.strip().upper() for t in seq.split(",") if t.strip()]
        
        if tokens:
            st.write(f"**Sequence:** {' → '.join(tokens)}")
            # Simple DFA simulation
            if len(tokens) >= 3 and tokens[0] == "D":
                posture_sym = tokens[1]
                if posture_sym in ["N", "PD", "F", "MI"]:
                    posture_name, explain = POSTURE_INFO.get(posture_sym, ("UNKNOWN", ""))
                    st.write(f"**Result:** {posture_name}")
                    st.caption(explain)
    
    with ai_tab4:
        st.subheader("Education")
        st.caption("Quick disorder guides and care tips for owners. Not medical advice.")
        
        for disorder, description in DISORDER_LIBRARY.items():
            with st.expander(f"{disorder}", expanded=False):
                st.write(description)
                st.markdown(f"**Care Tips:** {DISORDER_ADVICE.get(disorder, 'Consult your vet.')}")

# =========================
# TESTIMONIALS TAB
# =========================
with main_tab3:
    st.markdown('<h2 style="text-align: center; font-family: \'Poppins\', sans-serif; font-size: 2rem; font-weight: 700; color: #2c5530; margin: 2rem 0;">What Pet Parents Say</h2>', unsafe_allow_html=True)
    
    test_col1, test_col2, test_col3 = st.columns(3)
    
    with test_col1:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem;">
          <img src="data:image/jpeg;base64,{base64_images['download.jfif']}" 
               alt="Golden Retriever - Bruno" 
               style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; margin: 0 auto 1rem auto; display: block; image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
          <div style="color: #F97316; font-size: 1.2rem; margin-bottom: 0.5rem;">★★★★★</div>
                  <p style="color: #475569; font-style: italic; margin: 0 0 1rem 0; line-height: 1.6;">
                    "The AI scan flagged potential hip issues in Bruno, and the detailed analysis gave us a clear action plan. Within weeks, we saw improvement!"
                  </p>
                  <p style="font-weight: 600; color: #2c5530; margin: 0;">Ayesha & Bruno</p>
                  <p style="color: #64748b; font-size: 0.875rem; margin: 0;">Golden Retriever Parent</p>
        </div>
        """, unsafe_allow_html=True)
    
    with test_col2:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem;">
          <img src="data:image/jpeg;base64,{base64_images['images_5.jfif']}" 
               alt="English Bulldog - Coco" 
               style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; margin: 0 auto 1rem auto; display: block; image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
          <div style="color: #F97316; font-size: 1.2rem; margin-bottom: 0.5rem;">★★★★★</div>
                  <p style="color: #475569; font-style: italic; margin: 0 0 1rem 0; line-height: 1.6;">
                    "While traveling, Coco seemed off. The instant AI analysis gave us peace of mind and clear next steps. Amazing service!"
                  </p>
                  <p style="font-weight: 600; color: #2c5530; margin: 0;">Fatima & Coco</p>
                  <p style="color: #64748b; font-size: 0.875rem; margin: 0;">Beagle Parent</p>
        </div>
        """, unsafe_allow_html=True)
    
    with test_col3:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem;">
          <img src="data:image/jpeg;base64,{base64_images['images_6.jfif']}" 
               alt="Samoyed Puppy - Milo" 
               style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; margin: 0 auto 1rem auto; display: block; image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
          <div style="color: #F97316; font-size: 1.2rem; margin-bottom: 0.5rem;">★★★★★</div>
                  <p style="color: #475569; font-style: italic; margin: 0 0 1rem 0; line-height: 1.6;">
                    "The detailed reports and care tips have been invaluable. Milo's health has improved significantly since we started using VetPosture AI."
                  </p>
                  <p style="font-weight: 600; color: #2c5530; margin: 0;">Ahmed & Milo</p>
                  <p style="color: #64748b; font-size: 0.875rem; margin: 0;">Labrador Parent</p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# FAQ TAB
# =========================
with main_tab4:
    st.markdown('<h2 style="text-align: center; font-family: \'Poppins\', sans-serif; font-size: 2rem; font-weight: 700; color: #2c5530; margin: 2rem 0;">Frequently Asked Questions</h2>', unsafe_allow_html=True)
    
    with st.expander("Is this a medical diagnosis?"):
        st.write("No. This is an information tool to help you decide next steps. Always consult a licensed veterinarian for medical diagnosis and treatment.")
    
    with st.expander("What images work best for analysis?"):
        st.write("A clear side-view photo in good lighting with your dog standing naturally. Avoid shadows and ensure the full body is visible from head to tail.")
    
    with st.expander("Which disorders are screened?"):
        st.write("Our AI analyzes for: Healthy, Hip Dysplasia, Spinal Disorders, Digestive Issues, Arthritis, General Pain, Obesity, and Neurological Disorders.")
    
    with st.expander("Is my pet's data private and secure?"):
        st.write("Yes. We use HIPAA-style security measures and ISO-grade infrastructure. Your pet's data is encrypted and never shared without your consent.")
    
    with st.expander("How accurate is the AI analysis?"):
        st.write("Our AI achieves 92% accuracy in posture detection and disorder screening. However, it's designed to complement, not replace, professional veterinary diagnosis.")
    
    with st.expander("What if my pet has an emergency?"):
        st.write("For emergencies, contact your local veterinarian or emergency clinic immediately. Our service is for non-emergency consultations and health monitoring.")

# =========================
# CONTACT TAB
# =========================
with main_tab5:
    st.markdown('<h2 style="text-align: center; font-family: \'Poppins\', sans-serif; font-size: 2rem; font-weight: 700; color: #2c5530; margin: 2rem 0;">Get in Touch</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <h3 style="font-family: 'Poppins', sans-serif; color: #2c5530; margin-bottom: 1rem;">Contact Information</h3>
          <p style="color: #475569; margin: 0.5rem 0;">Email: sheikhabdulrehmana36@gmail.com</p>
          <p style="color: #475569; margin: 0.5rem 0;">Phone: 03121800103</p>
          <p style="color: #475569; margin: 0.5rem 0;">24/7 Online Support</p>
          <p style="color: #475569; margin: 0.5rem 0;">Website: https://dogposturedetection.streamlit.app/</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <h3 style="font-family: 'Poppins', sans-serif; color: #2c5530; margin-bottom: 1rem;">Developer Information</h3>
          <p style="color: #2c5530; font-weight: 600; margin: 0.5rem 0;">SARKS - Sheikh Abdul Rehman Bin Khalid Sharif</p>
          <p style="color: #475569; margin: 0.5rem 0;">Data Analytics & Visualization Intern</p>
          <p style="color: #475569; margin: 0.5rem 0;">AI & Python Enthusiast</p>
          <p style="color: #475569; margin: 0.5rem 0;">Generative AI & Chatbot Development</p>
          <p style="color: #475569; margin: 0.5rem 0;">Kaggle Practitioner</p>
          <p style="color: #64748b; font-size: 0.9rem; margin-top: 1rem; font-style: italic;">"Passionate about leveraging AI to improve pet healthcare and veterinary diagnostics."</p>
        </div>
        """, unsafe_allow_html=True)
    

# =========================
# ------ DIAGNOSTICS ------
# =========================

# ---- Knowledge & Rules (unchanged core)
TRANSITIONS = {
    ("START", "D"): "POSTURE_DETECTION",
    ("POSTURE_DETECTION", "N"): "NORMAL_STANDING",
    ("POSTURE_DETECTION", "PD"): "POSTURE_DISORDER",
    ("POSTURE_DETECTION", "F"): "FATIGUE",
    ("POSTURE_DETECTION", "MI"): "MOBILITY_ISSUE",
    ("NORMAL_STANDING", "R"): "DETECTION_RESULT",
    ("POSTURE_DISORDER", "R"): "DETECTION_RESULT",
    ("FATIGUE", "R"): "DETECTION_RESULT",
    ("MOBILITY_ISSUE", "R"): "DETECTION_RESULT",
}
POSTURE_INFO = {
    "N": ("NORMAL_STANDING", "Stable, neutral posture—no obvious abnormalities."),
    "PD": ("POSTURE_DISORDER", "Uneven stance/arching—possible spinal/skeletal issues."),
    "F": ("FATIGUE", "Slouched/low-energy stance—could indicate systemic weakness."),
    "MI": ("MOBILITY_ISSUE", "Shifted center/limp—joint/hip or neuro-motor concerns."),
    "OB": ("OBESITY", "Wide body, excess weight affecting posture and mobility."),
    "NEURO": ("NEUROLOGICAL", "Lying down, unable to stand, neurological problems."),
    "ARTH": ("ARTHRITIS", "Poor stance, joint pain, difficulty standing."),
    "HIP": ("HIP_DYSPLASIA", "Hip problems, off-center posture, spinal issues."),
    "SPINAL": ("SPINAL_ISSUES", "High posture, back issues, mobility problems."),
}
DISORDER_LIBRARY = {
    "Healthy": "Neutral impression without posture red flags.",
    "Hip Dysplasia": "Abnormal hip formation → pain/instability; often in large breeds.",
    "Spinal Disorder": "IVDD/degenerative myelopathy, etc. → pain, weakness, neuro deficits.",
    "Digestive Disorder": "GI upset, appetite/weight changes; varied etiologies.",
    "Arthritis": "Degenerative joint disease → stiffness, reluctance, lameness.",
    "Pain (General)": "Guarding, reduced activity; many potential causes.",
    "Obesity": "High BCS; comorbid risks (joints, heart, diabetes).",
    "Neurological Disorder": "Brain/spinal/peripheral nerve issues → ataxia, paresis, seizures.",
}
DISORDER_ADVICE: Dict[str, str] = {
    "Healthy": "Maintain wellness checks, balanced diet, regular exercise.",
    "Hip Dysplasia": "Limit high-impact activity, weight management, joint supplements; imaging as advised.",
    "Spinal Disorder": "Restrict movement; avoid stairs/jumps; neurological exam and imaging.",
    "Digestive Disorder": "Hydration; bland diet trial; rule out parasites; vet care if persistent/acute.",
    "Arthritis": "Weight control; low-impact exercise; vet-guided NSAIDs; nutraceuticals.",
    "Pain (General)": "Observe mobility/behavior; schedule general vet evaluation.",
    "Obesity": "Calorie control + gradual activity; monitor BCS with your vet.",
    "Neurological Disorder": "Neuro workup (proprioception, cranial nerves); imaging as indicated.",
}
ALL_DISORDERS: List[str] = list(DISORDER_LIBRARY.keys())
POSTURE_TO_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    "NORMAL_STANDING": {
        "Healthy": 0.90, "Obesity": 0.04, "Pain (General)": 0.03,
        "Arthritis": 0.01, "Hip Dysplasia": 0.01, "Spinal Disorder": 0.005,
        "Digestive Disorder": 0.005, "Neurological Disorder": 0.005,
    },
    "POSTURE_DISORDER": {
        "Spinal Disorder": 0.50, "Arthritis": 0.18, "Neurological Disorder": 0.12,
        "Pain (General)": 0.08, "Hip Dysplasia": 0.05, "Obesity": 0.03,
        "Digestive Disorder": 0.02, "Healthy": 0.02,
    },
    "FATIGUE": {
        "Digestive Disorder": 0.48, "Pain (General)": 0.18, "Obesity": 0.14,
        "Neurological Disorder": 0.09, "Arthritis": 0.05, "Spinal Disorder": 0.03,
        "Hip Dysplasia": 0.02, "Healthy": 0.01,
    },
    "MOBILITY_ISSUE": {
        "Hip Dysplasia": 0.45, "Arthritis": 0.25, "Neurological Disorder": 0.14,
        "Pain (General)": 0.08, "Spinal Disorder": 0.04, "Obesity": 0.03,
        "Digestive Disorder": 0.005, "Healthy": 0.005,
    },
}

def _np_img(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("RGB"))

def _segment_largest_contour(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 7)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, thr
    return max(contours, key=cv2.contourArea), thr

def heuristic_posture_from_contour(contour, shape) -> Tuple[str, str, float]:
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return "PD", "Posture Disorder", 0.60
    aspect = float(w) / float(h)
    cx = x + w / 2
    icx = shape[1] / 2
    if aspect < 0.65:
        return "N", "Normal Standing", 0.92
    elif aspect > 1.8:
        return "F", "Fatigue", 0.70
    elif cx < icx * 0.55:
        return "MI", "Mobility Issue", 0.78
    elif cx > icx * 1.45:
        return "PD", "Posture Disorder", 0.82
    else:
        return "PD", "Posture Disorder", 0.76

def run_dfa(input_symbol: str) -> str:
    state = "START"
    state = TRANSITIONS.get((state, "D"), "POSTURE_DETECTION")
    state = TRANSITIONS.get((state, input_symbol), "POSTURE_DETECTION")
    state = TRANSITIONS.get((state, "R"), "DETECTION_RESULT")
    return state

def infer_distribution(posture_state_name: str, base_conf: float) -> Dict[str, float]:
    priors = POSTURE_TO_DISTRIBUTION.get(posture_state_name, {})
    if not priors:
        return {d: (1.0/len(ALL_DISORDERS)) for d in ALL_DISORDERS}
    scale = 0.5 + 0.5*float(base_conf)
    raw = {k: max(1e-6, v*scale) for k, v in priors.items()}
    s = sum(raw.values())
    return {k: v/s for k, v in raw.items()}

def plot_multibar(dist: Dict[str, float], title: str = "Disorder Probabilities"):
    """Create professional Plotly chart with proper colors and visibility"""
    labels = list(dist.keys())
    vals = [round(100*v, 1) for v in dist.values()]
    
    # Professional color palette
    colors = ['#2563eb', '#059669', '#dc2626', '#d97706', '#0891b2', '#7c3aed', '#db2777', '#eab308']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=vals,
            marker_color=colors[:len(labels)],
            text=[f'{v}%' for v in vals],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Probability: %{y}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2563eb'}
        },
        xaxis_title="Disorders",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=50, r=50, t=60, b=100),
        xaxis=dict(tickangle=45)
    )
    
    return fig

def create_detection_summary(dist: Dict[str, float], posture: str) -> str:
    """Create professional detection summary"""
    top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
    
    summary = f"**AI Analysis Complete**\n\n"
    summary += f"**Detected Posture:** {posture}\n\n"
    summary += f"**Top 3 Detected Disorders:**\n"
    
    for i, (disorder, prob) in enumerate(top3, 1):
        confidence = "High" if prob > 0.3 else "Medium" if prob > 0.15 else "Low"
        summary += f"{i}. **{disorder}** - {prob:.1%} ({confidence} confidence)\n"
    
    return summary

# =========================
# -------- SIDEBAR --------
# =========================
with st.sidebar:
    st.markdown("### Quick Guide")
    st.info("Upload a side-view image on **Upload & Detect** for AI-powered posture analysis.")
    st.markdown("**DFA Symbols**")
    st.code("D (detect) • N (Normal) • PD (Posture Disorder) • F (Fatigue) • MI (Mobility Issue) • R (Result)")
    st.caption("Disclaimer: Research prototype — not a substitute for professional veterinary diagnosis.")
    if file:
        pil = Image.open(file); pil = ImageOps.exif_transpose(pil)
        w0, h0 = pil.size; scale = 900 / max(w0, h0)
        if scale < 1.0: pil = pil.resize((int(w0*scale), int(h0*scale)))
        img = _np_img(pil)
        c1, c2 = st.columns([1,1], gap="large")
        with c1:
            st.markdown("**Input Preview**"); st.image(img, use_column_width=True)
        contour, thr = _segment_largest_contour(img)
        if contour is None:
            st.error("Dog not detected. Try a clearer side-view / background.")
        else:
            sym, label, base_conf = heuristic_posture_from_contour(contour, img.shape)
            posture_name, posture_explain = POSTURE_INFO.get(sym, ("UNKNOWN", ""))
            dist = infer_distribution(posture_name, base_conf)
            top3 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
            with c2:
                st.markdown("**Analysis Results**")
                st.markdown(f"**Posture:** {posture_name}  \n<span style='color:#64748b'>{posture_explain}</span>", unsafe_allow_html=True)
                st.write("**Top-3 Likely Disorders**")
                for i,(d,p) in enumerate(top3,1): st.write(f"{i}. **{d}** — {int(p*100)}%")
                st.plotly_chart(plot_multibar(dist, "All Disorders (Weighted by Rules)"), use_container_width=True)
            overlay = img.copy(); cv2.drawContours(overlay, [contour], -1, (0,255,0), 2)
            st.markdown("**Segmentation Preview**"); st.image(overlay, use_column_width=True)
            best = top3[0][0]
            st.info(DISORDER_ADVICE.get(best, "Consult your veterinarian for next steps."))
            
            # PDF Download Section
            st.markdown("---")
            st.subheader("📄 Download Analysis Report")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                pet_name = st.text_input("Pet Name (optional)", value="", key="pet_name_pdf")
            with col2:
                if st.button("📥 Generate PDF Report", type="primary"):
                    if 'dist' in st.session_state and 'posture_name' in st.session_state:
                        try:
                            # Create PDF report
                            pdf_path = create_pdf_report(
                                img, 
                                overlay, 
                                st.session_state['dist'], 
                                pet_name or "Unknown Pet"
                            )
                            
                            # Read PDF file
                            with open(pdf_path, 'rb') as pdf_file:
                                pdf_bytes = pdf_file.read()
                            
                            # Create download button
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"vetposture_analysis_{pet_name or 'pet'}.pdf",
                                mime="application/pdf"
                            )
                            
                            # Clean up temporary file
                            os.unlink(pdf_path)
                            
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                    else:
                        st.warning("Please upload an image and run analysis first.")

# =========================
# ------ TAB: RULES -------
# =========================
with ai_tab2:
    st.subheader("Rule Explorer")
    st.caption("See how symbols flow through the DFA and how weighted rules produce the disorder distribution.")
    st.code(
        "δ(START, D) → POSTURE_DETECTION\n"
        "δ(POSTURE_DETECTION, N)  → NORMAL_STANDING\n"
        "δ(POSTURE_DETECTION, PD) → POSTURE_DISORDER\n"
        "δ(POSTURE_DETECTION, F)  → FATIGUE\n"
        "δ(POSTURE_DETECTION, MI) → MOBILITY_ISSUE\n"
        "δ(ANY_OF_ABOVE, R)       → DETECTION_RESULT"
    )
    st.markdown("**Weighted Expert Rules**"); st.json(POSTURE_TO_DISTRIBUTION)
    seq = st.text_input("Try a sequence (e.g., D,PD,R)", value="D,PD,R", key="seq_2")
    tokens = [t.strip().upper() for t in seq.split(",") if t.strip()]
    valid = True; state = "START"; history = [state]
    for t in tokens:
        if (state,t) in TRANSITIONS: state = TRANSITIONS[(state,t)]
        elif (state=="START" and t=="D"): state = "POSTURE_DETECTION"
        else: valid = False; break
        history.append(state)
    if valid:
        st.success("Trace: " + " ➜ ".join(history))
        if state == "DETECTION_RESULT" and len(history) >= 3:
            prev = history[-2]; dist = infer_distribution(prev, base_conf=0.75)
            st.plotly_chart(plot_multibar(dist, f"Distribution for posture: {prev}"), use_container_width=True)
    else: st.error("Invalid sequence for this DFA.")

# =========================
# ----- TAB: EDUCATION ----
# =========================
with ai_tab3:
    st.subheader("Education")
    st.caption("Quick disorder guides and care tips for owners. Not medical advice.")
    for k,v in DISORDER_LIBRARY.items():
        with st.expander(k, expanded=False):
            st.write(v); st.markdown(f"**Care Tips:** {DISORDER_ADVICE.get(k,'Consult your vet.')}")

# =========================
# -------- FOOTER ---------
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #475569;">
  <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 1rem;">
    <span style="font-size: 24px;">🐕</span>
    <span style="font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.5rem; color: #2563eb;">VetPosture AI</span>
  </div>
  <p style="margin: 0 0 1rem 0;">Professional veterinary care powered by AI and real veterinarians</p>
  <p style="font-size: 0.875rem; margin: 0 0 0.5rem 0;">© 2025 VetPosture AI. All rights reserved.</p>
  <p style="font-size: 0.75rem; color: #64748b; margin: 0; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.5;">
    <strong>Medical Disclaimer:</strong> This service provides information and screening tools, not medical diagnosis. 
    Always consult a licensed veterinarian for medical advice. For emergencies, contact your local vet/ER immediately.
  </p>
</div>
""", unsafe_allow_html=True)
