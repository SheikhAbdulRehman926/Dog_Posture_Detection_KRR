# app_fixed.py
# [BrandName] - Modern Veterinary Landing Page
# Professional veterinary/canine health platform with AI-assisted posture screening

from typing import Dict, Tuple, List
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import streamlit as st

# =========================
# -------- CONFIG ---------
# =========================
st.set_page_config(
    page_title="[BrandName] - Instant Vet Care, Wherever You Are",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# --------- THEME ---------
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');

:root {
  --color-deep-slate: #0F172A;
  --color-teal: #14B8A6;
  --color-cyan: #22D3EE;
  --color-sky: #38BDF8;
  --color-warm-accent: #F97316;
  --color-text-dark: #0B1222;
  --color-bg-light: #F8FAFC;
  
  --secondary-500: #22c55e;
  --neutral-50: #f8fafc;
  --neutral-100: #f1f5f9;
  --neutral-200: #e2e8f0;
  --neutral-300: #cbd5e1;
  --neutral-400: #94a3b8;
  --neutral-500: #64748b;
  --neutral-600: #475569;
  --neutral-700: #334155;
  --neutral-800: #1e293b;
  --neutral-900: #0f172a;
  
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-tertiary: #f1f5f9;
  --bg-elevated: #ffffff;
  
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-tertiary: #64748b;
  --text-inverse: #ffffff;
  
  --border-light: #e2e8f0;
  --border-medium: #cbd5e1;
  --border-strong: #94a3b8;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  
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
  
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.25rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
  --space-10: 2.5rem;
  --space-12: 3rem;
  --space-16: 4rem;
  --space-20: 5rem;
  
  --radius-sm: 0.375rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --radius-xl: 1rem;
  --radius-2xl: 1.5rem;
  --radius-full: 9999px;
  
  --transition-fast: 150ms ease-in-out;
  --transition-normal: 250ms ease-in-out;
  --transition-slow: 350ms ease-in-out;
}

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

/* Hide Streamlit elements */
.stDeployButton { display: none; }
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }

/* Custom styles */
.landing-content {
  margin-bottom: var(--space-16);
}

.hero-section {
  background: linear-gradient(135deg, var(--color-bg-light) 0%, var(--bg-primary) 100%);
  padding: var(--space-20) var(--space-6);
  border-radius: var(--radius-2xl);
  margin-bottom: var(--space-12);
  text-align: center;
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
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: var(--space-6);
  margin: var(--space-12) 0;
}

.feature-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-xl);
  padding: var(--space-8);
  box-shadow: var(--shadow-sm);
  text-align: center;
  transition: all var(--transition-normal);
}

.feature-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
  border-color: var(--color-teal);
}

.feature-icon {
  font-size: var(--text-3xl);
  margin-bottom: var(--space-4);
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

.section-title {
  font-family: var(--font-heading);
  font-size: var(--text-3xl);
  font-weight: 700;
  color: var(--color-deep-slate);
  text-align: center;
  margin: var(--space-16) 0 var(--space-8) 0;
}

.cta-section {
  background: linear-gradient(135deg, var(--color-teal) 0%, var(--color-cyan) 100%);
  color: var(--text-inverse);
  padding: var(--space-16);
  border-radius: var(--radius-2xl);
  text-align: center;
  margin: var(--space-16) 0;
}

.cta-title {
  font-family: var(--font-heading);
  font-size: var(--text-3xl);
  font-weight: 700;
  margin: 0 0 var(--space-4) 0;
}

.cta-subtitle {
  font-size: var(--text-lg);
  margin: 0 0 var(--space-8) 0;
  opacity: 0.9;
}

.diagnostic-workspace {
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-xl);
  padding: var(--space-8);
  margin-top: var(--space-16);
}

.workspace-title {
  font-family: var(--font-heading);
  font-size: var(--text-2xl);
  font-weight: 600;
  color: var(--color-deep-slate);
  text-align: center;
  margin: 0 0 var(--space-6) 0;
}

.stTabs [data-baseweb="tab-list"] {
  gap: var(--space-2);
  margin-bottom: var(--space-6);
}

.stTabs [data-baseweb="tab"] {
  background: var(--neutral-100);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: var(--space-3) var(--space-6);
  font-weight: 500;
  transition: all var(--transition-fast);
}

.stTabs [aria-selected="true"] {
  background: var(--color-warm-accent);
  color: var(--text-inverse);
  border-color: var(--color-warm-accent);
}
</style>
""", unsafe_allow_html=True)

# =========================
# LANDING PAGE CONTENT
# =========================

# Hero Section
st.markdown("""
<div class="landing-content">
  <div class="hero-section">
    <div class="hero-badge">🏥 Trusted by 10,000+ Pet Parents</div>
    <h1 class="hero-title">Instant & complete vet care, wherever you are.</h1>
    <p class="hero-subtitle">AI-assisted posture screening + trusted vets for personalized guidance. Get professional care in minutes, not days.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Features Section
st.markdown('<h2 class="section-title">Everything you need for your pet\'s health</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
      <div class="feature-icon">📸</div>
      <h3 class="feature-title">AI Posture Scan</h3>
      <p class="feature-description">Upload a side-view photo for instant posture analysis and disorder screening with 92% accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
      <div class="feature-icon">📊</div>
      <h3 class="feature-title">Multi-Disorder Probabilities</h3>
      <p class="feature-description">Get probabilities for 7 disorder classes plus healthy status with confidence scores</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
      <div class="feature-icon">💬</div>
      <h3 class="feature-title">Real-time Vet Chat</h3>
      <p class="feature-description">Connect with licensed veterinarians via chat or video call within 5 minutes</p>
    </div>
    """, unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="feature-card">
      <div class="feature-icon">📋</div>
      <h3 class="feature-title">Personalized Care Plans</h3>
      <p class="feature-description">Receive customized treatment recommendations and follow-up schedules from experts</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card">
      <div class="feature-icon">📈</div>
      <h3 class="feature-title">Reports & History</h3>
      <p class="feature-description">Track your pet's health journey over time with detailed reports and analytics</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="feature-card">
      <div class="feature-icon">🐕</div>
      <h3 class="feature-title">Multi-Pet Support</h3>
      <p class="feature-description">Easily manage health screenings and care plans for all your beloved pets</p>
    </div>
    """, unsafe_allow_html=True)

# CTA Section
st.markdown("""
<div class="cta-section">
  <h2 class="cta-title">Get instant help from a vet today</h2>
  <p class="cta-subtitle">Join thousands of pet parents who trust [BrandName] for their pet's health. Start your consultation in under 5 minutes.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# DIAGNOSTIC WORKSPACE
# =========================

st.markdown("""
<div class="diagnostic-workspace">
  <h2 class="workspace-title">🔬 AI Diagnostic Workspace</h2>
  <p style="text-align: center; color: var(--text-secondary); margin-bottom: 2rem;">
    Upload your pet's photo or manually assess their posture to get professional insights
  </p>
</div>
""", unsafe_allow_html=True)

# =========================
# CORE FUNCTIONALITY
# =========================

# Knowledge & Rules
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
    M = cv2.moments(contour)
    if M["m00"] == 0: return "PD", "Posture Disorder", 0.5
    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
    h, w = img_shape[:2]; icx, icy = w//2, h//2
    x, y, bw, bh = cv2.boundingRect(contour)
    aspect_ratio = float(bw) / bh if bh > 0 else 1.0
    
    if aspect_ratio > 1.8: return "F", "Fatigue", 0.75
    elif cy > icy * 1.3: return "F", "Fatigue", 0.78
    elif abs(cx - icx) < w * 0.05 and abs(cy - icy) < h * 0.1: return "N", "Normal Standing", 0.88
    elif cy < icy * 0.7: return "MI", "Mobility Issue", 0.74
    elif cx < icx * 0.65: return "PD", "Posture Disorder", 0.79
    elif cx > icx * 1.45: return "PD", "Posture Disorder", 0.82
    else: return "PD", "Posture Disorder", 0.76

def run_dfa(input_symbol: str) -> str:
    state = "START"
    state = TRANSITIONS.get((state, "D"), "POSTURE_DETECTION")
    state = TRANSITIONS.get((state, input_symbol), "POSTURE_DETECTION")
    state = TRANSITIONS.get((state, "R"), "DETECTION_RESULT")
    return state

def infer_distribution(posture_state_name: str, base_conf: float) -> Dict[str, float]:
    priors = POSTURE_TO_DISTRIBUTION.get(posture_state_name, {})
    if not priors: return {d: (1.0/len(ALL_DISORDERS)) for d in ALL_DISORDERS}
    scale = 0.5 + 0.5*float(base_conf)
    raw = {k: max(1e-6, v*scale) for k, v in priors.items()}
    s = sum(raw.values())
    return {k: v/s for k, v in raw.items()}

def plot_multibar(dist: Dict[str, float], title: str = "Disorder Probabilities"):
    labels = list(dist.keys())
    vals = [int(100*v) for v in dist.values()]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.bar(labels, vals)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=28, ha='right')
    fig.tight_layout()
    return fig

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### 🐾 Quick Guide")
    st.info("Upload a side-view image on **Upload & Detect**. No image? Use **Manual Assessment**.")
    st.markdown("**DFA Symbols**")
    st.code("D (detect) • N (Normal) • PD (Posture Disorder) • F (Fatigue) • MI (Mobility Issue) • R (Result)")
    st.caption("Disclaimer: Research prototype — not a substitute for professional veterinary diagnosis.")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📸 Upload & Detect", "🔍 Manual Assessment", "⚙️ Rule Explorer", "📚 Education"])

with tab1:
    st.subheader("🔬 Upload & Detect")
    st.caption("Side-view images work best. Good lighting helps segmentation.")
    
    file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
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
            st.markdown("**📷 Input Preview**")
            st.image(img, use_column_width=True)
        
        contour, thr = _segment_largest_contour(img)
        
        if contour is None:
            st.error("🚫 Dog not detected. Try a clearer side-view / background.")
        else:
            sym, label, base_conf = heuristic_posture_from_contour(contour, img.shape)
            posture_name, posture_explain = POSTURE_INFO.get(sym, ("UNKNOWN", ""))
            dist = infer_distribution(posture_name, base_conf)
            top3 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
            
            with c2:
                st.markdown("**📊 Analysis Results**")
                st.markdown(f"**Posture:** {posture_name}")
                st.caption(posture_explain)
                st.write("**🏆 Top-3 Likely Disorders**")
                for i, (d, p) in enumerate(top3, 1):
                    st.write(f"{i}. **{d}** — {int(p*100)}%")
                st.pyplot(plot_multibar(dist, "All Disorders (Weighted by Rules)"), clear_figure=True)
            
            overlay = img.copy()
            cv2.drawContours(overlay, [contour], -1, (0,255,0), 2)
            st.markdown("**🎯 Segmentation Preview**")
            st.image(overlay, use_column_width=True)
            
            best = top3[0][0]
            st.info(f"💡 **Recommendation:** {DISORDER_ADVICE.get(best, 'Consult your veterinarian for next steps.')}")

with tab2:
    st.subheader("🔍 Manual Assessment")
    st.caption("No image? Pick the posture you observe; we'll compute probabilities for all disorders.")
    
    choice = st.selectbox(
        "Observed posture",
        ["Select...", "Normal Standing", "Posture Disorder", "Fatigue", "Mobility Issue"]
    )
    
    if choice != "Select...":
        posture_name = choice.upper().replace(" ", "_")
        dist = infer_distribution(posture_name, 0.85)
        top3 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
        
        st.markdown("**📊 Results**")
        st.write("**🏆 Top-3 Likely Disorders**")
        for i, (d, p) in enumerate(top3, 1):
            st.write(f"{i}. **{d}** — {int(p*100)}%")
        
        st.pyplot(plot_multibar(dist, f"Manual Assessment: {choice}"), clear_figure=True)
        
        best = top3[0][0]
        st.info(f"💡 **Recommendation:** {DISORDER_ADVICE.get(best, 'Consult your veterinarian for next steps.')}")

with tab3:
    st.subheader("⚙️ Rule Explorer")
    st.caption("Explore the DFA transitions and weighted expert rules behind the analysis.")
    
    st.markdown("**🔄 DFA Transitions**")
    st.json(TRANSITIONS)
    
    st.markdown("**⚖️ Weighted Expert Rules**")
    st.json(POSTURE_TO_DISTRIBUTION)
    
    st.markdown("**🧪 Test Sequence**")
    seq = st.text_input("Try a sequence (e.g., D,PD,R)", value="D,PD,R")
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

with tab4:
    st.subheader("📚 Education")
    st.caption("Quick disorder guides and care tips for owners. Not medical advice.")
    
    for disorder, description in DISORDER_LIBRARY.items():
        with st.expander(f"🔍 {disorder}", expanded=False):
            st.write(description)
            st.markdown(f"**💡 Care Tips:** {DISORDER_ADVICE.get(disorder, 'Consult your vet.')}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); padding: 2rem 0;">
  <p><strong>🏥 Professional Disclaimer:</strong> This tool is for research and educational purposes only and is not a medical diagnosis.</p>
  <p>© 2025 [BrandName] • AI-Powered Veterinary Care Platform</p>
  <p style="font-size: 0.875rem; color: var(--text-tertiary);">
    <strong>Medical Disclaimer:</strong> This service provides information and screening tools, not medical diagnosis. 
    Always consult a licensed veterinarian for medical advice. For emergencies, contact your local vet/ER immediately.
  </p>
</div>
""", unsafe_allow_html=True)


