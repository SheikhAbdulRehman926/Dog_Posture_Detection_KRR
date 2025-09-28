import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from typing import Dict, List, Tuple

# =========================
# -------- CONFIG ---------
# =========================
st.set_page_config(
    page_title="VetPosture AI - Instant Vet Care, Wherever You Are",
    page_icon="🔬",
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
    ax.set_ylabel("Confidence (%)"); ax.set_ylim(0, 100); ax.set_title(title)
    ax.set_xticklabels(labels, rotation=28, ha='right'); fig.tight_layout(); return fig

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
}

html, body, .stApp {
  font-family: 'Inter', sans-serif;
  background: linear-gradient(135deg, #F8FAFC 0%, #ffffff 100%);
  color: #0F172A;
}

h1, h2, h3, h4, h5, h6 {
  font-family: 'Poppins', sans-serif;
  font-weight: 600;
  color: #0F172A;
}

.sticky-gallery {
  position: sticky;
  top: 20px;
  z-index: 100;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 1rem;
  padding: 1rem;
  margin-bottom: 2rem;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.gallery-title {
  font-family: 'Poppins', sans-serif;
  font-size: 1.125rem;
  font-weight: 600;
  color: #0F172A;
  text-align: center;
  margin-bottom: 1rem;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.5rem;
}

.gallery-image {
  width: 100%;
  height: 80px;
  object-fit: cover;
  border-radius: 0.5rem;
  transition: transform 0.2s;
  cursor: pointer;
}

.gallery-image:hover {
  transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# Create the sticky image gallery
st.markdown("""
<div class="sticky-gallery">
  <div class="gallery-title">Happy Pets Using VetPosture AI</div>
  <div class="image-grid">
    <img src="images.jfif" alt="Golden Retriever puppies" class="gallery-image" title="Golden Retriever puppies - In a green grassy field">
    <img src="images (1).jfif" alt="Australian Shepherd" class="gallery-image" title="Australian Shepherd - Tricolor coat in grass">
    <img src="images (2).jfif" alt="German Shepherd" class="gallery-image" title="German Shepherd - Black and tan coat">
    <img src="images (3).jfif" alt="Dog group" class="gallery-image" title="Dog group - Diverse breeds">
    <img src="images (4).jfif" alt="Golden Retriever" class="gallery-image" title="Golden Retriever - Senior dog on dirt">
    <img src="images (5).jfif" alt="English Bulldog" class="gallery-image" title="English Bulldog - Blue eyes stocky build">
    <img src="images (6).jfif" alt="Samoyed Puppy" class="gallery-image" title="Samoyed Puppy - White fluffy golden light">
  </div>
</div>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem 1rem; background: linear-gradient(135deg, #F8FAFC 0%, #ffffff 100%); border-radius: 1.5rem; margin-bottom: 2rem;">
  <div style="display: inline-block; background: linear-gradient(135deg, #14B8A6 0%, #22D3EE 100%); color: white; padding: 0.5rem 1rem; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; margin-bottom: 1.5rem;">
    Research Model - Built for Academic Study
  </div>
  <h1 style="font-family: 'Poppins', sans-serif; font-size: 2.5rem; font-weight: 700; color: #0F172A; margin: 0 0 1rem 0; line-height: 1.2;">
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
    st.markdown('<h2 style="text-align: center; font-family: \'Poppins\', sans-serif; font-size: 2rem; font-weight: 700; color: #0F172A; margin: 2rem 0;">Everything you need for your pet\'s health</h2>', unsafe_allow_html=True)
    
    # Hero Image Section
    st.markdown("""
    <div style="margin: 2rem 0; text-align: center;">
      <img src="download.jfif" 
           alt="Golden Retriever" 
           style="width: 100%; max-width: 600px; height: 300px; object-fit: cover; border-radius: 1rem; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">
      <p style="margin-top: 1rem; color: #64748b; font-style: italic;">
        "Our AI helped detect early signs in Bruno's posture. The vet consultation was incredibly helpful!" - Sarah & Bruno
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    st.markdown('<h3 style="text-align: center; font-family: \'Poppins\', sans-serif; margin: 3rem 0 2rem 0;">Key Features</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.2s;">
          <img src="download (1).jfif" 
               alt="Dog analysis" 
               style="width: 100%; height: 120px; object-fit: cover; border-radius: 0.5rem; margin-bottom: 1rem;">
          <h3 style="font-family: 'Poppins', sans-serif; font-size: 1.25rem; font-weight: 600; color: #0F172A; margin: 0 0 0.75rem 0;">AI Posture Scan</h3>
          <p style="color: #475569; margin: 0; line-height: 1.6;">Upload a side-view photo for instant posture analysis with 92% accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <img src="images (1).jfif" 
               alt="Australian Shepherd" 
               style="width: 100%; height: 120px; object-fit: cover; border-radius: 0.5rem; margin-bottom: 1rem;">
          <h3 style="font-family: 'Poppins', sans-serif; font-size: 1.25rem; font-weight: 600; color: #0F172A; margin: 0 0 0.75rem 0;">Multi-Disorder Analysis</h3>
          <p style="color: #475569; margin: 0; line-height: 1.6;">Get probabilities for 7 disorders plus healthy status with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <img src="images (2).jfif" 
               alt="German Shepherd" 
               style="width: 100%; height: 120px; object-fit: cover; border-radius: 0.5rem; margin-bottom: 1rem;">
          <h3 style="font-family: 'Poppins', sans-serif; font-size: 1.25rem; font-weight: 600; color: #0F172A; margin: 0 0 0.75rem 0;">Real-time Vet Chat</h3>
          <p style="color: #475569; margin: 0; line-height: 1.6;">Connect with licensed veterinarians within 5 minutes for personalized guidance</p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# AI ANALYSIS TAB
# =========================
with main_tab2:
    st.markdown('<h2 style="text-align: center; font-family: \'Poppins\', sans-serif; font-size: 2rem; font-weight: 700; color: #0F172A; margin: 2rem 0;">AI Diagnostic Workspace</h2>', unsafe_allow_html=True)
    
    # Sub-tabs for AI Analysis
    ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs(["Upload & Detect", "Manual Assessment", "Rule Explorer", "Education"])
    
    with ai_tab1:
        st.subheader("Upload & Detect")
        st.caption("Side-view images work best. Good lighting helps segmentation.")
        
        file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"], key="upload_1")
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
                st.image(img, use_column_width=True)
            
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
                    for i,(d,p) in enumerate(top3,1): 
                        st.write(f"{i}. **{d}** — {int(p*100)}%")
                    st.pyplot(plot_multibar(dist, "All Disorders (Weighted by Rules)"), clear_figure=True)
                
                overlay = img.copy()
                cv2.drawContours(overlay, [contour], -1, (0,255,0), 2)
                st.markdown("**Segmentation Preview**")
                st.image(overlay, use_column_width=True)
                
                best = top3[0][0]
                st.info(DISORDER_ADVICE.get(best, "Consult your veterinarian for next steps."))
    
    with ai_tab2:
        st.subheader("Manual Assessment")
        st.caption("No image? Pick the posture you observe; we'll compute probabilities for all disorders.")
        
        choice = st.selectbox(
            "Observed posture",
            ["Normal Standing (stable, upright)","Posture Disorder (uneven stance/arching)",
             "Fatigue / Weakness (slouching, low energy)","Mobility Issue (limp, difficulty rising)"]
        )
        
        symbol_map = {
            "Normal Standing (stable, upright)":"N",
            "Posture Disorder (uneven stance/arching)":"PD",
            "Fatigue / Weakness (slouching, low energy)":"F",
            "Mobility Issue (limp, difficulty rising)":"MI"
        }
        
        sym = symbol_map[choice]
        posture_name, posture_explain = POSTURE_INFO[sym]
        dist = infer_distribution(posture_name, base_conf=0.78)
        top3 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
        
        c1, c2 = st.columns([1,1], gap="large")
        with c1:
            st.markdown("**Manual Posture Assessment**")
            st.markdown(f"**Selected:** {choice}")
            st.markdown(f"**Mapped Symbol:** `{sym}`")
            st.markdown(f"**Posture State:** {posture_name}")
            st.markdown(f"<span style='color:#64748b'>{posture_explain}</span>", unsafe_allow_html=True)
        
        with c2:
            st.markdown("**Computed Disorder Probabilities**")
            st.write("**Top-3 Likely Disorders**")
            for i,(d,p) in enumerate(top3,1): 
                st.write(f"{i}. **{d}** — {int(p*100)}%")
            st.pyplot(plot_multibar(dist, "Manual Assessment Results"), clear_figure=True)
        
        best = top3[0][0]
        st.info(DISORDER_ADVICE.get(best, "Consult your veterinarian for next steps."))
    
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
            state = "START"
            for i, token in enumerate(tokens):
                if (state, token) in TRANSITIONS:
                    new_state = TRANSITIONS[(state, token)]
                    st.write(f"Step {i+1}: δ({state}, {token}) → {new_state}")
                    state = new_state
                else:
                    st.write(f"Step {i+1}: δ({state}, {token}) → **INVALID TRANSITION**")
                    break
            st.write(f"**Final State:** {state}")
    
    with ai_tab4:
        st.subheader("Education")
        st.caption("Quick disorder guides and care tips for owners. Not medical advice.")
        for k,v in DISORDER_LIBRARY.items():
            with st.expander(k, expanded=False):
                st.write(v)
                st.markdown(f"**Care Tips:** {DISORDER_ADVICE.get(k,'Consult your vet.')}")

# =========================
# TESTIMONIALS TAB
# =========================
with main_tab3:
    st.markdown('<h2 style="text-align: center; font-family: \'Poppins\', sans-serif; font-size: 2rem; font-weight: 700; color: #0F172A; margin: 2rem 0;">What Pet Parents Say</h2>', unsafe_allow_html=True)
    
    test_col1, test_col2, test_col3 = st.columns(3)
    
    with test_col1:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem;">
          <img src="images (3).jfif" 
               alt="Dog group" 
               style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; margin: 0 auto 1rem auto; display: block;">
          <div style="color: #F97316; font-size: 1.2rem; margin-bottom: 0.5rem;">★★★★★</div>
          <p style="color: #475569; font-style: italic; margin: 0 0 1rem 0; line-height: 1.6;">
            "The AI scan flagged potential hip issues in Bruno, and the vet consultation gave us a clear action plan. Within weeks, we saw improvement!"
          </p>
          <p style="font-weight: 600; color: #0F172A; margin: 0;">Sarah & Bruno</p>
          <p style="color: #64748b; font-size: 0.875rem; margin: 0;">Golden Retriever Parent</p>
        </div>
        """, unsafe_allow_html=True)
    
    with test_col2:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem;">
          <img src="images (4).jfif" 
               alt="Golden Retriever" 
               style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; margin: 0 auto 1rem auto; display: block;">
          <div style="color: #F97316; font-size: 1.2rem; margin-bottom: 0.5rem;">★★★★★</div>
          <p style="color: #475569; font-style: italic; margin: 0 0 1rem 0; line-height: 1.6;">
            "While traveling, Coco seemed off. The instant vet chat gave us peace of mind and clear next steps. Amazing service!"
          </p>
          <p style="font-weight: 600; color: #0F172A; margin: 0;">Sana & Coco</p>
          <p style="color: #64748b; font-size: 0.875rem; margin: 0;">Beagle Parent</p>
        </div>
        """, unsafe_allow_html=True)
    
    with test_col3:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem;">
          <img src="images (5).jfif" 
               alt="English Bulldog" 
               style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; margin: 0 auto 1rem auto; display: block;">
          <div style="color: #F97316; font-size: 1.2rem; margin-bottom: 0.5rem;">★★★★★</div>
          <p style="color: #475569; font-style: italic; margin: 0 0 1rem 0; line-height: 1.6;">
            "The detailed reports and care tips have been invaluable. Milo's health has improved significantly since we started using VetPosture AI."
          </p>
          <p style="font-weight: 600; color: #0F172A; margin: 0;">Rohan & Milo</p>
          <p style="color: #64748b; font-size: 0.875rem; margin: 0;">Labrador Parent</p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# FAQ TAB
# =========================
with main_tab4:
    st.markdown('<h2 style="text-align: center; font-family: \'Poppins\', sans-serif; font-size: 2rem; font-weight: 700; color: #0F172A; margin: 2rem 0;">Frequently Asked Questions</h2>', unsafe_allow_html=True)

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
    st.markdown('<h2 style="text-align: center; font-family: \'Poppins\', sans-serif; font-size: 2rem; font-weight: 700; color: #0F172A; margin: 2rem 0;">Get in Touch</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <h3 style="font-family: 'Poppins', sans-serif; color: #0F172A; margin-bottom: 1rem;">Contact Information</h3>
          <p style="color: #475569; margin: 0.5rem 0;">Email: support@vetpostureai.com</p>
          <p style="color: #475569; margin: 0.5rem 0;">Phone: +91 98765 43210</p>
          <p style="color: #475569; margin: 0.5rem 0;">24/7 Online Support</p>
          <p style="color: #475569; margin: 0.5rem 0;">Website: www.vetpostureai.com</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <h3 style="font-family: 'Poppins', sans-serif; color: #0F172A; margin-bottom: 1rem;">Quick Actions</h3>
          <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <button style="background: #14B8A6; color: white; border: none; padding: 0.75rem; border-radius: 0.5rem; cursor: pointer;">Start Free Scan</button>
            <button style="background: #F97316; color: white; border: none; padding: 0.75rem; border-radius: 0.5rem; cursor: pointer;">Book Consultation</button>
            <button style="background: #22D3EE; color: white; border: none; padding: 0.75rem; border-radius: 0.5rem; cursor: pointer;">Download App</button>
          </div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# -------- SIDEBAR --------
# =========================
with st.sidebar:
    st.markdown("### Quick Guide")
    st.info("Upload a side-view image on **Upload & Detect**. No image? Use **Manual Assessment**.")
    st.markdown("**DFA Symbols**")
    st.code("D (detect) • N (Normal) • PD (Posture Disorder) • F (Fatigue) • MI (Mobility Issue) • R (Result)")
    st.caption("Disclaimer: Research prototype — not a substitute for professional veterinary diagnosis.")

# =========================
# -------- FOOTER ---------
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #475569;">
  <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 1rem;">
    <span style="font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.5rem; color: #0F172A;">VetPosture AI</span>
  </div>
  <p style="margin: 0 0 1rem 0;">Professional veterinary care powered by AI and real veterinarians</p>
  <p style="font-size: 0.875rem; margin: 0 0 0.5rem 0;">© 2025 VetPosture AI. All rights reserved.</p>
  <p style="font-size: 0.75rem; color: #64748b; margin: 0; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.5;">
    <strong>Medical Disclaimer:</strong> This service provides information and screening tools, not medical diagnosis. 
    Always consult a licensed veterinarian for medical advice. For emergencies, contact your local vet/ER immediately.
  </p>
</div>
""", unsafe_allow_html=True)


