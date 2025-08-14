# app.py
# AI-Powered Resume Segregator & Candidate Ranking System
# Professional Resume Analysis Platform using Google Gemini AI
# 
# Features:
# - Intelligent Job Description Analysis
# - AI-Powered Resume Scoring & Ranking
# - Professional Dashboard & Analytics
# - Export & Reporting Capabilities
# - Advanced Filtering & Search
#
# Usage
# 1) Set environment variable:  export GOOGLE_API_KEY="<your_api_key>"
# 2) Install deps:            pip install -r requirements.txt
# 3) Run app:                 streamlit run app.py
#
# Notes
# - We do NOT send your files anywhere except to Google for generating embeddings/LLM responses, per your key usage.
# - For on-prem privacy, replace LLM/embedding calls with local models.
import io
import os
import re
import json
import math
import time
import base64
import zipfile
from typing import List, Dict, Any, Tuple
from datetime import datetime

import streamlit as st
import pandas as pd
from pypdf import PdfReader

# Google Generative AI (Gemini)
try:
    import google.generativeai as genai
except Exception as e:
    genai = None

###############################################################################
# ---------------------------- CONFIGURATION -------------------------------- #
###############################################################################
APP_TITLE = "AI-Powered Resume Segregator Pro"
APP_SUBTITLE = "Intelligent Candidate Ranking & Analysis Platform"
EMBED_MODEL = "text-embedding-004"         # Google text embedding model
GEN_MODEL = "gemini-1.5-flash"             # Fast, cost-effective for extraction
MAX_PAGES_PER_PDF = 20                      # Safety cap to avoid very large PDFs
MAX_CHARS_PER_DOC = 60_000                  # Truncate extremely large texts
SKILL_WEIGHT = 0.5                          # Weight for skill match in composite
SIMILARITY_WEIGHT = 0.5                     # Weight for semantic similarity

# Professional color scheme
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e", 
    "success": "#2ca02c",
    "warning": "#d62728",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40"
}

###############################################################################
# --------------------------- UTILS & HELPERS ------------------------------- #
###############################################################################

def ensure_google_configured() -> None:
    """Configure Google GenAI using env var; raise informative error otherwise."""
    if genai is None:
        raise RuntimeError(
            "google-generativeai not installed. Please install requirements.txt"
        )
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Export your Google API key before running."
        )
    genai.configure(api_key=api_key)


def read_pdf_bytes_to_text(pdf_bytes: bytes, max_pages: int = MAX_PAGES_PER_PDF) -> str:
    """Extract text from a PDF byte string with a page cap for performance."""
    text_parts: List[str] = []
    try:
        with io.BytesIO(pdf_bytes) as tmp:
            reader = PdfReader(tmp)
            n_pages = min(len(reader.pages), max_pages)
            for i in range(n_pages):
                try:
                    page = reader.pages[i]
                    text_parts.append(page.extract_text() or "")
                except Exception as e:
                    st.warning(f"Warning: Could not read page {i+1}: {str(e)}")
                    continue
        text = "\n".join(text_parts)
        # Normalize whitespace and truncate
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > MAX_CHARS_PER_DOC:
            text = text[:MAX_CHARS_PER_DOC]
            st.info(f"Document truncated to {MAX_CHARS_PER_DOC:,} characters for performance")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def unzip_and_collect_pdfs(zip_bytes: bytes) -> List[Tuple[str, bytes]]:
    """Return list of (filename, pdf_bytes) from a ZIP uploaded by the user."""
    files: List[Tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                fname = info.filename
                if not fname.lower().endswith((".pdf",)):
                    continue
                with zf.open(info) as f:
                    files.append((os.path.basename(fname), f.read()))
        return files
    except Exception as e:
        st.error(f"Error reading ZIP file: {str(e)}")
        return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    except Exception:
        return 0.0


def embed_text(text: str) -> List[float]:
    """Get embedding vector for text via Google GenAI."""
    ensure_google_configured()
    try:
        # Gemini embedding API returns dict with 'embedding'
        resp = genai.embed_content(model=EMBED_MODEL, content=text)
        # Response formats can vary; handle common shapes
        if isinstance(resp, dict) and "embedding" in resp:
            return resp["embedding"]
        if hasattr(resp, "embedding"):
            return resp.embedding
        raise RuntimeError("Unexpected embedding response format from Google API")
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return []


def extract_constraints_from_jd(jd_text: str) -> Dict[str, Any]:
    """Call Gemini to extract structured requirements from the JD."""
    ensure_google_configured()
    prompt = f"""
    You are an expert HR analyst with 15+ years of experience in talent acquisition and job analysis.
    
    Analyze the following Job Description and extract structured hiring requirements.
    Return STRICT JSON with the following keys:
    
    {{
        "role": "Job title/position",
        "must_have_skills": ["Critical skills required", "Essential qualifications"],
        "nice_to_have_skills": ["Preferred skills", "Bonus qualifications"],
        "min_years_experience": <integer if specified, else null>,
        "education": ["Required degrees", "Preferred education"],
        "locations": ["Work locations", "Remote options"],
        "certifications": ["Required certifications", "Preferred certs"],
        "keywords": ["Industry-specific terms", "Technical keywords"],
        "seniority_level": "Junior/Mid/Senior/Lead/Manager/Executive",
        "industry": "Technology/Finance/Healthcare/etc",
        "salary_range": "If mentioned, else null"
    }}
    
    Be precise and professional. If a field is missing in the JD, return an empty array or null accordingly.
    
    JD TEXT:
    {jd_text}
    
    JSON ONLY (no explanations):
    """
    
    try:
        model = genai.GenerativeModel(GEN_MODEL)
        out = model.generate_content(prompt)
        text = out.text if hasattr(out, "text") else str(out)
        
        # Try to locate JSON in the output
        match = re.search(r"\{[\s\S]*\}\s*$", text)
        raw_json = match.group(0) if match else text
        
        data = json.loads(raw_json)
        
        # Ensure all required fields exist
        required_fields = {
            "role", "must_have_skills", "nice_to_have_skills", 
            "min_years_experience", "education", "locations", 
            "certifications", "keywords", "seniority_level", 
            "industry", "salary_range"
        }
        
        for field in required_fields:
            if field not in data:
                if field in ["must_have_skills", "nice_to_have_skills", "education", "locations", "certifications", "keywords"]:
                    data[field] = []
                else:
                    data[field] = None
                    
        return data
        
    except Exception as e:
        st.warning(f"AI analysis failed, using fallback extraction: {str(e)}")
        # Fallback minimal schema
        return {
            "role": "Unknown",
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "min_years_experience": None,
            "education": [],
            "locations": [],
            "certifications": [],
            "keywords": [],
            "seniority_level": "Unknown",
            "industry": "Unknown",
            "salary_range": None
        }


def score_resume(
    resume_text: str,
    jd_text: str,
    jd_constraints: Dict[str, Any],
    jd_embedding: List[float],
) -> Dict[str, Any]:
    """Compute semantic similarity + rule-based matches to produce a composite score."""
    try:
        # Embedding similarity
        emb = embed_text(resume_text)
        if not emb:
            return {"error": "Failed to generate embeddings"}
            
        sim = cosine_similarity(emb, jd_embedding)

        # Skill coverage (improved token matching)
        def norm_tokens(xs: List[str]) -> List[str]:
            out = []
            for x in xs or []:
                x = re.sub(r"[^a-zA-Z0-9+#.]", " ", x or "").lower().strip()
                if x and len(x) > 2:  # Filter out very short tokens
                    out.append(x)
            return out

        must = set(norm_tokens(jd_constraints.get("must_have_skills", [])))
        nice = set(norm_tokens(jd_constraints.get("nice_to_have_skills", [])))

        res_tokens = set(re.split(r"[^a-zA-Z0-9+#.]+", resume_text.lower()))
        res_tokens = {t for t in res_tokens if len(t) > 2}  # Filter short tokens

        must_hits = len([s for s in must if s in res_tokens])
        nice_hits = len([s for s in nice if s in res_tokens])
        must_cov = (must_hits / max(len(must), 1))
        nice_cov = (nice_hits / max(len(nice), 1))

        # Experience heuristic: find years of experience mentions
        yrs = 0
        yr_matches = re.findall(r"(\d+)(?:\+)?\s*(?:years|yrs)\s*(?:of)?\s*experience", resume_text.lower())
        if yr_matches:
            yrs = max(int(x) for x in yr_matches)
        min_yrs = jd_constraints.get("min_years_experience") or 0
        exp_ok = 1.0 if yrs >= min_yrs else (yrs / max(min_yrs, 1))

        # Education matching
        edu_score = 0
        if jd_constraints.get("education"):
            resume_lower = resume_text.lower()
            for edu in jd_constraints["education"]:
                if edu.lower() in resume_lower:
                    edu_score += 1
            edu_score = min(edu_score / len(jd_constraints["education"]), 1.0)

        # Composite score with improved weighting
        skill_score = 0.7 * must_cov + 0.3 * nice_cov
        composite = (
            SIMILARITY_WEIGHT * sim + 
            SKILL_WEIGHT * (0.6 * skill_score + 0.3 * exp_ok + 0.1 * edu_score)
        )

        return {
            "semantic_similarity": round(sim, 3),
            "must_have_coverage": round(must_cov, 3),
            "nice_to_have_coverage": round(nice_cov, 3),
            "experience_ok": round(exp_ok, 3),
            "education_score": round(edu_score, 3),
            "estimated_years_experience": yrs,
            "composite_score": round(composite, 3),
            "skill_match_percentage": round(skill_score * 100, 1),
            "overall_fit": "Excellent" if composite > 0.8 else "Good" if composite > 0.6 else "Fair" if composite > 0.4 else "Poor"
        }
        
    except Exception as e:
        return {"error": f"Scoring failed: {str(e)}"}


def create_metrics_display(df: pd.DataFrame) -> None:
    """Display key metrics and insights."""
    if df.empty:
        return
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = df['composite_score'].mean()
        st.metric("Average Score", f"{avg_score:.2f}")
        
    with col2:
        top_score = df['composite_score'].max()
        st.metric("Top Score", f"{top_score:.2f}")
        
    with col3:
        total_candidates = len(df)
        st.metric("Total Candidates", total_candidates)
        
    with col4:
        excellent_candidates = len(df[df['composite_score'] > 0.8])
        st.metric("Excellent Fits", excellent_candidates)


def create_visualizations(df: pd.DataFrame) -> None:
    """Create professional charts and visualizations."""
    if df.empty:
        return
        
    st.subheader("📊 Analytics & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        st.write("**Score Distribution**")
        score_ranges = pd.cut(df['composite_score'], bins=[0, 0.4, 0.6, 0.8, 1.0], 
                             labels=['Poor (0-0.4)', 'Fair (0-0.6)', 'Good (0.6-0.8)', 'Excellent (0.8-1.0)'])
        score_counts = score_ranges.value_counts()
        st.bar_chart(score_counts)
        
    with col2:
        # Top skills coverage
        if 'must_have_coverage' in df.columns:
            st.write("**Must-Have Skills Coverage**")
            avg_must_cover = df['must_have_coverage'].mean()
            st.metric("Average Coverage", f"{avg_must_cover:.1%}")
            
            # Top performers
            top_3 = df.nlargest(3, 'composite_score')[['file_name', 'composite_score']]
            st.write("**Top 3 Candidates**")
            for idx, row in top_3.iterrows():
                st.write(f"• {row['file_name']}: {row['composite_score']:.2f}")


def is_resume_relevant(resume_text: str, jd_constraints: Dict[str, Any], threshold: float = 0.3) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if resume is relevant based on job description constraints.
    Returns (is_relevant, relevance_details)
    """
    try:
        resume_text_lower = resume_text.lower()
        relevance_score = 0.0
        total_checks = 0
        passed_checks = 0
        details = {
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "education_match": False,
            "experience_match": False,
            "overall_relevance": 0.0
        }
        
        # Check must-have skills
        if jd_constraints.get("must_have_skills"):
            total_checks += len(jd_constraints["must_have_skills"])
            for skill in jd_constraints["must_have_skills"]:
                skill_lower = skill.lower()
                # Use word boundary matching for more accurate results
                if re.search(r'\b' + re.escape(skill_lower) + r'\b', resume_text_lower):
                    details["must_have_skills"].append(skill)
                    passed_checks += 1
                    relevance_score += 0.4  # Higher weight for must-have skills
        
        # Check nice-to-have skills
        if jd_constraints.get("nice_to_have_skills"):
            for skill in jd_constraints["nice_to_have_skills"]:
                skill_lower = skill.lower()
                if re.search(r'\b' + re.escape(skill_lower) + r'\b', resume_text_lower):
                    details["nice_to_have_skills"].append(skill)
                    relevance_score += 0.2  # Lower weight for nice-to-have skills
        
        # Check education requirements
        if jd_constraints.get("education"):
            for edu in jd_constraints["education"]:
                if edu.lower() in resume_text_lower:
                    details["education_match"] = True
                    relevance_score += 0.2
                    break
        
        # Check experience requirements
        min_years = jd_constraints.get("min_years_experience")
        if min_years:
            yr_matches = re.findall(r'(\d+)(?:\+)?\s*(?:years|yrs)\s*(?:of)?\s*experience', resume_text_lower)
            if yr_matches:
                max_yrs = max(int(x) for x in yr_matches)
                if max_yrs >= min_years:
                    details["experience_match"] = True
                    relevance_score += 0.2
        
        # Calculate overall relevance
        if total_checks > 0:
            details["overall_relevance"] = min(relevance_score, 1.0)
            is_relevant = details["overall_relevance"] >= threshold
        else:
            # Fallback: if no specific constraints, use basic keyword matching
            basic_keywords = jd_constraints.get("keywords", [])
            if basic_keywords:
                for keyword in basic_keywords:
                    if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', resume_text_lower):
                        relevance_score += 0.1
                details["overall_relevance"] = min(relevance_score, 1.0)
                is_relevant = details["overall_relevance"] >= threshold
            else:
                # No constraints available, accept by default
                is_relevant = True
                details["overall_relevance"] = 0.5
        
        return is_relevant, details
        
    except Exception as e:
        st.warning(f"Error in relevance check: {str(e)}")
        return True, {"overall_relevance": 0.5, "error": str(e)}


def filter_relevant_resumes(resume_files: List[Tuple[str, bytes]], jd_constraints: Dict[str, Any], 
                           threshold: float = 0.3) -> Tuple[List[Tuple[str, bytes]], List[Tuple[str, str]]]:
    """
    Filter resumes based on relevance to job description.
    Returns (relevant_resumes, rejected_resumes_with_reasons)
    """
    relevant_resumes = []
    rejected_resumes = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (fname, pdf_bytes) in enumerate(resume_files):
        try:
            status_text.text(f"Analyzing relevance: {fname}")
            
            # Extract text from PDF
            rtext = read_pdf_bytes_to_text(pdf_bytes, max_pages=MAX_PAGES_PER_PDF)
            if not rtext:
                rejected_resumes.append((fname, "Could not extract text from PDF"))
                continue
            
            # Check relevance
            is_relevant, relevance_details = is_resume_relevant(rtext, jd_constraints, threshold)
            
            if is_relevant:
                relevant_resumes.append((fname, pdf_bytes))
            else:
                reason = f"Relevance score: {relevance_details['overall_relevance']:.2f} (below threshold {threshold})"
                if relevance_details.get("must_have_skills"):
                    reason += f" | Missing critical skills: {', '.join(relevance_details['must_have_skills'])}"
                rejected_resumes.append((fname, reason))
            
            progress_bar.progress((idx + 1) / len(resume_files))
            
        except Exception as e:
            rejected_resumes.append((fname, f"Error during analysis: {str(e)}"))
            progress_bar.progress((idx + 1) / len(resume_files))
    
    progress_bar.empty()
    status_text.empty()
    
    return relevant_resumes, rejected_resumes


###############################################################################
# --------------------------------- UI -------------------------------------- #
###############################################################################

# Page configuration
st.set_page_config(
    page_title="Resume Segregator Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1565c0, #e65100);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1> AI-Powered Resume Segregator Pro</h1>
    <p style="font-size: 1.2rem; margin: 0;">Intelligent Candidate Ranking & Analysis Platform</p>
    <p style="font-size: 1rem; margin: 0.5rem 0 0 0;">Powered by Google Gemini AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key status (hidden for security)
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        st.success("✅ Google API Key Configured")
        # Hidden API key display for security
        with st.expander("🔐 API Key Status (Click to view)", expanded=False):
            st.info("API key is properly configured and ready to use")
            if st.button("Show API Key (for debugging only)", type="secondary"):
                st.code(api_key, language="text")
                st.warning("⚠️ Keep this key secure and don't share it")
    else:
        st.error("❌ Google API Key Not Set")
        st.info("Set GOOGLE_API_KEY environment variable")
    
    st.markdown("---")
    
    # Performance settings
    st.subheader("📊 Performance Settings")
    max_pages = st.number_input("Max pages per PDF", min_value=1, max_value=100, value=MAX_PAGES_PER_PDF, key="max_pages")
    max_chars = st.number_input("Max chars per doc", min_value=5_000, max_value=200_000, value=MAX_CHARS_PER_DOC, step=5_000, key="max_chars")
    
    st.markdown("---")
    
    # Scoring weights
    st.subheader("⚖️ Scoring Weights")
    sim_w = st.slider("Similarity weight", 0.0, 1.0, SIMILARITY_WEIGHT, 0.05, key="sim_w")
    skill_w = st.slider("Skills weight", 0.0, 1.0, SKILL_WEIGHT, 0.05, key="skill_w")
    
    st.markdown("---")
    
    # Relevance filtering
    st.subheader("🔍 Relevance Filtering")
    relevance_threshold = st.slider("Relevance Threshold", 0.1, 0.9, 0.3, 0.1, 
                                   help="Resumes below this threshold will be automatically rejected")
    enable_filtering = st.checkbox("Enable Automatic Resume Filtering", value=True,
                                  help="Automatically reject irrelevant resumes before scoring")
    
    st.markdown("---")
    
    # App info
    st.subheader("ℹ️ About")
    st.markdown("""
    **Version:** 2.0 Pro  
    **AI Model:** Gemini 1.5 Flash  
    **Last Updated:** """ + datetime.now().strftime("%B %Y"))
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        st.rerun()

# Update globals from sidebar
MAX_PAGES_PER_PDF = st.session_state.get("max_pages", MAX_PAGES_PER_PDF)
MAX_CHARS_PER_DOC = st.session_state.get("max_chars", MAX_CHARS_PER_DOC)
SIMILARITY_WEIGHT = st.session_state.get("sim_w", SIMILARITY_WEIGHT)
SKILL_WEIGHT = st.session_state.get("skill_w", SKILL_WEIGHT)
RELEVANCE_THRESHOLD = st.session_state.get("relevance_threshold", 0.3)
ENABLE_FILTERING = st.session_state.get("enable_filtering", True)

# Main content area
st.markdown("### 📋 Upload Documents")

# File uploaders with better styling
col1, col2 = st.columns(2)

with col1:
    st.markdown("**📄 Job Description**")
    jd_file = st.file_uploader(
        "Upload Job Description (PDF)", 
        type=["pdf"], 
        accept_multiple_files=False,
        help="Single PDF file containing the job description"
    )
    if jd_file:
        st.success(f"✅ JD uploaded: {jd_file.name}")

with col2:
    st.markdown("**📁 Resume Collection**")
    zip_file = st.file_uploader(
        "Upload Resumes (ZIP of PDFs)", 
        type=["zip"], 
        accept_multiple_files=False,
        help="ZIP file containing multiple PDF resumes"
    )
    if zip_file:
        st.success(f"✅ Resumes uploaded: {zip_file.name}")

# Analysis button
st.markdown("---")
run_btn = st.button("🚀 Analyze & Rank Resumes", type="primary", use_container_width=True, help="Click to start the AI-powered analysis")

if run_btn:
    if not jd_file or not zip_file:
        st.error("❌ Please upload both the Job Description PDF and the ZIP of resumes.")
        st.stop()

    try:
        ensure_google_configured()
    except Exception as e:
        st.error(f"❌ Google API configuration error: {e}")
        st.stop()

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Extract JD text
    with st.spinner("📖 Reading Job Description PDF..."):
        status_text.text("Step 1/5: Reading Job Description...")
        jd_text = read_pdf_bytes_to_text(jd_file.read(), max_pages=MAX_PAGES_PER_PDF)
        if not jd_text:
            st.error("❌ Could not extract text from the JD PDF.")
            st.stop()
        progress_bar.progress(20)

    # Extract constraints
    with st.spinner("🤖 Extracting constraints from JD using Gemini AI..."):
        status_text.text("Step 2/5: AI Analysis of Job Requirements...")
        constraints = extract_constraints_from_jd(jd_text)
        progress_bar.progress(40)

    # Display parsed constraints
    st.markdown("---")
    st.subheader("🎯 Parsed Job Requirements")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Basic Information**")
        st.write(f"**Role:** {constraints.get('role', 'N/A')}")
        st.write(f"**Seniority:** {constraints.get('seniority_level', 'N/A')}")
        st.write(f"**Industry:** {constraints.get('industry', 'N/A')}")
        if constraints.get('min_years_experience'):
            st.write(f"**Min Experience:** {constraints.get('min_years_experience')} years")
    
    with col2:
        st.markdown("**📍 Location & Education**")
        if constraints.get('locations'):
            st.write(f"**Locations:** {', '.join(constraints['locations'])}")
        if constraints.get('education'):
            st.write(f"**Education:** {', '.join(constraints['education'])}")
    
    # Skills breakdown
    st.markdown("**🔧 Skills Analysis**")
    col1, col2 = st.columns(2)
    with col1:
        if constraints.get('must_have_skills'):
            st.markdown("**Must-Have Skills:**")
            for skill in constraints['must_have_skills']:
                st.write(f"• {skill}")
    with col2:
        if constraints.get('nice_to_have_skills'):
            st.markdown("**Nice-to-Have Skills:**")
            for skill in constraints['nice_to_have_skills']:
                st.write(f"• {skill}")

    # JD embedding
    with st.spinner("🔍 Generating JD embeddings..."):
        status_text.text("Step 3/5: Processing Job Description...")
        jd_emb = embed_text(jd_text)
        if not jd_emb:
            st.error("❌ Failed to generate JD embeddings.")
            st.stop()
        progress_bar.progress(60)

    # Unzip resumes
    with st.spinner("📦 Unzipping and reading resumes..."):
        status_text.text("Step 4/5: Processing Resume Files...")
        resume_files = unzip_and_collect_pdfs(zip_file.read())
        if not resume_files:
            st.error("❌ No PDF resumes found in the ZIP.")
            st.stop()
        progress_bar.progress(80)

    # Filter resumes based on relevance (if enabled)
    relevant_resumes = resume_files
    rejected_resumes = []
    
    if ENABLE_FILTERING:
        with st.spinner("🔍 Filtering resumes based on relevance..."):
            status_text.text("Step 4.5/5: Relevance Filtering...")
            relevant_resumes, rejected_resumes = filter_relevant_resumes(
                resume_files, constraints, RELEVANCE_THRESHOLD
            )
            
            if not relevant_resumes:
                st.error("❌ No resumes passed the relevance threshold. Consider lowering the threshold.")
                st.stop()
            
            # Display filtering results
            st.info(f"📊 **Filtering Results:** {len(relevant_resumes)} relevant resumes, {len(rejected_resumes)} rejected")
            
            if rejected_resumes:
                with st.expander(f"❌ Rejected Resumes ({len(rejected_resumes)})", expanded=False):
                    rejected_df = pd.DataFrame(rejected_resumes, columns=["File Name", "Rejection Reason"])
                    st.dataframe(rejected_df, use_container_width=True)
                    
                    # Download rejected resumes list
                    rejected_csv = rejected_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Rejected Resumes List (CSV)",
                        data=rejected_csv,
                        file_name=f"rejected_resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    # Score resumes (only relevant ones)
    status_text.text("Step 5/5: AI Scoring & Ranking...")
    rows = []
    
    for idx, (fname, pdf_bytes) in enumerate(relevant_resumes, start=1):
        try:
            rtext = read_pdf_bytes_to_text(pdf_bytes, max_pages=MAX_PAGES_PER_PDF)
            if not rtext:
                continue
            metrics = score_resume(rtext, jd_text, constraints, jd_emb)
            if "error" not in metrics:
                rows.append({"file_name": fname, **metrics, "resume_text_preview": rtext[:500]})
            else:
                rows.append({"file_name": fname, "error": metrics["error"]})
        except Exception as e:
            rows.append({"file_name": fname, "error": str(e)})
        
        progress_bar.progress(80 + (idx / len(relevant_resumes)) * 20)

    if not rows:
        st.warning("⚠️ No parseable resumes found. Please verify the PDFs are text-based (not scanned images).")
        st.stop()

    # Complete progress
    progress_bar.progress(100)
    status_text.text("✅ Analysis Complete!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

    # Create results dataframe
    df = pd.DataFrame(rows)
    
    # Sort by composite score desc
    if "composite_score" in df.columns:
        df = df.sort_values("composite_score", ascending=False)

    # Display metrics
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Show filtering summary if filtering was enabled
    if ENABLE_FILTERING and rejected_resumes:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Resumes", len(resume_files))
        with col2:
            st.metric("Relevant Resumes", len(relevant_resumes))
        with col3:
            st.metric("Rejected Resumes", len(rejected_resumes))
        with col4:
            rejection_rate = (len(rejected_resumes) / len(resume_files)) * 100
            st.metric("Rejection Rate", f"{rejection_rate:.1f}%")
        
        st.markdown("---")
    
    create_metrics_display(df)

    # Display ranked results
    st.subheader("🏆 Ranked Candidates")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.1)
    with col2:
        show_errors = st.checkbox("Show Failed Analyses", value=False)
    with col3:
        sort_by = st.selectbox("Sort By", ["composite_score", "semantic_similarity", "must_have_coverage"])
    
    # Apply filters
    filtered_df = df.copy()
    if "composite_score" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["composite_score"] >= min_score]
    
    if not show_errors:
        filtered_df = filtered_df[~filtered_df["file_name"].str.contains("error", na=False)]
    
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)
    
    # Display results
    if not filtered_df.empty:
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Rankings (CSV)",
                data=csv_bytes,
                file_name=f"resume_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with col2:
            # Top candidates summary
            top_k = st.number_input("How many top candidates to highlight?", min_value=1, max_value=min(20, len(filtered_df)), value=min(5, len(filtered_df)))
            
        # Top candidates display
        st.markdown("---")
        st.subheader(f"⭐ Top {top_k} Candidates")
        
        if "composite_score" in filtered_df.columns:
            top_candidates = filtered_df.head(int(top_k))
            cols = ["file_name", "composite_score", "semantic_similarity", "must_have_coverage", "nice_to_have_coverage", "estimated_years_experience", "overall_fit"]
            st.dataframe(top_candidates[cols], use_container_width=True)
        
        # Visualizations
        create_visualizations(filtered_df)
        
    else:
        st.warning("⚠️ No candidates match the current filters.")

    # Success message
    st.markdown("---")
    st.markdown("""
    <div class="success-box">
        <h4>🎉 Analysis Complete!</h4>
        <p>Your resume analysis has been completed successfully. The AI has ranked all candidates based on their fit for the position.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>AI-Powered Resume Segregator Pro | Built with Streamlit & Google Gemini AI</p>
    <p>Professional candidate ranking and analysis platform</p>
</div>
""", unsafe_allow_html=True)
