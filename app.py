import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intent-Aware Price Drop Predictor",
    page_icon="🎯",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f9; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0d2137 0%, #1e3a5f 50%, #2d6a9f 100%);
        border-radius: 20px;
        padding: 3rem 3.5rem;
        margin-bottom: 2.5rem;
        color: white;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0 0 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero .subtitle {
        font-size: 1.1rem;
        opacity: 0.82;
        margin: 0;
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.5;
    }
    .hero .badges {
        margin-top: 1.4rem;
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .hero .badge {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 999px;
        padding: 0.3rem 1rem;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* Section headings */
    .section-heading {
        font-size: 0.78rem;
        font-weight: 700;
        color: #2d6a9f;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0 0 1rem;
    }

    /* Page cards */
    .page-card {
        background: white;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        height: 100%;
        border-top: 4px solid transparent;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .page-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.11);
    }
    .pc-prediction { border-top-color: #2d6a9f; }
    .pc-recommend  { border-top-color: #28a745; }
    .pc-insights   { border-top-color: #fd7e14; }

    .pc-icon  { font-size: 2.4rem; margin-bottom: 0.75rem; line-height: 1; }
    .pc-title { font-size: 1.15rem; font-weight: 700; color: #1e3a5f; margin: 0 0 0.5rem; }
    .pc-desc  { font-size: 0.88rem; color: #555; line-height: 1.6; margin: 0 0 1rem; }

    .pc-steps { list-style: none; padding: 0; margin: 0; }
    .pc-steps li {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        font-size: 0.84rem;
        color: #444;
        padding: 0.35rem 0;
        border-bottom: 1px solid #f4f4f4;
        line-height: 1.45;
    }
    .pc-steps li:last-child { border-bottom: none; }
    .step-num {
        min-width: 22px;
        height: 22px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 700;
        color: white;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .sn-blue   { background: #2d6a9f; }
    .sn-green  { background: #28a745; }
    .sn-orange { background: #fd7e14; }

    .pc-nav-hint {
        display: inline-block;
        margin-top: 0.9rem;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        border: 1.5px solid;
    }
    .hint-blue   { color: #2d6a9f; border-color: #2d6a9f; background: #eef5fc; }
    .hint-green  { color: #28a745; border-color: #28a745; background: #edfbf1; }
    .hint-orange { color: #fd7e14; border-color: #fd7e14; background: #fff4ec; }

    /* How it works flow */
    .flow-wrap {
        background: white;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        margin-bottom: 2rem;
    }
    .flow-steps {
        display: flex;
        align-items: center;
        gap: 0;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .flow-step {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: #f0f4f9;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        flex: 1;
        min-width: 140px;
    }
    .flow-step .fnum {
        width: 28px; height: 28px;
        background: #1e3a5f;
        color: white;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.78rem; font-weight: 800;
        flex-shrink: 0;
    }
    .flow-step .ftext { font-size: 0.83rem; font-weight: 600; color: #1e3a5f; line-height: 1.3; }
    .flow-arrow { font-size: 1.2rem; color: #aac4de; padding: 0 0.4rem; flex-shrink: 0; }

    /* Requirements box */
    .req-box {
        background: #fffbf0;
        border: 1.5px solid #ffc107;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 2rem;
    }
    .req-box .req-title { font-size: 0.85rem; font-weight: 700; color: #856404; margin: 0 0 0.7rem; }
    .req-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.85rem;
        color: #555;
        padding: 0.25rem 0;
    }
    .req-icon { font-size: 1rem; flex-shrink: 0; }

    /* Sidebar styles */
    section[data-testid="stSidebar"] { background: #0d2137; }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li { color: rgba(255,255,255,0.82) !important; font-size: 0.88rem; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Navigator")
    st.markdown("---")
    st.markdown("""
**Pages in this app:**

🏠 **Home** ← *You are here*
> Overview, instructions & setup guide

🔮 **Prediction**
> Enter customer & product details to get purchase probability

💡 **Recommendation**
> Sweep discount values 5–50% and find the optimal offer

📊 **Insights**
> Explore feature importance, intent distribution & trends
""")
    st.markdown("---")
    st.markdown("**⚙️ Requirements**")
    st.markdown("""
- `model.pkl` in project root
- `data.csv` in project root
- Run `train.py` to generate both
""")
    st.markdown("---")
    st.markdown("""
<small style='opacity:0.5;'>
Intent-Aware Price Drop Predictor<br>
Powered by Random Forest
</small>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>&#127919; Intent-Aware Price Drop Predictor</h1>
    <p class="subtitle">
        A multi-page ML app that predicts purchase probability, classifies customer intent,
        and recommends the optimal discount to maximise conversions.
    </p>
    <div class="badges">
        <span class="badge">&#129302; Random Forest Model</span>
        <span class="badge">&#128202; Intent Classification</span>
        <span class="badge">&#127381; Discount Optimiser</span>
        <span class="badge">&#128200; Visual Insights</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── How it works ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-heading">&#9881; How It Works</p>', unsafe_allow_html=True)
st.markdown("""
<div class="flow-wrap">
    <div class="flow-steps">
        <div class="flow-step">
            <div class="fnum">1</div>
            <div class="ftext">Run train.py<br>to build model</div>
        </div>
        <span class="flow-arrow">&#8594;</span>
        <div class="flow-step">
            <div class="fnum">2</div>
            <div class="ftext">Enter customer &amp;<br>product inputs</div>
        </div>
        <span class="flow-arrow">&#8594;</span>
        <div class="flow-step">
            <div class="fnum">3</div>
            <div class="ftext">Get purchase<br>probability</div>
        </div>
        <span class="flow-arrow">&#8594;</span>
        <div class="flow-step">
            <div class="fnum">4</div>
            <div class="ftext">Classify customer<br>intent tier</div>
        </div>
        <span class="flow-arrow">&#8594;</span>
        <div class="flow-step">
            <div class="fnum">5</div>
            <div class="ftext">Find optimal<br>discount &amp; act</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Page guide cards ───────────────────────────────────────────────────────────
st.markdown('<p class="section-heading">&#128218; Page Guide</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
<div class="page-card pc-prediction">
    <div class="pc-icon">&#128302;</div>
    <p class="pc-title">Prediction Page</p>
    <p class="pc-desc">
        Enter customer profile and product details to get an instant
        purchase probability score and intent classification.
    </p>
    <ul class="pc-steps">
        <li>
            <span class="step-num sn-blue">1</span>
            Select <b>User Type</b> — new, returning, or loyal
        </li>
        <li>
            <span class="step-num sn-blue">2</span>
            Set <b>Purchase Frequency</b> and <b>Avg Order Value</b>
        </li>
        <li>
            <span class="step-num sn-blue">3</span>
            Choose <b>Product Category</b>, price, rating, and discount
        </li>
        <li>
            <span class="step-num sn-blue">4</span>
            Click <b>Predict</b> to see probability % and intent tier
        </li>
        <li>
            <span class="step-num sn-blue">5</span>
            Review the <b>Input Summary</b> and colour-coded result
        </li>
    </ul>
    <span class="pc-nav-hint hint-blue">&#129518; Pages / Prediction</span>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="page-card pc-recommend">
    <div class="pc-icon">&#128161;</div>
    <p class="pc-title">Recommendation Page</p>
    <p class="pc-desc">
        Automatically sweeps all discount levels (5%–50%) and identifies
        the offer that maximises purchase probability for this customer.
    </p>
    <ul class="pc-steps">
        <li>
            <span class="step-num sn-green">1</span>
            Uses the same inputs as the Prediction page
        </li>
        <li>
            <span class="step-num sn-green">2</span>
            Tests <b>8 discount tiers</b> from 5% to 50%
        </li>
        <li>
            <span class="step-num sn-green">3</span>
            Displays a <b>probability bar chart</b> for every tier
        </li>
        <li>
            <span class="step-num sn-green">4</span>
            Highlights the <b>best discount</b> and peak probability
        </li>
        <li>
            <span class="step-num sn-green">5</span>
            Shows a <b>curve chart</b> of discount % vs probability
        </li>
    </ul>
    <span class="pc-nav-hint hint-green">&#129518; Pages / Recommendation</span>
</div>
""", unsafe_allow_html=True)

with col3:
    st.markdown("""
<div class="page-card pc-insights">
    <div class="pc-icon">&#128200;</div>
    <p class="pc-title">Insights Page</p>
    <p class="pc-desc">
        Explore model behaviour, feature importance, and aggregate
        intent patterns across the full training dataset.
    </p>
    <ul class="pc-steps">
        <li>
            <span class="step-num sn-orange">1</span>
            View <b>feature importance</b> bar chart from the model
        </li>
        <li>
            <span class="step-num sn-orange">2</span>
            See <b>intent distribution</b> across all customers
        </li>
        <li>
            <span class="step-num sn-orange">3</span>
            Compare purchase rates by <b>user type</b> and category
        </li>
        <li>
            <span class="step-num sn-orange">4</span>
            Analyse how <b>discount % affects</b> conversion trends
        </li>
        <li>
            <span class="step-num sn-orange">5</span>
            Use findings to refine pricing and targeting strategy
        </li>
    </ul>
    <span class="pc-nav-hint hint-orange">&#129518; Pages / Insights</span>
</div>
""", unsafe_allow_html=True)

# ── Setup requirements ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-heading">&#128295; Setup &amp; Requirements</p>', unsafe_allow_html=True)

rc1, rc2 = st.columns([1, 1], gap="medium")

with rc1:
    st.markdown("""
<div class="req-box">
    <p class="req-title">&#9888;&#65039; Before You Start</p>
    <div class="req-item"><span class="req-icon">&#128196;</span> Make sure <b>train.py</b> has been run at least once</div>
    <div class="req-item"><span class="req-icon">&#129302;</span> <b>model.pkl</b> must exist in the project root folder</div>
    <div class="req-item"><span class="req-icon">&#128202;</span> <b>data.csv</b> must exist for the Insights page</div>
    <div class="req-item"><span class="req-icon">&#128230;</span> All pages share the same trained model automatically</div>
</div>
""", unsafe_allow_html=True)

with rc2:
    st.markdown("""
<div class="req-box" style="border-color:#2d6a9f; background:#eef5fc;">
    <p class="req-title" style="color:#1e3a5f;">&#128640; Quick Start</p>
    <div class="req-item"><span class="req-icon">&#9312;</span> Run <code>python train.py</code> in your terminal</div>
    <div class="req-item"><span class="req-icon">&#9313;</span> Run <code>streamlit run app.py</code></div>
    <div class="req-item"><span class="req-icon">&#9314;</span> Open the app in your browser (default: localhost:8501)</div>
    <div class="req-item"><span class="req-icon">&#9315;</span> Use the sidebar to navigate between pages</div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<br>
<center style='color:#aaa; font-size:0.8rem;'>
    Intent-Aware Price Drop Predictor &nbsp;&middot;&nbsp;
    Powered by Random Forest &nbsp;&middot;&nbsp;
    Built with Streamlit
</center>
""", unsafe_allow_html=True)