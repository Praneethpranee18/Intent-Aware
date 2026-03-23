import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="Purchase Predictor",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important;
}
[data-testid="stAppViewContainer"] > .main { background: #0d0f14 !important; }
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { background: #0d0f14 !important; }
.block-container { padding: 2.5rem 1.5rem 4rem !important; max-width: 780px !important; }

/* typography */
body, p, span, div, label, input {
    font-family: 'DM Sans', sans-serif !important;
    color: #e8eaf0;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

/* ── masthead ── */
.masthead {
    text-align: center;
    padding: 3.5rem 0 3rem;
    position: relative;
}
.masthead::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 70% 55% at 50% 0%, rgba(99,179,237,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.masthead .eyebrow {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 0.9rem;
}
.masthead h1 {
    font-size: clamp(2.2rem, 5vw, 3rem);
    font-weight: 400;
    color: #f0f4ff;
    line-height: 1.15;
    letter-spacing: -0.01em;
}
.masthead h1 em {
    font-style: italic;
    color: #93c5fd;
}
.masthead .sub {
    margin-top: 0.8rem;
    font-size: 0.9rem;
    color: #7a859e;
    font-weight: 300;
}

/* ── section labels ── */
.section-label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2535;
}

/* ── glass panel ── */
.glass {
    background: rgba(255,255,255,0.030);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(8px);
}

/* ── streamlit widget overrides ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #7a859e !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.3rem !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background: #131720 !important;
    border: 1px solid #2a3347 !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] input:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.12) !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #63b3ed !important;
}

/* predict button */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
    margin-top: 0.5rem !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(37,99,235,0.5) !important;
}

/* ── result card ── */
.result-card {
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
    animation: slideUp 0.45s cubic-bezier(0.16,1,0.3,1) forwards;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0.06;
    pointer-events: none;
    background-image:
        radial-gradient(circle at 20% 80%, white 1px, transparent 1px),
        radial-gradient(circle at 80% 20%, white 1px, transparent 1px),
        radial-gradient(circle at 50% 50%, white 0.5px, transparent 0.5px);
    background-size: 40px 40px, 60px 60px, 25px 25px;
}
.result-card.high { background: linear-gradient(135deg,#052e16 0%,#065f46 100%); border: 1px solid #059669; }
.result-card.mid  { background: linear-gradient(135deg,#1c1400 0%,#451a03 100%); border: 1px solid #d97706; }
.result-card.low  { background: linear-gradient(135deg,#1a0000 0%,#450a0a 100%); border: 1px solid #dc2626; }

.result-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}
.result-left .tier-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 0.4rem;
}
.result-left .tier-name {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    line-height: 1;
    color: #f0f4ff;
}
.result-left .tier-sub {
    margin-top: 0.45rem;
    font-size: 0.82rem;
    font-weight: 300;
    opacity: 0.65;
    max-width: 300px;
    line-height: 1.5;
}
.result-right .big-pct {
    font-family: 'DM Serif Display', serif;
    font-size: 4.5rem;
    line-height: 1;
    text-align: right;
}
.result-card.high .big-pct { color: #6ee7b7; }
.result-card.mid  .big-pct { color: #fcd34d; }
.result-card.low  .big-pct { color: #fca5a5; }
.result-right .pct-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.5;
    text-align: right;
}

/* progress track */
.prog-track {
    background: rgba(0,0,0,0.3);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
    margin-bottom: 1.25rem;
}
.prog-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(0.16,1,0.3,1);
}
.result-card.high .prog-fill { background: linear-gradient(90deg,#059669,#6ee7b7); }
.result-card.mid  .prog-fill { background: linear-gradient(90deg,#d97706,#fcd34d); }
.result-card.low  .prog-fill { background: linear-gradient(90deg,#dc2626,#fca5a5); }

/* action pill */
.action-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 999px;
    padding: 0.45rem 1rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: #c8d3e8;
    backdrop-filter: blur(4px);
}

/* ── metric chips ── */
.chips {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin: 1.2rem 0 0.5rem;
}
.chip {
    flex: 1;
    min-width: 120px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.chip .cv { font-size: 1.6rem; font-weight: 600; color: #93c5fd; line-height: 1; }
.chip .cl { font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; color: #4a5568; margin-top: 0.3rem; }

/* ── summary table ── */
.stbl { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
.stbl tr { border-bottom: 1px solid #1e2535; }
.stbl tr:last-child { border-bottom: none; }
.stbl td { padding: 0.55rem 0; font-size: 0.85rem; }
.stbl td:first-child { color: #4a5568; width: 55%; }
.stbl td:last-child  { color: #c8d3e8; font-weight: 500; text-align: right; }

/* hide streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

try:
    payload     = load_model()
    model       = payload["model"]
    le_user     = payload["le_user"]
    le_category = payload["le_category"]
    feat_cols   = payload["feature_columns"]
except FileNotFoundError:
    st.error("model.pkl not found — run train.py first.")
    st.stop()

# ── Masthead ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="eyebrow">◈ E-Commerce Intelligence</div>
    <h1>Purchase <em>Probability</em><br>Predictor</h1>
    <div class="sub">Enter customer and product details below to generate a prediction</div>
</div>
""", unsafe_allow_html=True)

# ── Form ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    user_type = st.selectbox(
        "User Type",
        ["new", "returning", "loyal"],
        index=1,
        help="Customer's relationship history with the store"
    )
    purchase_frequency = st.slider(
        "Purchase Frequency", 1, 20, 5,
        help="Number of purchases made in the past period"
    )
with col2:
    avg_order_value = st.number_input(
        "Avg Order Value ($)", 10.0, 500.0, 85.0, step=5.0,
        help="Customer's average historical spend per order"
    )
    rating = st.slider(
        "Product Rating", 1.0, 5.0, 4.0, step=0.1,
        help="Rating of the product being considered"
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Product Details</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    product_category = st.selectbox(
        "Category",
        ["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty", "Sports"],
    )
    product_price = st.number_input(
        "Product Price ($)", 5.0, 1500.0, 120.0, step=10.0
    )
with col4:
    discount_pct = st.select_slider(
        "Discount (%)",
        options=[0, 5, 10, 15, 20, 25, 30, 40, 50],
        value=10,
        help="Discount currently applied to this product"
    )
    effective_price = round(product_price * (1 - discount_pct / 100), 2)
    st.metric("Effective Price", f"${effective_price:,.2f}",
              delta=f"-${product_price - effective_price:,.2f}" if discount_pct > 0 else None)
st.markdown("</div>", unsafe_allow_html=True)

predict_btn = st.button("◈  Generate Prediction")

# ── Predict ────────────────────────────────────────────────────────────────────
if predict_btn:
    user_enc = int(le_user.transform([user_type])[0])
    cat_enc  = int(le_category.transform([product_category])[0])

    input_df = pd.DataFrame([{
        "user_type":           user_enc,
        "purchase_frequency":  purchase_frequency,
        "avg_order_value":     avg_order_value,
        "product_category":    cat_enc,
        "product_price":       product_price,
        "rating":              rating,
        "discount_percentage": discount_pct,
    }])[feat_cols]

    prob     = model.predict_proba(input_df)[0][1]
    prob_pct = round(prob * 100, 1)
    no_pct   = round(100 - prob_pct, 1)

    # tier classification
    if prob >= 0.70:
        tier      = "high"
        tier_name = "High Intent"
        tier_sub  = "Strong purchase signals detected. This customer is primed to convert."
        action    = "💡 Surface a personalised offer or limited-time deal to close the sale"
    elif prob >= 0.40:
        tier      = "mid"
        tier_name = "Medium Intent"
        tier_sub  = "Moderate interest detected. A targeted nudge may tip the balance."
        action    = "💡 Try a small discount, social proof, or free-shipping offer"
    else:
        tier      = "low"
        tier_name = "Low Intent"
        tier_sub  = "Weak purchase signals. Customer is likely browsing, not buying."
        action    = "💡 Retarget with content, wishlist reminders, or email nurture"

    # ── Result card ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="result-card ' + tier + '">'
        + '<div class="result-top">'
        +   '<div class="result-left">'
        +     '<div class="tier-label">Intent Classification</div>'
        +     '<div class="tier-name">' + tier_name + '</div>'
        +     '<div class="tier-sub">' + tier_sub + '</div>'
        +   '</div>'
        +   '<div class="result-right">'
        +     '<div class="big-pct">' + str(prob_pct) + '</div>'
        +     '<div class="pct-label">% probability</div>'
        +   '</div>'
        + '</div>'
        + '<div class="prog-track"><div class="prog-fill" style="width:' + str(prob_pct) + '%;"></div></div>'
        + '<div class="action-pill">' + action + '</div>'
        + '</div>',
        unsafe_allow_html=True
    )

    # ── Metric chips ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="chips">'
        + '<div class="chip"><div class="cv">' + str(prob_pct) + '%</div><div class="cl">Will Purchase</div></div>'
        + '<div class="chip"><div class="cv">' + str(no_pct) + '%</div><div class="cl">Will Not Purchase</div></div>'
        + '<div class="chip"><div class="cv">' + str(purchase_frequency) + 'x</div><div class="cl">Purchase Freq.</div></div>'
        + '<div class="chip"><div class="cv">$' + str(effective_price) + '</div><div class="cl">Effective Price</div></div>'
        + '</div>',
        unsafe_allow_html=True
    )

    # ── Input summary ─────────────────────────────────────────────────────────
    st.markdown('<div class="glass" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)

    rows = [
        ("User Type",           user_type.capitalize()),
        ("Purchase Frequency",  str(purchase_frequency) + " orders"),
        ("Avg Order Value",     f"${avg_order_value:,.2f}"),
        ("Product Category",    product_category),
        ("Product Price",       f"${product_price:,.2f}"),
        ("Effective Price",     f"${effective_price:,.2f}"),
        ("Rating",              f"{rating} / 5.0"),
        ("Discount Applied",    f"{discount_pct}%"),
    ]

    tbl = '<table class="stbl">'
    for label, val in rows:
        tbl += '<tr><td>' + label + '</td><td>' + val + '</td></tr>'
    tbl += '</table>'
    st.markdown(tbl, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif not predict_btn:
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 1rem; color:#2d3748;">
        <div style="font-size:2.5rem; margin-bottom:0.75rem;">◈</div>
        <div style="font-size:0.85rem; letter-spacing:0.1em; text-transform:uppercase;">
            Fill in the fields above and click Generate Prediction
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding-top:1.5rem;
            border-top:1px solid #1e2535; font-size:0.72rem;
            color:#2d3748; letter-spacing:0.08em; text-transform:uppercase;">
    Random Forest Model &nbsp;·&nbsp; Synthetic E-Commerce Data &nbsp;·&nbsp; ◈
</div>
""", unsafe_allow_html=True)