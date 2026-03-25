import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="Purchase Predictor",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] { background: #0d0f14 !important; }
[data-testid="stAppViewContainer"] > .main  { background: #0d0f14 !important; }
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { background: #090b10 !important; border-right: 1px solid #1a2035 !important; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 800px !important; }
body, p, span, div, label { font-family: 'DM Sans', sans-serif !important; color: #e8eaf0; }
h1,h2,h3,h4 { font-family: 'DM Serif Display', serif !important; }

/* ── sidebar ── */
[data-testid="stSidebarNav"] { display: none !important; }
.sb-brand { padding: 0.5rem 0.8rem 1.4rem; border-bottom: 1px solid #1a2035; margin-bottom: 1.2rem; }
.sb-logo  { font-size: 2rem; margin-bottom: 0.4rem; }
.sb-title { font-family:'DM Serif Display',serif; font-size:1.15rem; color:#f0f4ff; font-weight:700; }
.sb-sub   { font-size:0.7rem; color:#2d3a52; letter-spacing:0.06em; text-transform:uppercase; margin-top:0.2rem; }
.sb-section-label {
    font-size:0.6rem; font-weight:600; letter-spacing:0.2em;
    text-transform:uppercase; color:#2d3a52; margin:0 0 0.5rem 0.4rem;
}
.sb-nav-item {
    display:flex; align-items:center; gap:0.65rem;
    padding:0.55rem 0.8rem; border-radius:8px; margin-bottom:0.2rem;
    font-size:0.85rem; font-weight:500; color:#4a5568;
}
.sb-nav-item.active { background:rgba(99,179,237,0.12); color:#93c5fd; }
.sb-nav-item .ni    { font-size:1rem; width:22px; text-align:center; }
.sb-stats-box {
    background:rgba(255,255,255,0.03); border:1px solid #1a2035;
    border-radius:10px; padding:0.9rem 1rem; margin-top:1rem;
}
.sb-stat { margin-bottom:0.55rem; }
.sb-stat:last-child { margin-bottom:0; }
.sb-stat .sv { font-size:1.1rem; font-weight:600; color:#93c5fd; font-family:'DM Serif Display',serif; }
.sb-stat .sl { font-size:0.65rem; color:#2d3a52; text-transform:uppercase; letter-spacing:0.07em; }

/* ── masthead ── */
.masthead { text-align:center; padding:2.5rem 0 2.2rem; position:relative; }
.masthead::before {
    content:''; position:absolute; inset:0;
    background:radial-gradient(ellipse 70% 55% at 50% 0%, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events:none;
}
.eyebrow { font-size:0.68rem; font-weight:600; letter-spacing:0.2em; text-transform:uppercase; color:#63b3ed; margin-bottom:0.8rem; }
.masthead h1 { font-size:clamp(2rem,5vw,2.8rem); color:#f0f4ff; line-height:1.12; letter-spacing:-0.01em; }
.masthead h1 em { font-style:italic; color:#93c5fd; }
.masthead .sub { margin-top:0.7rem; font-size:0.88rem; color:#4a5568; font-weight:300; }

/* ── panels ── */
.glass {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:1.6rem 1.8rem; margin-bottom:1.2rem;
}
.section-label {
    font-size:0.63rem; font-weight:600; letter-spacing:0.18em;
    text-transform:uppercase; color:#2d3a52;
    margin-bottom:0.9rem; padding-bottom:0.5rem; border-bottom:1px solid #1a2035;
}

/* ── widgets ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    font-size:0.75rem !important; font-weight:500 !important;
    color:#4a5568 !important; letter-spacing:0.05em !important; text-transform:uppercase !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background:#0f1520 !important; border:1px solid #1e2d45 !important;
    border-radius:8px !important; color:#e8eaf0 !important;
}
[data-testid="stSlider"] > div > div > div > div { background:#63b3ed !important; }

/* ── button ── */
div[data-testid="stButton"] > button {
    width:100% !important;
    background:linear-gradient(135deg,#1d4ed8,#3b82f6) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    padding:0.85rem !important; font-family:'DM Sans',sans-serif !important;
    font-size:0.95rem !important; font-weight:600 !important;
    letter-spacing:0.04em !important; text-transform:uppercase !important;
    box-shadow:0 4px 20px rgba(37,99,235,0.35) !important; margin-top:0.5rem !important;
    transition:all 0.2s !important;
}
div[data-testid="stButton"] > button:hover { transform:translateY(-1px) !important; }

/* ── result card ── */
.result-card {
    border-radius:18px; padding:2.2rem; margin:1.2rem 0;
    animation:slideUp 0.4s cubic-bezier(0.16,1,0.3,1) forwards;
    position:relative; overflow:hidden;
}
@keyframes slideUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
.result-card.high { background:linear-gradient(135deg,#052e16,#065f46); border:1px solid #059669; }
.result-card.mid  { background:linear-gradient(135deg,#1c1400,#451a03); border:1px solid #d97706; }
.result-card.low  { background:linear-gradient(135deg,#1a0000,#450a0a); border:1px solid #dc2626; }
.rc-top { display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:1.2rem; }
.rc-badge { font-size:0.65rem; font-weight:600; letter-spacing:0.18em; text-transform:uppercase; opacity:0.55; margin-bottom:0.35rem; }
.rc-name  { font-family:'DM Serif Display',serif; font-size:1.9rem; color:#f0f4ff; line-height:1; }
.rc-sub   { font-size:0.8rem; opacity:0.6; margin-top:0.4rem; line-height:1.5; max-width:280px; }
.rc-pct   { font-family:'DM Serif Display',serif; font-size:4.2rem; line-height:1; text-align:right; }
.result-card.high .rc-pct { color:#6ee7b7; }
.result-card.mid  .rc-pct { color:#fcd34d; }
.result-card.low  .rc-pct { color:#fca5a5; }
.rc-plabel { font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; opacity:0.45; text-align:right; }
.prog-track { background:rgba(0,0,0,0.3); border-radius:999px; height:5px; overflow:hidden; margin-bottom:1rem; }
.prog-fill  { height:100%; border-radius:999px; }
.result-card.high .prog-fill { background:linear-gradient(90deg,#059669,#6ee7b7); }
.result-card.mid  .prog-fill { background:linear-gradient(90deg,#d97706,#fcd34d); }
.result-card.low  .prog-fill { background:linear-gradient(90deg,#dc2626,#fca5a5); }
.rc-action {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.1);
    border-radius:999px; padding:0.4rem 0.9rem;
    font-size:0.78rem; color:#c8d3e8;
}

/* ── metric chips ── */
.chips { display:flex; gap:0.7rem; flex-wrap:wrap; margin:1rem 0 0.5rem; }
.chip  {
    flex:1; min-width:110px; background:rgba(255,255,255,0.03);
    border:1px solid #1a2035; border-radius:12px; padding:0.9rem 1.1rem; text-align:center;
}
.chip .cv { font-size:1.5rem; font-weight:600; color:#93c5fd; line-height:1; }
.chip .cl { font-size:0.62rem; letter-spacing:0.1em; text-transform:uppercase; color:#2d3a52; margin-top:0.3rem; }

/* ── summary table ── */
.stbl { width:100%; border-collapse:collapse; }
.stbl tr { border-bottom:1px solid #1a2035; }
.stbl tr:last-child { border-bottom:none; }
.stbl td { padding:0.5rem 0; font-size:0.84rem; }
.stbl td:first-child { color:#4a5568; width:55%; }
.stbl td:last-child  { color:#c8d3e8; font-weight:500; text-align:right; }

#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-logo">🛒</div>
        <div class="sb-title">PurchaseIQ</div>
        <div class="sb-sub">E-Commerce Analytics Suite</div>
    </div>
    <div class="sb-section-label">Navigation</div>
    <div class="sb-nav-item active"><span class="ni">🎯</span> Prediction</div>
    <div class="sb-nav-item"><span class="ni">💡</span> Recommendation</div>
    <div class="sb-nav-item"><span class="ni">📊</span> Insights</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**ℹ️ About this page**")
    st.caption("Enter customer and product details to predict the probability of a purchase using a trained Random Forest model.")

    st.markdown("""
    <div class="sb-stats-box">
        <div class="sb-stat"><div class="sv">7</div><div class="sl">Input Features</div></div>
        <div class="sb-stat"><div class="sv">91%</div><div class="sl">Model Accuracy</div></div>
        <div class="sb-stat"><div class="sv">0.98</div><div class="sl">ROC-AUC Score</div></div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════
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
    st.sidebar.success("✅ Model loaded successfully")
except FileNotFoundError:
    st.error("⚠️ model.pkl not found — run train.py first.")
    st.stop()

# ══════════════════════════════════════════════════════════
# MASTHEAD
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="masthead">
    <div class="eyebrow">🎯 Purchase Probability Engine</div>
    <h1>Will This Customer <em>Buy?</em></h1>
    <div class="sub">Fill in the details below and get an instant purchase probability score</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# INPUTS
# ══════════════════════════════════════════════════════════
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="section-label">👤 Customer Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    user_type = st.selectbox(
        "User Type", ["new", "returning", "loyal"], index=1,
        help="🆕 New — first time | 🔄 Returning — came back | 💎 Loyal — frequent buyer"
    )
    purchase_frequency = st.slider("Purchase Frequency", 1, 20, 5, help="Orders placed in recent period")
with col2:
    avg_order_value = st.number_input("Avg Order Value ($)", 10.0, 500.0, 85.0, step=5.0,
                                      help="Customer's historical average spend per order")
    rating = st.slider("Product Rating ⭐", 1.0, 5.0, 4.0, step=0.1)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📦 Product Details</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    product_category = st.selectbox(
        "Category",
        ["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty", "Sports"],
        help="Product department"
    )
    product_price = st.number_input("Product Price ($)", 5.0, 1500.0, 120.0, step=10.0)
with col4:
    discount_pct = st.select_slider(
        "Discount (%)", options=[0, 5, 10, 15, 20, 25, 30, 40, 50], value=10,
        help="Current discount applied to the product"
    )
    effective_price = round(product_price * (1 - discount_pct / 100), 2)
    st.metric("💰 Effective Price", f"${effective_price:,.2f}",
              delta=f"-${product_price - effective_price:,.2f}" if discount_pct > 0 else None)
st.markdown("</div>", unsafe_allow_html=True)

# ── Input validation warnings ─────────────────────────────
col_w1, col_w2 = st.columns(2)
with col_w1:
    if user_type == "new" and discount_pct < 10:
        st.warning("⚠️ New customers convert better with ≥10% discount")
with col_w2:
    if rating < 2.5:
        st.warning("⚠️ Low product rating may suppress purchase probability")
if product_price > 500 and discount_pct == 0:
    st.info("💡 High-priced items typically benefit from at least a small discount")

predict_btn = st.button("🔮  Generate Prediction")

# ══════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════
if predict_btn:
    user_enc = int(le_user.transform([user_type])[0])
    cat_enc  = int(le_category.transform([product_category])[0])

    input_df = pd.DataFrame([{
        "user_type": user_enc, "purchase_frequency": purchase_frequency,
        "avg_order_value": avg_order_value, "product_category": cat_enc,
        "product_price": product_price, "rating": rating,
        "discount_percentage": discount_pct,
    }])[feat_cols]

    prob     = model.predict_proba(input_df)[0][1]
    prob_pct = round(prob * 100, 1)
    no_pct   = round(100 - prob_pct, 1)

    if prob >= 0.70:
        tier, icon, verdict = "high", "🟢", "High Intent"
        sub    = "Strong purchase signals — this customer is primed to convert."
        action = "💡 Show a personalised offer or limited-time deal to close the sale"
        st.success(f"✅ **High purchase likelihood detected** — {prob_pct}% probability. Prioritise this customer!")
    elif prob >= 0.40:
        tier, icon, verdict = "mid", "🟡", "Medium Intent"
        sub    = "Moderate interest — a small nudge may tip the balance."
        action = "💡 Try a targeted discount, reviews highlight, or free shipping offer"
        st.warning(f"⚡ **Moderate purchase likelihood** — {prob_pct}% probability. A nudge could convert this customer.")
    else:
        tier, icon, verdict = "low", "🔴", "Low Intent"
        sub    = "Weak signals — customer is likely browsing without buying intent."
        action = "💡 Retarget with content, wishlist reminders, or email nurture"
        st.error(f"❄️ **Low purchase likelihood** — {prob_pct}% probability. Consider a re-engagement campaign.")

    # Result card
    st.markdown(
        '<div class="result-card ' + tier + '">'
        + '<div class="rc-top">'
        +   '<div>'
        +     '<div class="rc-badge">Intent Classification</div>'
        +     '<div class="rc-name">' + icon + ' ' + verdict + '</div>'
        +     '<div class="rc-sub">' + sub + '</div>'
        +   '</div>'
        +   '<div>'
        +     '<div class="rc-pct">' + str(prob_pct) + '</div>'
        +     '<div class="rc-plabel">% probability</div>'
        +   '</div>'
        + '</div>'
        + '<div class="prog-track"><div class="prog-fill" style="width:' + str(prob_pct) + '%;"></div></div>'
        + '<div class="rc-action">' + action + '</div>'
        + '</div>',
        unsafe_allow_html=True
    )

    # Metric chips
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("🎯 Will Buy",      f"{prob_pct}%")
    col_m2.metric("🚫 Won't Buy",     f"{no_pct}%")
    col_m3.metric("📦 Order Freq.",   f"{purchase_frequency}x")
    col_m4.metric("💸 Eff. Price",    f"${effective_price}")

    # Input summary
    st.markdown('<div class="glass" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📋 Input Summary</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(
            '<table class="stbl">'
            + '<tr><td>👤 User Type</td><td>' + user_type.capitalize() + '</td></tr>'
            + '<tr><td>🔁 Purchase Freq.</td><td>' + str(purchase_frequency) + ' orders</td></tr>'
            + '<tr><td>💵 Avg Order Value</td><td>$' + f"{avg_order_value:,.2f}" + '</td></tr>'
            + '<tr><td>⭐ Rating</td><td>' + str(rating) + ' / 5.0</td></tr>'
            + '</table>',
            unsafe_allow_html=True
        )
    with col_s2:
        st.markdown(
            '<table class="stbl">'
            + '<tr><td>🏷️ Category</td><td>' + product_category + '</td></tr>'
            + '<tr><td>💲 Product Price</td><td>$' + f"{product_price:,.2f}" + '</td></tr>'
            + '<tr><td>🏷️ Discount</td><td>' + str(discount_pct) + '%</td></tr>'
            + '<tr><td>💰 Effective Price</td><td>$' + str(effective_price) + '</td></tr>'
            + '</table>',
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 1rem;color:#1e2d45;">
        <div style="font-size:3rem;margin-bottom:0.6rem;">🎯</div>
        <div style="font-size:0.82rem;letter-spacing:0.1em;text-transform:uppercase;color:#2d3a52;">
            Fill in the fields above and click Generate Prediction
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.2rem;
    border-top:1px solid #1a2035;font-size:0.7rem;color:#1e2d45;letter-spacing:0.08em;text-transform:uppercase;">
    🛒 PurchaseIQ &nbsp;·&nbsp; Random Forest Model &nbsp;·&nbsp; Synthetic E-Commerce Data
</div>
""", unsafe_allow_html=True)