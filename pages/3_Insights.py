import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Model Insights",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] { background: #f5f0e8 !important; }
[data-testid="stAppViewContainer"] > .main     { background: #f5f0e8 !important; }
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { background: #1c1917 !important; border-right:1px solid #292524 !important; }
.block-container { padding: 0 1.5rem 5rem !important; max-width: 840px !important; }
body, p, span, div { font-family: 'IBM Plex Sans', sans-serif !important; color: #1c1917; }
h1,h2,h3,h4 { font-family: 'Playfair Display', serif !important; }

/* ── sidebar ── */
[data-testid="stSidebarNav"] { display: none !important; }
.sb-brand { padding: 1rem 0.8rem 1.3rem; border-bottom:1px solid #292524; margin-bottom:1.2rem; }
.sb-logo  { font-size:2rem; margin-bottom:0.4rem; }
.sb-title { font-family:'Playfair Display',serif; font-size:1.1rem; color:#f5f5f0; font-weight:700; }
.sb-sub   { font-size:0.68rem; color:#44403c; letter-spacing:0.08em; text-transform:uppercase; margin-top:0.2rem; }
.sb-section-label {
    font-size:0.6rem; font-weight:600; letter-spacing:0.2em;
    text-transform:uppercase; color:#44403c; margin:0 0 0.5rem 0.4rem;
}
.sb-nav-item {
    display:flex; align-items:center; gap:0.65rem;
    padding:0.55rem 0.8rem; border-radius:8px; margin-bottom:0.2rem;
    font-size:0.85rem; font-weight:500; color:#57534e; font-family:'IBM Plex Sans',sans-serif;
}
.sb-nav-item.active { background:rgba(161,98,7,0.15); color:#d97706; }
.sb-feat-list { margin-top:1rem; }
.sb-feat-item {
    display:flex; justify-content:space-between; align-items:center;
    padding:0.45rem 0.5rem; border-radius:6px; margin-bottom:0.2rem;
    font-size:0.78rem; font-family:'IBM Plex Mono',monospace;
}
.sb-feat-item .fn { color:#78716c; }
.sb-feat-item .fv { color:#d97706; font-weight:600; }

/* ── masthead ── */
.masthead { border-bottom:2px solid #1c1917; padding:2.8rem 0 2rem; margin-bottom:2.5rem; position:relative; }
.mh-issue { font-family:'IBM Plex Mono',monospace !important; font-size:0.65rem; letter-spacing:0.15em; color:#78716c; text-transform:uppercase; margin-bottom:0.8rem; }
.masthead h1 { font-size:clamp(2.2rem,5.5vw,3.2rem); font-weight:700; color:#1c1917; line-height:1.05; letter-spacing:-0.02em; }
.masthead h1 em { font-style:italic; font-weight:400; color:#a16207; }
.masthead .sub { margin-top:0.75rem; font-size:0.9rem; color:#78716c; font-weight:300; max-width:500px; line-height:1.6; }

/* ── stat row ── */
.stat-row { display:flex; gap:0; border:1px solid #e7e5e4; border-radius:4px; overflow:hidden; margin-bottom:2rem; box-shadow:2px 2px 0 #e7e5e4; }
.stat-cell { flex:1; padding:1.2rem 1.3rem; background:white; border-right:1px solid #e7e5e4; text-align:center; }
.stat-cell:last-child { border-right:none; }
.stat-cell .sv { font-family:'Playfair Display',serif; font-size:1.9rem; font-weight:700; color:#1c1917; line-height:1; }
.stat-cell .sl { font-family:'IBM Plex Mono',monospace; font-size:0.58rem; letter-spacing:0.1em; text-transform:uppercase; color:#a8a29e; margin-top:0.3rem; }
.stat-cell.acc .sv { color:#a16207; }

/* ── section header ── */
.section-hd { display:flex; align-items:flex-start; gap:1.1rem; margin-bottom:1.3rem; padding-bottom:0.9rem; border-bottom:1px solid #e7e5e4; }
.section-num { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#a8a29e; margin-top:0.3rem; flex-shrink:0; }
.section-hd h2 { font-size:1.5rem; font-weight:700; color:#1c1917; line-height:1.2; }
.section-hd h2 em { font-style:italic; font-weight:400; }

/* ── panel ── */
.panel { background:white; border:1px solid #e7e5e4; border-radius:4px; padding:1.7rem 1.9rem; margin-bottom:1.3rem; box-shadow:2px 2px 0 #e7e5e4; }

/* ── importance rows ── */
.imp-row { display:flex; align-items:center; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid #f5f5f4; }
.imp-row:last-child { border-bottom:none; }
.imp-feat { width:175px; flex-shrink:0; font-size:0.79rem; font-weight:500; color:#44403c; font-family:'IBM Plex Mono',monospace; }
.imp-wrap { flex:1; background:#f5f0e8; border-radius:2px; height:10px; overflow:hidden; }
.imp-fill { height:100%; border-radius:2px; }
.imp-val  { width:42px; text-align:right; font-family:'IBM Plex Mono',monospace; font-size:0.77rem; font-weight:500; color:#1c1917; flex-shrink:0; }

/* ── insight cards ── */
.insight-do {
    background:#fefce8; border:1px solid #fde047; border-top:4px solid #ca8a04;
    border-radius:4px; padding:1.4rem 1.5rem; height:100%;
}
.insight-dont {
    background:#fafaf9; border:1px solid #e7e5e4; border-top:4px solid #78716c;
    border-radius:4px; padding:1.4rem 1.5rem; height:100%;
}
.ititle { font-family:'Playfair Display',serif; font-size:1rem; font-weight:700; margin-bottom:0.9rem; display:flex; align-items:center; gap:0.4rem; }
.insight-do .ititle   { color:#854d0e; }
.insight-dont .ititle { color:#44403c; }
.ipoint { display:flex; gap:0.55rem; margin-bottom:0.65rem; font-size:0.82rem; line-height:1.5; color:#44403c; }
.ipoint:last-child { margin-bottom:0; }
.ibullet { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; margin-top:0.15rem; flex-shrink:0; }
.insight-do .ibullet   { color:#ca8a04; }
.insight-dont .ibullet { color:#78716c; }

/* ── pullquote ── */
.pullquote { border-left:3px solid #a16207; padding:0.75rem 1.1rem; background:#fffbeb; margin:1.4rem 0; }
.pullquote p { font-family:'Playfair Display',serif !important; font-size:0.97rem; font-style:italic; color:#78716c; line-height:1.6; }

.sep { border:none; border-top:1px solid #e7e5e4; margin:2.2rem 0; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════
@st.cache_resource
def train_model():

    np.random.seed(42)
    n = 20000

    user_types = np.random.choice(['new', 'returning', 'loyal'], n, p=[0.3, 0.45, 0.25])
    categories = np.random.choice(
        ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Beauty', 'Sports'], n
    )

    category_price_map = {
        'Electronics': (80, 1200),
        'Clothing': (15, 300),
        'Home & Kitchen': (20, 500),
        'Books': (5, 60),
        'Beauty': (10, 200),
        'Sports': (25, 400),
    }

    purchase_frequency = np.where(
        user_types == 'loyal', np.random.randint(8, 20, n),
        np.where(user_types == 'returning', np.random.randint(3, 10, n),
                 np.random.randint(1, 4, n))
    )

    avg_order_value = np.where(
        user_types == 'loyal', np.random.uniform(80, 300, n),
        np.where(user_types == 'returning', np.random.uniform(40, 150, n),
                 np.random.uniform(20, 80, n))
    )

    product_price = np.array([
        np.random.uniform(*category_price_map[c]) for c in categories
    ])

    rating = np.random.uniform(1.0, 5.0, n)
    discount_pct = np.random.choice([0, 5, 10, 15, 20, 25, 30, 40, 50], n)

    score = (
        (user_types == 'loyal') * 0.35 +
        (user_types == 'returning') * 0.20 +
        (purchase_frequency / 20) * 0.15 +
        (rating / 5.0) * 0.20 +
        (discount_pct / 50) * 0.15 +
        np.random.uniform(0, 0.15, n)
    )

    purchase = (score > 0.45).astype(int)

    df = pd.DataFrame({
        'user_type': user_types,
        'purchase_frequency': purchase_frequency,
        'avg_order_value': avg_order_value,
        'product_category': categories,
        'product_price': product_price,
        'rating': rating,
        'discount_percentage': discount_pct,
        'purchase': purchase,
    })

    # Encoding
    le_user = LabelEncoder()
    le_category = LabelEncoder()

    df['user_type'] = le_user.fit_transform(df['user_type'])
    df['product_category'] = le_category.fit_transform(df['product_category'])

    X = df.drop(columns='purchase')
    y = df['purchase']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=12)
    model.fit(X_train, y_train)

    return model, le_user, le_category, list(X.columns)


# Load model
model, le_user, le_category, feat_cols = train_model()

st.sidebar.success("✅ Model trained successfully")

# ══════════════════════════════════════════════════════════
# PRE-COMPUTE
# ══════════════════════════════════════════════════════════
@st.cache_data
def compute_discount_curves():
    DISCOUNTS  = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    user_types = ["new", "returning", "loyal"]
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty", "Sports"]
    profiles   = [(3,50),(6,100),(14,180)]
    results    = {d: [] for d in DISCOUNTS}
    for ut in user_types:
        ue = int(le_user.transform([ut])[0])
        for cat in categories:
            ce = int(le_category.transform([cat])[0])
            for pf, aov in profiles:
                for d in DISCOUNTS:
                    row = pd.DataFrame([{"user_type":ue,"purchase_frequency":pf,"avg_order_value":aov,
                        "product_category":ce,"product_price":150.0,"rating":3.8,"discount_percentage":d}])[feat_cols]
                    results[d].append(model.predict_proba(row)[0][1])
    return DISCOUNTS, [round(np.mean(results[d])*100,1) for d in DISCOUNTS]

@st.cache_data
def compute_by_user_type():
    DISCOUNTS  = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    ce = int(le_category.transform(["Electronics"])[0])
    curves = {}
    for ut in ["new", "returning", "loyal"]:
        ue = int(le_user.transform([ut])[0])
        probs = []
        for d in DISCOUNTS:
            row = pd.DataFrame([{"user_type":ue,"purchase_frequency":6,"avg_order_value":100,
                "product_category":ce,"product_price":150.0,"rating":3.8,"discount_percentage":d}])[feat_cols]
            probs.append(round(model.predict_proba(row)[0][1]*100,1))
        curves[ut] = probs
    return DISCOUNTS, curves

@st.cache_data
def get_feature_importances():
    labels = {"purchase_frequency":"Purchase Frequency","user_type":"User Type",
              "avg_order_value":"Avg Order Value","rating":"Rating",
              "discount_percentage":"Discount %","product_price":"Product Price",
              "product_category":"Product Category"}
    return sorted([(labels.get(f,f), round(float(v)*100,1)) for f,v in zip(feat_cols, model.feature_importances_)],
                  key=lambda x: x[1], reverse=True)

disc_vals, avg_probs     = compute_discount_curves()
disc_vals_u, user_curves = compute_by_user_type()
feature_importances      = get_feature_importances()

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
    <div class="sb-nav-item"><span>🎯</span> Prediction</div>
    <div class="sb-nav-item"><span>💡</span> Recommendation</div>
    <div class="sb-nav-item active"><span>📊</span> Insights</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📊 Feature Weights**")

    feat_html = '<div class="sb-feat-list">'
    for feat, val in feature_importances:
        short = feat.replace("Purchase ", "").replace("Product ", "").replace("Avg Order ", "Avg ")
        feat_html += '<div class="sb-feat-item"><span class="fn">' + short + '</span><span class="fv">' + str(val) + '%</span></div>'
    feat_html += '</div>'
    st.markdown(feat_html, unsafe_allow_html=True)

    top_feat = feature_importances[0][0]
    st.info(f"🏆 **{top_feat}** is the strongest predictor at {feature_importances[0][1]}%")
    st.success("✅ Model loaded & analysed")

# ══════════════════════════════════════════════════════════
# MASTHEAD
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="masthead">
    <div class="mh-issue">📊 Model Intelligence Report &nbsp;·&nbsp; E-Commerce Analytics</div>
    <h1>Discount &amp; <em>Feature</em> Insights</h1>
    <div class="sub">A deep-dive into how discounts drive purchase probability and which features your model trusts most.</div>
</div>
""", unsafe_allow_html=True)

# Stat row
top_feat_val  = feature_importances[0][1]
prob_range    = round(max(avg_probs) - min(avg_probs), 1)
max_disc_val  = disc_vals[avg_probs.index(max(avg_probs))]

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("🏆 Top Feature",    f"{top_feat_val}%")
col_s2.metric("🔢 Features",       str(len(feature_importances)))
col_s3.metric("📈 Discount Impact", f"+{prob_range}%")
col_s4.metric("🎯 Best Avg Disc.", f"{max_disc_val}%")

st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 1 — Discount Charts
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="section-hd">
    <div class="section-num">01 /</div>
    <div><h2>Discount vs <em>Purchase Probability</em></h2></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)

COLORS = {"new":"#ef4444","returning":"#3b82f6","loyal":"#10b981"}
x_fine = np.linspace(min(disc_vals), max(disc_vals), 400)
y_avg  = np.interp(x_fine, disc_vals, avg_probs)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
fig.patch.set_facecolor("white")

ax = axes[0]
ax.set_facecolor("white")
ax.fill_between(x_fine, y_avg, alpha=0.07, color="#a16207")
ax.plot(x_fine, y_avg, color="#1c1917", linewidth=2.2, zorder=3)
ax.scatter(disc_vals, avg_probs, color="#a16207", s=55, zorder=5, edgecolors="white", linewidths=1.5)
for d, p in zip(disc_vals, avg_probs):
    ax.text(d, p+1.0, f"{p}%", ha="center", va="bottom", fontsize=6.5, color="#78716c")
ax.set_xlabel("Discount (%)", fontsize=9, color="#78716c", labelpad=5)
ax.set_ylabel("Purchase Probability (%)", fontsize=9, color="#78716c", labelpad=5)
ax.set_title("📊 Average — All Profiles", fontsize=10, fontweight="bold", color="#1c1917", pad=8)
ax.set_xticks(disc_vals)
ax.set_xticklabels([f"{d}%" for d in disc_vals], fontsize=7.5, color="#78716c")
ax.set_ylim(max(0,min(avg_probs)-8), min(105,max(avg_probs)+10))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax.tick_params(colors="#9ca3af", labelsize=7.5)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_edgecolor("#e7e5e4")
ax.grid(axis="y", linestyle="--", alpha=0.5, color="#f5f0e8")

ax2 = axes[1]
ax2.set_facecolor("white")
for ut, clr in COLORS.items():
    probs = user_curves[ut]
    y_u   = np.interp(x_fine, disc_vals_u, probs)
    ax2.plot(x_fine, y_u, color=clr, linewidth=1.8, label=ut.capitalize(), zorder=3)
    ax2.scatter(disc_vals_u, probs, color=clr, s=32, zorder=4, edgecolors="white", linewidths=1.2)
ax2.set_xlabel("Discount (%)", fontsize=9, color="#78716c", labelpad=5)
ax2.set_title("👥 By User Type", fontsize=10, fontweight="bold", color="#1c1917", pad=8)
ax2.set_xticks(disc_vals_u)
ax2.set_xticklabels([f"{d}%" for d in disc_vals_u], fontsize=7.5, color="#78716c")
ax2.set_ylim(0, 105)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax2.tick_params(colors="#9ca3af", labelsize=7.5)
for s in ["top","right"]: ax2.spines[s].set_visible(False)
for s in ["left","bottom"]: ax2.spines[s].set_edgecolor("#e7e5e4")
ax2.grid(axis="y", linestyle="--", alpha=0.5, color="#f5f0e8")
ax2.legend(fontsize=8, framealpha=0.9, edgecolor="#e7e5e4", loc="lower right")

plt.tight_layout(pad=1.5)
st.pyplot(fig)
plt.close(fig)
st.markdown("</div>", unsafe_allow_html=True)

# Key findings in columns
st.markdown("**📌 Key findings from discount analysis:**")
kf1, kf2, kf3 = st.columns(3)
kf1.info("📈 **Diminishing returns** above 25% — marginal gain drops sharply")
kf2.success("💎 **Loyal customers** convert at high rates even with 0% discount")
kf3.warning("🆕 **New customers** need 15–25% to overcome first-purchase hesitation")

st.markdown("""
<div class="pullquote">
    <p>"Discounts reliably increase purchase probability — but the marginal gain
    diminishes above 25%. Loyal customers convert at high rates even with minimal
    incentive; new customers require stronger nudges."</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 2 — Feature Importance
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="section-hd">
    <div class="section-num">02 /</div>
    <div><h2>Feature <em>Importance</em></h2></div>
</div>
""", unsafe_allow_html=True)

col_fi1, col_fi2 = st.columns([1, 1])

with col_fi1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**🔢 Importance Scores**")
    max_imp  = max(v for _, v in feature_importances)
    imp_html = ""
    for feat, val in feature_importances:
        w       = round((val / max_imp) * 100, 1)
        is_top  = (feat == feature_importances[0][0])
        bar_col = "#a16207" if is_top else "#1c1917"
        opacity = "1.0" if is_top else str(round(0.3 + 0.7*(val/max_imp), 2))
        imp_html += (
            '<div class="imp-row">'
            + '<span class="imp-feat">' + feat + '</span>'
            + '<span class="imp-wrap"><span class="imp-fill" style="width:' + str(w) + '%;background:' + bar_col + ';opacity:' + opacity + ';"></span></span>'
            + '<span class="imp-val">' + str(val) + '%</span>'
            + '</div>'
        )
    st.markdown(imp_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_fi2:
    fig2, ax3 = plt.subplots(figsize=(4.5, 3.5))
    fig2.patch.set_facecolor("white"); ax3.set_facecolor("white")
    feats  = [f for f, _ in reversed(feature_importances)]
    values = [v for _, v in reversed(feature_importances)]
    colors = ["#a16207" if f == feature_importances[0][0] else "#d6d3d1" for f in feats]
    bars   = ax3.barh(feats, values, color=colors, height=0.55, edgecolor="none")
    for bar, val in zip(bars, values):
        ax3.text(val+0.3, bar.get_y()+bar.get_height()/2, f"{val}%",
                 va="center", ha="left", fontsize=7.5, color="#44403c")
    ax3.set_xlabel("Importance (%)", fontsize=8.5, color="#78716c")
    ax3.set_xlim(0, max(values)*1.28)
    ax3.tick_params(axis="y", labelsize=8, colors="#44403c")
    ax3.tick_params(axis="x", labelsize=7, colors="#a8a29e")
    for s in ["top","right","left"]: ax3.spines[s].set_visible(False)
    ax3.spines["bottom"].set_edgecolor("#e7e5e4")
    ax3.grid(axis="x", linestyle="--", alpha=0.4, color="#f5f0e8")
    ax3.set_title("📊 Feature Importance", fontsize=9, fontweight="bold", color="#1c1917", pad=6)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# Feature insight callouts
top1, top2 = feature_importances[0], feature_importances[1]
fi1, fi2 = st.columns(2)
fi1.success(f"🏆 **{top1[0]}** dominates at {top1[1]}% — the single biggest conversion driver")
fi2.info(f"👤 **{top2[0]}** at {top2[1]}% — who the customer is matters enormously")

st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 3 — Business Playbook
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="section-hd">
    <div class="section-num">03 /</div>
    <div><h2>Business <em>Playbook</em></h2></div>
</div>
""", unsafe_allow_html=True)

DO_POINTS = [
    ("01","New customers on first purchase.","A 15–25% discount lowers the barrier and builds lifetime value."),
    ("02","High-priced products (Electronics, Sports).","Price sensitivity is strongest here — even 10% can tip a borderline decision."),
    ("03","Mid-rated products (3–4 stars).","A discount compensates for perceived risk and accelerates commitment."),
    ("04","Win-back campaigns for lapsed customers.","Returning users respond strongly to time-limited offers (20–30%)."),
    ("05","Cart abandonment recovery.","Higher conversion with a 10–15% nudge sent within 24 hours."),
]
DONT_POINTS = [
    ("01","Loyal, high-frequency customers.","They buy regardless — discounts erode margin without improving conversion."),
    ("02","Products with 4.5+ star ratings.","Social proof is already converting. Discounting trains customers to wait for deals."),
    ("03","Low average order value segments.","Discount cost may exceed lifetime value — focus on higher-AOV cohorts."),
    ("04","Above 30% on mass-market items.","Diminishing returns kick in sharply — deeper cuts only damage margin."),
    ("05","During peak demand periods.","High organic traffic converts without incentive — protect price anchoring."),
]

def make_points(points, bullet_color):
    html = ""
    for num, title, body in points:
        html += (
            '<div class="ipoint">'
            + '<span class="ibullet" style="color:' + bullet_color + ';">' + num + '</span>'
            + '<span><strong>' + title + '</strong> ' + body + '</span>'
            + '</div>'
        )
    return html

col_do, col_dont = st.columns(2)
with col_do:
    st.markdown(
        '<div class="insight-do">'
        + '<div class="ititle">✅ When to Give Discounts</div>'
        + make_points(DO_POINTS, "#ca8a04")
        + '</div>',
        unsafe_allow_html=True
    )
with col_dont:
    st.markdown(
        '<div class="insight-dont">'
        + '<div class="ititle">❌ When NOT to Give Discounts</div>'
        + make_points(DONT_POINTS, "#78716c")
        + '</div>',
        unsafe_allow_html=True
    )

st.markdown("""
<div class="pullquote" style="margin-top:1.4rem;">
    <p>"Purchase frequency and user type account for over 55% of model weight combined —
    meaning who the customer is matters far more than what discount you offer.
    Invest in customer relationships first; pricing strategy second."</p>
</div>
""", unsafe_allow_html=True)

# Bottom callout
bc1, bc2, bc3 = st.columns(3)
bc1.error("🚫 Never discount loyal high-freq buyers — protect margin")
bc2.warning("⚡ Medium-intent customers respond best to targeted nudges")
bc3.success("✅ Focus budget on new customers with high AOV potential")

st.markdown("""
<div style="margin-top:3rem;padding-top:1.4rem;border-top:2px solid #1c1917;
    display:flex;justify-content:space-between;
    font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#a8a29e;
    letter-spacing:0.1em;text-transform:uppercase;">
    <span>Random Forest Classifier</span>
    <span>📊 PurchaseIQ Insights</span>
    <span>Synthetic E-Commerce Data</span>
</div>
""", unsafe_allow_html=True)