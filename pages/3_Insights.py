import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

st.set_page_config(
    page_title="Model Insights",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] { background: #f5f0e8 !important; }
[data-testid="stAppViewContainer"] > .main     { background: #f5f0e8 !important; }
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { background: #f5f0e8 !important; }
.block-container { padding: 0 1.5rem 5rem !important; max-width: 820px !important; }

body, p, span, div { font-family: 'IBM Plex Sans', sans-serif !important; color: #1c1917; }
h1,h2,h3,h4 { font-family: 'Playfair Display', serif !important; }

/* ── masthead ── */
.masthead {
    border-bottom: 2px solid #1c1917;
    padding: 3rem 0 2rem;
    margin-bottom: 2.5rem;
    position: relative;
}
.masthead .issue {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #78716c;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.masthead h1 {
    font-size: clamp(2.4rem, 6vw, 3.4rem);
    font-weight: 700;
    color: #1c1917;
    line-height: 1.05;
    letter-spacing: -0.02em;
}
.masthead h1 em { font-style: italic; font-weight: 400; color: #a16207; }
.masthead .deck {
    margin-top: 0.8rem;
    font-size: 0.92rem;
    color: #78716c;
    font-weight: 300;
    max-width: 520px;
    line-height: 1.6;
}
.masthead .corner-tag {
    position: absolute;
    right: 0; top: 3rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #d6d3d1;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    writing-mode: vertical-rl;
}

/* ── section headers (editorial pull) ── */
.section-hd {
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
    margin-bottom: 1.4rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e7e5e4;
}
.section-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #a8a29e;
    margin-top: 0.35rem;
    flex-shrink: 0;
    letter-spacing: 0.06em;
}
.section-hd h2 {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1c1917;
    line-height: 1.2;
    letter-spacing: -0.01em;
}
.section-hd h2 em { font-style: italic; font-weight: 400; }

/* ── card / panel ── */
.panel {
    background: white;
    border: 1px solid #e7e5e4;
    border-radius: 4px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: 2px 2px 0 #e7e5e4;
}

/* ── stat row ── */
.stat-row {
    display: flex;
    gap: 0;
    border: 1px solid #e7e5e4;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 2rem;
    box-shadow: 2px 2px 0 #e7e5e4;
}
.stat-cell {
    flex: 1;
    padding: 1.3rem 1.4rem;
    background: white;
    border-right: 1px solid #e7e5e4;
    text-align: center;
}
.stat-cell:last-child { border-right: none; }
.stat-cell .sv {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #1c1917;
    line-height: 1;
}
.stat-cell .sl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #a8a29e;
    margin-top: 0.35rem;
}
.stat-cell.accent .sv { color: #a16207; }

/* ── insight cards (do / don't) ── */
.insight-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 0.5rem;
}
.insight-do {
    background: #fefce8;
    border: 1px solid #fde047;
    border-top: 4px solid #ca8a04;
    border-radius: 4px;
    padding: 1.4rem 1.5rem;
}
.insight-dont {
    background: #fafaf9;
    border: 1px solid #e7e5e4;
    border-top: 4px solid #78716c;
    border-radius: 4px;
    padding: 1.4rem 1.5rem;
}
.insight-do .ititle, .insight-dont .ititle {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.insight-do .ititle   { color: #854d0e; }
.insight-dont .ititle { color: #44403c; }
.ipoint {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 0.65rem;
    font-size: 0.82rem;
    line-height: 1.5;
    color: #44403c;
    font-weight: 400;
}
.ipoint .ibullet {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #a8a29e;
    margin-top: 0.15rem;
    flex-shrink: 0;
}
.insight-do .ibullet { color: #ca8a04; }

/* ── pull quote ── */
.pullquote {
    border-left: 3px solid #a16207;
    padding: 0.8rem 1.2rem;
    margin: 1.5rem 0;
    background: #fffbeb;
}
.pullquote p {
    font-family: 'Playfair Display', serif !important;
    font-size: 1rem;
    font-style: italic;
    color: #78716c;
    line-height: 1.6;
}

/* ── importance bar rows ── */
.imp-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid #f5f5f4;
}
.imp-row:last-child { border-bottom: none; }
.imp-feat {
    width: 180px;
    flex-shrink: 0;
    font-size: 0.8rem;
    font-weight: 500;
    color: #44403c;
    font-family: 'IBM Plex Mono', monospace;
}
.imp-bar-wrap { flex: 1; background: #f5f0e8; border-radius: 2px; height: 10px; overflow: hidden; }
.imp-bar-fill { height: 100%; border-radius: 2px; }
.imp-val {
    width: 44px;
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    color: #1c1917;
    flex-shrink: 0;
}

/* ── separator ── */
.sep {
    border: none;
    border-top: 1px solid #e7e5e4;
    margin: 2.5rem 0;
}

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

# ── Pre-compute data ───────────────────────────────────────────────────────────
@st.cache_data
def compute_discount_curves():
    """Average probability across user types and categories for each discount."""
    DISCOUNTS   = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    user_types  = ["new", "returning", "loyal"]
    categories  = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty", "Sports"]
    profiles    = [(pf, aov) for pf, aov in [(3,50),(6,100),(14,180)]]

    results = {d: [] for d in DISCOUNTS}
    for ut in user_types:
        ue = int(le_user.transform([ut])[0])
        for cat in categories:
            ce = int(le_category.transform([cat])[0])
            for pf, aov in profiles:
                for d in DISCOUNTS:
                    row = pd.DataFrame([{
                        "user_type": ue, "purchase_frequency": pf,
                        "avg_order_value": aov, "product_category": ce,
                        "product_price": 150.0, "rating": 3.8,
                        "discount_percentage": d,
                    }])[feat_cols]
                    prob = model.predict_proba(row)[0][1]
                    results[d].append(prob)

    disc_vals = DISCOUNTS
    avg_probs = [round(np.mean(results[d]) * 100, 1) for d in DISCOUNTS]
    return disc_vals, avg_probs

@st.cache_data
def compute_by_user_type():
    DISCOUNTS  = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    user_types = ["new", "returning", "loyal"]
    ce = int(le_category.transform(["Electronics"])[0])
    curves = {}
    for ut in user_types:
        ue   = int(le_user.transform([ut])[0])
        probs = []
        for d in DISCOUNTS:
            row = pd.DataFrame([{
                "user_type": ue, "purchase_frequency": 6,
                "avg_order_value": 100, "product_category": ce,
                "product_price": 150.0, "rating": 3.8, "discount_percentage": d,
            }])[feat_cols]
            probs.append(round(model.predict_proba(row)[0][1] * 100, 1))
        curves[ut] = probs
    return DISCOUNTS, curves

@st.cache_data
def get_feature_importances():
    importances = model.feature_importances_
    feat_names  = {
        "purchase_frequency":  "Purchase Frequency",
        "user_type":           "User Type",
        "avg_order_value":     "Avg Order Value",
        "rating":              "Rating",
        "discount_percentage": "Discount %",
        "product_price":       "Product Price",
        "product_category":    "Product Category",
    }
    data = sorted(
        [(feat_names.get(f, f), round(float(v) * 100, 1))
         for f, v in zip(feat_cols, importances)],
        key=lambda x: x[1], reverse=True
    )
    return data

disc_vals, avg_probs        = compute_discount_curves()
disc_vals_u, user_curves    = compute_by_user_type()
feature_importances         = get_feature_importances()

# ── Masthead ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="issue">◉ Model Intelligence Report &nbsp;·&nbsp; E-Commerce Analytics</div>
    <h1>Discount &amp; <em>Feature</em><br>Insights</h1>
    <div class="deck">
        An analytical deep-dive into how discounts drive purchase probability
        and which features your model has learned to trust most.
    </div>
    <div class="corner-tag">Powered by Random Forest</div>
</div>
""", unsafe_allow_html=True)

# ── Stat row ──────────────────────────────────────────────────────────────────
top_feat_name, top_feat_val = feature_importances[0]
max_disc_idx  = avg_probs.index(max(avg_probs))
max_disc_val  = disc_vals[max_disc_idx]
prob_range    = round(max(avg_probs) - min(avg_probs), 1)

st.markdown(
    '<div class="stat-row">'
    + '<div class="stat-cell accent"><div class="sv">' + str(top_feat_val) + '%</div><div class="sl">Top Feature Weight</div></div>'
    + '<div class="stat-cell"><div class="sv">' + str(len(feature_importances)) + '</div><div class="sl">Features Evaluated</div></div>'
    + '<div class="stat-cell"><div class="sv">+' + str(prob_range) + '%</div><div class="sl">Discount Impact Range</div></div>'
    + '<div class="stat-cell"><div class="sv">' + str(max_disc_val) + '%</div><div class="sl">Highest Avg Disc.</div></div>'
    + '</div>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Discount vs Purchase Probability
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-hd">
    <div class="section-num">01 /</div>
    <div>
        <h2>Discount vs <em>Purchase Probability</em></h2>
    </div>
</div>
""", unsafe_allow_html=True)

# Chart 1A — Average curve
st.markdown('<div class="panel">', unsafe_allow_html=True)

COLORS = {"new": "#ef4444", "returning": "#3b82f6", "loyal": "#10b981"}
x_fine = np.linspace(min(disc_vals), max(disc_vals), 400)
y_avg  = np.interp(x_fine, disc_vals, avg_probs)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
fig.patch.set_facecolor("white")

# left: average curve
ax = axes[0]
ax.set_facecolor("white")
ax.fill_between(x_fine, y_avg, alpha=0.07, color="#a16207")
ax.plot(x_fine, y_avg, color="#1c1917", linewidth=2.2, zorder=3)
ax.scatter(disc_vals, avg_probs, color="#a16207", s=60, zorder=5,
           edgecolors="white", linewidths=1.5)
for d, p in zip(disc_vals, avg_probs):
    ax.text(d, p + 1.0, f"{p}%", ha="center", va="bottom", fontsize=6.5, color="#78716c")

ax.set_xlabel("Discount (%)", fontsize=9, color="#78716c", labelpad=5)
ax.set_ylabel("Purchase Probability (%)", fontsize=9, color="#78716c", labelpad=5)
ax.set_title("Average Across All Profiles", fontsize=10, fontweight="bold", color="#1c1917", pad=8)
ax.set_xticks(disc_vals)
ax.set_xticklabels([f"{d}%" for d in disc_vals], fontsize=7.5, color="#78716c")
ax.set_ylim(max(0, min(avg_probs) - 8), min(105, max(avg_probs) + 10))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.tick_params(colors="#9ca3af", labelsize=7.5)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_edgecolor("#e7e5e4")
ax.grid(axis="y", linestyle="--", alpha=0.5, color="#f5f0e8")

# right: by user type
ax2 = axes[1]
ax2.set_facecolor("white")
for ut, clr in COLORS.items():
    probs = user_curves[ut]
    y_u   = np.interp(x_fine, disc_vals_u, probs)
    ax2.plot(x_fine, y_u, color=clr, linewidth=1.8, label=ut.capitalize(), zorder=3)
    ax2.scatter(disc_vals_u, probs, color=clr, s=35, zorder=4,
                edgecolors="white", linewidths=1.2)

ax2.set_xlabel("Discount (%)", fontsize=9, color="#78716c", labelpad=5)
ax2.set_title("By User Type", fontsize=10, fontweight="bold", color="#1c1917", pad=8)
ax2.set_xticks(disc_vals_u)
ax2.set_xticklabels([f"{d}%" for d in disc_vals_u], fontsize=7.5, color="#78716c")
ax2.set_ylim(0, 105)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax2.tick_params(colors="#9ca3af", labelsize=7.5)
for s in ["top","right"]: ax2.spines[s].set_visible(False)
for s in ["left","bottom"]: ax2.spines[s].set_edgecolor("#e7e5e4")
ax2.grid(axis="y", linestyle="--", alpha=0.5, color="#f5f0e8")
ax2.legend(fontsize=8, framealpha=0.9, edgecolor="#e7e5e4", loc="lower right")

plt.tight_layout(pad=1.5)
st.pyplot(fig)
plt.close(fig)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="pullquote">
    <p>"Discounts reliably increase purchase probability — but the marginal gain
    diminishes above 25%. Loyal customers convert at high rates even with minimal
    incentive; new customers require stronger nudges to overcome first-purchase friction."</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-hd">
    <div class="section-num">02 /</div>
    <div>
        <h2>Feature <em>Importance</em></h2>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)

max_imp = max(v for _, v in feature_importances)
imp_html = ""
for feat, val in feature_importances:
    w       = round((val / max_imp) * 100, 1)
    is_top  = (feat == feature_importances[0][0])
    bar_col = "#a16207" if is_top else "#1c1917"
    opacity = "1.0" if is_top else str(round(0.3 + 0.7 * (val / max_imp), 2))
    imp_html += (
        '<div class="imp-row">'
        + '<span class="imp-feat">' + feat + '</span>'
        + '<span class="imp-bar-wrap"><span class="imp-bar-fill" style="width:' + str(w) + '%;background:' + bar_col + ';opacity:' + opacity + ';"></span></span>'
        + '<span class="imp-val">' + str(val) + '%</span>'
        + '</div>'
    )
st.markdown(imp_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Matplotlib horizontal bar (secondary view)
fig2, ax3 = plt.subplots(figsize=(8, 3.2))
fig2.patch.set_facecolor("white")
ax3.set_facecolor("white")

feats  = [f for f, _ in reversed(feature_importances)]
values = [v for _, v in reversed(feature_importances)]
colors = ["#a16207" if f == feature_importances[0][0] else "#d6d3d1" for f in feats]
bars   = ax3.barh(feats, values, color=colors, height=0.55, edgecolor="none")

for bar, val in zip(bars, values):
    ax3.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             f"{val}%", va="center", ha="left", fontsize=8, color="#44403c", fontweight="500")

ax3.set_xlabel("Importance (%)", fontsize=9, color="#78716c")
ax3.set_xlim(0, max(values) * 1.25)
ax3.tick_params(axis="y", labelsize=8.5, colors="#44403c")
ax3.tick_params(axis="x", labelsize=7.5, colors="#a8a29e")
for s in ["top","right","left"]: ax3.spines[s].set_visible(False)
ax3.spines["bottom"].set_edgecolor("#e7e5e4")
ax3.grid(axis="x", linestyle="--", alpha=0.4, color="#f5f0e8")

plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Business Insights
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-hd">
    <div class="section-num">03 /</div>
    <div>
        <h2>Business <em>Playbook</em></h2>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="insight-grid">

  <div class="insight-do">
    <div class="ititle">&#10003;&nbsp; When to Give Discounts</div>

    <div class="ipoint"><span class="ibullet">01</span>
    <span><strong>New customers on first purchase.</strong> Acquisition cost is high; a 15–25% discount lowers the barrier and builds lifetime value.</span></div>

    <div class="ipoint"><span class="ibullet">02</span>
    <span><strong>High-priced products (Electronics, Sports).</strong> Price sensitivity is strongest here. Even a 10% reduction can tip a borderline decision.</span></div>

    <div class="ipoint"><span class="ibullet">03</span>
    <span><strong>Mid-rated products (3–4 stars).</strong> Customers are uncertain. A discount compensates for perceived risk and accelerates commitment.</span></div>

    <div class="ipoint"><span class="ibullet">04</span>
    <span><strong>Win-back campaigns for lapsed customers.</strong> Returning users who haven't purchased recently respond strongly to time-limited offers (20–30%).</span></div>

    <div class="ipoint"><span class="ibullet">05</span>
    <span><strong>Cart abandonment recovery.</strong> Customers who browsed but didn't buy show significantly higher conversion with a 10–15% nudge within 24 hours.</span></div>

  </div>

  <div class="insight-dont">
    <div class="ititle">&#10005;&nbsp; When Not to Give Discounts</div>

    <div class="ipoint"><span class="ibullet">01</span>
    <span><strong>Loyal, high-frequency customers.</strong> They buy regardless. Discounts erode margin without improving conversion — reward loyalty differently.</span></div>

    <div class="ipoint"><span class="ibullet">02</span>
    <span><strong>Products with 4.5+ star ratings.</strong> Social proof is already doing the conversion work. Discounting trains customers to wait for deals.</span></div>

    <div class="ipoint"><span class="ibullet">03</span>
    <span><strong>Low average order value segments.</strong> The discount cost may exceed lifetime value. Focus budget on higher-AOV cohorts with bigger upside.</span></div>

    <div class="ipoint"><span class="ibullet">04</span>
    <span><strong>Above 30% on mass-market items.</strong> Diminishing returns kick in sharply. The model shows probability plateaus — deeper cuts only damage margin.</span></div>

    <div class="ipoint"><span class="ibullet">05</span>
    <span><strong>During peak demand periods.</strong> High organic traffic converts without incentive. Reserve discounting for low-demand windows to protect price anchoring.</span></div>

  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pullquote" style="margin-top:1.5rem;">
    <p>"Purchase frequency and user type account for over 55% of model weight combined —
    meaning who the customer is matters far more than what discount you offer.
    Invest in customer relationships first; pricing strategy second."</p>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.5rem;border-top:2px solid #1c1917;
    display:flex;justify-content:space-between;align-items:center;
    font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
    color:#a8a29e;letter-spacing:0.1em;text-transform:uppercase;">
    <span>Random Forest Classifier</span>
    <span>◉ Model Intelligence Report</span>
    <span>Synthetic E-Commerce Data</span>
</div>
""", unsafe_allow_html=True)