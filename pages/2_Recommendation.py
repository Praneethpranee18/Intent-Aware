import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Discount Recommender",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] { background: #fafaf8 !important; }
[data-testid="stAppViewContainer"] > .main  { background: #fafaf8 !important; }
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { background: #fafaf8 !important; }
.block-container { padding: 0 1.5rem 4rem !important; max-width: 800px !important; }

body, p, span, div, label { font-family: 'Inter', sans-serif !important; color: #1a1a2e; }
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

/* ── masthead ── */
.masthead {
    background: #1a1a2e;
    margin: 0 -1.5rem 2.5rem;
    padding: 3.5rem 2.5rem 3rem;
    position: relative;
    overflow: hidden;
}
.masthead::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(245,158,11,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.masthead::before {
    content: '';
    position: absolute;
    left: 60%; bottom: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.masthead .tag {
    display: inline-block;
    background: rgba(245,158,11,0.15);
    border: 1px solid rgba(245,158,11,0.3);
    color: #fbbf24;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 1.1rem;
    font-family: 'Inter', sans-serif;
}
.masthead h1 {
    font-size: clamp(2rem, 5vw, 2.8rem);
    font-weight: 800;
    color: #f5f5f0;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.masthead h1 span { color: #fbbf24; }
.masthead .sub {
    margin-top: 0.7rem;
    font-size: 0.88rem;
    color: rgba(245,245,240,0.45);
    font-weight: 300;
    font-family: 'Inter', sans-serif;
}

/* ── section labels ── */
.slabel {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.slabel::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e5e7eb;
}

/* ── input panel ── */
.panel {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── streamlit widget overrides ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    color: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] input:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.12) !important;
    background: white !important;
}
[data-testid="stSlider"] > div > div > div > div { background: #f59e0b !important; }

/* cta button */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: #1a1a2e !important;
    color: #fbbf24 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin-top: 0.6rem !important;
}
div[data-testid="stButton"] > button:hover {
    background: #2d2d4e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(26,26,46,0.25) !important;
}

/* ── hero result ── */
.hero-result {
    background: #1a1a2e;
    border-radius: 18px;
    padding: 2.5rem;
    margin: 1.5rem 0 1.2rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlide 0.5s cubic-bezier(0.16,1,0.3,1) forwards;
}
@keyframes fadeSlide {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
.hero-result::before {
    content:'';
    position:absolute; inset:0;
    background: radial-gradient(ellipse 60% 80% at 85% 50%, rgba(245,158,11,0.12) 0%, transparent 65%);
}
.hr-disc {
    text-align: center;
    flex-shrink: 0;
    background: rgba(245,158,11,0.12);
    border: 2px solid rgba(245,158,11,0.3);
    border-radius: 16px;
    padding: 1.2rem 1.8rem;
    min-width: 130px;
}
.hr-disc .dlabel {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(251,191,36,0.6);
    font-family: 'Syne', sans-serif;
    margin-bottom: 0.3rem;
}
.hr-disc .dval {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #fbbf24;
    line-height: 1;
}
.hr-divider {
    width: 1px;
    height: 80px;
    background: rgba(255,255,255,0.08);
    flex-shrink: 0;
}
.hr-info { flex: 1; }
.hr-info .prob-row {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.hr-info .prob-big {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #10b981;
    line-height: 1;
}
.hr-info .prob-label {
    font-size: 0.72rem;
    color: rgba(245,245,240,0.45);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
}
.hr-info .gain-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #6ee7b7;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
    font-family: 'Inter', sans-serif;
}
.hr-info .rec-text {
    font-size: 0.82rem;
    color: rgba(245,245,240,0.55);
    line-height: 1.5;
    font-weight: 300;
    font-family: 'Inter', sans-serif;
}

/* ── discount breakdown table ── */
.disc-table { width: 100%; border-collapse: collapse; }
.disc-table th {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9ca3af;
    padding: 0 0 0.75rem;
    text-align: left;
    font-family: 'Syne', sans-serif;
    border-bottom: 1px solid #f3f4f6;
}
.disc-table th:last-child { text-align: right; }
.disc-table td {
    padding: 0.7rem 0;
    border-bottom: 1px solid #f9fafb;
    font-size: 0.85rem;
    vertical-align: middle;
}
.disc-table tr:last-child td { border-bottom: none; }
.disc-table .td-disc { font-weight: 600; color: #374151; width: 60px; }
.disc-table .td-bar  { padding-right: 1rem; }
.disc-table .td-pct  { font-weight: 700; color: #1a1a2e; text-align: right; width: 56px; }
.disc-table .td-tag  { text-align: right; width: 90px; }
.bar-wrap { background: #f3f4f6; border-radius: 999px; height: 8px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }
.tag-best {
    display: inline-block;
    background: #fef3c7;
    border: 1px solid #fcd34d;
    color: #92400e;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    letter-spacing: 0.06em;
    font-family: 'Syne', sans-serif;
}
.tag-curr {
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    letter-spacing: 0.06em;
    font-family: 'Syne', sans-serif;
}

/* ── insight strip ── */
.insight-strip {
    display: flex;
    gap: 0.8rem;
    margin: 1.2rem 0;
    flex-wrap: wrap;
}
.ins-card {
    flex: 1;
    min-width: 140px;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.ins-card .iv { font-family:'Syne',sans-serif; font-size:1.55rem; font-weight:800; color:#1a1a2e; line-height:1; }
.ins-card .il { font-size:0.65rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#9ca3af; margin-top:0.3rem; }
.ins-card.accent .iv { color:#f59e0b; }

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

DISCOUNT_OPTIONS = [5, 10, 15, 20, 25, 30, 40, 50]

def sweep_discounts(user_enc, cat_enc, pf, aov, pp, rating):
    rows = []
    for d in DISCOUNT_OPTIONS:
        row = pd.DataFrame([{
            "user_type": user_enc, "purchase_frequency": pf,
            "avg_order_value": aov, "product_category": cat_enc,
            "product_price": pp, "rating": rating,
            "discount_percentage": d,
        }])[feat_cols]
        prob = model.predict_proba(row)[0][1]
        rows.append({"discount": d, "probability": prob, "prob_pct": round(prob * 100, 1)})
    return pd.DataFrame(rows)

# ── Masthead ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="tag">◆ Discount Intelligence</div>
    <h1>Find the <span>Optimal</span><br>Discount</h1>
    <div class="sub">Sweep every discount tier and surface the one that maximises purchase probability</div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="slabel">Customer</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    user_type          = st.selectbox("User Type", ["new", "returning", "loyal"], index=1)
    purchase_frequency = st.slider("Purchase Frequency", 1, 20, 5)
with c2:
    avg_order_value = st.number_input("Avg Order Value ($)", 10.0, 500.0, 85.0, step=5.0)
    rating          = st.slider("Product Rating", 1.0, 5.0, 4.0, step=0.1)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="slabel">Product</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    product_category = st.selectbox("Category",
        ["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty", "Sports"])
    product_price = st.number_input("Product Price ($)", 5.0, 1500.0, 120.0, step=10.0)
with c4:
    current_discount = st.select_slider("Current Discount (%)",
        options=[0, 5, 10, 15, 20, 25, 30, 40, 50], value=10,
        help="Your current discount — used to compute the uplift from switching")
st.markdown("</div>", unsafe_allow_html=True)

run_btn = st.button("◆  Find Best Discount")

# ── Run sweep ──────────────────────────────────────────────────────────────────
if run_btn:
    user_enc = int(le_user.transform([user_type])[0])
    cat_enc  = int(le_category.transform([product_category])[0])

    df = sweep_discounts(user_enc, cat_enc, purchase_frequency,
                         avg_order_value, product_price, rating)

    best_row  = df.loc[df["probability"].idxmax()]
    best_disc = int(best_row["discount"])
    best_prob = best_row["prob_pct"]

    # current discount probability (use 0 if current_discount not in sweep)
    curr_row  = df[df["discount"] == current_discount]
    curr_prob = float(curr_row["prob_pct"].values[0]) if len(curr_row) else None

    gain      = round(best_prob - curr_prob, 1) if curr_prob is not None else None
    gain_str  = ("+" if gain >= 0 else "") + str(gain) + "%" if gain is not None else "n/a"
    gain_sign = gain >= 0 if gain is not None else True

    if best_disc == current_discount:
        rec_text = "Your current discount is already the optimal choice — no change needed."
    else:
        rec_text = (
            "Switch from " + str(current_discount) + "% to " + str(best_disc)
            + "% to achieve peak purchase probability."
            + (" Uplift: " + gain_str + " from current setting." if gain else "")
        )

    # ── Hero result ───────────────────────────────────────────────────────────
    gain_chip_html = ""
    if gain is not None and gain != 0:
        gain_chip_html = '<div class="gain-chip">&#9650; ' + gain_str + ' uplift vs current</div>'
    elif gain == 0:
        gain_chip_html = '<div class="gain-chip">&#10003; Already optimal</div>'

    st.markdown(
        '<div class="hero-result">'
        + '<div class="hr-disc">'
        +   '<div class="dlabel">Best Discount</div>'
        +   '<div class="dval">' + str(best_disc) + '%</div>'
        + '</div>'
        + '<div class="hr-divider"></div>'
        + '<div class="hr-info">'
        +   '<div class="prob-row">'
        +     '<div class="prob-big">' + str(best_prob) + '%</div>'
        +     '<div class="prob-label">peak probability</div>'
        +   '</div>'
        +   gain_chip_html
        +   '<div class="rec-text">' + rec_text + '</div>'
        + '</div>'
        + '</div>',
        unsafe_allow_html=True
    )

    # ── Insight strip ─────────────────────────────────────────────────────────
    min_prob  = df["prob_pct"].min()
    prob_range = round(best_prob - min_prob, 1)
    eff_price = round(product_price * (1 - best_disc / 100), 2)

    st.markdown(
        '<div class="insight-strip">'
        + '<div class="ins-card accent"><div class="iv">' + str(best_disc) + '%</div><div class="il">Recommended Discount</div></div>'
        + '<div class="ins-card"><div class="iv">' + str(best_prob) + '%</div><div class="il">Peak Probability</div></div>'
        + '<div class="ins-card"><div class="iv">$' + str(eff_price) + '</div><div class="il">Effective Price</div></div>'
        + '<div class="ins-card"><div class="iv">+' + str(prob_range) + '%</div><div class="il">Range (min→max)</div></div>'
        + '</div>',
        unsafe_allow_html=True
    )

    # ── Discount breakdown table ──────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="slabel">All Discount Levels</div>', unsafe_allow_html=True)

    max_p = df["probability"].max()
    rows_html = (
        '<table class="disc-table">'
        + '<tr>'
        +   '<th>Discount</th>'
        +   '<th style="width:100%">Probability</th>'
        +   '<th>Score</th>'
        +   '<th>Tag</th>'
        + '</tr>'
    )
    for _, r in df.iterrows():
        d   = int(r["discount"])
        pct = r["prob_pct"]
        w   = round((r["probability"] / max_p) * 100, 1)
        is_best = (d == best_disc)
        is_curr = (d == current_discount)

        if is_best:
            bar_col, tag_html = "#f59e0b", '<span class="tag-best">&#11088; Best</span>'
        elif is_curr and not is_best:
            bar_col, tag_html = "#3b82f6", '<span class="tag-curr">&#9664; Current</span>'
        else:
            bar_col, tag_html = "#d1d5db", ""

        rows_html += (
            '<tr>'
            + '<td class="td-disc">' + str(d) + '%</td>'
            + '<td class="td-bar"><div class="bar-wrap"><div class="bar-fill" style="width:' + str(w) + '%;background:' + bar_col + ';"></div></div></td>'
            + '<td class="td-pct">' + str(pct) + '%</td>'
            + '<td class="td-tag">' + tag_html + '</td>'
            + '</tr>'
        )
    rows_html += '</table>'
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Matplotlib chart ──────────────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Probability Curve</div>', unsafe_allow_html=True)

    disc_vals = df["discount"].tolist()
    prob_vals = df["prob_pct"].tolist()
    x_fine    = np.linspace(min(disc_vals), max(disc_vals), 400)
    y_fine    = np.interp(x_fine, disc_vals, prob_vals)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(x_fine, y_fine, alpha=0.08, color="#f59e0b")
    ax.plot(x_fine, y_fine, color="#1a1a2e", linewidth=2.2, zorder=3)
    ax.scatter(disc_vals, prob_vals, color="#1a1a2e", s=55, zorder=5, edgecolors="white", linewidths=1.5)

    # highlight current discount
    if current_discount in disc_vals:
        cp = prob_vals[disc_vals.index(current_discount)]
        ax.scatter([current_discount], [cp], color="#3b82f6", s=130, zorder=6,
                   edgecolors="white", linewidths=2)
        ax.annotate(f"Current\n{cp}%",
                    xy=(current_discount, cp), xytext=(-36, 12),
                    textcoords="offset points", fontsize=8, color="#1d4ed8",
                    arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.1))

    # highlight best
    bpv = prob_vals[disc_vals.index(best_disc)]
    ax.scatter([best_disc], [bpv], color="#f59e0b", s=180, zorder=7,
               edgecolors="white", linewidths=2, marker="*")
    ax.annotate(f"Best: {best_disc}%\n{bpv}%",
                xy=(best_disc, bpv), xytext=(8, 10),
                textcoords="offset points", fontsize=8.5, color="#92400e",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.2))

    for d, p in zip(disc_vals, prob_vals):
        if d not in (current_discount, best_disc):
            ax.text(d, p + 1.2, f"{p}%", ha="center", va="bottom", fontsize=7, color="#6b7280")

    ax.set_xlabel("Discount (%)", fontsize=9.5, color="#6b7280", labelpad=6)
    ax.set_ylabel("Purchase Probability (%)", fontsize=9.5, color="#6b7280", labelpad=6)
    ax.set_xticks(disc_vals)
    ax.set_xticklabels([f"{d}%" for d in disc_vals], fontsize=8.5, color="#6b7280")
    ax.set_ylim(max(0, min(prob_vals) - 8), min(105, max(prob_vals) + 10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(colors="#9ca3af", labelsize=8.5)
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]: ax.spines[spine].set_edgecolor("#f3f4f6")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#f3f4f6")
    ax.grid(axis="x", linestyle=":",  alpha=0.3, color="#f3f4f6")

    ax.legend(handles=[
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#3b82f6",
                   markersize=8, label=f"Current ({current_discount}%)"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="#f59e0b",
                   markersize=11, label=f"Best ({best_disc}%)"),
    ], fontsize=8, framealpha=0.9, edgecolor="#e5e7eb", loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2rem;color:#d1d5db;">
        <div style="font-size:2.5rem;margin-bottom:0.8rem;opacity:0.3;">◆</div>
        <div style="font-size:0.8rem;letter-spacing:0.12em;text-transform:uppercase;color:#9ca3af;">
            Configure inputs above and click Find Best Discount
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
    border-top:1px solid #f3f4f6;font-size:0.72rem;
    color:#d1d5db;letter-spacing:0.08em;text-transform:uppercase;
    font-family:'Inter',sans-serif;">
    Discount Sweep &nbsp;·&nbsp; 8 Tiers Tested &nbsp;·&nbsp; ◆
</div>
""", unsafe_allow_html=True)