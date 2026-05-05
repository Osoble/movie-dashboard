import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="CineAI - Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1,h2,h3,h4 { font-family: 'Syne', sans-serif; letter-spacing:-0.02em; }

.stApp { background:#09090f; color:#e2e0f0; }
[data-testid="stSidebar"] { background:#0f0f1a; border-right:1px solid #1a1a2e; }
[data-testid="stSidebar"] .stRadio label { color:#a0a0c0; font-size:0.9rem; }

.hero { background:linear-gradient(135deg,#1a0533 0%,#0d1a3a 50%,#001a1a 100%);
        border:1px solid #2a1a4a; border-radius:16px; padding:32px 36px; margin-bottom:28px; }
.hero h1 { font-size:2.4rem; font-weight:800; margin:0 0 8px;
           background:linear-gradient(90deg,#c084fc,#60a5fa,#34d399);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#8b8baa; font-size:1rem; margin:0; }

.stat-card { background:#0f0f1a; border:1px solid #1e1e35;
             border-radius:14px; padding:20px 22px; text-align:center; }
.stat-card .num { font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800;
                  background:linear-gradient(135deg,#c084fc,#60a5fa);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.stat-card .lbl { font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase;
                  color:#555575; margin-top:4px; }

.model-badge { display:inline-flex; align-items:center; gap:6px;
               background:#1a0533; border:1px solid #6d28d9;
               border-radius:20px; padding:4px 14px;
               font-size:0.75rem; color:#c084fc; font-weight:600; }
.best-model  { background:#0a2a0a; border:1px solid #16a34a; color:#4ade80; }

.rec-card { background:#0f0f1a; border:1px solid #1e1e35;
            border-left:3px solid #c084fc; border-radius:12px;
            padding:16px 20px; margin-bottom:10px;
            transition:transform 0.2s; }
.rec-card:hover { transform:translateX(4px); border-left-color:#60a5fa; }
.rec-rank { font-size:0.68rem; letter-spacing:0.14em; text-transform:uppercase;
            color:#c084fc; font-weight:700; }
.rec-title { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700;
             color:#e2e0f0; margin:5px 0 3px; }
.rec-genre { font-size:0.78rem; color:#555575; }
.rec-score { font-size:0.85rem; color:#60a5fa; margin-top:6px; font-weight:500; }

.filter-bar { background:#0f0f1a; border:1px solid #1e1e35; border-radius:12px;
              padding:18px 22px; margin-bottom:20px; }

.insight-box { background:linear-gradient(135deg,#0a1628,#1a0a28);
               border:1px solid #2a2a4a; border-radius:12px; padding:18px 22px;
               margin-top:16px; }
.insight-box h4 { font-family:'Syne',sans-serif; color:#c084fc; font-size:0.85rem;
                  letter-spacing:0.08em; text-transform:uppercase; margin:0 0 8px; }

.search-result { background:#0f0f1a; border:1px solid #1e1e35; border-radius:10px;
                 padding:12px 16px; margin-bottom:8px; cursor:pointer; }
.search-result:hover { border-color:#6d28d9; }

.tab-metric { background:#0f0f1a; border:1px solid #1e1e35; border-radius:10px;
              padding:14px 18px; }

div[data-testid="metric-container"] { background:#0f0f1a; border:1px solid #1e1e35;
                                       border-radius:10px; padding:12px 16px; }
.stButton > button { background:linear-gradient(135deg,#6d28d9,#2563eb);
                     color:#fff; border:none; border-radius:10px;
                     font-family:'Syne',sans-serif; font-weight:700;
                     letter-spacing:0.04em; padding:10px 28px; width:100%; }
.stButton > button:hover { opacity:0.9; }
.stSelectbox label, .stSlider label, .stNumberInput label,
.stMultiSelect label, .stTextInput label { color:#8b8baa; font-size:0.85rem; }

[data-baseweb="select"] { background:#0f0f1a; border-color:#1e1e35; }
.stProgress .st-bo { background:linear-gradient(90deg,#6d28d9,#2563eb); }

hr { border-color:#1e1e35; }

.section-title { font-family:'Syne',sans-serif; font-size:0.68rem;
                 letter-spacing:0.18em; text-transform:uppercase;
                 color:#555575; padding-bottom:10px;
                 border-bottom:1px solid #1e1e35; margin-bottom:18px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/"
BG        = "#09090f"
CARD      = "#0f0f1a"
GRID      = "#1e1e35"
PURPLE    = "#c084fc"
BLUE      = "#60a5fa"
GREEN     = "#4ade80"
TEXT      = "#e2e0f0"
MUTED     = "#555575"

def plot_layout(title="", h=380):
    return dict(
        title=dict(text=title, font=dict(family="Syne", size=14, color=TEXT)),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter", color=TEXT, size=12),
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, linecolor=GRID),
        margin=dict(l=16, r=16, t=44, b=16),
        height=h,
        legend=dict(bgcolor=CARD, bordercolor=GRID, borderwidth=1)
    )

# ── Load assets ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    assets, errors = {}, []
    files = {
        "model":        ("xgb_model.pkl",    "rb",  "pickle"),
        "results":      ("results.pkl",       "rb",  "pickle"),
        "movies":       ("movies_app.csv",    None,  "csv"),
        "feature_cols": ("feature_cols.pkl",  "rb",  "pickle"),
        "sample_fe":    ("sample_fe.csv",     None,  "csv"),
    }
    for key, (fname, mode, ftype) in files.items():
        path = os.path.join(CHECKPOINT_PATH, fname)
        try:
            if ftype == "pickle":
                with open(path, mode) as f:
                    assets[key] = pickle.load(f)
            else:
                assets[key] = pd.read_csv(path)
        except Exception as e:
            errors.append(f"{fname}: {e}")
    assets["errors"] = errors
    return assets

with st.spinner("Loading model and data..."):
    assets = load_assets()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## CineAI")
    st.markdown("<div class='section-title'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", [
        "Home",
        "Recommendation Engine",
        "Movie Explorer",
        "Model Comparison",
        "Feature Importance",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div class='section-title'>System Status</div>", unsafe_allow_html=True)
    items = [
        ("XGBoost Model",  "model"        in assets),
        ("Results",        "results"       in assets),
        ("Movies DB",      "movies"        in assets),
        ("Feature Cols",   "feature_cols"  in assets),
        ("Sample Data",    "sample_fe"     in assets),
    ]
    for label, ok in items:
        icon = "green" if ok else "red"
        dot  = "✓" if ok else "✗"
        st.markdown(f"<span style='color:{'#4ade80' if ok else '#f87171'}'>{dot}</span> <span style='color:#8b8baa;font-size:0.85rem'>{label}</span>", unsafe_allow_html=True)

    if assets["errors"]:
        st.error("Missing: " + ", ".join(assets["errors"]))

# ── Helper: results df ─────────────────────────────────────────────────────────
def get_results_df():
    results = assets.get("results", {})
    rows = []
    for model, metrics in results.items():
        rows.append({
            "Model":        model,
            "RMSE":         float(metrics.get("RMSE",       np.nan)),
            "MAE":          float(metrics.get("MAE",        np.nan)),
            "R2":           float(metrics.get("R2",         metrics.get("R²", np.nan))),
            "NDCG@10":      float(metrics.get("NDCG@10",   np.nan)),
            "Train Time (s)": float(metrics.get("Train_Time_s", np.nan)),
        })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""<div class='hero'>
        <h1>CineAI Movie Recommender</h1>
        <p>A machine learning powered recommendation system built on 10 million MovieLens ratings.
        Explore models, discover movies, and get personalized recommendations.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats = [("10M+", "Training Ratings"), ("62K", "Unique Movies"),
             ("162K", "Users"), ("0.8551", "Best RMSE (XGBoost)")]
    for col, (num, lbl) in zip([c1,c2,c3,c4], stats):
        with col:
            st.markdown(f"<div class='stat-card'><div class='num'>{num}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("<div class='section-title'>Model Pipeline</div>", unsafe_allow_html=True)
        tiers = [
            ("Tier 1", "Baseline", "Linear Regression, Ridge, Decision Tree, KNN", "#6d28d9"),
            ("Tier 2", "Ensemble", "Random Forest, XGBoost", "#2563eb"),
            ("Tier 3", "Collab. Filtering", "SVD Matrix Factorization (k=50)", "#0891b2"),
            ("Tier 4", "Neural CF", "TensorFlow Keras (optional)", "#059669"),
        ]
        for tier, name, models, color in tiers:
            st.markdown(f"""<div style='background:{CARD};border:1px solid {GRID};border-left:3px solid {color};
                border-radius:10px;padding:12px 16px;margin-bottom:8px'>
                <span style='font-size:0.68rem;color:{color};letter-spacing:0.1em;text-transform:uppercase;font-weight:700'>{tier} — {name}</span>
                <div style='font-size:0.85rem;color:{MUTED};margin-top:3px'>{models}</div>
            </div>""", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='section-title'>Feature Engineering</div>", unsafe_allow_html=True)
        features = [
            ("User Features",    "Avg rating, rating count, user bias",        12),
            ("Movie Features",   "Avg rating, popularity log, tail flag",       12),
            ("Genre SVD",        "TF-IDF + TruncatedSVD (10 components)",       10),
            ("Genome SVD",       "Movie-tag matrix SVD (15 components)",        15),
            ("User Tag SVD",     "User tag TF-IDF + SVD (10 components)",       10),
            ("K-Means Cluster",  "Genre-based movie cluster (20 groups)",        1),
        ]
        total = sum(n for _,_,n in features)
        for name, desc, n in features:
            pct = n / total
            st.markdown(f"""<div style='margin-bottom:10px'>
                <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                    <span style='font-size:0.85rem;color:{TEXT}'>{name}</span>
                    <span style='font-size:0.75rem;color:{MUTED}'>{n} features</span>
                </div>
                <div style='background:{GRID};border-radius:4px;height:6px'>
                    <div style='background:linear-gradient(90deg,#6d28d9,#2563eb);width:{pct*100:.0f}%;height:6px;border-radius:4px'></div>
                </div>
                <div style='font-size:0.75rem;color:{MUTED};margin-top:3px'>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style='background:#1a0a28;border:1px solid #6d28d9;border-radius:10px;
            padding:10px 14px;margin-top:8px;text-align:center'>
            <span style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:{PURPLE}'>{total}</span>
            <span style='color:{MUTED};font-size:0.8rem;margin-left:8px'>Total Features</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Recommendation Engine":
    st.markdown("# Recommendation Engine")
    st.markdown("<div class='section-title'>Personalized movie predictions using XGBoost</div>", unsafe_allow_html=True)

    if not all(k in assets for k in ["model", "movies", "sample_fe", "feature_cols"]):
        st.error("One or more required files are missing.")
        st.stop()

    model        = assets["model"]
    movies_df    = assets["movies"]
    sample_fe    = assets["sample_fe"]
    feature_cols = assets["feature_cols"]

    st.markdown("<div class='filter-bar'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        available_users = sorted(sample_fe["userId"].unique())
        user_id = st.selectbox("Select User ID", available_users)
    with col2:
        n_recs = st.slider("Number of recommendations", 5, 25, 10)
    with col3:
        all_genres = sorted(set(
            g for genres in movies_df["genres"].dropna()
            for g in genres.split("|") if g != "(no genres listed)"
        ))
        genre_filter = st.multiselect("Filter by genre (optional)", all_genres)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Get My Recommendations"):
        user_data = sample_fe[sample_fe["userId"] == user_id].copy()
        if user_data.empty:
            st.error(f"No data found for user {user_id}.")
        else:
            already_rated  = set(user_data["movieId"].values)
            candidate_pool = movies_df[~movies_df["movieId"].isin(already_rated)].copy()

            if genre_filter:
                mask = candidate_pool["genres"].apply(
                    lambda g: any(gf in g.split("|") for gf in genre_filter)
                )
                candidate_pool = candidate_pool[mask]

            if candidate_pool.empty:
                st.warning("No movies found matching your genre filter.")
            else:
                user_profile  = user_data[feature_cols].mean()
                candidates_fe = sample_fe[
                    sample_fe["movieId"].isin(candidate_pool["movieId"])
                ].drop_duplicates("movieId").copy()

                for col in feature_cols:
                    if "user" in col:
                        candidates_fe[col] = user_profile.get(col, candidates_fe[col].mean())

                missing = [c for c in feature_cols if c not in candidates_fe.columns]
                for c in missing:
                    candidates_fe[c] = 0

                preds = model.predict(candidates_fe[feature_cols])
                candidates_fe["predicted_rating"] = np.clip(preds, 0.5, 5.0)

                top_recs = (
                    candidates_fe[["movieId", "predicted_rating"]]
                    .sort_values("predicted_rating", ascending=False)
                    .head(n_recs)
                    .merge(movies_df, on="movieId", how="left")
                )

                # User stats
                u_avg  = float(user_data["user_avg_rating"].iloc[0]) if "user_avg_rating" in user_data.columns else 3.5
                u_cnt  = int(user_data["user_rating_count"].iloc[0]) if "user_rating_count" in user_data.columns else 0
                u_bias = float(user_data["user_bias"].iloc[0]) if "user_bias" in user_data.columns else 0.0

                st.markdown("---")
                ua, ub, uc = st.columns(3)
                with ua:
                    st.markdown(f"<div class='stat-card'><div class='num' style='font-size:1.4rem'>{u_avg:.2f}</div><div class='lbl'>Avg Rating Given</div></div>", unsafe_allow_html=True)
                with ub:
                    st.markdown(f"<div class='stat-card'><div class='num' style='font-size:1.4rem'>{u_cnt:,}</div><div class='lbl'>Movies Rated</div></div>", unsafe_allow_html=True)
                with uc:
                    bias_label = "above average" if u_bias > 0 else "below average"
                    st.markdown(f"<div class='stat-card'><div class='num' style='font-size:1.4rem'>{u_bias:+.2f}</div><div class='lbl'>User Bias ({bias_label})</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                left_col, right_col = st.columns([3, 2])

                with left_col:
                    st.markdown(f"<div class='section-title'>Top {n_recs} picks for User {user_id}</div>", unsafe_allow_html=True)
                    for i, row in enumerate(top_recs.itertuples(), 1):
                        title   = getattr(row, "title", f"Movie {row.movieId}")
                        genres  = getattr(row, "genres", "").replace("|", " · ")
                        rating  = row.predicted_rating
                        pct     = (rating - 0.5) / 4.5
                        bar_w   = int(pct * 100)
                        color   = PURPLE if i <= 3 else BLUE
                        st.markdown(f"""<div class='rec-card'>
                            <div class='rec-rank'>#{i}</div>
                            <div class='rec-title'>{title}</div>
                            <div class='rec-genre'>{genres}</div>
                            <div style='display:flex;align-items:center;gap:10px;margin-top:8px'>
                                <div style='flex:1;background:{GRID};border-radius:4px;height:5px'>
                                    <div style='background:linear-gradient(90deg,{color},#2563eb);
                                        width:{bar_w}%;height:5px;border-radius:4px'></div>
                                </div>
                                <span class='rec-score'>{rating:.2f}/5.0</span>
                            </div>
                        </div>""", unsafe_allow_html=True)

                with right_col:
                    st.markdown("<div class='section-title'>Rating Distribution</div>", unsafe_allow_html=True)
                    fig = go.Figure(go.Bar(
                        x=top_recs["predicted_rating"].round(2),
                        y=[r.title[:25] + "..." if len(r.title) > 25 else r.title
                           for r in top_recs.itertuples()],
                        orientation="h",
                        marker=dict(
                            color=top_recs["predicted_rating"],
                            colorscale=[[0,"#6d28d9"],[1,"#2563eb"]],
                            line=dict(width=0)
                        ),
                        text=[f"{r:.2f}" for r in top_recs["predicted_rating"]],
                        textposition="outside",
                        textfont=dict(color=TEXT, size=11),
                    ))
                    layout = plot_layout("", h=max(300, n_recs * 30))
                    layout["xaxis"]["range"] = [0, 5.5]
                    layout["xaxis"]["title"] = "Predicted Rating"
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)

                    if genre_filter:
                        genre_counts = {}
                        for row in top_recs.itertuples():
                            for g in str(getattr(row, "genres", "")).split("|"):
                                genre_counts[g] = genre_counts.get(g, 0) + 1
                        gc_df = pd.Series(genre_counts).sort_values(ascending=False).head(8)
                        fig2  = go.Figure(go.Bar(
                            x=gc_df.values, y=gc_df.index, orientation="h",
                            marker=dict(color=PURPLE, line=dict(width=0)),
                        ))
                        layout2 = plot_layout("Genre Breakdown", h=260)
                        fig2.update_layout(**layout2)
                        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MOVIE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Movie Explorer":
    st.markdown("# Movie Explorer")
    st.markdown("<div class='section-title'>Search and explore the movie database</div>", unsafe_allow_html=True)

    if "movies" not in assets or "sample_fe" not in assets:
        st.error("movies_app.csv or sample_fe.csv not loaded.")
        st.stop()

    movies_df = assets["movies"]
    sample_fe = assets["sample_fe"]

    movies_stats = sample_fe.groupby("movieId").agg(
        avg_rating=("rating", "mean") if "rating" in sample_fe.columns else ("movie_avg_rating", "mean"),
        rating_count=("movie_rating_count", "mean"),
    ).reset_index() if "movie_avg_rating" in sample_fe.columns else None

    if movies_stats is not None:
        movies_df = movies_df.merge(movies_stats, on="movieId", how="left")

    st.markdown("<div class='filter-bar'>", unsafe_allow_html=True)
    sc1, sc2 = st.columns([3, 2])
    with sc1:
        search_query = st.text_input("Search movies by title", placeholder="e.g. Inception, Toy Story...")
    with sc2:
        all_genres = sorted(set(
            g for genres in movies_df["genres"].dropna()
            for g in genres.split("|") if g != "(no genres listed)"
        ))
        genre_pick = st.multiselect("Filter by genre", all_genres)
    st.markdown("</div>", unsafe_allow_html=True)

    filtered = movies_df.copy()
    if search_query:
        filtered = filtered[filtered["title"].str.contains(search_query, case=False, na=False)]
    if genre_pick:
        mask = filtered["genres"].apply(
            lambda g: any(gf in str(g).split("|") for gf in genre_pick)
        )
        filtered = filtered[mask]

    filtered = filtered.head(50)

    st.markdown(f"<div class='section-title'>Showing {len(filtered)} movies</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 3])

    with col_a:
        for row in filtered.head(20).itertuples():
            title  = getattr(row, "title", f"Movie {row.movieId}")
            genres = getattr(row, "genres", "").replace("|", " · ")
            avg    = getattr(row, "avg_rating", None)
            cnt    = getattr(row, "rating_count", None)
            rating_str = f"{avg:.2f} avg" if avg and not np.isnan(avg) else ""
            count_str  = f"· {int(cnt):,} ratings" if cnt and not np.isnan(cnt) else ""
            st.markdown(f"""<div class='search-result'>
                <div style='font-family:Syne,sans-serif;font-weight:700;color:{TEXT};font-size:0.95rem'>{title}</div>
                <div style='font-size:0.75rem;color:{MUTED};margin-top:3px'>{genres}</div>
                <div style='font-size:0.75rem;color:{PURPLE};margin-top:4px'>{rating_str} {count_str}</div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-title'>Genre Distribution in Results</div>", unsafe_allow_html=True)
        genre_exp = (
            filtered["genres"].str.split("|").explode()
            .value_counts()
            .head(15)
        )
        genre_exp = genre_exp[genre_exp.index != "(no genres listed)"]

        colors_list = [PURPLE if i == 0 else "#2a1a4a" for i in range(len(genre_exp))]
        fig = go.Figure(go.Bar(
            x=genre_exp.values, y=genre_exp.index, orientation="h",
            marker=dict(color=colors_list, line=dict(width=0)),
            text=genre_exp.values, textposition="outside",
            textfont=dict(color=TEXT, size=11),
        ))
        fig.update_layout(**plot_layout("Genres in Search Results", h=420))
        st.plotly_chart(fig, use_container_width=True)

        if "avg_rating" in filtered.columns and filtered["avg_rating"].notna().sum() > 5:
            st.markdown("<div class='section-title'>Rating Distribution</div>", unsafe_allow_html=True)
            fig2 = go.Figure(go.Histogram(
                x=filtered["avg_rating"].dropna(),
                nbinsx=20,
                marker=dict(color=PURPLE, line=dict(width=0)),
            ))
            fig2.update_layout(**plot_layout("Avg Rating Distribution", h=220))
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown("# Model Comparison")
    st.markdown("<div class='section-title'>All 7 models evaluated on 200k validation rows</div>", unsafe_allow_html=True)

    if "results" not in assets:
        st.error("results.pkl not loaded.")
        st.stop()

    df = get_results_df()
    best_rmse = df.loc[df["RMSE"].idxmin(), "Model"]

    c1, c2, c3, c4 = st.columns(4)
    for col, (metric, best_fn, label) in zip([c1,c2,c3,c4], [
        ("RMSE","idxmin","Best RMSE"), ("MAE","idxmin","Best MAE"),
        ("R2","idxmax","Best R2"), ("NDCG@10","idxmax","Best NDCG@10")
    ]):
        idx   = getattr(df[metric], best_fn)()
        val   = df.loc[idx, metric]
        model = df.loc[idx, "Model"]
        with col:
            st.markdown(f"""<div class='stat-card'>
                <div class='num' style='font-size:1.5rem'>{val:.4f}</div>
                <div class='lbl'>{label}</div>
                <div style='font-size:0.72rem;color:{PURPLE};margin-top:6px'>{model}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Bar Charts", "Radar Chart", "Full Table"])

    with tab1:
        m1, m2 = st.columns(2)
        for col, metric in zip([m1, m2], ["RMSE", "MAE"]):
            with col:
                ds = df.sort_values(metric)
                clrs = [GREEN if r.Model == best_rmse else "#2a1a4a" for r in ds.itertuples()]
                fig = go.Figure(go.Bar(
                    x=ds[metric], y=ds["Model"], orientation="h",
                    marker=dict(color=clrs, line=dict(width=0)),
                    text=[f"{v:.4f}" for v in ds[metric]],
                    textposition="outside", textfont=dict(color=TEXT, size=10),
                ))
                fig.update_layout(**plot_layout(metric, h=320))
                st.plotly_chart(fig, use_container_width=True)

        m3, m4 = st.columns(2)
        for col, metric in zip([m3, m4], ["R2", "NDCG@10"]):
            with col:
                ds = df.sort_values(metric)
                clrs = [GREEN if r.Model == best_rmse else "#2a1a4a" for r in ds.itertuples()]
                fig = go.Figure(go.Bar(
                    x=ds[metric], y=ds["Model"], orientation="h",
                    marker=dict(color=clrs, line=dict(width=0)),
                    text=[f"{v:.4f}" for v in ds[metric]],
                    textposition="outside", textfont=dict(color=TEXT, size=10),
                ))
                fig.update_layout(**plot_layout(metric, h=320))
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Normalized performance across all metrics</div>", unsafe_allow_html=True)
        df_norm = df.copy()
        df_norm["RMSE_n"]   = 1 - (df["RMSE"] - df["RMSE"].min()) / (df["RMSE"].max() - df["RMSE"].min() + 1e-9)
        df_norm["MAE_n"]    = 1 - (df["MAE"]  - df["MAE"].min())  / (df["MAE"].max()  - df["MAE"].min()  + 1e-9)
        df_norm["R2_n"]     = (df["R2"]     - df["R2"].min())     / (df["R2"].max()     - df["R2"].min()     + 1e-9)
        df_norm["NDCG_n"]   = (df["NDCG@10"] - df["NDCG@10"].min()) / (df["NDCG@10"].max() - df["NDCG@10"].min() + 1e-9)
        cats = ["RMSE (inv)", "MAE (inv)", "R2", "NDCG@10"]

        palette = [PURPLE, BLUE, GREEN, "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"]
        fig = go.Figure()
        for i, row in df_norm.iterrows():
            vals = [row["RMSE_n"], row["MAE_n"], row["R2_n"], row["NDCG_n"]]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=row["Model"],
                line=dict(color=palette[i % len(palette)], width=2),
                fillcolor="rgba(100,100,200,0.08)",
                opacity=0.85
            ))
        fig.update_layout(
            polar=dict(
                bgcolor=CARD,
                radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID, linecolor=GRID, color=MUTED),
                angularaxis=dict(gridcolor=GRID, linecolor=GRID, color=TEXT)
            ),
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="Inter", color=TEXT),
            legend=dict(bgcolor=CARD, bordercolor=GRID, borderwidth=1),
            height=480, margin=dict(l=60,r=60,t=40,b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-title'>Complete metrics table</div>", unsafe_allow_html=True)
        display = df.set_index("Model").round(4)
        st.dataframe(
            display.style
            .highlight_min(subset=["RMSE","MAE"], color="#1a2a1a")
            .highlight_max(subset=["R2","NDCG@10"], color="#1a2a1a"),
            use_container_width=True
        )
        st.markdown(f"""<div class='insight-box'>
            <h4>Key Insight</h4>
            <p style='color:#a0a0c0;font-size:0.88rem;margin:0'>
            XGBoost achieves the best RMSE of {df['RMSE'].min():.4f}, outperforming all baseline models.
            The gap between XGBoost and Linear Regression ({df.loc[df['Model'].str.contains('Linear'),'RMSE'].values[0] - df['RMSE'].min():.4f} RMSE)
            demonstrates the value of ensemble learning for collaborative filtering tasks.
            </p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    st.markdown("# Feature Importance")
    st.markdown("<div class='section-title'>XGBoost model, feature importance by gain</div>", unsafe_allow_html=True)

    if "model" not in assets or "feature_cols" not in assets:
        st.error("xgb_model.pkl or feature_cols.pkl not loaded.")
        st.stop()

    model        = assets["model"]
    feature_cols = assets["feature_cols"]

    fa, fb = st.columns([3, 1])
    with fa:
        n_top = st.slider("Number of top features", 5, len(feature_cols), 20)
    with fb:
        sort_by = st.selectbox("Sort by", ["Importance", "Feature Group"])

    feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    def get_group(f):
        if "genome_svd" in f: return "Genome SVD"
        if "genre_svd"  in f: return "Genre SVD"
        if "tag_svd"    in f: return "Tag SVD"
        if "user"       in f: return "User Features"
        if "movie"      in f: return "Movie Features"
        return "Other"

    top = feat_imp.head(n_top)
    groups = [get_group(f) for f in top.index]
    group_colors = {
        "Genome SVD":      PURPLE,
        "Genre SVD":       BLUE,
        "Tag SVD":         GREEN,
        "User Features":   "#f59e0b",
        "Movie Features":  "#ef4444",
        "Other":           MUTED,
    }
    bar_colors = [group_colors.get(g, MUTED) for g in groups]

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Bar(
            x=top.values[::-1], y=top.index[::-1], orientation="h",
            marker=dict(color=bar_colors[::-1], line=dict(width=0)),
            text=[f"{v:.4f}" for v in top.values[::-1]],
            textposition="outside", textfont=dict(color=TEXT, size=10),
        ))
        fig.update_layout(**plot_layout("Feature Importance (XGBoost)", h=max(380, n_top * 26)))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("<div class='section-title'>By Feature Group</div>", unsafe_allow_html=True)
        group_imp = {}
        for feat, imp in feat_imp.head(n_top).items():
            g = get_group(feat)
            group_imp[g] = group_imp.get(g, 0) + imp
        g_df = pd.Series(group_imp).sort_values(ascending=False)

        fig2 = go.Figure(go.Pie(
            labels=g_df.index, values=g_df.values,
            hole=0.5,
            marker=dict(colors=[group_colors.get(g, MUTED) for g in g_df.index],
                        line=dict(color=BG, width=2)),
            textfont=dict(color=TEXT, size=11),
        ))
        fig2.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="Inter", color=TEXT),
            legend=dict(bgcolor=CARD, bordercolor=GRID, borderwidth=1),
            height=300, margin=dict(l=16,r=16,t=30,b=16),
            showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='section-title'>Top 3 Most Important</div>", unsafe_allow_html=True)
        for feat, imp in feat_imp.head(3).items():
            g     = get_group(feat)
            color = group_colors.get(g, MUTED)
            pct   = imp / feat_imp.sum() * 100
            st.markdown(f"""<div style='background:{CARD};border:1px solid {GRID};
                border-left:3px solid {color};border-radius:10px;
                padding:12px 16px;margin-bottom:8px'>
                <div style='font-family:Syne,sans-serif;font-weight:700;color:{TEXT};font-size:0.9rem'>{feat}</div>
                <div style='font-size:0.75rem;color:{MUTED};margin-top:2px'>{g}</div>
                <div style='font-size:0.85rem;color:{color};margin-top:4px'>{pct:.1f}% of total importance</div>
            </div>""", unsafe_allow_html=True)
