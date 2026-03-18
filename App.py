import gc
import numpy as np
import pandas as pd
import streamlit as st
import sys
import altair as alt

import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -------------------------------------------
# Streamlit Page Config
# -------------------------------------------
st.set_page_config(page_title="Absence Analysis", layout="wide")

st.title("Absence Analysis")
st.caption(
    "Upload a CSV with columns: 'Industry Number' (PII), "
    "'Absense Occasions', 'Days Absent', and 'Group Shaft Name'. "
    "PII is anonymized immediately."
)

# -------------------------------------------
# Helper Functions for Grouped Average Plot
# -------------------------------------------

def find_categorical(df, threshold=20):
    cat_cols = [
        col for col in df.columns 
        if df[col].dtype == "object" or df[col].nunique() < threshold
    ]
    return cat_cols

@st.cache_data
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize/standardize column names to match canonical expectations.
    Supports slight variations like 'Absence' vs 'Absense'.
    """
    # Build a lookup by normalized name (lower, stripped, single-space)
    def norm(s: str) -> str:
        return " ".join(str(s).strip().lower().split())

    normalized_map = {norm(c): c for c in df.columns}

    # Canonical names we want in the final df
    want = {
        "industry number": "Industry Number",
        "absense occasions": "Absense Occasions",  # as specified in the prompt
        "absence occasions": "Absense Occasions",   # accept common alternative spelling
        "days absent": "Days Absent",
        "bradford score": "Bradford Score",
        "overtime avg (12 months)": "Overtime Avg (12 Months)",

    }

    # Determine which actual columns correspond to the canonical ones
    rename_from = {}
    for k, canonical in want.items():
        if k in normalized_map:
            rename_from[normalized_map[k]] = canonical

    # Apply renaming where possible
    df2 = df.rename(columns=rename_from)
    return df2

def check_required_columns(df: pd.DataFrame):
    required = ["Industry Number", "Absense Occasions", "Days Absent"]
    missing = [c for c in required if c not in df.columns]
    return missing

@st.cache_data
def anonymize_industry_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 'Industry Number' values with sequential integers starting at 1.
    Does not retain any mapping or the original values.
    """
    # Factorize to sequential ids; discard the uniques array afterward
    codes, uniques = pd.factorize(df["Industry Number"], sort=False)
    df["Industry Number"] = codes + 1

    # Explicitly drop references to original unique values and trigger GC
    del uniques
    gc.collect()
    return df

@st.cache_data
def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data
def aggregate_stats(df: pd.DataFrame, group_col) -> pd.DataFrame:
    """
    Group by group_col and compute mean/median/std for the two numeric measures.
    Also include a count per group (number of rows contributed).
    """
    grouped = df.groupby(group_col, dropna=False)

    agg_df = grouped.agg({
        "Absense Occasions": ["mean", "median", "std"],
        "Days Absent": ["mean", "median", "std"]
    })
    # Flatten multiindex columns
    agg_df.columns = [
        f"{col[0]}_{col[1]}" for col in agg_df.columns.to_flat_index()
    ]

    # Add count per group (size)
    agg_df["count"] = grouped.size()

    # Rename to cleaner column names
    rename_map = {
        "Absense Occasions_mean": "avg_absense_occasions",
        "Absense Occasions_median": "median_absense_occasions",
        "Absense Occasions_std": "std_absense_occasions",
        "Days Absent_mean": "avg_days_absent",
        "Days Absent_median": "median_days_absent",
        "Days Absent_std": "std_days_absent",
    }
    agg_df = agg_df.rename(columns=rename_map)

    # Optional: sort by avg_days_absent descending
    agg_df = agg_df.sort_values(by="avg_days_absent", ascending=False)

    return agg_df.reset_index()

@st.cache_data
def auto_thin_points(df: pd.DataFrame,
                     x_col: str,
                     y_col: str,
                     weight_col: str = "count") -> pd.DataFrame:
    """
    Automatically reduce points if there are too many or if they are too close.
    Strategy:
    - Normalize x/y into [0,1] range.
    - Dynamic max points cap based on number of groups.
    - Greedy selection based on 'weight_col' (keep higher weight first).
    - Enforce a minimum Euclidean distance in normalized space.
    """

    n = len(df)
    if n == 0:
        return df

    # Dynamic heuristics for max points and spacing
    if n <= 100:
        max_points = 100
        min_dist = 0.0           # show all, no thinning
    elif n <= 500:
        max_points = 120
        min_dist = 0.02
    elif n <= 2000:
        max_points = 150
        min_dist = 0.03
    else:
        max_points = 200
        min_dist = 0.04

    # Normalize x and y to [0, 1] for distance checks
    x = df[x_col].astype(float)
    y = df[y_col].astype(float)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # avoid divide by zero; if no spread, normalization collapses
    x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else pd.Series(0.5, index=df.index)
    y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else pd.Series(0.5, index=df.index)

    df_norm = df.copy()
    df_norm["_xn"] = x_norm
    df_norm["_yn"] = y_norm

    # Sort by weight descending so we keep the densest / most representative groups first
    if weight_col in df_norm.columns:
        df_norm = df_norm.sort_values(by=weight_col, ascending=False)
    else:
        df_norm = df_norm.sample(frac=1.0, random_state=42)  # fallback random order

    selected_idx = []
    selected_coords = []

    # Greedy selection ensuring spacing
    for idx, row in df_norm.iterrows():
        if len(selected_idx) >= max_points:
            break

        xn, yn = row["_xn"], row["_yn"]

        if min_dist <= 0:
            selected_idx.append(idx)
            selected_coords.append((xn, yn))
            continue

        # Check spacing from all previously selected points
        too_close = False
        for (sx, sy) in selected_coords:
            if (xn - sx) ** 2 + (yn - sy) ** 2 < (min_dist ** 2):
                too_close = True
                break

        if not too_close:
            selected_idx.append(idx)
            selected_coords.append((xn, yn))

    thinned = df_norm.loc[selected_idx].drop(columns=["_xn", "_yn"])
    return thinned

# -------------------------------------------
# File Uploader (CSV only)
# -------------------------------------------
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is None:
    st.info("Please upload a CSV to begin.")
    st.stop()

# -------------------------------------------
# Read and limit to only necessary columns early
# -------------------------------------------
try:
    # Read CSV into a temporary df
    df_full = pd.read_csv(uploaded)

    # Normalize column names to capture minor variations; then keep only needed columns
    df_full = normalize_columns(df_full)
    df_full = coerce_numeric(df_full,["Absense Occasions", "Days Absent", "Bradford Score", "Overtime Avg (12 Months)"])
    missing = check_required_columns(df_full)
    if missing:
        st.error(
            "The uploaded file is missing required columns: "
            + ", ".join([f"'{m}'" for m in missing])
        )
        st.stop()
except Exception as e:
    st.stop()

# Keep only the necessary columns; minimize exposure to any extra data
group_col = st.selectbox(
        "Choose a grouping column:",
        find_categorical(df_full),
    )
df = df_full[["Industry Number", "Absense Occasions", "Days Absent", f"{group_col}"]].copy()

# Explicitly discard the broader df to avoid retaining extra columns/PII
df_full = anonymize_industry_number(df_full)
gc.collect()

# --- Count group sizes ---
group_counts = df[group_col].value_counts().sort_values()

# --- Ask user if small groups should be excluded ---
st.write("### Group Distribution")
st.bar_chart(group_counts)

exclude_small = st.checkbox("Exclude groups with a low number of records?", value=False)

if exclude_small:
    threshold = st.number_input(
        "Minimum number of records required per group:",
        min_value=1,
        max_value=int(group_counts.max()),
        value=5,
        step=1,
    )
    
    # remove groups below threshold
    valid_groups = group_counts[group_counts >= threshold].index
    df = df[df[group_col].isin(valid_groups)]

    st.success(f"Filtered dataset now includes {len(valid_groups)} groups.")
    st.write("Remaining groups:", list(valid_groups))

# -------------------------------------------
# Anonymize PII and enforce numeric types
# -------------------------------------------
df = coerce_numeric(df, ["Absense Occasions", "Days Absent"])
# Optional: show a peek of the anonymized data (no original PII)
with st.expander("Table Imported (with anonymized PII)", expanded=False):
    st.dataframe(df_full)

# -------------------------------------------
# Aggregate statistics
# -------------------------------------------
agg_df = aggregate_stats(df, group_col)

st.subheader("Group-level Statistics")
st.caption(f"Averages, medians, and standard deviations per {group_col}.")
st.dataframe(agg_df, width='stretch')

# -------------------------------------------
# Scatter Plot: Average Absense Occasions vs Average Days Absent
# -------------------------------------------
st.subheader("2D Scatter: Average Absense Occasions vs Average Days Absent")

plot_df = agg_df[[group_col, "avg_absense_occasions", "avg_days_absent", "count"]].dropna()

if len(plot_df) == 0:
    st.info("No points to plot (insufficient or all-null aggregates).")
    st.stop()

# Auto-thin points for readability
plot_thin = auto_thin_points(
    plot_df,
    x_col="avg_absense_occasions",
    y_col="avg_days_absent",
    weight_col="count"
)

# Compute overall averages (for reference lines)
overall_x = plot_df["avg_absense_occasions"].mean()
overall_y = plot_df["avg_days_absent"].mean()

# Base scatter
points = (
    alt.Chart(plot_thin)
      .mark_circle(size=80, opacity=0.85, color="#2E7D32")
      .encode(
          x=alt.X("avg_absense_occasions:Q", title="Average Absense Occasions"),
          y=alt.Y("avg_days_absent:Q", title="Average Days Absent"),
          tooltip=[
              alt.Tooltip(f"{group_col}:N", title="Group"),
              alt.Tooltip("avg_absense_occasions:Q", title="Avg Absense Occasions", format=".2f"),
              alt.Tooltip("avg_days_absent:Q", title="Avg Days Absent", format=".2f"),
              alt.Tooltip("count:Q", title="Records in Group"),
          ]
      )
)

# Vertical line at overall Absense Occasions mean
vline = (
    alt.Chart(pd.DataFrame({'x': [overall_x]}))
      .mark_rule(color='red', strokeDash=[6, 4])
      .encode(x='x:Q')
)

# Horizontal line at overall Days Absent mean
hline = (
    alt.Chart(pd.DataFrame({'y': [overall_y]}))
      .mark_rule(color='red', strokeDash=[6, 4])
      .encode(y='y:Q')
)

# Combine layers
final_chart = (points + vline + hline).properties(height=500).interactive()

st.altair_chart(final_chart, width='stretch')

# -------------------------------------------
# Group Similarity via Clustering
# -------------------------------------------
st.subheader("Statistical Grouping: Merge Close Groups")

# Attempt to import scikit-learn for K-Means and silhouette
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_OK = True
except Exception as _e:
    SKLEARN_OK = False
    st.warning(
        "scikit-learn is not available; K-Means clustering requires scikit-learn. "
        "Please install/repair scikit-learn to enable this feature."
    )

if SKLEARN_OK:
    # -----------------------------
    # Sidebar controls
    # -----------------------------
    st.sidebar.subheader("K-Means Clustering (Auto-k via Silhouette)")
    k_min, k_max = st.sidebar.slider(
        "Range of k (clusters) to try",
        min_value=2, max_value=15, value=(2, 8), step=1,
        help="We will fit K-Means for each k in this range and choose the best silhouette score."
    )
    standardize = st.sidebar.checkbox(
        "Standardize features (recommended)",
        value=True,
        help="Scale features to zero-mean/unit-variance so neither dominates."
    )
    weight_by_count = st.sidebar.checkbox(
        "Weight by number of records in group",
        value=True,
        help="Give more influence to groups with more employee records."
    )
    min_n_per_group = st.sidebar.number_input(
        "Minimum records per group to include",
        min_value=1, max_value=1000, value=5, step=1,
        help="Groups with fewer rows than this are excluded from clustering."
    )
    random_state = st.sidebar.number_input(
        "Random seed (reproducibility)",
        min_value=0, max_value=10_000, value=42, step=1
    )
    n_init = st.sidebar.number_input(
        "n_init (K-Means restarts)",
        min_value=1, max_value=100, value=10, step=1,
        help="Number of times K-Means will run with different centroid seeds."
    )
    show_sil_details = st.sidebar.checkbox(
        "Show silhouette details", value=False,
        help="Display silhouette score by k and a small diagnostic chart."
    )

    # -----------------------------
    # Prepare group-level features
    # -----------------------------
    # Expect agg_df to exist with columns:
    # [group_col, "avg_absense_occasions", "avg_days_absent", "count"]
    features_df = agg_df[[
        group_col, "avg_absense_occasions", "avg_days_absent", "count"
    ]].copy()

    # Exclude small groups from clustering, but keep them in mapping as "Excluded"
    eligible_mask = features_df["count"] >= min_n_per_group
    eligible_df = features_df.loc[eligible_mask].reset_index(drop=True)
    excluded_df = features_df.loc[~eligible_mask].reset_index(drop=True)

    if len(eligible_df) < 2:
        st.info("Not enough eligible groups (need at least 2) to run clustering.")
    else:
        # Features: two dimensions (avg_absense_occasions, avg_days_absent)
        X = eligible_df[["avg_absense_occasions", "avg_days_absent"]].values

        # Optional standardization
        if standardize:
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(X)
        else:
            X_proc = X

        # Cap k range to the number of eligible samples
        max_k_possible = max(2, min(k_max, len(eligible_df)))
        min_k_possible = max(2, min(k_min, max_k_possible))
        if (min_k_possible, max_k_possible) != (k_min, k_max):
            st.warning(
                f"Adjusted k-range to ({min_k_possible}, {max_k_possible}) due to number of eligible groups "
                f"({len(eligible_df)})."
            )

        # Sample weights (by group size) if selected
        sample_w = eligible_df["count"].values if weight_by_count else None

        # -----------------------------
        # Search k with silhouette
        # -----------------------------
        sil_rows = []
        best_k = None
        best_sil = -1.0
        best_km = None
        best_labels = None

        for k_try in range(min_k_possible, max_k_possible + 1):
            # Fit K-Means
            km = KMeans(
                n_clusters=k_try,
                random_state=int(random_state),
                n_init=int(n_init)
            )
            try:
                if sample_w is not None:
                    km.fit(X_proc, sample_weight=sample_w)
                else:
                    km.fit(X_proc)
            except TypeError:
                # Older scikit-learn may not support sample_weight; fallback
                if sample_w is not None:
                    st.warning("Your scikit-learn build does not support sample_weight for K-Means. Fitting without weights.")
                km.fit(X_proc)

            labels_try = km.labels_.astype(int)

            # Compute silhouette on the processed features (silhouette_score has no sample_weight)
            # Requires at least 2 clusters and not all points in a single cluster (guaranteed by KMeans if k <= n_samples)
            try:
                sil = float(silhouette_score(X_proc, labels_try, metric="euclidean"))
            except Exception as _e:
                sil = float("nan")

            sil_rows.append({"k": k_try, "silhouette": sil})

            # Track best (break ties by smaller k)
            if np.isfinite(sil) and (sil > best_sil or (np.isclose(sil, best_sil) and (best_k is None or k_try < best_k))):
                best_sil = sil
                best_k = k_try
                best_km = km
                best_labels = labels_try

        sil_df = pd.DataFrame(sil_rows).sort_values("k").reset_index(drop=True)

        if best_k is None or best_km is None or best_labels is None:
            st.info("Could not determine a valid number of clusters from silhouette scores.")
        else:
            st.success(f"Selected k = {best_k} (silhouette = {best_sil:.3f})")

            # Assign labels for eligible groups
            label_names = {i: f"Cluster {i+1}" for i in range(best_k)}
            eligible_df["Overall Grouping"] = [label_names[i] for i in best_labels]

            # Mark excluded groups
            if len(excluded_df) > 0:
                excluded_df["Overall Grouping"] = "Excluded (Too few records)"

            # Combine back for mapping & plotting
            mapping_df = pd.concat(
                [eligible_df, excluded_df],
                axis=0, ignore_index=True
            )[[group_col, "Overall Grouping", "avg_absense_occasions", "avg_days_absent", "count"]]

            # --- Ensure consistent dtypes for mapping_df ---
            # Coerce text columns to str; fill NaNs so downstream UI is stable.
            mapping_df[group_col] = (
                mapping_df[group_col]
                .astype(str)  # convert mixed types to string
                .str.strip()
                .replace({"nan": ""})  # if NaN became 'nan'
            )

            mapping_df["Overall Grouping"] = (
                mapping_df["Overall Grouping"]
                .astype(str)
                .str.strip()
                .replace({"nan": ""})
            )

            # -----------------------------
            # Output mapping table
            # -----------------------------
            st.markdown("### Overall Grouping Membership (K-Means, Auto-k via Silhouette)")
            st.dataframe(
                mapping_df.sort_values(["Overall Grouping", group_col]).reset_index(drop=True),
                width='stretch'
            )

            # -----------------------------
            # (Optional) Silhouette diagnostics
            # -----------------------------
            if show_sil_details and len(sil_df) > 0:
                with st.expander("Silhouette details", expanded=False):
                    st.dataframe(
                        sil_df.style.format({"silhouette": "{:.3f}"}),
                        width='stretch'
                    )
                    try:
                        sil_chart = (
                            alt.Chart(sil_df)
                              .mark_line(point=True)
                              .encode(
                                  x=alt.X("k:Q", title="k (number of clusters)"),
                                  y=alt.Y("silhouette:Q", title="Silhouette score"),
                                  tooltip=[alt.Tooltip("k:Q"), alt.Tooltip("silhouette:Q", format=".3f")]
                              )
                              .properties(height=260)
                              .interactive()
                        )
                        st.altair_chart(sil_chart, width='stretch')
                    except Exception:
                        pass

            # -----------------------------
            # Scatter by cluster (same look & feel)
            # -----------------------------
            st.markdown("### Scatter by Overall Grouping")
            plot_groups = mapping_df.copy()

            # Auto-thin to keep chart readable (assumes auto_thin_points exists)
            plot_groups_thin = auto_thin_points(
                plot_groups.rename(columns={
                    "avg_absense_occasions": "x",
                    "avg_days_absent": "y"
                }),
                x_col="x", y_col="y", weight_col="count"
            ).rename(columns={"x": "avg_absense_occasions", "y": "avg_days_absent"})

            color_field = alt.Color("Overall Grouping:N", title="Overall Grouping")
            points2 = (
                alt.Chart(plot_groups_thin)
                  .mark_circle(size=90, opacity=0.85)
                  .encode(
                      x=alt.X("avg_absense_occasions:Q", title="Average Absense Occasions"),
                      y=alt.Y("avg_days_absent:Q", title="Average Days Absent"),
                      color=color_field,
                      tooltip=[
                          alt.Tooltip(f"{group_col}:N", title="Group"),
                          alt.Tooltip("Overall Grouping:N"),
                          alt.Tooltip("avg_absense_occasions:Q", title="Avg Absense Occasions", format=".2f"),
                          alt.Tooltip("avg_days_absent:Q", title="Avg Days Absent", format=".2f"),
                          alt.Tooltip("count:Q", title="Records in Group"),
                      ]
                  )
                  .properties(height=520)
                  .interactive()
            )

            # Global mean lines
            overall_x = plot_groups["avg_absense_occasions"].mean()
            overall_y = plot_groups["avg_days_absent"].mean()
            vline2 = alt.Chart(pd.DataFrame({'x': [overall_x]})).mark_rule(color='red', strokeDash=[6,4]).encode(x='x:Q')
            hline2 = alt.Chart(pd.DataFrame({'y': [overall_y]})).mark_rule(color='red', strokeDash=[6,4]).encode(y='y:Q')

            st.altair_chart(points2 + vline2 + hline2, width='stretch')

            # -----------------------------
            # Download mapping
            # -----------------------------
            with st.expander("Download Overall Grouping Membership"):
                st.download_button(
                    label="Download group-to-overall-group mapping (CSV)",
                    data=mapping_df.sort_values(["Overall Grouping", f"{group_col}"]).to_csv(index=False),
                    file_name=f"overall_groupings_kmeans_k{best_k}.csv",
                    mime="text/csv",
                )

# ---------------------------------------------
# Per-Cluster / Per-Group Distribution Viewer
# ---------------------------------------------
st.markdown("## Distribution Explorer: Days Absent & Absense Occasions")

# Guardrails for required data
required_cols_ind = {f"{group_col}", "Days Absent", "Absense Occasions"}
required_cols_map = {f"{group_col}", "Overall Grouping"}

if not required_cols_ind.issubset(set(df.columns)) or not required_cols_map.issubset(set(mapping_df.columns)):
    st.info("Required columns not found to build distribution plots. "
            "Please ensure 'df' has individual-level 'Days Absent' and 'Absense Occasions' "
            f"and 'mapping_df' contains '{group_col}' and 'Overall Grouping'.")
else:
    select_mode = st.radio(
        "View by",
        options=["Cluster", "Single Group"],
        index=0,
        help="Choose a K-Means cluster or a specific group."
    )

    # Build choices
    cluster_choices = sorted([c for c in mapping_df["Overall Grouping"].unique()
                              if c.startswith("Cluster")])
    group_choices = sorted(mapping_df[f"{group_col}"].unique())

    if select_mode == "Cluster":
        if len(cluster_choices) == 0:
            st.warning("No clusters available. Run clustering first.")
        else:
            st.caption("Select a cluster to view its duration distribution.")

            picked_cluster = st.selectbox(
                "Choose a cluster:",
                cluster_choices,
            )

            # Groups inside the chosen cluster
            groups_in_cluster = mapping_df.loc[
                mapping_df["Overall Grouping"] == picked_cluster, f"{group_col}"
            ].unique().tolist()

            st.caption(f"Selected **{picked_cluster}** · {len(groups_in_cluster)} group(s)")
            filt_groups = groups_in_cluster

    else:
        st.caption("Select a group to view its duration distribution.")

        picked_group = st.selectbox(
            "Choose a group:",
            group_choices,
        )
        st.caption(f"Selected **{picked_group}**")
        filt_groups = [picked_group]

    # Histogram options
    bins = st.slider("Number of histogram bins", min_value=10, max_value=80, value=30, step=1)
    opacity = 0.8

    # --- Build individual-level subset from df_full so we can include Bradford & Overtime ---
    # Start with group column + all four potential metrics
    needed_cols = [f"{group_col}", "Days Absent", "Absense Occasions",
                   "Bradford Score", "Overtime Avg (12 Months)"]
    present_cols = [c for c in needed_cols if c in df_full.columns]

    sub = (
        df_full.loc[df_full[f"{group_col}"].isin(filt_groups), present_cols]
               .copy()
    )

    # Coerce present numeric cols
    for c in ["Days Absent", "Absense Occasions", "Bradford Score", "Overtime Avg (12 Months)"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # Drop rows that are completely NaN across the four metrics (keep rows if at least one metric exists)
    metric_cols = [c for c in ["Days Absent", "Absense Occasions", "Bradford Score", "Overtime Avg (12 Months)"] if c in sub.columns]
    if len(metric_cols) == 0:
        st.info("No metric columns available to plot.")
    else:
        sub = sub.dropna(subset=metric_cols, how="all")

    n_rows = len(sub)
    if n_rows == 0:
        st.info("No individual-level rows found for the current selection.")
    else:
        # Utility: histogram builder (unchanged except now reusable)
        def hist_chart(df_in: pd.DataFrame, value_col: str, title: str):
            import numpy as np
            vals = df_in[value_col].dropna().values
            if vals.size == 0 or np.all(vals == vals[0]):
                # Handle degenerate case
                return alt.Chart(pd.DataFrame({value_col: vals if vals.size else [0]})).mark_text(
                    text="Insufficient variation to plot"
                ).properties(height=260)

            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5

            base = alt.Chart(pd.DataFrame({value_col: vals}))
            hist = (
                base.mark_bar(opacity=opacity)
                    .encode(
                        x=alt.X(f"{value_col}:Q",
                                bin=alt.Bin(extent=[vmin, vmax], maxbins=bins),
                                title=title),
                        y=alt.Y("count():Q", title="Count"),
                        tooltip=[
                            alt.Tooltip(f"{value_col}:Q", title=title, format=".2f"),
                            alt.Tooltip("count():Q", title="Count")
                        ]
                    )
            )
            return hist.properties(height=300)

        # --- Build the four charts (only for columns that are present) ---
        charts = []
        titles = {
            "Days Absent": "Days Absent",
            "Absense Occasions": "Absense Occasions",
            "Bradford Score": "Bradford Score",
            "Overtime Avg (12 Months)": "Overtime Avg (12 Months)",
        }

        # Existing two
        if "Days Absent" in sub.columns:
            charts.append(hist_chart(sub, "Days Absent", titles["Days Absent"]))
        if "Absense Occasions" in sub.columns:
            charts.append(hist_chart(sub, "Absense Occasions", titles["Absense Occasions"]))

        # NEW two
        if "Bradford Score" in sub.columns:
            charts.append(hist_chart(sub, "Bradford Score", titles["Bradford Score"]))
        if "Overtime Avg (12 Months)" in sub.columns:
            charts.append(hist_chart(sub, "Overtime Avg (12 Months)", titles["Overtime Avg (12 Months)"]))

        # Layout: render sequentially (Streamlit will stack them)
        st.markdown("### Distributions")
        for ch in charts:
            st.altair_chart(ch, width='stretch')

        # Small numeric summary over whichever metrics are present
        with st.expander("Summary stats for current selection", expanded=False):
            def summarize(col):
                s = sub[col].dropna()
                if s.empty:
                    return pd.DataFrame({"metric": ["count","mean","std","min","25%","50%","75%","max"], col: [0, *[float("nan")]*7]})
                return pd.DataFrame({
                    "metric": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
                    col: [
                        int(s.count()),
                        float(s.mean()),
                        float(s.std(ddof=1)) if s.count() > 1 else float("nan"),
                        float(s.min()),
                        float(s.quantile(0.25)),
                        float(s.quantile(0.50)),
                        float(s.quantile(0.75)),
                        float(s.max()),
                    ]
                })

            # Merge summaries for all present metric columns
            summary = None
            for c in ["Days Absent", "Absense Occasions", "Bradford Score", "Overtime Avg (12 Months)"]:
                if c in sub.columns:
                    summary = summarize(c) if summary is None else summary.merge(summarize(c), on="metric")

            if summary is not None:
                st.dataframe(
                    summary.style.format({
                        "Days Absent": "{:.2f}",
                        "Absense Occasions": "{:.2f}",
                        "Bradford Score": "{:.2f}",
                        "Overtime Avg (12 Months)": "{:.2f}",
                    }),
                    width='stretch'
                )

# -------------------------------------------
# Correlation Heat Map (Absense, Days, Bradford, Overtime)
# -------------------------------------------
st.subheader("Correlation Heat Map")

# Determine which of the extra columns exist in the uploaded data
corr_candidates = [
    "Absense Occasions",
    "Days Absent",
    "Bradford Score",
    "Overtime Avg (12 Months)",
]
available_cols = [c for c in corr_candidates if c in df_full.columns]

if len(available_cols) < 3:
    st.info(
        "To show a meaningful correlation matrix, we need at least 3 of the following columns: "
        "'Absense Occasions', 'Days Absent', 'Bradford Score', 'Overtime Avg (12 Months)'. "
        f"Currently found: {', '.join(available_cols) if available_cols else 'none'}"
    )
else:
    # Choose correlation method
    corr_method = st.selectbox(
        "Correlation method",
        options=["pearson", "spearman", "kendall"],
        index=0,
        help=(
            "Pearson: linear correlation (default). "
            "Spearman: rank correlation (robust to non-linear monotonic relations). "
            "Kendall: rank correlation (more conservative on small samples)."
        ),
    )

    # Work on a safe copy; coerce to numeric
    corr_df = df_full[available_cols].copy()
    for c in available_cols:
        corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")

    # Drop rows where all are NaN (keeps rows where at least one value exists)
    corr_df = corr_df.dropna(how="all")

    # If too few valid rows, abort
    min_rows = 3  # correlation needs at least a few non-null rows
    if len(corr_df.dropna(how="any")) < min_rows and corr_method == "pearson":
        st.warning(
            "Not enough fully non-null rows to compute Pearson correlation reliably. "
            "Try 'Spearman' or 'Kendall', or ensure data completeness."
        )

    # Compute correlation matrix on columns pairwise (pandas handles pairwise NaNs)
    try:
        corr_mat = corr_df.corr(method=corr_method)
    except Exception as e:
        st.error(f"Could not compute correlation: {e}")
        corr_mat = None

    if corr_mat is None or corr_mat.empty:
        st.info("No correlation matrix could be computed from the available data.")
    else:
        # Prepare for Altair heat map (melt to long form)
        corr_long = (
            corr_mat.reset_index()
                    .melt(id_vars="index", var_name="Variable 2", value_name="Correlation")
                    .rename(columns={"index": "Variable 1"})
        )

        # Force categorical order to keep a square grid in a consistent order
        order = available_cols  # use the same order as columns were found
        corr_long["Variable 1"] = pd.Categorical(corr_long["Variable 1"], categories=order, ordered=True)
        corr_long["Variable 2"] = pd.Categorical(corr_long["Variable 2"], categories=order, ordered=True)

        # Build Altair heat map with labels
        base = alt.Chart(corr_long)

        heat = (
            base.mark_rect()
                .encode(
                    x=alt.X("Variable 1:O", title="", sort=order),
                    y=alt.Y("Variable 2:O", title="", sort=order),
                    color=alt.Color(
                        "Correlation:Q",
                        scale=alt.Scale(domain=[-1, 0, 1], range=["#b2182b", "#f7f7f7", "#2166ac"]),
                        legend=alt.Legend(title="Correlation")
                    ),
                    tooltip=[
                        alt.Tooltip("Variable 1:N"),
                        alt.Tooltip("Variable 2:N"),
                        alt.Tooltip("Correlation:Q", format=".2f")
                    ]
                )
                .properties(height=360)
        )

        # Text annotations on top of the heat cells
        text = (
            base.mark_text(baseline='middle')
                .encode(
                    x=alt.X("Variable 1:O", sort=order),
                    y=alt.Y("Variable 2:O", sort=order),
                    text=alt.Text("Correlation:Q", format=".2f"),
                    color=alt.condition(
                        alt.datum.Correlation >= 0.5,
                        alt.value("white"),  # high positive -> white text
                        alt.value("black")   # else black text
                    )
                )
        )

        st.caption(f"Correlation method: **{corr_method.capitalize()}**")
        st.altair_chart((heat + text), width='stretch')

# -------------------------------------------
# Segmented Correlation Heat Maps (by Cluster or by Group)
# -------------------------------------------
st.markdown("### Segmented Correlation Heat Maps")

# Helper to build a correlation heat map for a subset DataFrame
def _corr_heat_for_subset(sub_df: pd.DataFrame, title: str):
    # Ensure we only use the numeric candidates you already identified
    if len(available_cols) < 3:
        st.info(f"'{title}': need at least 3 numeric columns to compute correlation.")
        return

    sub = sub_df[available_cols].copy()
    for c in available_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # Drop rows that are entirely NaN across the columns, as done in the overall heat map
    sub = sub.dropna(how="all")

    # If too few valid rows, warn but still try (Spearman/Kendall can work with partial NaNs pairwise)
    min_rows = 3
    if len(sub.dropna(how="any")) < min_rows and corr_method == "pearson":
        st.warning(
            f"'{title}': Not enough fully non-null rows to compute Pearson reliably. "
            "Spearman or Kendall may be more robust."
        )

    try:
        cm = sub.corr(method=corr_method)
    except Exception as e:
        st.warning(f"'{title}': could not compute correlation ({e}).")
        return

    if cm is None or cm.empty:
        st.info(f"'{title}': correlation matrix is empty.")
        return

    # Melt to long form for Altair
    corr_long = (
        cm.reset_index()
          .melt(id_vars="index", var_name="Variable 2", value_name="Correlation")
          .rename(columns={"index": "Variable 1"})
    )

    # Keep a stable order across charts
    order = available_cols
    corr_long["Variable 1"] = pd.Categorical(corr_long["Variable 1"], categories=order, ordered=True)
    corr_long["Variable 2"] = pd.Categorical(corr_long["Variable 2"], categories=order, ordered=True)

    base = alt.Chart(corr_long)

    heat = (
        base.mark_rect()
            .encode(
                x=alt.X("Variable 1:O", title="", sort=order),
                y=alt.Y("Variable 2:O", title="", sort=order),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(domain=[-1, 0, 1], range=["#b2182b", "#f7f7f7", "#2166ac"]),
                    legend=alt.Legend(title="Correlation")
                ),
                tooltip=[
                    alt.Tooltip("Variable 1:N"),
                    alt.Tooltip("Variable 2:N"),
                    alt.Tooltip("Correlation:Q", format=".2f")
                ]
            )
            .properties(height=320)
    )

    text = (
        base.mark_text(baseline='middle')
            .encode(
                x=alt.X("Variable 1:O", sort=order),
                y=alt.Y("Variable 2:O", sort=order),
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.condition(
                    alt.datum.Correlation >= 0.5,
                    alt.value("white"),
                    alt.value("black")
                )
            )
    )

    st.markdown(f"**{title}**")
    st.altair_chart(heat + text, width='stretch')

# User chooses segmentation mode
seg_mode = st.radio(
    "Show correlations by:",
    options=["Cluster", "Group"],
    index=0,
    horizontal=True,
    help="Pick 'Cluster' to see heat maps per K-Means cluster; or 'Group' to see them per group."
)

# ----------------------
# Segment by Cluster
# ----------------------
if seg_mode == "Cluster":
    # Require clustering output
    if "mapping_df" not in locals() or mapping_df is None or "Overall Grouping" not in mapping_df.columns:
        st.info("No clustering results available. Run the K-Means section above to assign clusters first.")
    else:
        # Which clusters to show
        cluster_list = (
            mapping_df["Overall Grouping"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        if len(cluster_list) == 0:
            st.info("No clusters found to display.")
        else:
            chosen_clusters = st.multiselect(
                "Clusters to display:",
                options=cluster_list,
                default=cluster_list[: min(3, len(cluster_list))],
                help="Select one or more clusters. Use fewer selections for faster rendering."
            )

            if len(chosen_clusters) == 0:
                st.info("Pick at least one cluster.")
            else:
                # Render each cluster in its own tab
                tabs = st.tabs(chosen_clusters)
                for tab, clus in zip(tabs, chosen_clusters):
                    with tab:
                        groups_in_cluster = (
                            mapping_df.loc[mapping_df["Overall Grouping"] == clus, group_col]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                        seg_df = df_full[df_full[group_col].astype(str).isin(groups_in_cluster)].copy()
                        st.caption(f"{clus} · {len(groups_in_cluster)} group(s) · {len(seg_df)} row(s)")
                        _corr_heat_for_subset(seg_df, title=f"{clus}")

# ----------------------
# Segment by Group
# ----------------------
else:
    # All groups from the current dataset
    group_list = (
        df_full[group_col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    if len(group_list) == 0:
        st.info("No groups found to display.")
    else:
        # Let user pick multiple groups (tabs keep it readable)
        default_selection = group_list[: min(5, len(group_list))]
        chosen_groups = st.multiselect(
            "Groups to display:",
            options=group_list,
            default=default_selection,
            help="Select one or more groups. Use fewer for faster rendering."
        )

        if len(chosen_groups) == 0:
            st.info("Pick at least one group.")
        else:
            tabs = st.tabs([str(g) for g in chosen_groups])
            for tab, g in zip(tabs, chosen_groups):
                with tab:
                    seg_df = df_full[df_full[group_col].astype(str) == str(g)].copy()
                    st.caption(f"Group: {g} · {len(seg_df)} row(s)")
                    _corr_heat_for_subset(seg_df, title=f"Group: {g}")

import streamlit as st
import pandas as pd
import io
import plotly.express as px
from datetime import timedelta

st.set_page_config(page_title="Occasional Leave Timeseries", layout="wide")

st.title("Occasional Leave — Daily Time Series with Weekend Highlighting")

st.markdown(
    """
This app uploads a CSV, filters **Attendance = "Leave Occasional"**, and plots the daily count.
Weekend days (Saturday/Sunday) are shown with **light vertical shading**.
"""
)

# --- Sidebar controls ---
with st.sidebar:
    st.header("Options")
    dayfirst = st.checkbox("Interpret dates as day-first (e.g., 31/12/2025)", value=True)
    case_insensitive = st.checkbox("Case-insensitive match for Attendance", value=True)
    try_contains = st.checkbox('Fallback: match Attendance that *contains* "leave" & "occasional"', value=True)
    weekend_color = st.color_picker("Weekend highlight color", value="#D3D3D3")
    weekend_opacity = st.slider("Weekend highlight opacity", 0.05, 0.6, 0.2, 0.05)

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if not uploaded:
    st.info("Please upload the CSV to proceed.")
    st.stop()

# --- Load CSV robustly ---
@st.cache_data(show_spinner=False)
def load_csv(file_like) -> pd.DataFrame:
    # Try utf-8-sig first then a common fallback
    for enc in ("utf-8-sig", "ISO-8859-1"):
        file_like.seek(0)
        try:
            df = pd.read_csv(file_like, encoding=enc)
            return df
        except Exception:
            continue
    # Last resort: default read (may still fail but will raise)
    file_like.seek(0)
    return pd.read_csv(file_like)

df = load_csv(uploaded)

def anonymize_industry_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 'Industry Number' values with sequential integers starting at 1.
    Does not retain any mapping or the original values.
    """
    # Factorize to sequential ids; discard the uniques array afterward
    codes, uniques = pd.factorize(df["Industry Number"], sort=False)
    df["Industry Number"] = codes + 1

    # Explicitly drop references to original unique values and trigger GC
    del uniques
    gc.collect()
    return df

df = anonymize_industry_number(df)

# Normalize headers (trim surrounding spaces)
df.columns = df.columns.astype(str).str.strip()

# Rename duplicate columns by appending integers: Name1, Name2, ...
def rename_duplicates_with_numbers(cols):
    """
    Return a new list of column names where duplicates are renamed
    with 1-based integer suffixes in the order they appear.
      e.g., ['Attendance', 'Attendance', 'Shift ', 'Shift']
      --> ['Attendance1', 'Attendance2', 'Shift1', 'Shift2']
    """
    counts = {}
    new_cols = []
    for c in cols:
        base = c.strip()
        # Start counting from 1 for the first occurrence
        counts[base] = counts.get(base, 0) + 1
        new_name = f"{base}{counts[base]}"
        new_cols.append(new_name)
    return new_cols

df.columns = rename_duplicates_with_numbers(df.columns.tolist())

st.subheader("Preview")
#st.dataframe(df.head(10), width='stretch')

# --- Identify date column candidates ---
# Preference order: 'Formula Date' then 'Date', then any column with 'date' in the name,
# else any column that can parse to datetime with reasonable success.
lower_cols = [c.lower() for c in df.columns]
date_candidates = []

if "formula date" in lower_cols:
    date_candidates.append(df.columns[lower_cols.index("formula date")])
if "date" in lower_cols:
    date_candidates.append(df.columns[lower_cols.index("date")])

# Add any other columns containing 'date' not already included
for c in df.columns:
    if "date" in c.lower() and c not in date_candidates:
        date_candidates.append(c)

# If still empty, try all columns: keep those that coerce to datetime without too many NaT
if not date_candidates:
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)
            # If at least 50% parse successfully, consider it a candidate
            if parsed.notna().mean() >= 0.5:
                date_candidates.append(c)
        except Exception:
            pass

if not date_candidates:
    st.error(
        "No suitable date column found. Please ensure your file has a 'Date' or 'Formula Date' column "
        "or another date-like column."
    )
    st.stop()

# --- Attendance column(s) ---
att_candidates = [c for c in df.columns if c.strip().casefold() == "attendance".casefold()]
if not att_candidates:
    # Try softer search if duplicated/renamed
    att_candidates = [c for c in df.columns if "attendance" in c.strip().casefold()]

if not att_candidates:
    st.error(
        "No 'Attendance' column found. Please ensure your file contains an 'Attendance' column."
    )
    st.stop()

# Let user choose explicitly if multiple
col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox("Select the Date column", options=date_candidates, index=0)
with col2:
    att_col = st.selectbox("Select the Attendance column", options=att_candidates, index=0)

# --- Parse the selected date column ---
date_series = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)
if date_series.notna().sum() == 0:
    st.error(
        f"Could not parse any dates from '{date_col}'. "
        "Try toggling the 'day-first' option or choose a different date column."
    )
    st.stop()

df["_date"] = date_series.dt.floor("D")
df = df[df["_date"].notna()].copy()

# --- Build Attendance filter for "Leave Occasional" ---
att_ser = df[att_col].astype(str).str.strip()

if case_insensitive:
    base_mask = att_ser.str.casefold() == "leave occasional".casefold()
else:
    base_mask = att_ser == "Leave Occasional"

# Optional fallback to catch common variants like "Occasional Leave" or extra spaces
if try_contains and base_mask.sum() == 0:
    # contains both 'leave' and 'occasional' in any order, case-insensitive
    ci = att_ser.str.casefold()
    base_mask = ci.str.contains("leave") & ci.str.contains("occasional")

occ = df[base_mask].copy()

if occ.empty:
    st.warning(
        "No rows matched 'Leave Occasional' with the current settings. "
        "Try enabling the fallback 'contains' option or check which Attendance column is correct."
    )

# --- Aggregate to daily counts (including zeros for missing days) ---
if occ.empty:
    # Even if empty, build an empty index range from full df to render axes nicely
    date_min = df["_date"].min()
    date_max = df["_date"].max()
    if pd.isna(date_min) or pd.isna(date_max):
        st.stop()
    all_days = pd.date_range(date_min, date_max, freq="D")
    daily = pd.Series(0, index=all_days, name="Occasional Leaves")
else:
    grouped = occ.groupby("_date").size().rename("Occasional Leaves")
    date_min = df["_date"].min()
    date_max = df["_date"].max()
    all_days = pd.date_range(date_min, date_max, freq="D")
    daily = grouped.reindex(all_days, fill_value=0)

# --- Plotly time series ---
fig = px.line(
    x=daily.index,
    y=daily.values,
    markers=True,
    title="Occasional Leave — Daily Count",
    labels={"x": "Date", "y": "Occasional Leaves"},
)
fig.update_traces(line=dict(width=2), marker=dict(size=6))
fig.update_layout(
    hovermode="x unified",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Occasional Leaves", rangemode="tozero"),
    margin=dict(l=40, r=40, t=60, b=40),
)

# --- Weekend highlighting (Saturday=5, Sunday=6) ---
weekend_days = daily.index[daily.index.weekday >= 5]
for d in weekend_days:
    # shade each weekend day from midnight to midnight next day
    fig.add_vrect(
        x0=d,
        x1=d + pd.Timedelta(days=1),
        fillcolor=weekend_color,
        opacity=weekend_opacity,
        layer="below",
        line_width=0,
    )

st.plotly_chart(fig, width='stretch')

# --- Summary metrics ---
total_days = int(daily.shape[0])
total_occ = int(daily.sum())
with st.container():
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Days in Range", f"{total_days:,}")
    c2.metric("Total Occasional Leaves", f"{total_occ:,}")
    c3.metric("Max in a Day", f"{int(daily.max()):,}")

st.caption(
    f"Using date column **{date_col}** and attendance column **{att_col}**. "
    f"Date parsing is {'day-first' if dayfirst else 'month-first'}."
)
