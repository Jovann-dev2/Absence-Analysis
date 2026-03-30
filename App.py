from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Absence Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Constants
# =========================================================
APP_TITLE = "Absence Analysis"

COL_EMPLOYEE_ID = "Industry Number"
COL_ABSENCE_OCCASIONS = "Absense Occasions"  # kept to match input expectation
COL_DAYS_ABSENT = "Days Absent"
COL_GROUPING = "Group Shaft Name"
COL_REPORTING_DISCIPLINE = "Reporting Discipline"
COL_BRADFORD = "Bradford Score"
COL_OVERTIME = "Overtime Avg (12 Months)"
COL_CLUSTER = "Overall Grouping"
COL_INDIVIDUAL_CLUSTER = "Individual Cluster"

REQUIRED_COLUMNS = [
    COL_EMPLOYEE_ID,
    COL_ABSENCE_OCCASIONS,
    COL_DAYS_ABSENT,
]

OPTIONAL_NUMERIC_COLUMNS = [
    COL_BRADFORD,
    COL_OVERTIME,
]

NUMERIC_COLUMNS = [
    COL_ABSENCE_OCCASIONS,
    COL_DAYS_ABSENT,
    *OPTIONAL_NUMERIC_COLUMNS,
]

COLUMN_ALIASES = {
    "industry number": COL_EMPLOYEE_ID,
    "absense occasions": COL_ABSENCE_OCCASIONS,
    "absence occasions": COL_ABSENCE_OCCASIONS,
    "days absent": COL_DAYS_ABSENT,
    "group shaft name": COL_GROUPING,
    "reporting discipline": COL_REPORTING_DISCIPLINE,
    "bradford score": COL_BRADFORD,
    "overtime avg (12 months)": COL_OVERTIME,
}

EXCLUDED_GROUP_COLUMNS = {
    COL_EMPLOYEE_ID,
    COL_ABSENCE_OCCASIONS,
    COL_DAYS_ABSENT,
    COL_BRADFORD,
    COL_OVERTIME,
    COL_CLUSTER,
    COL_INDIVIDUAL_CLUSTER,
}


# =========================================================
# Data classes
# =========================================================
@dataclass(frozen=True)
class ClusterConfig:
    k_min: int
    k_max: int
    standardize: bool
    weight_by_count: bool
    min_records_per_group: int
    random_state: int
    n_init: int


@dataclass(frozen=True)
class IndividualClusterConfig:
    k_min: int
    k_max: int
    standardize: bool
    random_state: int
    n_init: int


# =========================================================
# Utility functions
# =========================================================
def normalize_header_name(name: str) -> str:
    """Normalize raw column header text for alias matching."""
    return " ".join(str(name).strip().lower().split())


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known columns to canonical names while leaving others unchanged."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        normalized = normalize_header_name(col)
        if normalized in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[normalized]
    return df.rename(columns=rename_map)


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    """Return missing required columns."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def anonymize_employee_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace employee identifier values with sequential integers starting at 1.
    The original mapping is not retained.
    """
    result = df.copy()
    codes = pd.factorize(result[COL_EMPLOYEE_ID], sort=False)[0]
    result[COL_EMPLOYEE_ID] = codes + 1
    return result


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert listed columns to numeric if they exist."""
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def sanitize_text_column(series: pd.Series) -> pd.Series:
    """Convert values to stripped strings while preserving missing values."""
    return series.astype("string").fillna(pd.NA).str.strip()


def get_grouping_candidates(
    df: pd.DataFrame,
    min_unique: int = 2,
    max_unique: int = 450,
) -> list[str]:
    """
    Return sensible grouping columns:
    - not in excluded set
    - between min_unique and max_unique distinct values
    - groupby-safe
    """
    valid_columns: list[str] = []

    for col in df.columns:
        if col in EXCLUDED_GROUP_COLUMNS:
            continue

        try:
            nunique = df[col].nunique(dropna=True)
        except Exception:
            continue

        if nunique < min_unique or nunique > max_unique:
            continue

        try:
            _ = df.groupby(col, dropna=False).size()
        except Exception:
            continue

        valid_columns.append(col)

    return valid_columns


def filter_reporting_discipline(df: pd.DataFrame, enabled: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows where Reporting Discipline contains 'unfit' or 'maternity'.
    Returns:
        filtered_df,
        excluded_value_counts_df
    """
    if not enabled or COL_REPORTING_DISCIPLINE not in df.columns:
        empty = pd.DataFrame(columns=[COL_REPORTING_DISCIPLINE, "rows_excluded"])
        return df.copy(), empty

    result = df.copy()
    discipline = sanitize_text_column(result[COL_REPORTING_DISCIPLINE])
    mask_exclude = discipline.str.contains(r"unfit|maternity", case=False, na=False)

    excluded_counts = (
        discipline[mask_exclude]
        .dropna()
        .replace("", pd.NA)
        .dropna()
        .value_counts(dropna=False)
        .rename_axis(COL_REPORTING_DISCIPLINE)
        .reset_index(name="rows_excluded")
    )

    filtered = result.loc[~mask_exclude].copy()
    return filtered, excluded_counts


def apply_min_group_size_filter(
    df: pd.DataFrame,
    group_col: str,
    enabled: bool,
    threshold: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Optionally exclude groups below a minimum record count."""
    group_counts = df[group_col].value_counts(dropna=False)

    if not enabled:
        return df.copy(), group_counts

    valid_groups = group_counts[group_counts >= threshold].index
    filtered = df[df[group_col].isin(valid_groups)].copy()
    filtered_counts = filtered[group_col].value_counts(dropna=False)
    return filtered, filtered_counts


def compute_group_statistics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute grouped descriptive statistics."""
    grouped = df.groupby(group_col, dropna=False)

    agg = grouped.agg(
        avg_absense_occasions=(COL_ABSENCE_OCCASIONS, "mean"),
        median_absense_occasions=(COL_ABSENCE_OCCASIONS, "median"),
        std_absense_occasions=(COL_ABSENCE_OCCASIONS, "std"),
        avg_days_absent=(COL_DAYS_ABSENT, "mean"),
        median_days_absent=(COL_DAYS_ABSENT, "median"),
        std_days_absent=(COL_DAYS_ABSENT, "std"),
        count=(COL_EMPLOYEE_ID, "size"),
    ).reset_index()

    return agg.sort_values(by="avg_days_absent", ascending=False, na_position="last").reset_index(drop=True)


def thin_points(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    weight_col: str = "count",
) -> pd.DataFrame:
    """
    Greedy point thinning to keep scatter plots readable.
    Uses normalized distance in 2D space and prioritizes higher-weight points.
    """
    if df.empty:
        return df.copy()

    n = len(df)
    if n <= 100:
        max_points, min_dist = 100, 0.0
    elif n <= 500:
        max_points, min_dist = 120, 0.02
    elif n <= 2000:
        max_points, min_dist = 150, 0.03
    else:
        max_points, min_dist = 200, 0.04

    result = df.copy()

    x = pd.to_numeric(result[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(result[y_col], errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(x) & np.isfinite(y)
    result = result.loc[valid_mask].copy()
    x = x[valid_mask]
    y = y[valid_mask]

    if len(result) == 0:
        return result

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    x_norm = np.full_like(x, 0.5, dtype=float) if x_range == 0 else (x - x.min()) / x_range
    y_norm = np.full_like(y, 0.5, dtype=float) if y_range == 0 else (y - y.min()) / y_range

    if weight_col in result.columns:
        weights = pd.to_numeric(result[weight_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        order = np.argsort(-weights)
    else:
        rng = np.random.default_rng(42)
        order = rng.permutation(len(result))

    selected_positions: list[int] = []
    selected_coords: list[tuple[float, float]] = []

    for idx in order:
        if len(selected_positions) >= max_points:
            break

        xn = x_norm[idx]
        yn = y_norm[idx]

        if min_dist <= 0:
            selected_positions.append(idx)
            selected_coords.append((xn, yn))
            continue

        too_close = any((xn - sx) ** 2 + (yn - sy) ** 2 < min_dist ** 2 for sx, sy in selected_coords)
        if not too_close:
            selected_positions.append(idx)
            selected_coords.append((xn, yn))

    return result.iloc[selected_positions].copy()


def build_scatter_chart(
    df: pd.DataFrame,
    group_col: str,
    color_field: str | None = None,
    title: str | None = None,
    show_median_lines: bool = True,
) -> alt.Chart:
    """
    Create a scatter plot of average absence occasions vs average days absent.
    Optionally adds median reference lines.
    """
    cols = [group_col, "avg_absense_occasions", "avg_days_absent", "count"]
    if color_field and color_field not in cols:
        cols.append(color_field)

    plot_df = df[cols].dropna(subset=["avg_absense_occasions", "avg_days_absent"]).copy()
    plot_df = thin_points(plot_df, "avg_absense_occasions", "avg_days_absent", "count")

    if plot_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point()

    tooltips = [
        alt.Tooltip(f"{group_col}:N", title="Group"),
        alt.Tooltip("avg_absense_occasions:Q", title="Avg Absense Occasions", format=".2f"),
        alt.Tooltip("avg_days_absent:Q", title="Avg Days Absent", format=".2f"),
        alt.Tooltip("count:Q", title="Records in Group"),
    ]

    if color_field:
        tooltips.append(alt.Tooltip(f"{color_field}:N", title="Cluster"))

    encodings = {
        "x": alt.X("avg_absense_occasions:Q", title="Average Absense Occasions"),
        "y": alt.Y("avg_days_absent:Q", title="Average Days Absent"),
        "tooltip": tooltips,
    }

    if color_field:
        encodings["color"] = alt.Color(f"{color_field}:N", title=color_field)

    points = alt.Chart(plot_df).mark_circle(size=90, opacity=0.85).encode(**encodings)

    layers = [points]

    if show_median_lines:
        median_x = float(df["avg_absense_occasions"].median(skipna=True))
        median_y = float(df["avg_days_absent"].median(skipna=True))

        vline_df = pd.DataFrame({"x": [median_x], "label": ["Median Absense Occasions"]})
        hline_df = pd.DataFrame({"y": [median_y], "label": ["Median Days Absent"]})

        vline = (
            alt.Chart(vline_df)
            .mark_rule(color="red", strokeDash=[6, 4], size=2)
            .encode(
                x="x:Q",
                tooltip=[
                    alt.Tooltip("label:N", title="Reference"),
                    alt.Tooltip("x:Q", title="Median Absense Occasions", format=".2f"),
                ],
            )
        )

        hline = (
            alt.Chart(hline_df)
            .mark_rule(color="red", strokeDash=[6, 4], size=2)
            .encode(
                y="y:Q",
                tooltip=[
                    alt.Tooltip("label:N", title="Reference"),
                    alt.Tooltip("y:Q", title="Median Days Absent", format=".2f"),
                ],
            )
        )

        layers.extend([vline, hline])

    chart = alt.layer(*layers).properties(height=500)

    if title:
        chart = chart.properties(title=title)

    return chart.interactive()


def build_histogram(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    bins: int = 30,
    integer_bins: bool = False,
) -> alt.Chart:
    """Create a histogram for a numeric column."""
    values = pd.to_numeric(df[value_col], errors="coerce").dropna().to_numpy()

    if len(values) == 0:
        return alt.Chart(pd.DataFrame({"message": ["No data available"]})).mark_text(size=14).encode(text="message:N")

    if np.all(values == values[0]):
        return alt.Chart(pd.DataFrame({"message": ["Insufficient variation to plot"]})).mark_text(size=14).encode(
            text="message:N"
        )

    hist_df = pd.DataFrame({value_col: values})

    if integer_bins:
        bin_def = alt.Bin(step=1)
        tooltip_fields = [
            alt.Tooltip(f"{value_col}_bin_start:Q", title=title, format=".0f"),
            alt.Tooltip("count():Q", title="Count"),
        ]
        axis_def = alt.Axis(tickMinStep=1, format=".0f")
    else:
        bin_def = alt.Bin(maxbins=bins)
        tooltip_fields = [
            alt.Tooltip(f"{value_col}_bin_start:Q", title="Bin start", format=".2f"),
            alt.Tooltip(f"{value_col}_bin_end:Q", title="Bin end", format=".2f"),
            alt.Tooltip("count():Q", title="Count"),
        ]
        axis_def = alt.Axis()

    return (
        alt.Chart(hist_df)
        .transform_bin(
            as_=[f"{value_col}_bin_start", f"{value_col}_bin_end"],
            field=value_col,
            bin=bin_def,
        )
        .mark_bar(opacity=0.8, binSpacing=1)   # <- tiny gap between bars
        .encode(
            x=alt.X(
                f"{value_col}_bin_start:Q",
                bin="binned",                  # <- important
                title=title,
                axis=axis_def,
            ),
            x2=f"{value_col}_bin_end:Q",
            y=alt.Y("count():Q", title="Count"),
            tooltip=tooltip_fields,
        )
        .properties(height=280)
    )


def build_group_distribution_chart(group_counts: pd.Series, group_col: str) -> alt.Chart:
    """Build a descending group count bar chart (not alphabetical)."""
    if group_counts.empty:
        return alt.Chart(pd.DataFrame({"message": ["No data available"]})).mark_text(size=14).encode(text="message:N")

    plot_df = (
        group_counts.rename_axis(group_col)
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Record Count"),
            y=alt.Y(f"{group_col}:N", title=group_col, sort="-x"),
            tooltip=[
                alt.Tooltip(f"{group_col}:N", title="Group"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=max(320, min(900, len(plot_df) * 22)))
    )


def build_correlation_heatmap(df: pd.DataFrame, columns: list[str], method: str) -> alt.Chart | None:
    """Build an annotated Altair correlation heatmap."""
    if len(columns) < 3:
        return None

    corr_df = df[columns].copy()
    for col in columns:
        corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")
    corr_df = corr_df.dropna(how="all")

    if corr_df.empty:
        return None

    corr = corr_df.corr(method=method)
    if corr.empty:
        return None

    corr_long = (
        corr.reset_index()
        .melt(id_vars="index", var_name="Variable 2", value_name="Correlation")
        .rename(columns={"index": "Variable 1"})
    )

    corr_long["Variable 1"] = pd.Categorical(corr_long["Variable 1"], categories=columns, ordered=True)
    corr_long["Variable 2"] = pd.Categorical(corr_long["Variable 2"], categories=columns, ordered=True)

    base = alt.Chart(corr_long)

    heat = (
        base.mark_rect()
        .encode(
            x=alt.X("Variable 1:O", title="", sort=columns),
            y=alt.Y("Variable 2:O", title="", sort=columns),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(domain=[-1, 0, 1], range=["#b2182b", "#f7f7f7", "#2166ac"]),
                legend=alt.Legend(title="Correlation"),
            ),
            tooltip=[
                alt.Tooltip("Variable 1:N"),
                alt.Tooltip("Variable 2:N"),
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        .properties(height=360)
    )

    labels = (
        base.mark_text(baseline="middle")
        .encode(
            x=alt.X("Variable 1:O", sort=columns),
            y=alt.Y("Variable 2:O", sort=columns),
            text=alt.Text("Correlation:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Correlation >= 0.5,
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    return heat + labels


def summarize_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return summary statistics for selected numeric columns."""
    summaries: list[pd.DataFrame] = []

    for col in columns:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            summary = pd.DataFrame(
                {
                    "metric": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
                    col: [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                }
            )
        else:
            summary = pd.DataFrame(
                {
                    "metric": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
                    col: [
                        int(series.count()),
                        float(series.mean()),
                        float(series.std(ddof=1)) if series.count() > 1 else np.nan,
                        float(series.min()),
                        float(series.quantile(0.25)),
                        float(series.quantile(0.50)),
                        float(series.quantile(0.75)),
                        float(series.max()),
                    ],
                }
            )
        summaries.append(summary)

    if not summaries:
        return pd.DataFrame()

    result = summaries[0]
    for summary in summaries[1:]:
        result = result.merge(summary, on="metric", how="outer")

    return result
def compute_overtime_summary(df: pd.DataFrame, high_threshold: float) -> dict[str, float | int] | None:
    """Return key overtime summary metrics for the provided dataframe."""
    if COL_OVERTIME not in df.columns:
        return None

    series = pd.to_numeric(df[COL_OVERTIME], errors="coerce").dropna()
    if series.empty:
        return None

    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=1)) if series.count() > 1 else np.nan,
        "min": float(series.min()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
        "pct_zero": float((series == 0).mean() * 100),
        "pct_high": float((series >= high_threshold).mean() * 100),
    }


def compute_group_overtime_statistics(
    df: pd.DataFrame,
    group_col: str,
    high_threshold: float,
) -> pd.DataFrame:
    """Compute group-level overtime statistics."""
    if COL_OVERTIME not in df.columns:
        return pd.DataFrame()

    work_df = df[[COL_EMPLOYEE_ID, group_col, COL_OVERTIME]].copy()
    work_df[group_col] = sanitize_text_column(work_df[group_col])
    work_df[COL_OVERTIME] = pd.to_numeric(work_df[COL_OVERTIME], errors="coerce")

    total_rows = (
        work_df.groupby(group_col, dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )

    valid_df = work_df.dropna(subset=[COL_OVERTIME]).copy()
    if valid_df.empty:
        return pd.DataFrame()

    grouped = valid_df.groupby(group_col, dropna=False)

    stats_df = grouped.agg(
        overtime_mean=(COL_OVERTIME, "mean"),
        overtime_median=(COL_OVERTIME, "median"),
        overtime_std=(COL_OVERTIME, "std"),
        overtime_min=(COL_OVERTIME, "min"),
        overtime_p90=(COL_OVERTIME, lambda s: s.quantile(0.90)),
        overtime_p95=(COL_OVERTIME, lambda s: s.quantile(0.95)),
        overtime_max=(COL_OVERTIME, "max"),
        overtime_non_null=(COL_OVERTIME, "count"),
        pct_zero_overtime=(COL_OVERTIME, lambda s: float((s == 0).mean() * 100)),
        pct_high_overtime=(COL_OVERTIME, lambda s: float((s >= high_threshold).mean() * 100)),
    ).reset_index()

    result = total_rows.merge(stats_df, on=group_col, how="left")
    return result.sort_values(
        by=["overtime_median", "overtime_mean"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_ranked_overtime_group_chart(
    stats_df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    metric_title: str,
    chart_title: str,
    top_n: int = 20,
) -> alt.Chart:
    """Build a ranked bar chart for a selected overtime metric by group."""
    if stats_df.empty or metric_col not in stats_df.columns:
        return alt.Chart(pd.DataFrame({"message": ["No data available"]})).mark_text(size=14).encode(text="message:N")

    plot_df = (
        stats_df[[group_col, metric_col, "row_count"]]
        .dropna(subset=[metric_col])
        .sort_values(metric_col, ascending=False)
        .head(top_n)
        .copy()
    )

    if plot_df.empty:
        return alt.Chart(pd.DataFrame({"message": ["No data available"]})).mark_text(size=14).encode(text="message:N")

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric_col}:Q", title=metric_title),
            y=alt.Y(f"{group_col}:N", title=group_col, sort="-x"),
            tooltip=[
                alt.Tooltip(f"{group_col}:N", title="Group"),
                alt.Tooltip(f"{metric_col}:Q", title=metric_title, format=".2f"),
                alt.Tooltip("row_count:Q", title="Rows"),
            ],
        )
        .properties(height=max(320, min(800, len(plot_df) * 24)), title=chart_title)
    )


def build_overtime_band_chart(df: pd.DataFrame) -> alt.Chart:
    """Build a fixed-band overtime distribution chart."""
    if COL_OVERTIME not in df.columns:
        return alt.Chart(pd.DataFrame({"message": ["No overtime column available"]})).mark_text(size=14).encode(
            text="message:N"
        )

    series = pd.to_numeric(df[COL_OVERTIME], errors="coerce").dropna()
    if series.empty:
        return alt.Chart(pd.DataFrame({"message": ["No overtime data available"]})).mark_text(size=14).encode(
            text="message:N"
        )

    bins = [-np.inf, 0, 10, 20, 40, np.inf]
    labels = ["0", ">0 to 10", "10 to 20", "20 to 40", "40+"]

    band_df = pd.DataFrame({COL_OVERTIME: series})
    band_df["Overtime Band"] = pd.cut(
        band_df[COL_OVERTIME],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    plot_df = (
        band_df["Overtime Band"]
        .value_counts(dropna=False)
        .reindex(labels, fill_value=0)
        .rename_axis("Overtime Band")
        .reset_index(name="count")
    )

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Overtime Band:N", sort=labels, title="Overtime Band"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=[
                alt.Tooltip("Overtime Band:N", title="Overtime Band"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=300, title="Overtime Band Distribution")
    )


def build_overtime_boxplot(
    df: pd.DataFrame,
    group_col: str,
    top_groups: list[str],
    title: str,
) -> alt.Chart:
    """Build a boxplot of overtime for selected groups."""
    if COL_OVERTIME not in df.columns:
        return alt.Chart(pd.DataFrame({"message": ["No overtime column available"]})).mark_text(size=14).encode(
            text="message:N"
        )

    plot_df = df[df[group_col].astype(str).isin([str(g) for g in top_groups])].copy()
    plot_df = plot_df.dropna(subset=[COL_OVERTIME])

    if plot_df.empty:
        return alt.Chart(pd.DataFrame({"message": ["No data available"]})).mark_text(size=14).encode(text="message:N")

    return (
        alt.Chart(plot_df)
        .mark_boxplot(size=28)
        .encode(
            x=alt.X(f"{group_col}:N", title=group_col, sort=top_groups),
            y=alt.Y(f"{COL_OVERTIME}:Q", title=COL_OVERTIME),
            tooltip=[
                alt.Tooltip(f"{group_col}:N", title="Group"),
                alt.Tooltip(f"{COL_OVERTIME}:Q", title=COL_OVERTIME, format=".2f"),
            ],
        )
        .properties(height=420, title=title)
    )

def run_kmeans_clustering(
    agg_df: pd.DataFrame,
    group_col: str,
    config: ClusterConfig,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, int | None, float | None]:
    """
    Run K-Means on grouped averages and choose the best k via silhouette score.

    Returns:
        mapping_df,
        silhouette_results_df,
        best_k,
        best_silhouette
    """
    features = agg_df[[group_col, "avg_absense_occasions", "avg_days_absent", "count"]].copy()

    eligible_mask = features["count"] >= config.min_records_per_group
    eligible_df = features.loc[eligible_mask].reset_index(drop=True)
    excluded_df = features.loc[~eligible_mask].reset_index(drop=True)

    if len(eligible_df) < 3:
        return None, None, None, None

    X = eligible_df[["avg_absense_occasions", "avg_days_absent"]].to_numpy(dtype=float)

    if config.standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X

    n_samples = len(eligible_df)
    min_k = max(2, config.k_min)
    max_k = min(config.k_max, n_samples - 1)

    if min_k > max_k:
        return None, None, None, None

    sample_weight = eligible_df["count"].to_numpy(dtype=float) if config.weight_by_count else None

    silhouette_rows: list[dict[str, float | int]] = []
    best_k: int | None = None
    best_score = -np.inf
    best_labels: np.ndarray | None = None

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=config.random_state,
            n_init=config.n_init,
        )

        try:
            if sample_weight is not None:
                model.fit(X_proc, sample_weight=sample_weight)
            else:
                model.fit(X_proc)
        except TypeError:
            model.fit(X_proc)

        labels = model.labels_.astype(int)

        try:
            score = float(silhouette_score(X_proc, labels, metric="euclidean"))
        except Exception:
            score = np.nan

        silhouette_rows.append({"k": k, "silhouette": score})

        if np.isfinite(score) and (
            score > best_score or (np.isclose(score, best_score) and (best_k is None or k < best_k))
        ):
            best_k = k
            best_score = score
            best_labels = labels

    if best_k is None or best_labels is None:
        return None, pd.DataFrame(silhouette_rows), None, None

    label_names = {i: f"Cluster {i + 1}" for i in range(best_k)}
    eligible_df[COL_CLUSTER] = [label_names[i] for i in best_labels]

    if not excluded_df.empty:
        excluded_df[COL_CLUSTER] = "Excluded (Too few records)"

    mapping_df = pd.concat([eligible_df, excluded_df], ignore_index=True)
    mapping_df[group_col] = sanitize_text_column(mapping_df[group_col]).astype("string")

    silhouette_df = pd.DataFrame(silhouette_rows).sort_values("k").reset_index(drop=True)
    return mapping_df, silhouette_df, best_k, float(best_score)


def run_individual_kmeans_clustering(
    df_group: pd.DataFrame,
    config: IndividualClusterConfig,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, int | None, float | None]:
    """
    Cluster individual records within a single selected group using:
    - x: Absense Occasions
    - y: Days Absent

    Returns:
        clustered_df,
        silhouette_df,
        best_k,
        best_silhouette
    """
    work_df = df_group.copy()
    work_df = work_df.dropna(subset=[COL_ABSENCE_OCCASIONS, COL_DAYS_ABSENT]).reset_index(drop=True)

    if len(work_df) < 3:
        return None, None, None, None

    X = work_df[[COL_ABSENCE_OCCASIONS, COL_DAYS_ABSENT]].to_numpy(dtype=float)

    if config.standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X

    n_samples = len(work_df)
    min_k = max(2, config.k_min)
    max_k = min(config.k_max, n_samples - 1)

    if min_k > max_k:
        return None, None, None, None

    silhouette_rows: list[dict[str, float | int]] = []
    best_k: int | None = None
    best_score = -np.inf
    best_labels: np.ndarray | None = None

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=config.random_state,
            n_init=config.n_init,
        )
        model.fit(X_proc)
        labels = model.labels_.astype(int)

        try:
            score = float(silhouette_score(X_proc, labels, metric="euclidean"))
        except Exception:
            score = np.nan

        silhouette_rows.append({"k": k, "silhouette": score})

        if np.isfinite(score) and (
            score > best_score or (np.isclose(score, best_score) and (best_k is None or k < best_k))
        ):
            best_k = k
            best_score = score
            best_labels = labels

    if best_k is None or best_labels is None:
        return None, pd.DataFrame(silhouette_rows), None, None

    label_names = {i: f"Cluster {i + 1}" for i in range(best_k)}
    work_df[COL_INDIVIDUAL_CLUSTER] = [label_names[i] for i in best_labels]

    silhouette_df = pd.DataFrame(silhouette_rows).sort_values("k").reset_index(drop=True)
    return work_df, silhouette_df, best_k, float(best_score)


def build_individual_scatter_chart(
    df: pd.DataFrame,
    group_col: str,
    color_field: str | None = None,
    title: str | None = None,
    show_median_lines: bool = True,
    median_x: float | None = None,
    median_y: float | None = None,
) -> alt.Chart:
    """
    Scatter plot of individual records in a chosen group.
    Optionally adds median reference lines for Absense Occasions and Days Absent.
    """
    plot_df = df.dropna(subset=[COL_ABSENCE_OCCASIONS, COL_DAYS_ABSENT]).copy()

    if plot_df.empty:
        return alt.Chart(pd.DataFrame({"message": ["No data available"]})).mark_text(size=14).encode(text="message:N")

    tooltip_fields = [
        alt.Tooltip(f"{COL_EMPLOYEE_ID}:N", title=COL_EMPLOYEE_ID),
        alt.Tooltip(f"{group_col}:N", title="Group"),
        alt.Tooltip(f"{COL_ABSENCE_OCCASIONS}:Q", title=COL_ABSENCE_OCCASIONS, format=".2f"),
        alt.Tooltip(f"{COL_DAYS_ABSENT}:Q", title=COL_DAYS_ABSENT, format=".2f"),
    ]

    if COL_OVERTIME in plot_df.columns:
        tooltip_fields.append(alt.Tooltip(f"{COL_OVERTIME}:Q", title=COL_OVERTIME, format=".2f"))

    if color_field and color_field in plot_df.columns:
        tooltip_fields.append(alt.Tooltip(f"{color_field}:N", title=color_field))

    encodings = {
        "x": alt.X(f"{COL_ABSENCE_OCCASIONS}:Q", title=COL_ABSENCE_OCCASIONS),
        "y": alt.Y(f"{COL_DAYS_ABSENT}:Q", title=COL_DAYS_ABSENT),
        "tooltip": tooltip_fields,
    }

    if color_field and color_field in plot_df.columns:
        encodings["color"] = alt.Color(f"{color_field}:N", title=color_field)

    points = (
        alt.Chart(plot_df)
        .mark_circle(size=75, opacity=0.8)
        .encode(**encodings)
    )

    layers = [points]

    if show_median_lines:
        if median_x is None:
            median_x = pd.to_numeric(plot_df[COL_ABSENCE_OCCASIONS], errors="coerce").median(skipna=True)
        if median_y is None:
            median_y = pd.to_numeric(plot_df[COL_DAYS_ABSENT], errors="coerce").median(skipna=True)

        if pd.notna(median_x):
            vline_df = pd.DataFrame(
                {
                    "x": [float(median_x)],
                    "label": [f"Median {COL_ABSENCE_OCCASIONS}"],
                }
            )
            vline = (
                alt.Chart(vline_df)
                .mark_rule(color="red", strokeDash=[6, 4], size=2)
                .encode(
                    x="x:Q",
                    tooltip=[
                        alt.Tooltip("label:N", title="Reference"),
                        alt.Tooltip("x:Q", title=f"Median {COL_ABSENCE_OCCASIONS}", format=".2f"),
                    ],
                )
            )
            layers.append(vline)

        if pd.notna(median_y):
            hline_df = pd.DataFrame(
                {
                    "y": [float(median_y)],
                    "label": [f"Median {COL_DAYS_ABSENT}"],
                }
            )
            hline = (
                alt.Chart(hline_df)
                .mark_rule(color="red", strokeDash=[6, 4], size=2)
                .encode(
                    y="y:Q",
                    tooltip=[
                        alt.Tooltip("label:N", title="Reference"),
                        alt.Tooltip("y:Q", title=f"Median {COL_DAYS_ABSENT}", format=".2f"),
                    ],
                )
            )
            layers.append(hline)

    chart = alt.layer(*layers).properties(height=450)

    if title:
        chart = chart.properties(title=title)

    return chart.interactive()


# =========================================================
# Cached file loader
# =========================================================
@st.cache_data(show_spinner=False)
def load_and_prepare_csv(file_bytes: bytes, anonymize_ids: bool) -> pd.DataFrame:
    """
    Read, standardize, validate, coerce numeric columns,
    and optionally anonymize employee IDs.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = standardize_column_names(df)

    missing = validate_required_columns(df)
    if missing:
        raise ValueError(
            "The uploaded file is missing required columns: "
            + ", ".join(f"'{col}'" for col in missing)
        )

    df = coerce_numeric_columns(df, NUMERIC_COLUMNS)

    if anonymize_ids:
        df = anonymize_employee_ids(df)

    return df


# =========================================================
# UI helpers
# =========================================================
def render_dataset_overview(df: pd.DataFrame, group_col: str) -> None:
    """Render top-level dataset summary metrics."""
    total_rows = len(df)
    unique_groups = df[group_col].nunique(dropna=True)
    avg_absence = df[COL_ABSENCE_OCCASIONS].mean(skipna=True)
    avg_days = df[COL_DAYS_ABSENT].mean(skipna=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{total_rows:,}")
    c2.metric("Groups", f"{unique_groups:,}")
    c3.metric("Avg Absense Occasions", f"{avg_absence:.2f}" if pd.notna(avg_absence) else "N/A")
    c4.metric("Avg Days Absent", f"{avg_days:.2f}" if pd.notna(avg_days) else "N/A")


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Data Options")

anonymize_ids = st.sidebar.checkbox(
    "Anonymize Industry Number values on load",
    value=True,
    help="If selected, Industry Number values will be replaced with sequential integers. If not selected, original values are retained.",
)

st.sidebar.header("Clustering Options")

show_clustering_parameters = st.sidebar.checkbox(
    "Edit clustering parameters",
    value=False,
    help="Show advanced clustering parameters in the sidebar. If not selected, default clustering parameters are used.",
)

# Default clustering settings
group_cluster_k_range = (2, 8)
group_cluster_standardize = True
group_cluster_weight_by_count = True
group_cluster_min_records = 5
group_cluster_random_state = 42
group_cluster_n_init = 10

individual_cluster_k_range = (2, 6)
individual_cluster_standardize = True
individual_cluster_random_state = 42
individual_cluster_n_init = 10

if show_clustering_parameters:
    st.sidebar.markdown("### Group Clustering Parameters")
    group_cluster_k_range = st.sidebar.slider(
        "Group clustering k-range",
        min_value=2,
        max_value=15,
        value=(2, 8),
        help="K-Means will be run for each k in this range, and the best silhouette score will be selected.",
    )
    group_cluster_standardize = st.sidebar.checkbox(
        "Standardize group features",
        value=True,
        help="Recommended when features are on different scales.",
    )
    group_cluster_weight_by_count = st.sidebar.checkbox(
        "Weight group clustering by group size",
        value=True,
        help="Gives larger groups more influence in the model fit.",
    )
    group_cluster_min_records = st.sidebar.number_input(
        "Minimum records per group for clustering",
        min_value=1,
        max_value=1000,
        value=5,
        step=1,
    )
    group_cluster_random_state = st.sidebar.number_input(
        "Group clustering random seed",
        min_value=0,
        max_value=10000,
        value=42,
        step=1,
    )
    group_cluster_n_init = st.sidebar.number_input(
        "Group clustering n_init",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Number of centroid initializations.",
    )

    st.sidebar.markdown("### Individual Clustering Parameters")
    individual_cluster_k_range = st.sidebar.slider(
        "Individual clustering k-range",
        min_value=2,
        max_value=10,
        value=(2, 6),
        help="K-Means will be run for each k in this range, and the best silhouette score will be selected.",
    )
    individual_cluster_standardize = st.sidebar.checkbox(
        "Standardize individual features",
        value=True,
        help="Standardize Absense Occasions and Days Absent before clustering.",
    )
    individual_cluster_random_state = st.sidebar.number_input(
        "Individual clustering random seed",
        min_value=0,
        max_value=10000,
        value=42,
        step=1,
    )
    individual_cluster_n_init = st.sidebar.number_input(
        "Individual clustering n_init",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

group_cluster_config = ClusterConfig(
    k_min=int(group_cluster_k_range[0]),
    k_max=int(group_cluster_k_range[1]),
    standardize=bool(group_cluster_standardize),
    weight_by_count=bool(group_cluster_weight_by_count),
    min_records_per_group=int(group_cluster_min_records),
    random_state=int(group_cluster_random_state),
    n_init=int(group_cluster_n_init),
)

individual_cluster_config = IndividualClusterConfig(
    k_min=int(individual_cluster_k_range[0]),
    k_max=int(individual_cluster_k_range[1]),
    standardize=bool(individual_cluster_standardize),
    random_state=int(individual_cluster_random_state),
    n_init=int(individual_cluster_n_init),
)


# =========================================================
# Main app
# =========================================================
st.title(APP_TITLE)
st.caption(
    "Upload a CSV with columns such as "
    f"'{COL_EMPLOYEE_ID}', '{COL_ABSENCE_OCCASIONS}', '{COL_DAYS_ABSENT}', and a categorical grouping column."
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

try:
    raw_bytes = uploaded_file.getvalue()
    df_all = load_and_prepare_csv(raw_bytes, anonymize_ids)
except Exception as exc:
    st.error(f"Could not read the uploaded file. Details: {exc}")
    st.stop()

group_candidates = get_grouping_candidates(df_all)
if not group_candidates:
    st.error(
        "No valid grouping columns were found. "
        "Please ensure the file contains at least one categorical/text column with a reasonable number of distinct values."
    )
    st.stop()

# ---------------------------------------------------------
# Sidebar analysis controls
# ---------------------------------------------------------
st.sidebar.header("Analysis Controls")

group_col = st.sidebar.selectbox(
    "Grouping column",
    options=group_candidates,
    help="Select the categorical column used to compare groups.",
)

exclude_unfit = st.sidebar.checkbox(
    "Exclude 'unfit' / 'maternity' records",
    value=False,
    help="Remove rows where Reporting Discipline contains 'unfit' or 'maternity' (case-insensitive).",
)

exclude_small_groups = st.sidebar.checkbox(
    "Exclude small groups",
    value=False,
    help="Remove groups with too few records before group-level analysis.",
)

current_group_counts = df_all[group_col].value_counts(dropna=False)
max_threshold = int(current_group_counts.max()) if not current_group_counts.empty else 1

min_group_size = st.sidebar.number_input(
    "Minimum records per group",
    min_value=1,
    max_value=max(1, max_threshold),
    value=min(5, max(1, max_threshold)),
    step=1,
    disabled=not exclude_small_groups,
)

# ---------------------------------------------------------
# Apply filters
# ---------------------------------------------------------
df_filtered, excluded_values_df = filter_reporting_discipline(df_all, exclude_unfit)
df_filtered, group_counts = apply_min_group_size_filter(
    df_filtered,
    group_col,
    enabled=exclude_small_groups,
    threshold=int(min_group_size),
)

# Ensure group column is clean for UI behavior
df_filtered[group_col] = sanitize_text_column(df_filtered[group_col])

if df_filtered.empty:
    st.warning("No rows remain after applying the selected filters.")
    st.stop()

# Core working dataframe
base_columns = [COL_EMPLOYEE_ID, group_col, COL_ABSENCE_OCCASIONS, COL_DAYS_ABSENT]
additional_columns = [col for col in OPTIONAL_NUMERIC_COLUMNS if col in df_filtered.columns]
df = df_filtered[base_columns + additional_columns].copy()

# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------
render_dataset_overview(df, group_col)

if exclude_unfit and not excluded_values_df.empty:
    with st.expander("Excluded Reporting Discipline values", expanded=False):
        st.caption(
            f"Excluded **{int(excluded_values_df['rows_excluded'].sum()):,}** row(s) "
            f"across **{len(excluded_values_df):,}** distinct Reporting Discipline value(s)."
        )
        st.dataframe(excluded_values_df, use_container_width=True)

st.subheader("Group Distribution")
group_distribution_chart = build_group_distribution_chart(group_counts, group_col)
st.altair_chart(group_distribution_chart, use_container_width=True)
st.caption("Record counts per group after applying the selected filters, sorted in descending order.")

preview_label = "Preview of imported data"
if anonymize_ids:
    preview_label += " (PII anonymized)"

with st.expander(preview_label, expanded=False):
    st.dataframe(df.head(200), use_container_width=True)

# ---------------------------------------------------------
# Group statistics
# ---------------------------------------------------------
agg_df = compute_group_statistics(df, group_col)

st.subheader("Group-level Statistics")
st.caption(f"Averages, medians, standard deviations, and row counts by '{group_col}'.")
st.dataframe(agg_df, use_container_width=True)

st.download_button(
    label="Download group-level statistics (CSV)",
    data=agg_df.to_csv(index=False),
    file_name="group_level_statistics.csv",
    mime="text/csv",
)

# ---------------------------------------------------------
# Tabs for analysis sections
# ---------------------------------------------------------
tab_scatter, tab_clustering, tab_distributions, tab_overtime, tab_correlations = st.tabs(
    ["Scatter", "Clustering", "Distributions", "Overtime", "Correlations"]
)

mapping_df: pd.DataFrame | None = None

# =========================================================
# Scatter tab
# =========================================================
with tab_scatter:
    st.subheader("2D Scatter: Average Absense Occasions vs Average Days Absent")

    if agg_df[["avg_absense_occasions", "avg_days_absent"]].dropna().empty:
        st.info("No valid aggregated points are available to plot.")
    else:
        scatter_chart = build_scatter_chart(
            df=agg_df,
            group_col=group_col,
            title="Group Averages",
            show_median_lines=True,
        )
        st.altair_chart(scatter_chart, use_container_width=True)

# =========================================================
# Clustering tab
# =========================================================
with tab_clustering:
    st.subheader("Statistical Grouping (K-Means)")

    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn is not available in this environment, so clustering cannot be run.")
    else:
        if not show_clustering_parameters:
            st.caption("Using default clustering parameters. Enable 'Edit clustering parameters' in the sidebar to customize them.")
        else:
            st.caption("Using clustering parameters configured in the sidebar.")

        mapping_df, silhouette_df, best_k, best_score = run_kmeans_clustering(
            agg_df=agg_df,
            group_col=group_col,
            config=group_cluster_config,
        )

        if mapping_df is None or best_k is None or best_score is None:
            st.info(
                "Not enough eligible groups to run silhouette-based clustering. "
                "At least 3 eligible groups are required."
            )
        else:
            st.success(f"Selected k = {best_k} (silhouette = {best_score:.3f})")

            st.markdown("### Group-to-Cluster Mapping")
            view_df = mapping_df.sort_values([COL_CLUSTER, group_col]).reset_index(drop=True)
            st.dataframe(view_df, use_container_width=True)

            st.download_button(
                label="Download cluster mapping (CSV)",
                data=view_df.to_csv(index=False),
                file_name=f"overall_groupings_kmeans_k{best_k}.csv",
                mime="text/csv",
            )

            with st.expander("Silhouette diagnostics", expanded=False):
                if silhouette_df is not None and not silhouette_df.empty:
                    st.dataframe(silhouette_df, use_container_width=True)
                    sil_chart = (
                        alt.Chart(silhouette_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("k:Q", title="k (number of clusters)"),
                            y=alt.Y("silhouette:Q", title="Silhouette score"),
                            tooltip=[
                                alt.Tooltip("k:Q"),
                                alt.Tooltip("silhouette:Q", format=".3f"),
                            ],
                        )
                        .properties(height=260)
                        .interactive()
                    )
                    st.altair_chart(sil_chart, use_container_width=True)

            st.markdown("### Scatter by Cluster")
            st.caption("Includes median reference lines for Avg Absense Occasions and Avg Days Absent.")
            cluster_plot_df = mapping_df.copy()
            cluster_chart = build_scatter_chart(
                df=cluster_plot_df,
                group_col=group_col,
                color_field=COL_CLUSTER,
                title="Clustered Group Averages",
                show_median_lines=True,
            )
            st.altair_chart(cluster_chart, use_container_width=True)

# =========================================================
# Distributions tab
# =========================================================
with tab_distributions:
    st.subheader("Distribution Explorer")

    view_mode = st.radio(
        "View distributions by",
        options=["Single Group", "Cluster"],
        horizontal=True,
    )

    histogram_bins = st.slider(
        "Number of histogram bins",
        min_value=10,
        max_value=80,
        value=30,
        step=1,
    )

    selected_groups: list[str] = []
    selected_group: str | None = None

    if view_mode == "Single Group":
        group_options = (
            df[group_col]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        if not group_options:
            st.info("No groups are available.")
        else:
            selected_group = st.selectbox("Select a group", options=group_options)
            selected_groups = [selected_group]

    else:
        if mapping_df is None or COL_CLUSTER not in mapping_df.columns:
            st.info("Run clustering first to enable cluster-level distribution views.")
        else:
            cluster_options = (
                mapping_df[COL_CLUSTER]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )

            if not cluster_options:
                st.info("No clusters are available.")
            else:
                selected_cluster = st.selectbox("Select a cluster", options=cluster_options)
                selected_groups = (
                    mapping_df.loc[mapping_df[COL_CLUSTER].astype(str) == selected_cluster, group_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                st.caption(f"{selected_cluster} contains {len(selected_groups)} group(s).")

    if selected_groups:
        subset = df[df[group_col].astype(str).isin(selected_groups)].copy()

        # Bradford removed from Distribution Explorer charts and summary
        distribution_metric_columns = [
            col for col in [COL_DAYS_ABSENT, COL_ABSENCE_OCCASIONS, COL_OVERTIME]
            if col in subset.columns
        ]
        subset = subset.dropna(subset=distribution_metric_columns, how="all")

        if subset.empty:
            st.info("No individual-level rows are available for the current selection.")
        else:
            st.caption(f"Selected rows: **{len(subset):,}**")

            charts: list[alt.Chart] = []
            if COL_DAYS_ABSENT in subset.columns:
                charts.append(build_histogram(subset, COL_DAYS_ABSENT, COL_DAYS_ABSENT, bins=histogram_bins))
            if COL_ABSENCE_OCCASIONS in subset.columns:
                charts.append(
                    build_histogram(
                        subset,
                        COL_ABSENCE_OCCASIONS,
                        COL_ABSENCE_OCCASIONS,
                        bins=histogram_bins,
                        integer_bins=True,
                    )
                )
            if COL_OVERTIME in subset.columns:
                charts.append(build_histogram(subset, COL_OVERTIME, COL_OVERTIME, bins=histogram_bins))

            st.markdown("### Distributions")
            for chart in charts:
                st.altair_chart(chart, use_container_width=True)

            with st.expander("Summary statistics for current selection", expanded=False):
                summary_df = summarize_numeric_columns(
                    subset,
                    [COL_DAYS_ABSENT, COL_ABSENCE_OCCASIONS, COL_OVERTIME],
                )
                if summary_df.empty:
                    st.info("No summary statistics are available.")
                else:
                    st.dataframe(summary_df.round(2), use_container_width=True)

            # -------------------------------------------------
            # Individual scatter + clustering for selected group
            # -------------------------------------------------
            if view_mode == "Single Group" and selected_group is not None:
                st.markdown("### Individual Scatter Plot")
                st.caption(
                    "Includes a vertical median line for Absense Occasions and a horizontal median line "
                    "for Days Absent for the selected group."
                )

                single_group_df = df[df[group_col].astype(str) == str(selected_group)].copy()

                # Calculate medians once for the selected group and reuse for both plots
                valid_single_group_df = single_group_df.dropna(subset=[COL_ABSENCE_OCCASIONS, COL_DAYS_ABSENT]).copy()
                median_absense_occasions = (
                    float(valid_single_group_df[COL_ABSENCE_OCCASIONS].median())
                    if not valid_single_group_df.empty
                    else None
                )
                median_days_absent = (
                    float(valid_single_group_df[COL_DAYS_ABSENT].median())
                    if not valid_single_group_df.empty
                    else None
                )

                individual_scatter = build_individual_scatter_chart(
                    single_group_df,
                    group_col=group_col,
                    title=f"Individuals in {selected_group}",
                    show_median_lines=True,
                    median_x=median_absense_occasions,
                    median_y=median_days_absent,
                )
                st.altair_chart(individual_scatter, use_container_width=True)

                st.markdown("### Clustered Individual Scatter Plot")
                st.caption(
                    "Uses the same selected-group medians as reference lines."
                )

                if not SKLEARN_AVAILABLE:
                    st.warning("scikit-learn is not available in this environment, so individual clustering cannot be run.")
                else:
                    valid_points = single_group_df.dropna(subset=[COL_ABSENCE_OCCASIONS, COL_DAYS_ABSENT]).shape[0]

                    if valid_points < 3:
                        st.info("At least 3 individual records with valid Absense Occasions and Days Absent are required.")
                    else:
                        if not show_clustering_parameters:
                            st.caption("Using default individual clustering parameters from the sidebar settings.")
                        else:
                            st.caption("Using individual clustering parameters configured in the sidebar.")

                        adjusted_individual_config = IndividualClusterConfig(
                            k_min=individual_cluster_config.k_min,
                            k_max=min(individual_cluster_config.k_max, max(2, valid_points - 1)),
                            standardize=individual_cluster_config.standardize,
                            random_state=individual_cluster_config.random_state,
                            n_init=individual_cluster_config.n_init,
                        )

                        clustered_individuals_df, individual_silhouette_df, best_individual_k, best_individual_score = (
                            run_individual_kmeans_clustering(
                                df_group=single_group_df,
                                config=adjusted_individual_config,
                            )
                        )

                        if (
                            clustered_individuals_df is None
                            or best_individual_k is None
                            or best_individual_score is None
                        ):
                            st.info("Unable to cluster individual records for the selected group.")
                        else:
                            st.success(
                                f"Selected k = {best_individual_k} "
                                f"(silhouette = {best_individual_score:.3f}) for individuals in '{selected_group}'."
                            )

                            clustered_scatter = build_individual_scatter_chart(
                                clustered_individuals_df,
                                group_col=group_col,
                                color_field=COL_INDIVIDUAL_CLUSTER,
                                title=f"Clustered Individuals in {selected_group}",
                                show_median_lines=True,
                                median_x=median_absense_occasions,
                                median_y=median_days_absent,
                            )
                            st.altair_chart(clustered_scatter, use_container_width=True)

                            with st.expander("Individual clustering diagnostics", expanded=False):
                                if individual_silhouette_df is not None and not individual_silhouette_df.empty:
                                    st.dataframe(individual_silhouette_df, use_container_width=True)

                                    sil_chart = (
                                        alt.Chart(individual_silhouette_df)
                                        .mark_line(point=True)
                                        .encode(
                                            x=alt.X("k:Q", title="k (number of clusters)"),
                                            y=alt.Y("silhouette:Q", title="Silhouette score"),
                                            tooltip=[
                                                alt.Tooltip("k:Q"),
                                                alt.Tooltip("silhouette:Q", format=".3f"),
                                            ],
                                        )
                                        .properties(height=260)
                                        .interactive()
                                    )
                                    st.altair_chart(sil_chart, use_container_width=True)

                            with st.expander("Clustered individual records", expanded=False):
                                display_cols = [
                                    col for col in [
                                        COL_EMPLOYEE_ID,
                                        group_col,
                                        COL_ABSENCE_OCCASIONS,
                                        COL_DAYS_ABSENT,
                                        COL_OVERTIME,
                                        COL_INDIVIDUAL_CLUSTER,
                                    ]
                                    if col in clustered_individuals_df.columns
                                ]
                                st.dataframe(
                                    clustered_individuals_df[display_cols].sort_values(
                                        by=[COL_INDIVIDUAL_CLUSTER, COL_ABSENCE_OCCASIONS, COL_DAYS_ABSENT]
                                    ),
                                    use_container_width=True,
                                )
                                
# =========================================================
# Overtime tab
# =========================================================
with tab_overtime:
    st.subheader("Overtime Analysis")

    if COL_OVERTIME not in df.columns:
        st.info(
            f"No '{COL_OVERTIME}' column is available in the uploaded dataset, so overtime analytics cannot be shown."
        )
    else:
        overtime_valid_series = pd.to_numeric(df[COL_OVERTIME], errors="coerce").dropna()

        if overtime_valid_series.empty:
            st.info("The overtime column exists, but no valid numeric overtime values are available after cleaning.")
        else:
            overtime_view_mode = st.radio(
                "Analyse overtime for",
                options=["All Filtered Data", "Single Group", "Cluster"],
                horizontal=True,
            )

            overtime_subset = df.copy()
            overtime_selection_label = "All filtered data"

            if overtime_view_mode == "Single Group":
                overtime_group_options = (
                    df[group_col]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist()
                )

                if not overtime_group_options:
                    st.info("No groups are available for overtime analysis.")
                    overtime_subset = pd.DataFrame()
                else:
                    overtime_selected_group = st.selectbox(
                        "Select a group for overtime analysis",
                        options=overtime_group_options,
                        key="overtime_single_group_select",
                    )
                    overtime_subset = df[df[group_col].astype(str) == str(overtime_selected_group)].copy()
                    overtime_selection_label = f"Group: {overtime_selected_group}"

            elif overtime_view_mode == "Cluster":
                if mapping_df is None or COL_CLUSTER not in mapping_df.columns:
                    st.info("Run clustering first to enable cluster-level overtime analysis.")
                    overtime_subset = pd.DataFrame()
                else:
                    overtime_cluster_options = (
                        mapping_df[COL_CLUSTER]
                        .dropna()
                        .astype(str)
                        .sort_values()
                        .unique()
                        .tolist()
                    )

                    if not overtime_cluster_options:
                        st.info("No clusters are available.")
                        overtime_subset = pd.DataFrame()
                    else:
                        overtime_selected_cluster = st.selectbox(
                            "Select a cluster for overtime analysis",
                            options=overtime_cluster_options,
                            key="overtime_cluster_select",
                        )

                        overtime_selected_groups = (
                            mapping_df.loc[
                                mapping_df[COL_CLUSTER].astype(str) == str(overtime_selected_cluster),
                                group_col,
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                        overtime_subset = df[df[group_col].astype(str).isin(overtime_selected_groups)].copy()
                        overtime_selection_label = (
                            f"Cluster: {overtime_selected_cluster} "
                            f"({len(overtime_selected_groups)} group(s))"
                        )

            if overtime_subset.empty:
                st.info("No overtime data is available for the current selection.")
            else:
                overtime_subset = overtime_subset.dropna(subset=[COL_OVERTIME]).copy()

                if overtime_subset.empty:
                    st.info("No valid overtime values are available for the current selection.")
                else:
                    st.caption(f"Current selection: **{overtime_selection_label}** · **{len(overtime_subset):,}** row(s)")

                    max_overtime_value = float(pd.to_numeric(df[COL_OVERTIME], errors="coerce").dropna().max())
                    slider_max = max(1.0, float(np.ceil(max_overtime_value)))
                    default_threshold = min(20.0, slider_max)

                    high_overtime_threshold = st.slider(
                        "High overtime threshold",
                        min_value=0.0,
                        max_value=slider_max,
                        value=float(default_threshold),
                        step=1.0,
                        help="Used to calculate the share of employees/records at or above this overtime level.",
                    )

                    overtime_summary = compute_overtime_summary(
                        overtime_subset,
                        high_threshold=high_overtime_threshold,
                    )

                    if overtime_summary is None:
                        st.info("No overtime summary could be calculated for the current selection.")
                    else:
                        c1, c2, c3, c4, c5, c6 = st.columns(6)
                        c1.metric("Records with overtime", f"{overtime_summary['count']:,}")
                        c2.metric("Mean overtime", f"{overtime_summary['mean']:.2f}")
                        c3.metric("Median overtime", f"{overtime_summary['median']:.2f}")
                        c4.metric("P90 overtime", f"{overtime_summary['p90']:.2f}")
                        c5.metric("% zero overtime", f"{overtime_summary['pct_zero']:.1f}%")
                        c6.metric(
                            f"% ≥ {high_overtime_threshold:.0f}",
                            f"{overtime_summary['pct_high']:.1f}%",
                        )

                    st.markdown("### Overtime Distribution")
                    dist_col1, dist_col2 = st.columns(2)

                    with dist_col1:
                        overtime_hist = build_histogram(
                            overtime_subset,
                            COL_OVERTIME,
                            COL_OVERTIME,
                            bins=30,
                        )
                        st.altair_chart(overtime_hist, use_container_width=True)

                    with dist_col2:
                        overtime_band_chart = build_overtime_band_chart(overtime_subset)
                        st.altair_chart(overtime_band_chart, use_container_width=True)

                    with st.expander("Overtime summary statistics", expanded=False):
                        overtime_summary_df = summarize_numeric_columns(
                            overtime_subset,
                            [COL_OVERTIME],
                        )
                        if overtime_summary_df.empty:
                            st.info("No summary statistics are available.")
                        else:
                            st.dataframe(overtime_summary_df.round(2), use_container_width=True)

                    # -------------------------------------------------
                    # Group-level overtime analysis
                    # -------------------------------------------------
                    st.markdown("### Group-Level Overtime Analysis")

                    if overtime_view_mode == "Single Group":
                        st.caption("A single group is selected, so group-to-group overtime comparisons are not shown.")
                    else:
                        group_scope_df = overtime_subset.copy()

                        overtime_group_stats_df = compute_group_overtime_statistics(
                            group_scope_df,
                            group_col=group_col,
                            high_threshold=high_overtime_threshold,
                        )

                        if overtime_group_stats_df.empty:
                            st.info("No group-level overtime statistics are available.")
                        else:
                            st.dataframe(overtime_group_stats_df.round(2), use_container_width=True)

                            st.download_button(
                                label="Download group-level overtime statistics (CSV)",
                                data=overtime_group_stats_df.to_csv(index=False),
                                file_name="group_level_overtime_statistics.csv",
                                mime="text/csv",
                            )

                            chart_col1, chart_col2 = st.columns(2)

                            with chart_col1:
                                median_chart = build_ranked_overtime_group_chart(
                                    overtime_group_stats_df,
                                    group_col=group_col,
                                    metric_col="overtime_median",
                                    metric_title="Median Overtime",
                                    chart_title="Top Groups by Median Overtime",
                                    top_n=20,
                                )
                                st.altair_chart(median_chart, use_container_width=True)

                            with chart_col2:
                                high_share_chart = build_ranked_overtime_group_chart(
                                    overtime_group_stats_df,
                                    group_col=group_col,
                                    metric_col="pct_high_overtime",
                                    metric_title=f"% ≥ {high_overtime_threshold:.0f}",
                                    chart_title="Top Groups by High Overtime Share",
                                    top_n=20,
                                )
                                st.altair_chart(high_share_chart, use_container_width=True)

                            top_groups_for_boxplot = (
                                overtime_group_stats_df.sort_values("row_count", ascending=False)[group_col]
                                .dropna()
                                .astype(str)
                                .head(12)
                                .tolist()
                            )

                            if len(top_groups_for_boxplot) >= 2:
                                overtime_boxplot = build_overtime_boxplot(
                                    group_scope_df,
                                    group_col=group_col,
                                    top_groups=top_groups_for_boxplot,
                                    title="Overtime Box Plot for Largest Groups",
                                )
                                st.altair_chart(overtime_boxplot, use_container_width=True)

                    # -------------------------------------------------
                    # Highest-overtime records
                    # -------------------------------------------------
                    st.markdown("### Highest Overtime Records")

                    top_n_records = st.slider(
                        "Number of highest-overtime records to show",
                        min_value=5,
                        max_value=100,
                        value=20,
                        step=5,
                        key="top_overtime_records_slider",
                    )

                    highest_overtime_df = (
                        overtime_subset[
                            [
                                col for col in [
                                    COL_EMPLOYEE_ID,
                                    group_col,
                                    COL_ABSENCE_OCCASIONS,
                                    COL_DAYS_ABSENT,
                                    COL_OVERTIME,
                                ]
                                if col in overtime_subset.columns
                            ]
                        ]
                        .copy()
                        .sort_values(by=COL_OVERTIME, ascending=False)
                        .head(top_n_records)
                        .reset_index(drop=True)
                    )

                    st.dataframe(highest_overtime_df, use_container_width=True)
                    
# =========================================================
# Correlations tab
# =========================================================
with tab_correlations:
    st.subheader("Correlation Analysis")

    correlation_columns = [
        col for col in [COL_ABSENCE_OCCASIONS, COL_DAYS_ABSENT, COL_BRADFORD, COL_OVERTIME]
        if col in df.columns
    ]

    if len(correlation_columns) < 3:
        st.info(
            "At least 3 numeric columns are needed to render a meaningful correlation heatmap. "
            f"Currently available: {', '.join(correlation_columns) if correlation_columns else 'none'}"
        )
    else:
        corr_method = st.selectbox(
            "Correlation method",
            options=["pearson", "spearman", "kendall"],
            index=0,
            help=(
                "Pearson measures linear correlation. "
                "Spearman measures rank-based monotonic relationships. "
                "Kendall is a more conservative rank-based measure."
            ),
        )

        st.markdown("### Overall Correlation Heat Map")
        overall_heatmap = build_correlation_heatmap(df, correlation_columns, corr_method)
        if overall_heatmap is None:
            st.info("No overall correlation matrix could be computed.")
        else:
            st.altair_chart(overall_heatmap, use_container_width=True)

        st.markdown("### Segmented Correlation Heat Maps")
        segment_mode = st.radio(
            "Segment correlations by",
            options=["Group", "Cluster"],
            horizontal=True,
        )

        if segment_mode == "Group":
            group_options = (
                df[group_col]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )

            if not group_options:
                st.info("No groups are available.")
            else:
                chosen_groups = st.multiselect(
                    "Groups to display",
                    options=group_options,
                    default=group_options[: min(5, len(group_options))],
                )

                if not chosen_groups:
                    st.info("Select at least one group.")
                else:
                    tabs = st.tabs(chosen_groups)
                    for tab, chosen_group in zip(tabs, chosen_groups):
                        with tab:
                            seg_df = df[df[group_col].astype(str) == str(chosen_group)].copy()
                            st.caption(f"Group: {chosen_group} · {len(seg_df):,} row(s)")
                            chart = build_correlation_heatmap(seg_df, correlation_columns, corr_method)
                            if chart is None:
                                st.info("No correlation matrix could be computed for this group.")
                            else:
                                st.altair_chart(chart, use_container_width=True)

        else:
            if mapping_df is None or COL_CLUSTER not in mapping_df.columns:
                st.info("Run clustering first to enable cluster-level correlation views.")
            else:
                cluster_options = (
                    mapping_df[COL_CLUSTER]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist()
                )

                if not cluster_options:
                    st.info("No clusters are available.")
                else:
                    chosen_clusters = st.multiselect(
                        "Clusters to display",
                        options=cluster_options,
                        default=cluster_options[: min(3, len(cluster_options))],
                    )

                    if not chosen_clusters:
                        st.info("Select at least one cluster.")
                    else:
                        tabs = st.tabs(chosen_clusters)
                        for tab, chosen_cluster in zip(tabs, chosen_clusters):
                            with tab:
                                groups_in_cluster = (
                                    mapping_df.loc[mapping_df[COL_CLUSTER].astype(str) == chosen_cluster, group_col]
                                    .dropna()
                                    .astype(str)
                                    .unique()
                                    .tolist()
                                )
                                seg_df = df[df[group_col].astype(str).isin(groups_in_cluster)].copy()
                                st.caption(
                                    f"{chosen_cluster} · {len(groups_in_cluster):,} group(s) · {len(seg_df):,} row(s)"
                                )
                                chart = build_correlation_heatmap(seg_df, correlation_columns, corr_method)
                                if chart is None:
                                    st.info("No correlation matrix could be computed for this cluster.")
                                else:
                                    st.altair_chart(chart, use_container_width=True)
