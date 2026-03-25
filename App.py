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
