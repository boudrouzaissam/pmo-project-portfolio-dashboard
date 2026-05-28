import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="PMO Project Portfolio & Budget Tracking",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def money_millions(value):
    """Format a monetary value in millions."""
    if pd.isna(value):
        return "N/A"
    return f"${value/1_000_000:,.1f}M"


def number_fmt(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def pct_fmt(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def ratio_fmt(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def clean_column_name(col):
    return (
        col.strip()
        .lower()
        .replace("%", "_pct")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load, clean and enrich the project management dataset."""
    raw = pd.read_csv(path)
    df = raw.copy()

    # 1) Standardize column names
    df.columns = [clean_column_name(c) for c in df.columns]

    # 2) Normalize text values
    text_cols = [
        "project_name", "project_description", "project_type", "project_manager",
        "region", "department", "complexity", "status", "phase"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 3) Harmonize status labels
    if "status" in df.columns:
        df["status"] = (
            df["status"]
            .str.replace("In - Progress", "In-Progress", regex=False)
            .str.replace("In Progress", "In-Progress", regex=False)
        )

    # 4) Convert financial variables
    for col in ["project_cost", "project_benefit"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) Convert completion percentage
    if "completion_pct" in df.columns:
        df["completion_pct"] = (
            df["completion_pct"].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["completion_pct"] = pd.to_numeric(df["completion_pct"], errors="coerce")

    # 6) Convert dates
    for col in ["start_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 7) Feature engineering
    df["net_benefit"] = df["project_benefit"] - df["project_cost"]
    df["benefit_cost_ratio"] = np.where(
        df["project_cost"] > 0,
        df["project_benefit"] / df["project_cost"],
        np.nan
    )
    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days

    # 8) Outlier flags using IQR rule
    for col in ["project_cost", "project_benefit", "benefit_cost_ratio", "duration_days"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)

    # 9) Risk flag: concrete PMO monitoring logic
    median_cost = df["project_cost"].median()
    median_duration = df["duration_days"].median()
    df["risk_flag"] = np.where(
        (df["status"].isin(["Cancelled", "On-Hold"])) |
        ((df["complexity"] == "High") & (df["project_cost"] > median_cost)) |
        ((df["benefit_cost_ratio"] < 1.75) & (df["project_cost"] > median_cost)) |
        ((df["duration_days"] > median_duration) & (df["status"].isin(["In-Progress", "On-Hold"]))),
        "To Monitor",
        "Normal"
    )

    return df


def iqr_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [
        ("project_cost", "Project Cost"),
        ("project_benefit", "Project Benefit"),
        ("benefit_cost_ratio", "Benefit-Cost Ratio"),
        ("duration_days", "Duration Days"),
    ]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        rows.append({
            "Variable": label,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "Lower Bound": round(lower, 2),
            "Upper Bound": round(upper, 2),
            "Outliers": outliers,
        })
    return pd.DataFrame(rows)


def interpretation_box(title: str, text: str):
    st.markdown(
        f"""
        <div style="background-color:#FFFFFF; padding:1rem; border-radius:0.75rem; border-left:6px solid #2563EB; margin-top:0.6rem; margin-bottom:1rem; box-shadow:0 1px 4px rgba(15,23,42,0.08);">
        <strong>{title}</strong><br>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(number: str, question: str, purpose: str):
    st.markdown(f"## {number}. {question}")
    st.write(purpose)


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
DEFAULT_DATA_PATH = "Project Management Dataset.csv"

st.sidebar.title("PMO Dashboard")
st.sidebar.write("Use the filters below to explore the project portfolio.")

uploaded_file = st.sidebar.file_uploader("Upload a project CSV file", type=["csv"])
if uploaded_file is not None:
    data_path = uploaded_file
else:
    data_path = DEFAULT_DATA_PATH

df = load_and_clean_data(data_path)

# Sidebar filters
statuses = sorted(df["status"].dropna().unique())
departments = sorted(df["department"].dropna().unique())
regions = sorted(df["region"].dropna().unique())
years = sorted(df["year"].dropna().unique())

selected_status = st.sidebar.multiselect("Status", statuses, default=statuses)
selected_departments = st.sidebar.multiselect("Department", departments, default=departments)
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
selected_years = st.sidebar.multiselect("Year", years, default=years)

filtered = df[
    df["status"].isin(selected_status) &
    df["department"].isin(selected_departments) &
    df["region"].isin(selected_regions) &
    df["year"].isin(selected_years)
].copy()

# ------------------------------------------------------------
# Title and project definition
# ------------------------------------------------------------
st.title("PMO Project Portfolio & Budget Tracking Analysis")
st.caption("A data-driven project coordination tool for monitoring costs, expected benefits, delivery status, and portfolio risk signals.")

st.markdown("""
### Project definition
This project analyzes a portfolio of **99 projects** to support **PMO reporting, project coordination, and budget tracking**. 
The analysis focuses on concrete management questions: which projects consume the most resources, which projects generate the strongest expected value, which departments are more exposed financially, and which projects should be monitored because of delivery or risk signals.

### Data source and scope
The dataset was obtained from **Kaggle** as a CSV file named **Project Management Dataset.csv**. Since the source does not clearly identify a specific real company or organization, the dataset is treated here as an **anonymized project portfolio** used to demonstrate PMO analytics and project coordination reporting.
""")

# ------------------------------------------------------------
# Table of actions / processing plan
# ------------------------------------------------------------
st.markdown("## Data Processing Roadmap")
processing_steps = pd.DataFrame({
    "Step": [
        "1. Import data",
        "2. Standardize column names",
        "3. Clean text categories",
        "4. Convert financial variables",
        "5. Convert completion rate",
        "6. Convert dates",
        "7. Create analytical variables",
        "8. Check data quality",
        "9. Detect outliers",
        "10. Build KPIs",
        "11. Build section-level analyses",
        "12. Generate PMO interpretations",
    ],
    "Action in Python": [
        "Read the CSV file with pandas.",
        "Convert names such as ' Project Cost ' into clean names such as project_cost.",
        "Remove spaces and harmonize labels such as In - Progress into In-Progress.",
        "Transform Project Cost and Project Benefit from text to numeric values.",
        "Convert Completion% from text format such as '77%' into numeric values.",
        "Convert Start Date and End Date into date variables.",
        "Create net benefit, benefit-cost ratio, duration days, and risk flag.",
        "Check missing values, duplicates, inconsistent values and data types.",
        "Use the IQR rule to flag unusual cost, benefit, ratio or duration values.",
        "Calculate total projects, total cost, total benefit, average completion and risk counts.",
        "Analyze the portfolio by status, department, region, project type, manager and phase.",
        "Add concrete text interpretations below each graph or table.",
    ],
    "Why it matters for the project": [
        "Creates the working dataframe.",
        "Avoids coding errors caused by spaces and inconsistent column names.",
        "Makes categories reliable for charts and filters.",
        "Allows sums, averages, ratios and financial KPIs.",
        "Allows progress analysis and delivery monitoring.",
        "Allows duration analysis and time-based charts.",
        "Creates the core indicators used in the dashboard.",
        "Ensures the analysis is based on reliable data.",
        "Prevents unusual values from being ignored or misinterpreted.",
        "Provides the executive overview of the portfolio.",
        "Answers the main PMO and coordination questions.",
        "Makes the dashboard decision-oriented, not only descriptive.",
    ]
})
st.dataframe(processing_steps, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# KPI calculations
# ------------------------------------------------------------
total_projects = len(filtered)
total_cost = filtered["project_cost"].sum()
total_benefit = filtered["project_benefit"].sum()
net_benefit = filtered["net_benefit"].sum()
global_bcr = total_benefit / total_cost if total_cost > 0 else np.nan
avg_completion = filtered["completion_pct"].mean()
completed_projects = int((filtered["status"] == "Completed").sum())
cancelled_projects = int((filtered["status"] == "Cancelled").sum())
on_hold_projects = int((filtered["status"] == "On-Hold").sum())
high_complexity_projects = int((filtered["complexity"] == "High").sum())
risk_projects = int((filtered["risk_flag"] == "To Monitor").sum())

# ------------------------------------------------------------
# Section 1: Portfolio overview
# ------------------------------------------------------------
section_header(
    "1",
    "What is the overall size and financial value of the project portfolio?",
    "This section summarizes the number of projects, total costs, expected benefits, net benefit and the overall benefit-cost ratio."
)

cols = st.columns(5)
cols[0].metric("Total Projects", number_fmt(total_projects))
cols[1].metric("Total Cost", money_millions(total_cost))
cols[2].metric("Expected Benefit", money_millions(total_benefit))
cols[3].metric("Net Benefit", money_millions(net_benefit))
cols[4].metric("Benefit-Cost Ratio", ratio_fmt(global_bcr))

interpretation_box(
    "Interpretation",
    f"The filtered portfolio contains <b>{total_projects}</b> projects with a total cost of <b>{money_millions(total_cost)}</b> and expected benefits of <b>{money_millions(total_benefit)}</b>. The global benefit-cost ratio is <b>{ratio_fmt(global_bcr)}</b>, meaning that each dollar invested is associated with approximately <b>{ratio_fmt(global_bcr)}</b> dollars of expected benefit. This is positive at the portfolio level, but the following sections check whether this value is concentrated in specific departments, statuses or project types."
)

# ------------------------------------------------------------
# Section 2: Status and delivery monitoring
# ------------------------------------------------------------
section_header(
    "2",
    "What is the current delivery status of the projects?",
    "This section checks whether the portfolio is mainly completed, in progress, cancelled or on hold."
)

status_summary = (
    filtered.groupby("status", as_index=False)
    .agg(projects=("project_name", "count"), avg_completion=("completion_pct", "mean"), total_cost=("project_cost", "sum"))
    .sort_values("projects", ascending=False)
)

col1, col2 = st.columns([1.15, 1])
with col1:
    fig = px.bar(
        status_summary,
        x="status",
        y="projects",
        text="projects",
        title="Number of Projects by Status",
        labels={"status": "Status", "projects": "Number of Projects"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="Projects")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.dataframe(status_summary.assign(total_cost_m=status_summary["total_cost"].map(money_millions), avg_completion_pct=status_summary["avg_completion"].map(pct_fmt))[["status", "projects", "avg_completion_pct", "total_cost_m"]], use_container_width=True, hide_index=True)

if total_projects > 0:
    cancelled_share = cancelled_projects / total_projects * 100
    on_hold_share = on_hold_projects / total_projects * 100
else:
    cancelled_share = on_hold_share = 0
interpretation_box(
    "Interpretation",
    f"The portfolio includes <b>{completed_projects}</b> completed projects, <b>{cancelled_projects}</b> cancelled projects and <b>{on_hold_projects}</b> on-hold projects. Cancelled projects represent <b>{cancelled_share:.1f}%</b> of the filtered portfolio and on-hold projects represent <b>{on_hold_share:.1f}%</b>. For a PMO, this is a direct delivery signal: the dashboard should not only show value, but also identify where planned projects are not being delivered."
)

# ------------------------------------------------------------
# Section 3: Department concentration
# ------------------------------------------------------------
section_header(
    "3",
    "Which departments concentrate the most projects, costs and expected benefits?",
    "This section identifies the departments with the highest project volume, budget exposure and expected value."
)

dept_summary = (
    filtered.groupby("department", as_index=False)
    .agg(
        projects=("project_name", "count"),
        total_cost=("project_cost", "sum"),
        total_benefit=("project_benefit", "sum"),
        avg_completion=("completion_pct", "mean"),
        cancelled=("status", lambda x: (x == "Cancelled").sum()),
        on_hold=("status", lambda x: (x == "On-Hold").sum()),
    )
)
dept_summary["net_benefit"] = dept_summary["total_benefit"] - dept_summary["total_cost"]
dept_summary["benefit_cost_ratio"] = dept_summary["total_benefit"] / dept_summary["total_cost"]
dept_summary = dept_summary.sort_values("total_cost", ascending=False)

tab1, tab2, tab3 = st.tabs(["Budget exposure", "Project count", "Status heatmap"])
with tab1:
    fig = px.bar(
        dept_summary,
        x="department",
        y="total_cost",
        text=dept_summary["total_cost"].map(lambda x: f"{x/1_000_000:.1f}M"),
        title="Total Project Cost by Department",
        labels={"department": "Department", "total_cost": "Total Cost"},
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    fig = px.bar(
        dept_summary.sort_values("projects", ascending=False),
        x="department",
        y="projects",
        text="projects",
        title="Number of Projects by Department",
        labels={"department": "Department", "projects": "Projects"},
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    heat = filtered.pivot_table(index="department", columns="status", values="project_name", aggfunc="count", fill_value=0)
    fig = px.imshow(
        heat,
        text_auto=True,
        aspect="auto",
        title="Heatmap: Project Status by Department",
        labels=dict(x="Status", y="Department", color="Projects"),
    )
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    dept_summary.assign(
        total_cost_m=dept_summary["total_cost"].map(money_millions),
        total_benefit_m=dept_summary["total_benefit"].map(money_millions),
        net_benefit_m=dept_summary["net_benefit"].map(money_millions),
        bcr=dept_summary["benefit_cost_ratio"].map(ratio_fmt),
        avg_completion_pct=dept_summary["avg_completion"].map(pct_fmt),
    )[["department", "projects", "total_cost_m", "total_benefit_m", "net_benefit_m", "bcr", "avg_completion_pct", "cancelled", "on_hold"]],
    use_container_width=True,
    hide_index=True,
)

if not dept_summary.empty:
    top_cost_dept = dept_summary.iloc[0]
    best_ratio_dept = dept_summary.sort_values("benefit_cost_ratio", ascending=False).iloc[0]
    interpretation_box(
        "Interpretation",
        f"<b>{top_cost_dept['department']}</b> has the highest budget exposure with <b>{money_millions(top_cost_dept['total_cost'])}</b> in project costs. However, the best relative value is observed in <b>{best_ratio_dept['department']}</b>, with a benefit-cost ratio of <b>{ratio_fmt(best_ratio_dept['benefit_cost_ratio'])}</b>. This distinction matters: the department with the highest budget is not necessarily the department that generates the strongest expected return per dollar invested."
    )

# ------------------------------------------------------------
# Section 4: Project type analysis
# ------------------------------------------------------------
section_header(
    "4",
    "Which project types dominate the portfolio and where is the expected value concentrated?",
    "This section compares income generation, cost reduction, process improvement and working capital projects."
)

type_summary = (
    filtered.groupby("project_type", as_index=False)
    .agg(projects=("project_name", "count"), total_cost=("project_cost", "sum"), total_benefit=("project_benefit", "sum"), avg_bcr=("benefit_cost_ratio", "mean"))
)
type_summary["net_benefit"] = type_summary["total_benefit"] - type_summary["total_cost"]

col1, col2 = st.columns(2)
with col1:
    fig = px.pie(type_summary, names="project_type", values="projects", hole=0.45, title="Project Types by Count")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(
        type_summary.sort_values("net_benefit", ascending=False),
        x="project_type",
        y="net_benefit",
        text=type_summary.sort_values("net_benefit", ascending=False)["net_benefit"].map(lambda x: f"{x/1_000_000:.1f}M"),
        title="Net Benefit by Project Type",
        labels={"project_type": "Project Type", "net_benefit": "Net Benefit"},
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

if not type_summary.empty:
    dominant_type = type_summary.sort_values("projects", ascending=False).iloc[0]
    top_value_type = type_summary.sort_values("net_benefit", ascending=False).iloc[0]
    interpretation_box(
        "Interpretation",
        f"The most frequent project type is <b>{dominant_type['project_type']}</b> with <b>{int(dominant_type['projects'])}</b> projects. The highest estimated net benefit is generated by <b>{top_value_type['project_type']}</b>, with approximately <b>{money_millions(top_value_type['net_benefit'])}</b>. This helps distinguish operational volume from expected financial contribution."
    )

# ------------------------------------------------------------
# Section 5: Value and profitability
# ------------------------------------------------------------
section_header(
    "5",
    "Which projects have the strongest and weakest benefit-cost ratios?",
    "This section identifies high-value projects and projects with weaker expected returns."
)

col1, col2 = st.columns(2)
with col1:
    top_ratio = filtered.sort_values("benefit_cost_ratio", ascending=False).head(10)
    fig = px.bar(
        top_ratio,
        x="benefit_cost_ratio",
        y="project_name",
        orientation="h",
        text="benefit_cost_ratio",
        title="Top 10 Projects by Benefit-Cost Ratio",
        labels={"benefit_cost_ratio": "Benefit-Cost Ratio", "project_name": "Project"},
        hover_data=["department", "status", "project_cost", "project_benefit"]
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    low_ratio = filtered.sort_values("benefit_cost_ratio", ascending=True).head(10)
    fig = px.bar(
        low_ratio,
        x="benefit_cost_ratio",
        y="project_name",
        orientation="h",
        text="benefit_cost_ratio",
        title="Bottom 10 Projects by Benefit-Cost Ratio",
        labels={"benefit_cost_ratio": "Benefit-Cost Ratio", "project_name": "Project"},
        hover_data=["department", "status", "project_cost", "project_benefit"]
    )
    fig.update_layout(yaxis={"categoryorder": "total descending"})
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

if not top_ratio.empty and not low_ratio.empty:
    best = top_ratio.iloc[0]
    weakest = low_ratio.iloc[0]
    interpretation_box(
        "Interpretation",
        f"The strongest expected return is observed for <b>{best['project_name']}</b>, with a benefit-cost ratio of <b>{ratio_fmt(best['benefit_cost_ratio'])}</b>. The weakest ratio is observed for <b>{weakest['project_name']}</b>, with <b>{ratio_fmt(weakest['benefit_cost_ratio'])}</b>. A low ratio does not automatically mean the project is bad, but it indicates that the project creates less expected value per dollar invested and should be compared with higher-return alternatives."
    )

# ------------------------------------------------------------
# Section 6: Complexity and delivery risk
# ------------------------------------------------------------
section_header(
    "6",
    "How complex is the portfolio and where are the main PMO risk signals?",
    "This section combines complexity, status, cost and benefit-cost ratio to identify projects that require closer monitoring."
)

cols = st.columns(4)
cols[0].metric("High Complexity", number_fmt(high_complexity_projects))
cols[1].metric("Cancelled", number_fmt(cancelled_projects))
cols[2].metric("On-Hold", number_fmt(on_hold_projects))
cols[3].metric("To Monitor", number_fmt(risk_projects))

col1, col2 = st.columns(2)
with col1:
    complexity_summary = filtered.groupby("complexity", as_index=False).agg(projects=("project_name", "count"), avg_cost=("project_cost", "mean"), avg_bcr=("benefit_cost_ratio", "mean"))
    fig = px.bar(
        complexity_summary,
        x="complexity",
        y="projects",
        text="projects",
        title="Projects by Complexity Level",
        labels={"complexity": "Complexity", "projects": "Projects"},
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    risk_status_heat = filtered.pivot_table(index="risk_flag", columns="status", values="project_name", aggfunc="count", fill_value=0)
    fig = px.imshow(
        risk_status_heat,
        text_auto=True,
        aspect="auto",
        title="Heatmap: Risk Flag by Status",
        labels=dict(x="Status", y="Risk Flag", color="Projects"),
    )
    st.plotly_chart(fig, use_container_width=True)

risk_table = filtered[filtered["risk_flag"] == "To Monitor"].copy()
risk_table = risk_table.sort_values(["project_cost", "benefit_cost_ratio"], ascending=[False, True]).head(15)
st.markdown("### Projects requiring closer PMO monitoring")
st.dataframe(
    risk_table.assign(
        project_cost_m=risk_table["project_cost"].map(money_millions),
        project_benefit_m=risk_table["project_benefit"].map(money_millions),
        bcr=risk_table["benefit_cost_ratio"].map(ratio_fmt),
        duration=risk_table["duration_days"].map(lambda x: f"{x:.0f} days" if pd.notna(x) else "N/A"),
        completion=risk_table["completion_pct"].map(pct_fmt),
    )[["project_name", "department", "status", "complexity", "project_cost_m", "project_benefit_m", "bcr", "completion", "duration"]],
    use_container_width=True,
    hide_index=True,
)

interpretation_box(
    "Interpretation",
    f"The risk logic flags <b>{risk_projects}</b> projects as requiring closer monitoring. This does not mean that all these projects are failures. It means that they combine one or more PMO warning signs: cancelled or on-hold status, high complexity with above-median cost, low benefit-cost ratio with above-median cost, or longer duration while still active/on hold."
)

# ------------------------------------------------------------
# Section 7: Time and duration analysis
# ------------------------------------------------------------
section_header(
    "7",
    "How does project activity evolve over time and which projects take longer?",
    "This section uses the year, month, start date and end date variables to analyze the portfolio over time."
)

yearly = filtered.groupby("year", as_index=False).agg(projects=("project_name", "count"), total_cost=("project_cost", "sum"), total_benefit=("project_benefit", "sum"))
yearly["net_benefit"] = yearly["total_benefit"] - yearly["total_cost"]

col1, col2 = st.columns(2)
with col1:
    fig = px.line(
        yearly,
        x="year",
        y="projects",
        markers=True,
        title="Number of Projects by Year",
        labels={"year": "Year", "projects": "Projects"},
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(
        filtered,
        x="status",
        y="duration_days",
        title="Project Duration by Status",
        labels={"status": "Status", "duration_days": "Duration in Days"},
    )
    st.plotly_chart(fig, use_container_width=True)

avg_duration = filtered["duration_days"].mean()
max_duration_project = filtered.sort_values("duration_days", ascending=False).head(1)
if not max_duration_project.empty:
    longest = max_duration_project.iloc[0]
    interpretation_box(
        "Interpretation",
        f"The average project duration in the filtered data is <b>{avg_duration:.1f} days</b>. The longest project is <b>{longest['project_name']}</b>, with <b>{longest['duration_days']:.0f} days</b>. Long duration is not automatically an error or a negative outcome, but it should be monitored when combined with high cost, on-hold status or weak benefit-cost ratio."
    )

# ------------------------------------------------------------
# Section 8: Project manager workload and monitoring
# ------------------------------------------------------------
section_header(
    "8",
    "How is the project workload distributed across project managers?",
    "This section is not used to judge individuals. It shows workload concentration and monitoring needs by project manager."
)

manager_summary = (
    filtered.groupby("project_manager", as_index=False)
    .agg(
        projects=("project_name", "count"),
        total_cost=("project_cost", "sum"),
        avg_completion=("completion_pct", "mean"),
        to_monitor=("risk_flag", lambda x: (x == "To Monitor").sum()),
        cancelled=("status", lambda x: (x == "Cancelled").sum()),
    )
    .sort_values("projects", ascending=False)
)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(
        manager_summary,
        x="project_manager",
        y="projects",
        text="projects",
        title="Project Count by Project Manager",
        labels={"project_manager": "Project Manager", "projects": "Projects"},
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.scatter(
        manager_summary,
        x="projects",
        y="total_cost",
        size="to_monitor",
        hover_name="project_manager",
        title="Workload, Budget Exposure and Monitoring Needs",
        labels={"projects": "Number of Projects", "total_cost": "Total Cost", "to_monitor": "Projects to Monitor"},
    )
    st.plotly_chart(fig, use_container_width=True)

if not manager_summary.empty:
    busiest = manager_summary.iloc[0]
    interpretation_box(
        "Interpretation",
        f"<b>{busiest['project_manager']}</b> manages the highest number of projects in the filtered portfolio, with <b>{int(busiest['projects'])}</b> projects. This section should be interpreted as workload monitoring, not personal performance evaluation. In a PMO context, workload concentration can indicate where additional coordination support or follow-up capacity may be needed."
    )

# ------------------------------------------------------------
# Section 9: Outlier and data quality checks
# ------------------------------------------------------------
section_header(
    "9",
    "Are there missing values, duplicates or outliers that may affect the analysis?",
    "This section documents the quality checks before interpreting the results."
)

missing = df.isna().sum().reset_index()
missing.columns = ["Variable", "Missing Values"]
duplicates = int(df.duplicated().sum())
outlier_summary = iqr_outlier_summary(df)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Missing values")
    st.dataframe(missing[missing["Missing Values"] > 0], use_container_width=True, hide_index=True)
    if missing["Missing Values"].sum() == 0:
        st.success("No missing values detected after cleaning.")
    st.markdown(f"**Duplicate rows:** {duplicates}")
with col2:
    st.markdown("### IQR outlier summary")
    st.dataframe(outlier_summary, use_container_width=True, hide_index=True)

outlier_duration = int(outlier_summary.loc[outlier_summary["Variable"] == "Duration Days", "Outliers"].iloc[0])
interpretation_box(
    "Interpretation",
    f"The data quality check is part of the project, not a technical detail. Outliers are not automatically deleted. They are first flagged and interpreted. For example, the IQR rule detects <b>{outlier_duration}</b> duration outlier(s). In PMO analysis, a long project can be a meaningful management signal rather than an error, especially if it is costly or still active."
)

# ------------------------------------------------------------
# Section 10: Practical PMO recommendations
# ------------------------------------------------------------
st.markdown("## 10. Practical PMO Recommendations")

recommendations = [
    "Monitor cancelled and on-hold projects separately from completed and in-progress projects, because they represent delivery and governance signals.",
    "Prioritize detailed review of high-cost projects with weak benefit-cost ratios, especially when they are also high complexity.",
    "Use department-level analysis to identify where budget exposure is concentrated and whether the expected value justifies the investment.",
    "Use the risk table as a monthly PMO follow-up list for projects requiring attention.",
    "Do not remove duration outliers automatically; review them as potential indicators of longer delivery cycles or implementation challenges.",
]
for rec in recommendations:
    st.markdown(f"- {rec}")

# ------------------------------------------------------------
# Downloadable cleaned data
# ------------------------------------------------------------
st.markdown("## Download Cleaned Dataset")
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered cleaned dataset as CSV",
    data=csv,
    file_name="pmo_project_portfolio_cleaned_filtered.csv",
    mime="text/csv",
)

st.caption("Built with Python, pandas, Plotly and Streamlit.")
