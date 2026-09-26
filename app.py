"""National Medicaid coverage continuity dashboard (observed CMS data)."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA = Path(__file__).parent / "data" / "cms_state_month.csv"
METRICS = {
    "Procedural disenrollments / renewals due (%)": "procedural_rate",
    "Renewal rate (%)": "renewal_rate",
    "Automatic renewals / renewals due (%)": "ex_parte_rate",
    "Medicaid + CHIP enrollment": "enrollment",
    "Medicaid enrollment": "medicaid_enrollment",
    "CHIP enrollment": "chip_enrollment",
    "Calls per 1,000 enrolled": "call_rate",
    "Call wait (minutes)": "wait_minutes",
}


@st.cache_data
def load_data():
    frame = pd.read_csv(DATA, dtype={"state": "string", "month": "string"})
    frame["date"] = pd.to_datetime(frame["month"], format="%Y%m")
    for numerator, denominator, output, scale in [
        ("procedural", "renewal_due", "procedural_rate", 100),
        ("renewed", "renewal_due", "renewal_rate", 100),
        ("ex_parte", "renewal_due", "ex_parte_rate", 100),
        ("call_volume", "enrollment", "call_rate", 1000),
    ]:
        frame[output] = frame[numerator].div(frame[denominator].where(frame[denominator] > 0)) * scale
    return frame


@st.cache_data
def load_program_data():
    medicare = pd.read_csv(DATA.parent / "medicare_state.csv", dtype={"state": "string", "month": "string"})
    mltss = pd.read_csv(DATA.parent / "mltss_annual.csv")
    # CMS annual table may append a numbered footnote to a state name.
    mltss["state_name"] = mltss["state_name_source"].str.replace(r"\d+$", "", regex=True).str.strip()
    for field in ["comprehensive_mltss", "mltss_only"]:
        mltss[field] = pd.to_numeric(mltss[field], errors="coerce")
    return medicare, mltss


@st.cache_data
def load_program_history():
    medicare = pd.read_csv(DATA.parent / "medicare_history.csv", dtype={"state": "string", "month": "string"})
    medicare["date"] = pd.to_datetime(medicare["month"], format="%Y%m")
    mltss = pd.read_csv(DATA.parent / "mltss_history.csv")
    mltss["state_name"] = mltss["state_name_source"].str.replace(r"\d+$", "", regex=True).str.strip()
    return medicare, mltss


@st.cache_data
def load_policy_data():
    history = pd.read_csv(DATA.parent / "policy_history.csv", dtype={"state": "string"})
    events = pd.read_csv(DATA.parent / "policy_events.csv", dtype={"state": "string", "month": "string"})
    return history, events


@st.cache_data
def load_renewal_revisions():
    return pd.read_csv(DATA.parent / "renewal_revisions.csv", dtype={"state": "string", "month": "string"})


def forecast_enrollment(series, origin, horizon, model="Seasonal regression"):
    """Forecast one state and program from information available at origin."""
    values = series.dropna().sort_index()
    values = values[values.index <= origin]
    dates = pd.date_range(origin + pd.DateOffset(months=1), periods=horizon, freq="MS")
    if model == "Seasonal regression":
        training = values.loc[origin - pd.DateOffset(months=35):origin]
        if len(training) < 30 or origin not in training.index:
            return None
        # Linear month-to-month trend plus fixed calendar-month effects.
        offset = (training.index.year - origin.year) * 12 + training.index.month - origin.month
        month_effects = np.eye(12)[training.index.month - 1, 1:]
        design = np.column_stack([np.ones(len(training)), offset, month_effects])
        coefficients = np.linalg.lstsq(design, training.to_numpy(dtype=float), rcond=None)[0]
        future_offset = (dates.year - origin.year) * 12 + dates.month - origin.month
        future_design = np.column_stack([np.ones(len(dates)), future_offset,
                                         np.eye(12)[dates.month - 1, 1:]])
        return pd.Series(np.maximum(0, future_design @ coefficients), index=dates)
    if len(values) < 20 or len(values.loc[origin - pd.DateOffset(months=23):origin]) < 20:
        return None
    recent = pd.date_range(origin - pd.DateOffset(months=5), origin, freq="MS")
    changes = []
    for month in recent:
        previous = month - pd.DateOffset(years=1)
        if month in values.index and previous in values.index:
            changes.append(values.loc[month] - values.loc[previous])
    if len(changes) < 4:
        return None
    annual_change = float(pd.Series(changes).median())
    if any(month - pd.DateOffset(years=1) not in values.index for month in dates):
        return None
    return pd.Series({month: max(0, values.loc[month - pd.DateOffset(years=1)] + annual_change)
                      for month in dates}, dtype="float64")


def forecast_backtest(series, origin, model):
    """Score six historical one-month-ahead forecasts, using only data then known."""
    errors = []
    actuals = []
    for target in pd.date_range(origin - pd.DateOffset(months=5), origin, freq="MS"):
        if target not in series.index or pd.isna(series.loc[target]):
            continue
        estimate = forecast_enrollment(series, target - pd.DateOffset(months=1), 1, model)
        if estimate is not None:
            errors.append(abs(float(series.loc[target]) - float(estimate.iloc[0])))
            actuals.append(float(series.loc[target]))
    return (sum(errors) / sum(actuals) * 100, len(errors)) if len(errors) >= 4 and sum(actuals) else (None, len(errors))


st.set_page_config(page_title="Medicaid Coverage Continuity", layout="wide")
df = load_data()
st.title("National Medicaid Coverage Continuity Monitor")
st.caption("App update: September 25, 2026 • Medicare and managed LTSS outlook + responsive map")
st.caption("Observed CMS state reporting | 50 states + DC | Medicaid and CHIP")
st.info("Explore historical renewal friction and operational measures. These data do not estimate losses caused by Public Law 119-21 or identify individuals subject to future requirements.")

months = sorted(
    month for month, group in df.groupby("month")
    if (group["renewal_due"].notna() & group["enrollment"].notna()).sum() >= 45
)
if not months:
    st.error("No reporting month has sufficient data in the bundled snapshot.")
    st.stop()

with st.sidebar:
    st.header("Explore")
    selected_month = st.selectbox("Reporting month", months, index=len(months) - 1,
                                  format_func=lambda value: pd.to_datetime(value, format="%Y%m").strftime("%B %Y"))
    selected_state = st.selectbox("Select state for detail", ["None"] + sorted(df["state"].dropna().unique()),
                                  index=1 + sorted(df["state"].dropna().unique()).index("TX"))
    expanded_only = st.checkbox("Expansion states only")
    st.caption("Rates use beneficiaries due for renewal as the denominator. Missing values stay missing. National percentages are ratios of reported totals.")

period = df[df["month"] == selected_month].copy()
history = df[df["month"] <= selected_month].copy()
if expanded_only:
    period = period[period["expanded"] == "Y"]
    history = history[history["expanded"] == "Y"]
headline_period = period if selected_state == "None" else period[period["state"] == selected_state]
valid = headline_period.dropna(subset=["renewal_due", "procedural"])
valid = valid[valid["renewal_due"] > 0]
due = valid["renewal_due"].sum()
rate = 100 * valid["procedural"].sum() / due if due else None
a, b, c = st.columns(3)
scope = "Reporting states" if selected_state == "None" else f"{selected_state} reporting"
a.metric(scope, f"{len(valid)} of {len(headline_period)}")
b.metric(f"{selected_state if selected_state != 'None' else 'Reporting states'}: renewals due", f"{due:,.0f}")
c.metric(f"{selected_state if selected_state != 'None' else 'Reporting states'}: procedural / due",
         f"{rate:.1f}%" if rate is not None else "N/A")

st.subheader("What changed in the CMS data?")
st.caption("State counts below compare with the immediately preceding month. The investigation queue further down compares with the same month last year.")
previous_month = (pd.to_datetime(selected_month, format="%Y%m") - pd.DateOffset(months=1)).strftime("%Y%m")
if selected_state == "None":
    st.info("Choose a state in the sidebar for a month-by-month explanation and product counts.")
else:
    current_state = period[period["state"] == selected_state]
    previous_state = df[(df["state"] == selected_state) & (df["month"] == previous_month)]
    if current_state.empty:
        st.info("No state record is available in this reporting month.")
    else:
        now = current_state.iloc[0]
        before = previous_state.iloc[0] if not previous_state.empty else None
        measures = [
            ("Medicaid enrolled", "medicaid_enrollment", "people"),
            ("CHIP enrolled", "chip_enrollment", "people"),
            ("Renewals due", "renewal_due", "people"),
            ("Automatically renewed", "ex_parte", "people"),
            ("Procedurally disenrolled", "procedural", "people"),
            ("Renewals pending", "pending", "people"),
            ("Procedural / due", "procedural_rate", "percent"),
            ("Automatic renewal / due", "ex_parte_rate", "percent"),
        ]
        rows = []
        for title, field, unit in measures:
            value = now[field]
            previous = before[field] if before is not None else float("nan")
            rows.append({"Measure": title, "Current": value, "Previous": previous,
                         "Change": value - previous if pd.notna(value) and pd.notna(previous) else float("nan"),
                         "Unit": unit})
        changes = pd.DataFrame(rows)
        for column in ["Current", "Previous", "Change"]:
            changes[column] = changes.apply(lambda row: (f"{row[column]:+,.1f} pp" if column == "Change" else f"{row[column]:,.1f}%")
                if row["Unit"] == "percent" else (f"{row[column]:+,.0f}" if column == "Change" else f"{row[column]:,.0f}")
                if pd.notna(row[column]) else "—", axis=1)
        st.dataframe(changes[["Measure", "Current", "Previous", "Change"]].rename(columns={
            "Current": pd.to_datetime(selected_month, format="%Y%m").strftime("%b %Y"),
            "Previous": pd.to_datetime(previous_month, format="%Y%m").strftime("%b %Y")}),
            hide_index=True, width="stretch")
        st.caption("CMS renewal outcomes are statewide totals, without a Medicaid-versus-CHIP split. The two enrollment rows show product enrollment changes, not product-specific disenrollments. Month-to-month differences also reflect reporting and pending cases.")
        if pd.notna(now["renewal_footnote"]):
            st.warning(f"CMS procedural footnote: {now['renewal_footnote']}")
        if pd.notna(now["renewal_due_footnote"]):
            st.warning(f"CMS renewals-due footnote: {now['renewal_due_footnote']}")

st.subheader("Did CMS revise this month's renewal outcomes?")
if selected_state == "None":
    st.info("Choose a state to compare its original and updated CMS renewal outcomes.")
else:
    revisions = load_renewal_revisions()
    audited = revisions[(revisions["state"] == selected_state) & (revisions["month"] == selected_month)]
    if audited.empty or pd.isna(audited.iloc[0]["updated_due"]):
        st.info("Only the original (O) CMS renewal report is available for this state and month in the bundled snapshot. Treat pending cases and recent rates as provisional.")
    else:
        revision = audited.iloc[0]
        comparisons = [("Renewals due", "due"), ("Procedural disenrollments", "procedural"),
                       ("Pending renewals", "pending"), ("Automatic renewals", "auto_renewed")]
        revision_rows = []
        for title, field in comparisons:
            original, updated = revision[f"original_{field}"], revision[f"updated_{field}"]
            revision_rows.append({"Measure": title, "Original report": original, "Updated report": updated,
                                  "Revision": updated - original if pd.notna(original) and pd.notna(updated) else float("nan")})
        revision_table = pd.DataFrame(revision_rows)
        st.dataframe(revision_table.style.format({"Original report": "{:,.0f}", "Updated report": "{:,.0f}",
            "Revision": "{:+,.0f}"}, na_rep="—"), hide_index=True, width="stretch")
        st.caption("Revisions can resolve initially pending cases or correct earlier reports. An updated count does not identify why any one person changed status.")
    st.markdown("[CMS original and updated eligibility processing data](https://data.medicaid.gov/dataset/5abea2e0-3f8e-4b49-a50d-d63d5fd9103c)")

st.subheader("What changed in state policy or operations?")
policy_history, policy_events = load_policy_data()
st.markdown("**Federal rule context for all states and DC**")
st.write("**2024 Medicaid and CHIP eligibility and enrollment final rule:** Effective June 3, 2024, with different state compliance dates. Public Law 119-21 paused implementation and enforcement of **certain provisions whose compliance dates fell after July 4, 2025**; it did not cancel the whole final rule. The moratorium runs through September 30, 2034. This is applicable federal context for May 2026, not evidence it caused any state's monthly change.")
st.markdown("[CMS final rule fact sheet](https://www.cms.gov/newsroom/fact-sheets/streamlining-medicaid-childrens-health-insurance-program-and-basic-health-program-application) · [CMS bulletin specifying the moratorium and affected provisions](https://www.medicaid.gov/federal-policy-guidance/downloads/cib11182025.pdf)")
st.write("**Community engagement interim final rule:** Issued June 1, 2026. Its work requirement generally starts January 1, 2027, or earlier only if a state elects early implementation. It cannot explain a May 2026 outcome. [CMS rule fact sheet](https://www.cms.gov/newsroom/fact-sheets/medicaid-community-engagement-requirement-certain-individuals-interim-final-rule-comment-period-cms).")
if selected_state == "None":
    st.info("Choose a state to see dated, source-linked policy or operations events.")
else:
    state_name = df.loc[df["state"] == selected_state, "state_name"].iloc[0]
    dated_events = policy_events[(policy_events["state"] == selected_state) & (policy_events["month"] == selected_month)]
    st.markdown(f"**{state_name}: verified policy evidence for {pd.to_datetime(selected_month, format='%Y%m').strftime('%B %Y')}**")
    if dated_events.empty:
        st.info("No state-specific eligibility or renewal event has been verified for this month in the project's register. State policy research is incomplete; this is not evidence that no change occurred.")
    else:
        for _, event in dated_events.iterrows():
            st.write(f"**Verified record:** {event['event']}")
            st.caption(f"Affected group: {event['affected_population']}")
            st.caption("Document published: " + (str(event["source_published_date"]) if pd.notna(event["source_published_date"]) else "Not recorded"))
            st.caption("Effective date: " + (str(event["effective_date"]) if pd.notna(event["effective_date"]) else "Not independently established"))
            st.caption("State implementation: " + (str(event["state_implementation_date"]) if pd.notna(event["state_implementation_date"]) else "Not independently established"))
            st.caption(event["relationship"])
            st.markdown(f"[Official document]({event['source_url']})")
    with st.expander("Find and verify additional state documents"):
        st.write(f"Search for {state_name} and check each document's subject, approval date, effective date, and affected population. A federal final rule does not imply a separate state final rule.")
        st.markdown("[CMS State Plan Amendments (filter by state)](https://www.medicaid.gov/medicaid/medicaid-state-plan-amendments) · [CMS Section 1115 state waivers](https://www.medicaid.gov/medicaid/section-1115-demonstrations)")
        st.caption("State regulations and agency notices may also document changes. Only a relevant, dated source should be added to the verified record above; a document's approval date can differ from its effective date.")
    state_history = policy_history[policy_history["state"] == selected_state]
    if not state_history.empty:
        entry = state_history.iloc[0]
        with st.expander("Earlier CMS renewal strategy record (December 2024)"):
            st.write(f"CMS listed {int(entry['total_waivers'])} approved unwinding flexibilities for {state_name}, including {int(entry['auto_renewal_waivers'])} designed to increase automatic renewal. This is a December 2024 snapshot, **not** a change dated {selected_month} and not evidence these strategies affected this month's outcomes.")
            st.markdown(f"[CMS state-by-state waiver table]({entry['source_url']})")
    if selected_state == "MA" and selected_month == "202605":
        st.caption("For Massachusetts May 2026, the state eligibility memo and eligibility letter listings were checked; no relevant May renewal-rule change was established. The listed work and six-month renewal changes start in 2027.")
        st.markdown("[MassHealth eligibility memos](https://www.mass.gov/lists/eligibility-operations-memos-by-year) · [2026 eligibility letters](https://www.mass.gov/lists/2026-masshealth-eligibility-letters) · [Federal change dates](https://www.mass.gov/federal-changes-affecting-masshealth-medicaid-members)")
st.caption("Policy register coverage is incomplete. An event's date, affected group, and source are shown only when verified; calendar alignment is not a causal estimate.")

st.subheader("Four program landscape")
medicare, mltss = load_program_data()
medicare_month = medicare[medicare["month"] == selected_month]
mltss_year = int(mltss["year"].max())
st.caption(f"Medicaid and CHIP: {pd.to_datetime(selected_month, format='%Y%m').strftime('%B %Y')}. "
           f"Medicare: {pd.to_datetime(selected_month, format='%Y%m').strftime('%B %Y')} if available. "
           f"Managed LTSS: {mltss_year} annual snapshot. Program counts overlap; do not add them together.")
if medicare_month.empty:
    st.warning("Medicare is available in the bundled snapshot for May 2026 only. Choose May 2026 to compare state counts.")
profile = period[["state", "state_name", "medicaid_enrollment", "chip_enrollment"]].merge(
    medicare_month[["state", "medicare_enrollment", "medicare_medicaid_duals"]], on="state", how="left")
profile = profile.merge(mltss[["state_name", "comprehensive_mltss", "mltss_only", "notes"]],
                        on="state_name", how="left")
if selected_state != "None":
    state_program = profile[profile["state"] == selected_state]
    if not state_program.empty:
        item = state_program.iloc[0]
        state_products = pd.DataFrame([
            {"Product / category": "Medicaid", "Enrolled": item["medicaid_enrollment"], "Period": selected_month},
            {"Product / category": "CHIP", "Enrolled": item["chip_enrollment"], "Period": selected_month},
            {"Product / category": "Medicare", "Enrolled": item["medicare_enrollment"],
             "Period": selected_month if pd.notna(item["medicare_enrollment"]) else "Unavailable for this month"},
            {"Product / category": "Comprehensive managed LTSS", "Enrolled": item["comprehensive_mltss"], "Period": str(mltss_year)},
            {"Product / category": "MLTSS-only", "Enrolled": item["mltss_only"], "Period": str(mltss_year)},
        ])
        st.markdown(f"**{item['state_name']}: product snapshot**")
        state_products["Enrolled"] = state_products["Enrolled"].map(
            lambda value: f"{value:,.0f}" if pd.notna(value) else "Unavailable")
        st.dataframe(state_products, hide_index=True, width="stretch")
        st.caption("Managed LTSS is part of Medicaid and may overlap with Medicare. Do not add these rows together. Only Medicaid and CHIP have a comparable prior-month enrollment in the bundled data.")
choices = {"Medicaid": "medicaid_enrollment", "Medicare": "medicare_enrollment",
           "CHIP": "chip_enrollment", "Comprehensive managed LTSS (annual)": "comprehensive_mltss",
           "MLTSS-only program (annual)": "mltss_only"}
program = st.selectbox("Program map", list(choices), key="program_map")
field = choices[program]
mapped = profile[profile[field].notna()]
st.caption(f"{len(mapped)} of {len(profile)} jurisdictions have a numeric {program} count for this view. "
           "Blank means unavailable; it does not mean zero.")
if not mapped.empty:
    fig = px.choropleth(profile, locations="state", locationmode="USA-states", color=field,
                         scope="usa", hover_name="state_name", color_continuous_scale="Teal",
                         labels={field: f"{program} enrolled"})
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=5, b=0))
    st.plotly_chart(fig, width="stretch")
st.dataframe(profile.rename(columns={"state_name": "State", "medicaid_enrollment": "Medicaid",
              "medicare_enrollment": "Medicare", "chip_enrollment": "CHIP",
              "medicare_medicaid_duals": "Medicare-Medicaid duals",
              "comprehensive_mltss": "Comprehensive MLTSS", "mltss_only": "MLTSS only"})[
                  ["State", "Medicaid", "Medicare", "CHIP",
                   "Medicare-Medicaid duals", "Comprehensive MLTSS", "MLTSS only"]],
             hide_index=True, width="stretch")
st.caption("Managed LTSS is a Medicaid service delivery category, including beneficiaries who may also have Medicare. "
           "Comprehensive MLTSS and MLTSS-only are shown separately because most MLTSS-only cells are unavailable. "
           "This table does not count all LTSS users, including fee-for-service LTSS. "
           "State footnotes and missing values require review before drawing conclusions.")

st.subheader("Enrollment forecast by state and program")
st.caption("Exploratory Medicaid, CHIP, and Medicare enrollment forecasts. Choose a state in the sidebar to update "
           "the figures below. These models do not estimate renewal outcomes or policy effects.")
forecast_product = st.selectbox("Program to forecast", ["Medicaid", "CHIP", "Medicare"], key="forecast_product")
forecast_field = {"Medicaid": "medicaid_enrollment", "CHIP": "chip_enrollment",
                  "Medicare": "medicare_enrollment"}[forecast_product]
forecast_model = st.selectbox("Projection model", ["Seasonal regression", "Seasonal baseline"], key="forecast_model",
                              help="Regression fits a monthly trend and calendar-month effects over up to 36 months. "
                                   "Baseline repeats last year's count plus a recent median annual change.")
forecast_horizon = st.slider("Months ahead", 3, 12, 6, key="forecast_horizon")
origin = pd.to_datetime(selected_month, format="%Y%m")
medicare_history, mltss_history = load_program_history()
if forecast_product == "Medicare":
    names = df[["state", "state_name"]].drop_duplicates("state")
    forecast_source = medicare_history.merge(names, on="state", how="left")
    forecast_source = forecast_source[forecast_source["date"] <= origin]
    if expanded_only:
        forecast_source = forecast_source[forecast_source["state"].isin(period["state"])]
else:
    forecast_source = history
forecast_rows = []
backtests = {}
focus = selected_state if selected_state != "None" else None
forecast_history = forecast_source if focus is None else forecast_source[forecast_source["state"] == focus]
for state_code, group in forecast_history.groupby("state"):
    series = group.set_index("date")[forecast_field].sort_index()
    projection = forecast_enrollment(series, origin, forecast_horizon, forecast_model)
    if projection is None:
        continue
    backtests[state_code] = forecast_backtest(series, origin, forecast_model)
    for date, estimate in projection.items():
        forecast_rows.append({"state": state_code, "state_name": group["state_name"].iloc[-1],
                              "date": date, "forecast": estimate})
forecast_frame = pd.DataFrame(forecast_rows)
if forecast_frame.empty:
    st.info("This reporting month does not have enough complete monthly enrollment history for a forecast. "
            "Choose a later month or refresh the CMS data.")
else:
    included = forecast_frame["state"].nunique()
    st.caption(f"{included} jurisdiction(s) with sufficient history through {origin:%B %Y}. "
               "Missing jurisdictions are excluded; expansion-only filtering applies when selected. "
               "Recent preliminary counts may be revised by CMS.")
    current = forecast_frame[forecast_frame["date"] == forecast_frame["date"].min()]["forecast"].sum()
    last = forecast_frame[forecast_frame["date"] == forecast_frame["date"].max()]["forecast"].sum()
    st.metric(f"{focus or 'Forecastable states'}: {forecast_product} next month", f"{current:,.0f}")
    st.metric(f"{focus or 'Forecastable states'}: {forecast_product} at {forecast_horizon} months", f"{last:,.0f}")
    if focus and focus in backtests:
        error, n = backtests[focus]
        if error is not None:
            st.metric(f"{focus} historical one-month error", f"{error:.1f}%",
                      help=f"Weighted absolute percentage error across {n} prior one-month forecasts. "
                           "This is a diagnostic, not a future prediction interval.")
            alternative = "Seasonal baseline" if forecast_model == "Seasonal regression" else "Seasonal regression"
            comparison_error, comparison_n = forecast_backtest(
                forecast_history[forecast_history["state"] == focus].set_index("date")[forecast_field].sort_index(),
                origin, alternative)
            if comparison_error is not None:
                st.caption(f"Same holdout, {alternative.lower()}: {comparison_error:.1f}% "
                           f"across {comparison_n} months. Lower historical error is preferable, "
                           "but neither check establishes long-horizon accuracy.")
                if error > comparison_error:
                    st.warning(f"{forecast_model} had higher recent one-month error than {alternative.lower()} "
                               f"for {focus}. Compare both before interpreting the projection.")
        else:
            st.caption(f"{focus}: too few historical holdout months to report forecast error.")
    display_states = list(forecast_frame["state"].unique())
    observed = forecast_source[forecast_source["state"].isin(display_states) & (forecast_source["date"] >= origin - pd.DateOffset(months=24))]
    observed = observed.groupby("date", as_index=False)[forecast_field].sum(min_count=1)
    projected = forecast_frame[forecast_frame["state"].isin(display_states)].groupby("date", as_index=False)["forecast"].sum()
    chart_forecast = go.Figure()
    chart_forecast.add_trace(go.Scatter(x=observed["date"], y=observed[forecast_field],
                                        mode="lines+markers", name="Observed CMS enrollment"))
    chart_forecast.add_trace(go.Scatter(x=projected["date"], y=projected["forecast"],
                                        mode="lines+markers", line=dict(dash="dash"), name="Exploratory forecast"))
    chart_forecast.update_layout(yaxis_title=f"{forecast_product} enrolled", xaxis_title="Month",
                                 hovermode="x unified", height=360)
    st.plotly_chart(chart_forecast, width="stretch")
    table = forecast_frame.pivot(index=["state", "state_name"], columns="date", values="forecast")
    table.columns = [date.strftime("%b %Y") for date in table.columns]
    table = table.reset_index().rename(columns={"state": "Code", "state_name": "State"})
    st.dataframe(table.style.format({column: "{:,.0f}" for column in table.columns[2:]}, na_rep="—"),
                 hide_index=True, width="stretch")
    st.caption("The all-state chart sums only forecastable jurisdictions. The historical error checks one-month "
               "predictions, not the entire selected horizon. Forecast errors, reporting revisions, "
               "policy changes, and population shifts can make actual enrollment differ substantially.")

st.subheader("Managed LTSS annual outlook")
st.caption("Managed LTSS is annual Medicaid reporting, latest 2024. This is a separate yearly outlook; "
           "it cannot be interpreted as a monthly forecast or added to Medicaid and Medicare totals.")
ltss_category = st.selectbox("Managed LTSS category", ["Comprehensive managed LTSS", "MLTSS-only"], key="ltss_category")
ltss_field = {"Comprehensive managed LTSS": "comprehensive_mltss", "MLTSS-only": "mltss_only"}[ltss_category]
ltss_state = focus or st.selectbox("State for annual outlook", sorted(df["state"].dropna().unique()),
                                   index=sorted(df["state"].dropna().unique()).index("TX"), key="ltss_state")
state_names = df[["state", "state_name"]].drop_duplicates("state")
state_name = state_names.loc[state_names["state"] == ltss_state, "state_name"].iloc[0]
annual = mltss_history[mltss_history["state_name"] == state_name].copy()
annual[ltss_field] = pd.to_numeric(annual[ltss_field], errors="coerce")
annual = annual.dropna(subset=[ltss_field]).sort_values("year")
if len(annual) < 6 or annual["year"].max() != mltss_history["year"].max():
    st.info(f"{state_name} has insufficient recent annual {ltss_category} observations for an outlook. "
            "Missing CMS cells are not assumed to be zero.")
else:
    years = annual["year"].to_numpy(dtype=float)
    counts = annual[ltss_field].to_numpy(dtype=float)
    future_years = np.arange(int(years[-1]) + 1, int(years[-1]) + 4)
    slope, intercept = np.polyfit(years - years[-1], counts, 1)
    annual_projection = np.maximum(0, intercept + slope * (future_years - years[-1]))
    annual_plot = go.Figure()
    annual_plot.add_trace(go.Scatter(x=years.astype(int), y=counts, mode="lines+markers", name="CMS annual count"))
    annual_plot.add_trace(go.Scatter(x=future_years, y=annual_projection, mode="lines+markers",
                                     line=dict(dash="dash"), name="Linear annual outlook"))
    annual_plot.update_layout(yaxis_title=f"{ltss_category} enrolled", xaxis_title="Year", height=330)
    st.plotly_chart(annual_plot, width="stretch")
    st.dataframe(pd.DataFrame({"Year": future_years, "Exploratory enrolled": annual_projection}).style.format(
        {"Exploratory enrolled": "{:,.0f}"}), hide_index=True, width="stretch")
    st.caption(f"{len(annual)} nonmissing annual observations. Linear trend only; state program changes, "
               "reporting gaps, and older source years may make these projections unreliable. "
               "The first outlook year is already past the latest monthly data and is not a CMS observation.")

st.subheader("State map and ranking")
label = st.selectbox("Measure shown in the map and ranking below", list(METRICS), key="map_metric")
metric = METRICS[label]
map_fig = px.choropleth(period, locations="state", locationmode="USA-states", color=metric,
                         scope="usa", hover_name="state_name", hover_data={metric: ":,.1f", "state": False},
                         color_continuous_scale="Blues", labels={metric: label})
map_fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(map_fig, width="stretch")

st.subheader("State comparison")
columns = {
    "state_name": "State", "enrollment": "Enrolled", "medicaid_enrollment": "Medicaid enrolled",
    "chip_enrollment": "CHIP enrolled", "renewal_due": "Due",
    "procedural": "Procedural", "procedural_rate": "Procedural %",
    "renewal_rate": "Renewed %", "call_rate": "Calls / 1k",
    "wait_minutes": "Wait min", "enrollment_version": "Enrollment version",
    "renewal_version": "Renewal version",
}
ranking = period.sort_values(metric, ascending=False, na_position="last")[list(columns)].rename(columns=columns)
st.dataframe(ranking.style.format({"Enrolled": "{:,.0f}", "Medicaid enrolled": "{:,.0f}",
                                   "CHIP enrolled": "{:,.0f}", "Due": "{:,.0f}",
                                   "Procedural": "{:,.0f}", "Procedural %": "{:.1f}",
                                   "Renewed %": "{:.1f}", "Calls / 1k": "{:.1f}",
                                   "Wait min": "{:.1f}"}, na_rep="—"),
             hide_index=True, width="stretch")


st.subheader("Enrollment by program")
st.caption("Medicaid and CHIP counts are reported separately by CMS. Renewal outcomes below are statewide totals and cannot be assigned to either program from these files.")
program_counts = headline_period[["medicaid_enrollment", "chip_enrollment"]].sum(min_count=1)
covered = headline_period[["medicaid_enrollment", "chip_enrollment"]].notna().sum()
left, right = st.columns(2)
left.metric(f"{selected_state if selected_state != 'None' else 'Reporting states'}: Medicaid enrollment",
            f"{program_counts['medicaid_enrollment']:,.0f}",
            help=f"{covered['medicaid_enrollment']} of {len(headline_period)} jurisdictions reporting")
right.metric(f"{selected_state if selected_state != 'None' else 'Reporting states'}: CHIP enrollment",
             f"{program_counts['chip_enrollment']:,.0f}",
             help=f"{covered['chip_enrollment']} of {len(headline_period)} jurisdictions reporting")

st.subheader("Investigation queue")
st.caption("Compare the selected month with the same month one year earlier. Changes are descriptive; a policy or operational cause requires separate evidence. Automatic renewal % uses everyone due for renewal as its denominator.")
prior_month = (pd.to_datetime(selected_month, format="%Y%m") - pd.DateOffset(years=1)).strftime("%Y%m")
prior = df[df["month"] == prior_month].copy()
if expanded_only:
    prior = prior[prior["expanded"] == "Y"]
if prior.empty:
    st.info("A comparable month one year earlier is unavailable.")
else:
    fields = ["state", "renewal_due", "procedural", "pending", "ex_parte", "procedural_rate", "ex_parte_rate", "renewal_footnote", "ex_parte_footnote"]
    comparison = period[fields + ["state_name"]].merge(prior[fields], on="state", how="inner", suffixes=("", "_prior"))
    comparison = comparison.dropna(subset=["renewal_due", "procedural", "renewal_due_prior", "procedural_prior"])
    comparison = comparison[(comparison["renewal_due"] > 0) & (comparison["renewal_due_prior"] > 0)].copy()
    comparison["Rate change (pp)"] = comparison["procedural_rate"] - comparison["procedural_rate_prior"]
    comparison["Count change"] = comparison["procedural"] - comparison["procedural_prior"]
    comparison["Auto renewal change (pp)"] = comparison["ex_parte_rate"] - comparison["ex_parte_rate_prior"]
    comparison["Pending change"] = comparison["pending"] - comparison["pending_prior"]
    comparison["Review CMS notes"] = comparison[["renewal_footnote", "renewal_footnote_prior", "ex_parte_footnote", "ex_parte_footnote_prior"]].notna().any(axis=1)
    queue_cols = {"state_name": "State", "procedural_rate": "Procedural %", "Rate change (pp)": "Rate change (pp)",
                  "procedural": "Procedural count", "Count change": "Count change", "renewal_due": "Due",
                  "ex_parte_rate": "Auto renewal %", "Auto renewal change (pp)": "Auto renewal change (pp)",
                  "pending": "Pending", "Pending change": "Pending change", "Review CMS notes": "Review CMS notes"}
    queue = comparison.sort_values("Rate change (pp)", ascending=False)[list(queue_cols)].rename(columns=queue_cols)
    st.dataframe(queue.style.format({"Procedural %": "{:.1f}", "Rate change (pp)": "{:+.1f}",
        "Procedural count": "{:,.0f}", "Count change": "{:+,.0f}", "Due": "{:,.0f}",
        "Auto renewal %": "{:.1f}", "Auto renewal change (pp)": "{:+.1f}",
        "Pending": "{:,.0f}", "Pending change": "{:+,.0f}"}, na_rep="—"), hide_index=True, width="stretch")
    st.caption("A flagged note may concern either comparison month or the automatic renewal measure. Read its exact wording under Definitions and source notes, or in the CMS source.")
    if selected_state != "None":
        selected = comparison[comparison["state"] == selected_state]
        if not selected.empty:
            item = selected.iloc[0]
            st.write(f"**{item['state_name']} ({selected_month[:4]} vs {prior_month[:4]})**: procedural rate {item['procedural_rate_prior']:.1f}% → {item['procedural_rate']:.1f}% ({item['Rate change (pp)']:+.1f} pp); automatic renewal rate {item['ex_parte_rate_prior']:.1f}% → {item['ex_parte_rate']:.1f}% ({item['Auto renewal change (pp)']:+.1f} pp)." if pd.notna(item["ex_parte_rate"]) and pd.notna(item["ex_parte_rate_prior"]) else f"**{item['state_name']}**: automatic renewal data unavailable for this comparison.")
            for note_col, title in [("renewal_footnote", "Current procedural note"), ("renewal_footnote_prior", "Prior procedural note"), ("ex_parte_footnote", "Current automatic renewal note"), ("ex_parte_footnote_prior", "Prior automatic renewal note")]:
                if pd.notna(item[note_col]):
                    st.caption(f"{title}: {item[note_col]}")

st.subheader("National trend and selected state")
trend = history.dropna(subset=["renewal_due", "procedural"])
trend = trend[trend["renewal_due"] > 0]
summary = trend.groupby("date", as_index=False).agg(procedural=("procedural", "sum"),
                                                     renewal_due=("renewal_due", "sum"),
                                                     reporting_states=("state", "nunique"))
summary["rate"] = 100 * summary["procedural"] / summary["renewal_due"]
chart = go.Figure()
chart.add_trace(go.Scatter(x=summary["date"], y=summary["rate"], mode="lines+markers",
                           name="National reporting states", customdata=summary["reporting_states"],
                           hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<br>%{customdata} reporting states<extra></extra>"))
if selected_state != "None":
    state_trend = trend[trend["state"] == selected_state]
    chart.add_trace(go.Scatter(x=state_trend["date"], y=state_trend["procedural_rate"],
                               mode="lines+markers", name=selected_state))
chart.update_layout(yaxis_title="Procedural disenrollments / renewal due (%)", yaxis_rangemode="tozero",
                    xaxis_title="Reporting month", hovermode="x unified", height=380)
st.plotly_chart(chart, width="stretch")
if selected_state != "None":
    operation = history[(history["state"] == selected_state) & history["renewal_due"].gt(0)].copy()
    if not operation.empty and operation["ex_parte_rate"].notna().any():
        operation["pending_rate"] = 100 * operation["pending"] / operation["renewal_due"]
        view = operation[operation["date"] >= operation["date"].max() - pd.DateOffset(months=30)]
        ops_chart = go.Figure()
        for column, title in [("ex_parte_rate", "Automatic renewal / due"),
                              ("procedural_rate", "Procedural disenrollment / due"),
                              ("pending_rate", "Pending renewal / due")]:
            ops_chart.add_trace(go.Scatter(x=view["date"], y=view[column], mode="lines+markers", name=title))
        ops_chart.update_layout(title=f"{selected_state}: renewal outcomes over time", yaxis_title="Share of renewals due (%)",
                                xaxis_title="Reporting month", hovermode="x unified", height=360)
        st.plotly_chart(ops_chart, width="stretch")
        st.caption("These outcomes use the same due-for-renewal denominator. They are parts of the renewal cohort, not proof that a change in one caused a change in another.")

with st.expander("Definitions, caveats, and source notes"):
    st.write("A procedural disenrollment is an administrative renewal outcome. It does not prove ineligibility or a future law's effect. The national rate divides the sum of procedural disenrollments by the sum due for renewal among states reporting both values in each month. Reporting coverage changes over time.")
    st.write("Call center measures have state-specific footnotes and may not be comparable. Reporting can be revised. Check source notes before using any state comparison in a presentation.")
    st.dataframe(period[["state_name", "enrollment_version", "renewal_version", "enrollment_footnote", "medicaid_footnote", "chip_footnote", "renewal_due_footnote", "renewal_footnote", "ex_parte_footnote"]],
                 hide_index=True, width="stretch")
    st.markdown("[CMS enrollment data](https://data.medicaid.gov/dataset/6165f45b-ca93-5bb5-9d06-db29c692a360) · [CMS renewal data](https://data.medicaid.gov/dataset/5abea2e0-3f8e-4b49-a50d-d63d5fd9103c) · [CMS policy guidance](https://www.medicaid.gov/resources-for-states/working-families-tax-cut-legislation/community-engagement)")
    st.markdown("[CMS Medicare monthly enrollment](https://data.cms.gov/summary-statistics-on-beneficiary-enrollment/medicare-and-medicaid-reports/medicare-monthly-enrollment) · [CMS managed LTSS annual enrollment](https://data.medicaid.gov/dataset/5394bcab-c748-5e4b-af07-b5bf77ed3aa3)")
