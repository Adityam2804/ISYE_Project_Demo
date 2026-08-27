import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ISYE 2530 Project Journey Demo",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"

DATASETS = {
    "Supply Chain & Logistics": {
        "file": "supply_chain_orders.csv",
        "subtitle": "Supply Chain & Logistics: From Replenishment Data to a Defensible Shipment Plan",
        "scenario": (
            "A distribution network receives replenishment requests from several warehouses. "
            "The next outbound truck cannot satisfy every request, so the team must decide which "
            "replenishment orders should move first."
        ),
        "decision_makers": [
            "Distribution planning manager",
            "Warehouse operations manager",
            "Supply chain analyst",
        ],
        "objectives": [
            "Prioritize high-risk replenishment orders under truck capacity",
            "Protect the most customer demand with limited capacity",
            "Balance stockout risk and customer demand impact",
        ],
        "horizons": ["Next outbound truck", "Next 24 hours", "This week"],
        "decision_object": "one replenishment order",
        "evidence": ["stockout risk", "customer demand impact", "shipment volume", "product category", "warehouse"],
        "limitations": ["supplier delays", "lane disruptions", "service-level contracts", "inventory substitutions", "transportation cost"],
        "raw_to_standard": {
            "order_id": "request_id",
            "warehouse": "location",
            "product": "item",
            "category": "category",
            "units_needed": "quantity_needed",
            "space_per_unit": "weight_per_unit",
            "cost_per_unit": "cost_per_unit",
            "customer_demand": "people_served",
            "stockout_risk": "urgency",
            "need_by_date": "request_date",
        },
        "category_map": {
            "Servo Motors": "Components",
            "Sensor Kits": "Electronics",
            "Safety Gloves": "Safety",
            "Packing Film": "Packaging",
            "Bearings": "Components",
        },
        "entity_labels": {"location": "Warehouse", "item": "Product", "request": "ReplenishmentRequest"},
        "metric_labels": {"people": "Demand represented", "urgency": "Stockout risk", "capacity": "Truck capacity"},
        "query_names": ["Highest stockout-risk orders", "Demand by warehouse", "Units required by category"],
        "issue_labels": {
            "Location whitespace": "Warehouse whitespace",
            "Missing category": "Missing product category",
            "Missing people served": "Missing customer demand",
            "Invalid quantity": "Invalid units needed",
            "Urgency outside 1–10": "Stockout risk outside 1–10",
            "Duplicate request ID": "Duplicate order ID",
        },
    },
    "Healthcare Systems": {
        "file": "healthcare_service_requests.csv",
        "subtitle": "Healthcare Systems: From Service Requests to a Defensible Capacity Decision",
        "scenario": (
            "A regional care network receives non-identifiable service-capacity requests from several facilities. "
            "Limited mobile clinical capacity means not every request can be served in the next allocation window."
        ),
        "decision_makers": [
            "Regional capacity coordinator",
            "Clinical operations manager",
            "Health systems analyst",
        ],
        "objectives": [
            "Prioritize urgent service requests under limited mobile capacity",
            "Serve the largest number of patients with available capacity",
            "Balance clinical urgency and patients affected",
        ],
        "horizons": ["Next allocation window", "Next 48 hours", "This week"],
        "decision_object": "one service-capacity request",
        "evidence": ["clinical urgency", "patients affected", "resource hours", "service line", "facility"],
        "limitations": ["clinical complexity", "staff skill mix", "travel time", "equity obligations", "local surge capacity"],
        "raw_to_standard": {
            "case_id": "request_id",
            "facility": "location",
            "service": "item",
            "department": "category",
            "slots_needed": "quantity_needed",
            "hours_per_slot": "weight_per_unit",
            "cost_per_slot": "cost_per_unit",
            "patients_affected": "people_served",
            "clinical_urgency": "urgency",
            "request_date": "request_date",
        },
        "category_map": {
            "Cardiology Consults": "Cardiology",
            "Imaging Slots": "Radiology",
            "Infusion Sessions": "Oncology",
            "Physical Therapy": "Rehabilitation",
            "Respiratory Therapy": "Pulmonary",
        },
        "entity_labels": {"location": "Facility", "item": "Service", "request": "ServiceRequest"},
        "metric_labels": {"people": "Patients represented", "urgency": "Clinical urgency", "capacity": "Available service hours"},
        "query_names": ["Highest-urgency service requests", "Patients affected by facility", "Slots requested by department"],
        "issue_labels": {
            "Location whitespace": "Facility whitespace",
            "Missing category": "Missing department",
            "Missing people served": "Missing patients affected",
            "Invalid quantity": "Invalid slots needed",
            "Urgency outside 1–10": "Clinical urgency outside 1–10",
            "Duplicate request ID": "Duplicate case ID",
        },
    },
    "Humanitarian Operations": {
        "file": "humanitarian_requests.csv",
        "subtitle": "Humanitarian Operations: From Messy Requests to a Defensible Relief Shipment",
        "scenario": (
            "A humanitarian logistics coordinator receives requests for water, food, medical, shelter, "
            "and hygiene supplies from several districts. One shipment cannot carry everything."
        ),
        "decision_makers": [
            "Humanitarian logistics coordinator",
            "Field operations manager",
            "Regional program director",
        ],
        "objectives": [
            "Prioritize urgent requests under shipment capacity",
            "Reach as many people as possible",
            "Balance urgency and people served",
        ],
        "horizons": ["Next shipment", "Next 48 hours", "This week"],
        "decision_object": "one relief request",
        "evidence": ["urgency", "people served", "shipment weight", "category", "location"],
        "limitations": ["road access", "equity commitments", "perishability", "donor restrictions", "security conditions", "local inventory"],
        "raw_to_standard": {},
        "category_map": {
            "Water Kits": "Water",
            "Rice Packs": "Food",
            "Medical Kits": "Medical",
            "Blankets": "Shelter",
            "Hygiene Kits": "Hygiene",
            "Tarps": "Shelter",
        },
        "entity_labels": {"location": "Location", "item": "ReliefItem", "request": "Request"},
        "metric_labels": {"people": "People represented", "urgency": "Urgency", "capacity": "Shipment capacity"},
        "query_names": ["Highest-urgency requests", "Requests by location", "Total quantity by category"],
        "issue_labels": {
            "Location whitespace": "Location whitespace",
            "Missing category": "Missing category",
            "Missing people served": "Missing people served",
            "Invalid quantity": "Invalid quantity",
            "Urgency outside 1–10": "Urgency outside 1–10",
            "Duplicate request ID": "Duplicate request ID",
        },
    },
}


@st.cache_data
def load_dataset(dataset_name):
    cfg = DATASETS[dataset_name]
    return pd.read_csv(DATA_DIR / cfg["file"])


def infer_dataset_name(uploaded_name, df):
    """Infer which teaching track an uploaded example file belongs to."""
    uploaded_name = (uploaded_name or "").lower()
    cols = set(df.columns)

    for name, cfg in DATASETS.items():
        if cfg["file"].lower() == uploaded_name:
            return name

    if {"order_id", "warehouse", "product", "stockout_risk"}.issubset(cols):
        return "Supply Chain & Logistics"
    if {"case_id", "facility", "service", "clinical_urgency"}.issubset(cols):
        return "Healthcare Systems"
    if {"request_id", "location", "item", "urgency"}.issubset(cols):
        return "Humanitarian Operations"
    return None


def standardize_raw(raw, cfg):
    df = raw.copy()
    if cfg["raw_to_standard"]:
        df = df.rename(columns=cfg["raw_to_standard"])
    return df


def clean_data(raw_standard, cfg):
    """Illustrative cleaning with an audit log and removed-row record."""
    df = raw_standard.copy()
    log = []

    before = df["location"].astype(str)
    df["location"] = df["location"].astype(str).str.strip()
    changed = int((before != df["location"]).sum())
    if changed:
        log.append({"Cleaning action": "Trim location text", "Rows affected": changed, "Reason": "Remove accidental leading/trailing spaces"})

    df["request_date"] = pd.to_datetime(df["request_date"], errors="coerce")
    numeric_cols = ["request_id", "quantity_needed", "weight_per_unit", "cost_per_unit", "people_served", "urgency"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_cat = df["category"].isna() | (df["category"].fillna("").astype(str).str.strip() == "")
    cat_count = int(missing_cat.sum())
    if cat_count:
        df.loc[missing_cat, "category"] = df.loc[missing_cat, "item"].map(cfg["category_map"])
        log.append({"Cleaning action": "Fill missing category", "Rows affected": cat_count, "Reason": "Recover category from the known item/service definition"})

    missing_people = df["people_served"].isna()
    people_count = int(missing_people.sum())
    if people_count:
        item_medians = df.groupby("item")["people_served"].transform("median")
        overall_median = df["people_served"].median()
        df.loc[missing_people, "people_served"] = item_medians[missing_people].fillna(overall_median)
        log.append({"Cleaning action": "Impute missing impact measure", "Rows affected": people_count, "Reason": "Use median for the same item/service; fallback to overall median"})

    invalid_urgency = df["urgency"].notna() & ~df["urgency"].between(1, 10)
    urgency_count = int(invalid_urgency.sum())
    if urgency_count:
        df.loc[invalid_urgency, "urgency"] = df.loc[invalid_urgency, "urgency"].clip(1, 10)
        log.append({"Cleaning action": "Correct urgency/risk range", "Rows affected": urgency_count, "Reason": "The demo scale is defined from 1 to 10"})

    remove_mask = (
        df["request_id"].isna() | df["location"].isna() | df["item"].isna()
        | df["quantity_needed"].isna() | (df["quantity_needed"] <= 0) | df["request_date"].isna()
    )
    removed_invalid = df[remove_mask].copy()
    if len(removed_invalid):
        removed_invalid["removal_reason"] = "Missing/invalid required field or non-positive quantity"
        log.append({"Cleaning action": "Remove invalid records", "Rows affected": len(removed_invalid), "Reason": "Required fields must be valid; quantity must be positive"})
    df = df[~remove_mask].copy()

    duplicate_mask = df.duplicated(subset=["request_id"], keep="first")
    removed_dupes = df[duplicate_mask].copy()
    if len(removed_dupes):
        removed_dupes["removal_reason"] = "Duplicate record ID"
        log.append({"Cleaning action": "Remove duplicate records", "Rows affected": len(removed_dupes), "Reason": "The ID should uniquely identify one decision record"})
    df = df[~duplicate_mask].copy()

    removed = pd.concat([removed_invalid, removed_dupes], ignore_index=True)
    return df.reset_index(drop=True), pd.DataFrame(log), removed


def build_relational_tables(clean):
    locations = clean[["location"]].drop_duplicates().sort_values("location").reset_index(drop=True)
    locations.insert(0, "location_id", range(1, len(locations) + 1))

    items = clean[["item", "category", "weight_per_unit", "cost_per_unit"]].drop_duplicates().sort_values("item").reset_index(drop=True)
    items.insert(0, "item_id", range(1, len(items) + 1))

    requests = clean.merge(locations, on="location", how="left").merge(
        items, on=["item", "category", "weight_per_unit", "cost_per_unit"], how="left"
    )[["request_id", "location_id", "item_id", "quantity_needed", "people_served", "urgency", "request_date"]].copy()
    return locations, items, requests


def calculate_metrics(clean, urgency_weight=60):
    df = clean.copy()
    people_weight = 100 - urgency_weight
    df["total_weight"] = df["quantity_needed"] * df["weight_per_unit"]
    df["estimated_cost"] = df["quantity_needed"] * df["cost_per_unit"]
    max_people = max(df["people_served"].max(), 1)
    df["priority_score"] = ((urgency_weight / 100) * (df["urgency"] / 10.0) + (people_weight / 100) * (df["people_served"] / max_people)) * 100
    df["benefit_points"] = df["priority_score"].round().astype(int)
    return df


def knapsack_select(df, capacity):
    if df.empty:
        out = df.copy(); out["selected"] = False; return out
    weights = df["total_weight"].round().astype(int).tolist()
    values = df["benefit_points"].astype(int).tolist()
    cap = int(capacity); n = len(df)
    dp = [[0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(cap + 1):
            dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - w] + v) if w <= c else dp[i - 1][c]
    chosen = []; c = cap
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            chosen.append(i - 1); c -= weights[i - 1]
    out = df.copy(); out["selected"] = False
    if chosen: out.loc[out.index[chosen], "selected"] = True
    return out


def issue_rows(raw_standard, issue_key):
    if issue_key == "Missing category":
        mask = raw_standard["category"].isna() | (raw_standard["category"].fillna("").astype(str).str.strip() == "")
    elif issue_key == "Missing people served":
        mask = pd.to_numeric(raw_standard["people_served"], errors="coerce").isna()
    elif issue_key == "Invalid quantity":
        mask = pd.to_numeric(raw_standard["quantity_needed"], errors="coerce") <= 0
    elif issue_key == "Urgency outside 1–10":
        urgency = pd.to_numeric(raw_standard["urgency"], errors="coerce"); mask = urgency.notna() & ~urgency.between(1, 10)
    elif issue_key == "Duplicate request ID":
        mask = raw_standard.duplicated(subset=["request_id"], keep=False)
    else:
        mask = raw_standard["location"].astype(str) != raw_standard["location"].astype(str).str.strip()
    return raw_standard[mask]


def build_demo_database(locations, items, requests, cfg):
    conn = sqlite3.connect(":memory:")
    locations.to_sql(cfg["entity_labels"]["location"], conn, index=False, if_exists="replace")
    items.to_sql(cfg["entity_labels"]["item"], conn, index=False, if_exists="replace")
    req = requests.copy(); req["request_date"] = req["request_date"].astype(str)
    req.to_sql(cfg["entity_labels"]["request"], conn, index=False, if_exists="replace")
    return conn


def sql_queries(cfg):
    ltab = cfg["entity_labels"]["location"]
    itab = cfg["entity_labels"]["item"]
    rtab = cfg["entity_labels"]["request"]
    q1, q2, q3 = cfg["query_names"]
    return {
        q1: f'''SELECT r.request_id, l.location, i.item, r.urgency, r.people_served\nFROM {rtab} r\nJOIN {ltab} l ON r.location_id = l.location_id\nJOIN {itab} i ON r.item_id = i.item_id\nORDER BY r.urgency DESC, r.people_served DESC;''',
        q2: f'''SELECT l.location, COUNT(*) AS request_count, SUM(r.people_served) AS impact_represented\nFROM {rtab} r\nJOIN {ltab} l ON r.location_id = l.location_id\nGROUP BY l.location\nORDER BY request_count DESC, impact_represented DESC;''',
        q3: f'''SELECT i.category, SUM(r.quantity_needed) AS total_quantity_requested\nFROM {rtab} r\nJOIN {itab} i ON r.item_id = i.item_id\nGROUP BY i.category\nORDER BY total_quantity_requested DESC;''',
    }


def recommendation_table(solution):
    out = solution.copy()
    out["recommendation"] = out["selected"].map({True: "Include in next allocation", False: "Hold for review / later allocation"})
    return out.sort_values(["selected", "priority_score"], ascending=[False, False])


STAGES = ["Overview", "M1 — Define the Problem", "M2 — Prepare the Information", "M3 — Support the Decision", "M4 — Build the User Experience", "Final Project View"]
if "stage" not in st.session_state: st.session_state.stage = STAGES[0]
if "dataset_name" not in st.session_state: st.session_state.dataset_name = "Humanitarian Operations"
if "uploaded_dataset" not in st.session_state: st.session_state.uploaded_dataset = None
if "uploaded_filename" not in st.session_state: st.session_state.uploaded_filename = None


def go_next():
    i = STAGES.index(st.session_state.stage)
    if i < len(STAGES) - 1: st.session_state.stage = STAGES[i + 1]


def go_previous():
    i = STAGES.index(st.session_state.stage)
    if i > 0: st.session_state.stage = STAGES[i - 1]


# ---------- Shared shell ----------
st.title("ISYE 2530 — Student Project Journey")
with st.sidebar:
    st.markdown("### Walkthrough controls")
    mode = st.radio("Experience", ["Guided walkthrough", "Free exploration"])
    selected_stage = st.radio("Project stage", STAGES, index=STAGES.index(st.session_state.stage))
    st.session_state.stage = selected_stage
    st.markdown("---")
    st.caption("Illustrative teaching demo. The three datasets are intentionally small and contain planted data-quality issues so each milestone is visible in class.")

stage = st.session_state.stage
stage_no = STAGES.index(stage)
st.progress(stage_no / (len(STAGES) - 1), text=f"Journey progress: {stage_no} of {len(STAGES)-1} milestones completed")

# Dataset selection is available globally once chosen in M1, but M1 is where the student formally chooses and inspects it.
dataset_name = st.session_state.dataset_name
cfg = DATASETS[dataset_name]
if st.session_state.uploaded_dataset is not None:
    raw_original = st.session_state.uploaded_dataset.copy()
else:
    raw_original = load_dataset(dataset_name)
raw = standardize_raw(raw_original, cfg)
clean, cleaning_log, removed_rows = clean_data(raw, cfg)
locations, items, requests = build_relational_tables(clean)


# ---------- Overview ----------
if stage == "Overview":
    st.header("One Project, Four Connected Milestones")
    st.write("Students pick a dataset file, inspect it, and then carry that same dataset from inspection through a working decision-support interface.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Example datasets", 3)
    c2.metric("Milestones", 4)
    c3.metric("Final deliverable", "Interactive app")

    cards = [
        ("M1", "Choose + Define", "Pick and inspect a dataset file first; then define the user and recurring decision(s)."),
        ("M2", "Prepare", "Clean, validate, relationally structure, and query the chosen dataset."),
        ("M3", "Decide", "Turn the prepared information into transparent decision support."),
        ("M4", "Communicate", "Build an interface where a user can inspect, challenge, and act."),
    ]
    cols = st.columns(4)
    for col, (m, verb, body) in zip(cols, cards):
        with col:
            st.markdown(f"### {m} · {verb}"); st.write(body)

    st.markdown("### Three example project tracks")
    for name, dc in DATASETS.items():
        with st.expander(name):
            preview = load_dataset(name)
            st.write(dc["scenario"])
            st.dataframe(preview.head(6), width="stretch", hide_index=True)

    st.success("Day-one message: the project starts with choosing and understanding data—not with coding—and every later milestone builds on that choice.")


# ---------- M1 ----------
elif stage == "M1 — Define the Problem":
    st.header("Milestone 1 — Define the Problem")
    st.write("Start by choosing a dataset and inspecting what it contains. Only after that should the team define the user, recurring decision(s), objectives, and evidence.")

    st.markdown("### 1. Pick a dataset file")
    st.write("Choose a CSV file first. The app will inspect the file and determine which project context it represents.")

    uploaded_file = st.file_uploader(
        "Choose dataset CSV",
        type=["csv"],
        accept_multiple_files=False,
        help="For this demo, upload one of the three example CSVs supplied in the datasets folder.",
    )

    with st.expander("Need one of the three example datasets?"):
        st.caption("These are the same files bundled under the project datasets/ folder.")
        sample_cols = st.columns(3)
        for col, (name, dc) in zip(sample_cols, DATASETS.items()):
            sample_path = DATA_DIR / dc["file"]
            with col:
                st.markdown(f"**{name}**")
                st.download_button(
                    "Download sample CSV",
                    data=sample_path.read_bytes(),
                    file_name=dc["file"],
                    mime="text/csv",
                    key=f"download_{dc['file']}",
                    use_container_width=True,
                )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
            st.stop()

        inferred = infer_dataset_name(uploaded_file.name, uploaded_df)
        if inferred is None:
            st.error("This file does not match any of the three example dataset schemas. Please upload one of the supplied example CSV files.")
            st.stop()

        changed = (
            st.session_state.uploaded_filename != uploaded_file.name
            or st.session_state.dataset_name != inferred
        )
        st.session_state.dataset_name = inferred
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.uploaded_dataset = uploaded_df
        if changed:
            st.rerun()

    if st.session_state.uploaded_dataset is None:
        st.info("Pick a CSV above to begin Milestone 1.")
        st.stop()

    chosen = st.session_state.dataset_name
    cfg = DATASETS[chosen]
    raw_original = st.session_state.uploaded_dataset.copy()
    raw = standardize_raw(raw_original, cfg)
    clean, cleaning_log, removed_rows = clean_data(raw, cfg)
    locations, items, requests = build_relational_tables(clean)

    st.success(f"Loaded **{st.session_state.uploaded_filename}** · detected context: **{chosen}**")
    st.markdown("### 2. Inspect the dataset before defining the problem")
    a, b, c, d = st.columns(4)
    a.metric("Rows", len(raw_original))
    b.metric("Columns", len(raw_original.columns))
    c.metric("Missing cells", int(raw_original.isna().sum().sum()))
    d.metric("Duplicate IDs", int(raw.duplicated(subset=["request_id"]).sum()))
    st.dataframe(raw_original, width="stretch", hide_index=True)

    with st.expander("What do these fields appear to support?"):
        st.write(cfg["scenario"])
        st.write("Potential evidence: " + ", ".join(cfg["evidence"]) + ".")

    st.markdown("### 3. Define the decision context from the chosen data")
    left, right = st.columns([1, 1.35])
    with left:
        persona = st.selectbox("Choose the decision-maker", cfg["decision_makers"], key=f"persona_{chosen}")
        objective = st.selectbox("Primary decision objective", cfg["objectives"], key=f"objective_{chosen}")
        horizon = st.radio("Decision horizon", cfg["horizons"], horizontal=True, key=f"horizon_{chosen}")
    with right:
        st.markdown("#### Project definition generated from those choices")
        st.markdown(f"- **Dataset / track:** {chosen}")
        st.markdown(f"- **Intended user:** {persona}")
        st.markdown(f"- **Recurring Decision(s):** {objective}")
        st.markdown(f"- **Decision horizon:** {horizon}")
        st.markdown(f"- **Decision object:** {cfg['decision_object']}")
        st.markdown(f"- **Evidence:** {', '.join(cfg['evidence'])}")
        st.markdown(f"- **Known limitation:** the dataset does not capture {', '.join(cfg['limitations'][:3])}")

    st.info("M1 sequence: **choose → inspect → judge suitability → define the decision problem**. The problem statement should be grounded in what the selected data can actually support.")


# ---------- M2 ----------
elif stage == "M2 — Prepare the Information":
    st.header("Milestone 2 — Prepare the Information")
    st.caption(f"Working dataset: {dataset_name}")
    st.write("The cleaning rules, relational structure, and SQL questions now follow the dataset chosen in M1.")

    m1, m2, m3, m4 = st.columns(4)
    missing_values = int(raw.isna().sum().sum()) + int((raw["category"].fillna("").astype(str).str.strip() == "").sum())
    m1.metric("Raw rows", len(raw)); m2.metric("Missing values", missing_values)
    m3.metric("Duplicate IDs", int(raw.duplicated(subset=["request_id"]).sum())); m4.metric("Rows retained", len(clean), delta=len(clean) - len(raw))

    st.markdown("### 1. Find a data-quality problem")
    issue_map = cfg["issue_labels"]
    displayed_issues = list(issue_map.values())
    selected_issue_label = st.selectbox("Inspect one issue", displayed_issues)
    issue_key = next(k for k, v in issue_map.items() if v == selected_issue_label)
    affected = issue_rows(raw, issue_key)
    st.caption(f"{len(affected)} row(s) affected")
    st.dataframe(affected, width="stretch", hide_index=True)

    st.markdown("### 2. See the cleaning decisions and audit trail")
    st.dataframe(cleaning_log, width="stretch", hide_index=True)

    st.markdown("### 3. Validate the cleaned result")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Clean rows", len(clean))
    v2.metric("Missing required values", int(clean[["request_id", "location", "item", "quantity_needed"]].isna().sum().sum()))
    v3.metric("Duplicate IDs", int(clean.duplicated(subset=["request_id"]).sum()))
    v4.metric("Risk/urgency outside 1–10", int((~clean["urgency"].between(1, 10)).sum()))

    st.markdown("### 4. Turn the spreadsheet into relational tables")
    a, b, c = st.columns(3)
    with a:
        st.markdown(f"**{cfg['entity_labels']['location']}**"); st.dataframe(locations, width="stretch", hide_index=True)
    with b:
        st.markdown(f"**{cfg['entity_labels']['item']}**"); st.dataframe(items, width="stretch", hide_index=True)
    with c:
        st.markdown(f"**{cfg['entity_labels']['request']}**"); st.dataframe(requests, width="stretch", hide_index=True)

    st.markdown("### 5. Query the chosen dataset")
    queries = sql_queries(cfg)
    query_name = st.selectbox("Run a prepared SQL question", list(queries.keys()))
    st.code(queries[query_name], language="sql")
    conn = build_demo_database(locations, items, requests, cfg)
    query_result = pd.read_sql_query(queries[query_name], conn); conn.close()
    st.dataframe(query_result, width="stretch", hide_index=True)


# ---------- M3 ----------
elif stage == "M3 — Support the Decision":
    st.header("Milestone 3 — Support the Decision")
    st.caption(f"Working dataset: {dataset_name}")
    st.write(f"The objective now reflects the {dataset_name.lower()} context. Change the assumptions and see whether the recommended allocation changes.")

    control, result_col = st.columns([0.9, 1.6])
    with control:
        urgency_weight = st.slider(f"Weight on {cfg['metric_labels']['urgency'].lower()} (%)", 0, 100, 60, 5)
        st.caption(f"Impact weight automatically becomes {100 - urgency_weight}%")
        max_cap = max(500, int((clean["quantity_needed"] * clean["weight_per_unit"]).sum() * 0.8))
        default_cap = min(max_cap, max(100, int(max_cap * 0.55)))
        capacity = st.slider(cfg["metric_labels"]["capacity"], 100, max_cap, default_cap, 50)
        selected_locations = st.multiselect(f"Eligible {cfg['entity_labels']['location'].lower()}s", sorted(clean["location"].unique()), default=sorted(clean["location"].unique()))
        selected_categories = st.multiselect("Eligible categories", sorted(clean["category"].unique()), default=sorted(clean["category"].unique()))

    metrics = calculate_metrics(clean, urgency_weight)
    scenario = metrics[metrics["location"].isin(selected_locations) & metrics["category"].isin(selected_categories)].copy()
    solution = knapsack_select(scenario, capacity); selected = solution[solution["selected"]]

    with result_col:
        a, b, c, d = st.columns(4)
        a.metric("Records selected", len(selected))
        b.metric("Capacity used", f"{int(selected['total_weight'].sum()):,} / {capacity:,}" if not selected.empty else f"0 / {capacity:,}")
        c.metric(cfg["metric_labels"]["people"], f"{int(selected['people_served'].sum()):,}" if not selected.empty else "0")
        d.metric("Estimated cost", f"${selected['estimated_cost'].sum():,.0f}" if not selected.empty else "$0")
        if not scenario.empty:
            st.dataframe(recommendation_table(solution)[["request_id", "location", "item", "urgency", "people_served", "total_weight", "priority_score", "recommendation"]], width="stretch", hide_index=True)
        else:
            st.warning("Choose at least one eligible location and category.")

    st.markdown("### What-if: compare against the default 60/40 policy")
    default_solution = knapsack_select(calculate_metrics(clean, 60)[lambda x: x["location"].isin(selected_locations) & x["category"].isin(selected_categories)], capacity)
    default_ids = set(default_solution.loc[default_solution["selected"], "request_id"].tolist()); scenario_ids = set(selected["request_id"].tolist())
    changed_ids = sorted(default_ids.symmetric_difference(scenario_ids))
    if urgency_weight == 60: st.info("This is the default 60/40 policy. Move the weight slider to test sensitivity.")
    elif changed_ids: st.warning(f"Changing the weights changes the recommendation for ID(s): {changed_ids}")
    else: st.success("The recommendation remains stable under this weight change for the current capacity and filters.")


# ---------- M4 ----------
elif stage == "M4 — Build the User Experience":
    st.header("Milestone 4 — Build the User Experience")
    st.caption(f"Working dataset: {dataset_name}")
    st.write("The interface, labels, filters, and review language follow the domain selected in M1.")

    left, right = st.columns([0.85, 1.55])
    with left:
        max_cap = max(500, int((clean["quantity_needed"] * clean["weight_per_unit"]).sum() * 0.8))
        capacity = st.slider(cfg["metric_labels"]["capacity"], 100, max_cap, min(max_cap, int(max_cap * 0.55)), 50, key="m4_capacity")
        urgency_weight = st.slider(f"{cfg['metric_labels']['urgency']} emphasis", 0, 100, 60, 5, key="m4_urgency")
        categories = sorted(clean["category"].unique())
        selected_categories = st.multiselect("Categories", categories, default=categories)
        minimum_urgency = st.slider(f"Minimum {cfg['metric_labels']['urgency'].lower()} to consider", 1, 10, 1)

    metrics = calculate_metrics(clean, urgency_weight)
    filtered = metrics[metrics["category"].isin(selected_categories) & (metrics["urgency"] >= minimum_urgency)].copy()
    with right:
        if filtered.empty:
            st.warning("No records match the current filters.")
        else:
            solution = knapsack_select(filtered, capacity); selected = solution[solution["selected"]]
            a, b, c, d = st.columns(4)
            a.metric("Considered", len(filtered)); b.metric("Recommended", len(selected)); c.metric("Capacity used", f"{int(selected['total_weight'].sum()):,}"); d.metric(cfg["metric_labels"]["people"], f"{int(selected['people_served'].sum()):,}")
            st.markdown("#### Recommended next allocation")
            st.dataframe(selected[["request_id", "location", "item", "category", "urgency", "people_served", "total_weight", "priority_score"]].sort_values("priority_score", ascending=False), width="stretch", hide_index=True)

    if not filtered.empty:
        st.markdown("### Human review and override")
        solution = knapsack_select(filtered, capacity)
        review = recommendation_table(solution)[["request_id", "location", "item", "urgency", "people_served", "priority_score", "recommendation"]].copy()
        review["human_review"] = "Accept model recommendation"
        edited = st.data_editor(review, width="stretch", hide_index=True, disabled=["request_id", "location", "item", "urgency", "people_served", "priority_score", "recommendation"], column_config={"human_review": st.column_config.SelectboxColumn("Human review", options=["Accept model recommendation", "Escalate for review", "Override — include", "Override — hold"])}, key="human_review_editor")
        overrides = edited[edited["human_review"] != "Accept model recommendation"]
        if overrides.empty: st.info("No human overrides recorded. Change one row to demonstrate that the model supports—not replaces—judgment.")
        else: st.warning(f"{len(overrides)} record(s) have a human review action different from the model default.")

        st.markdown(f"### Priority by {cfg['entity_labels']['location'].lower()}")
        st.bar_chart(solution.groupby("location")["priority_score"].mean().sort_values(ascending=False))
        limitations = st.multiselect("Which missing considerations should the user be reminded about?", cfg["limitations"], default=cfg["limitations"][:3])
        if limitations: st.warning("Not represented in this demo model: " + ", ".join(limitations) + ".")


# ---------- Final ----------
else:
    st.header("Final Project View — Put the Whole System Together")
    st.caption(f"Working dataset: {dataset_name}")
    st.write(cfg["scenario"])

    max_cap = max(500, int((clean["quantity_needed"] * clean["weight_per_unit"]).sum() * 0.8))
    c0, c1, c2 = st.columns(3)
    with c0: capacity = st.number_input(cfg["metric_labels"]["capacity"], 100, max_cap, min(max_cap, int(max_cap * 0.55)), 50)
    with c1: urgency_weight = st.slider(f"{cfg['metric_labels']['urgency']} weight", 0, 100, 60, 5, key="final_urgency")
    with c2: min_urgency = st.slider(f"Minimum {cfg['metric_labels']['urgency'].lower()}", 1, 10, 1, key="final_min_urgency")

    eligible = calculate_metrics(clean, urgency_weight)
    eligible = eligible[eligible["urgency"] >= min_urgency].copy()
    solution = knapsack_select(eligible, capacity); selected = solution[solution["selected"]]

    a, b, c, d, e = st.columns(5)
    a.metric("Capacity", f"{capacity:,}"); b.metric("Capacity used", f"{int(selected['total_weight'].sum()):,}")
    c.metric("Selected records", len(selected)); d.metric(cfg["metric_labels"]["people"], f"{int(selected['people_served'].sum()):,}"); e.metric("Estimated cost", f"${selected['estimated_cost'].sum():,.0f}")
    st.dataframe(recommendation_table(solution)[["request_id", "location", "item", "urgency", "people_served", "total_weight", "estimated_cost", "priority_score", "recommendation"]], width="stretch", hide_index=True)
    st.download_button("Download recommended allocation CSV", data=selected.to_csv(index=False).encode("utf-8"), file_name=f"{dataset_name.lower().replace(' ', '_').replace('&', 'and')}_recommended_allocation.csv", mime="text/csv")

    st.markdown("### What students built across the semester")
    st.markdown("""
1. **M1 — Choose + Define:** pick and inspect a dataset file, judge suitability, identify the user, recurring decision(s), objective, and limitations
2. **M2 — Prepare:** clean, validate, relationally structure, and query the selected dataset
3. **M3 — Decide:** create transparent measures, assumptions, what-if analysis, and optimization
4. **M4 — Communicate:** turn the decision logic into an inspectable, usable interface with human review
    """)
    st.error("Responsible-use reminder: this application supports a decision; it does not replace professional judgment or guarantee that every real-world consideration is represented.")


# ---------- Guided navigation ----------
if mode == "Guided walkthrough":
    st.markdown("---")
    p, n = st.columns(2)
    with p:
        if stage_no > 0: st.button("← Previous stage", on_click=go_previous, use_container_width=True)
    with n:
        if stage_no < len(STAGES) - 1: st.button("Next stage →", on_click=go_next, use_container_width=True, type="primary")
