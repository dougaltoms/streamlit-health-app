from snowflake.snowpark import Session
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="Customer Insights Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------- Connection ----------
@st.cache_resource(ttl=1)
def get_session() -> Session:
    try:
        connection_parameters = {
            "account": st.secrets["account"],
            "user": st.secrets["user"],
            "password": st.secrets["password"],
            "role": st.secrets["role"],
            "warehouse": st.secrets["warehouse"],
            "database": st.secrets["database"],
            "schema": st.secrets["schema"],
        }
        session = Session.builder.configs(connection_parameters).create()
        return session
    except Exception as e:
        st.error(f"Failed to connect to Snowflake. Error: {e}")
        st.stop()

# ---------- App Header ----------
st.title(f"Customer Insights Dashboard")
st.subheader('Internal Usage')

# ---------- Data Load ----------
try:
    snowpark_session = get_session()
    TABLE_NAME = "CUSTOMER_BI_DATA"

    try:
        df = snowpark_session.table(TABLE_NAME).to_pandas()
        if df.empty:
            st.warning(f"The table `{TABLE_NAME}` appears to be empty.")
            st.stop()

        # Types
        if 'START_DATE_LOCAL' in df.columns:
            df['START_DATE_LOCAL'] = pd.to_datetime(df['START_DATE_LOCAL'], errors='coerce')
        if 'SIGNUP_DATE' in df.columns:
            df['SIGNUP_DATE'] = pd.to_datetime(df['SIGNUP_DATE'], errors='coerce')

        if 'AGE' not in df.columns:
            df['AGE'] = np.random.randint(18, 81, size=len(df))

        for col in ['COUNTRY', 'CITY', 'SUBSCRIPTION_PLAN', 'SUBSCRIPTION_STATUS', 'CHURN_REASON']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')

    except Exception:
        st.warning(f"The table `{TABLE_NAME}` does not seem to exist.")
        st.stop()

except Exception as e:
    st.error(f"An error occurred during app initialization: {e}")
    st.stop()


# ---------- Sidebar Filters ----------
st.sidebar.header("Dashboard Filters")

min_age = int(df['AGE'].min()); max_age = int(df['AGE'].max())
age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

continent_map = {
    'USA': 'North America', 'UK': 'Europe', 'Canada': 'North America',
    'Australia': 'Australia', 'France': 'Europe', 'Japan': 'Asia',
    'Germany': 'Europe', 'Brazil': 'South America', 'India': 'Asia', 'Netherlands': 'Europe'
}
df['CONTINENT'] = df['COUNTRY'].map(continent_map).fillna('Other')
all_continents = df['CONTINENT'].dropna().unique().tolist()
selected_continents = st.sidebar.segmented_control(
    "Select Continents",
    options=all_continents, selection_mode="multi", default=all_continents
)

all_activity_types = df['ACTIVITY_TYPE'].dropna().unique().tolist() if 'ACTIVITY_TYPE' in df.columns else []
selected_activities = st.sidebar.pills(
    'Sport Type',
    options=all_activity_types, selection_mode="multi", default=all_activity_types
) if all_activity_types else []

mask = (
    (df['AGE'].between(age_range[0], age_range[1])) &
    (df['CONTINENT'].isin(selected_continents))
)
if len(selected_activities) and 'ACTIVITY_TYPE' in df.columns:
    mask &= df['ACTIVITY_TYPE'].isin(selected_activities)

filtered_df = df[mask].copy()

# ---------- Chat to Data (unchanged structure except schema text) ----------
with st.expander('Chat to your data'):
    # st.header("Ask me anything")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help?"}
        ]

    @st.cache_data()
    def generate_sql_query(user_question: str):
        prompt = f"""
You are an expert SQL assistant specializing in customer & subscription BI (revenue, churn, signups, support).
Generate a single, valid Snowflake SQL query. Output ONLY the SQL.

Schema for '{TABLE_NAME}':

create or replace TABLE CUSTOMER_BI_DATA (
  USER_ID NUMBER(38,0),
  AGE NUMBER(38,0),
  CITY VARCHAR(50),
  COUNTRY VARCHAR(50),
  SIGNUP_DATE DATE,
  SUBSCRIPTION_PLAN VARCHAR(20),
  SUBSCRIPTION_STATUS VARCHAR(20),
  MONTHLY_REVENUE FLOAT,
  CHURN_REASON VARCHAR(50),
  SUPPORT_TICKETS_LAST_YEAR NUMBER(38,0),
  ACTIVITY_ID NUMBER(38,0),
  ACTIVITY_TYPE VARCHAR(50),
  START_DATE_LOCAL TIMESTAMP_NTZ(9),
  DEVICE_TYPE VARCHAR(20),
  APP_VERSION VARCHAR(10),
  DISTANCE_KM FLOAT,
  MOVING_TIME_SEC NUMBER(38,0),
  ELAPSED_TIME_SEC NUMBER(38,0),
  TOTAL_ELEVATION_GAIN_M FLOAT,
  AVG_SPEED_KPH FLOAT,
  HAS_HEARTRATE BOOLEAN,
  AVG_HEARTRATE NUMBER(38,0),
  MAX_HEARTRATE NUMBER(38,0),
  CALORIES_KJ FLOAT,
  KUDOS NUMBER(38,0)
);

User Question: "{user_question}"
SQL Query:
        """
        cleaned = prompt.replace("'", "''")
        cmd = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('snowflake-arctic', '{cleaned}') as response"
        try:
            r = snowpark_session.sql(cmd).to_pandas()
            sql_query = r['RESPONSE'][0].strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:-4].strip()
            return sql_query
        except Exception as e:
            st.error(f"Error generating SQL with Cortex: {e}")
            return None

    def get_natural_language_answer(user_question: str, sql_query: str, results_df: pd.DataFrame):
        prompt = f"""
Provide a concise answer based on the data.

Question: "{user_question}"

SQL:
`{sql_query}`

Data:
{results_df.to_string(index=False)}

Answer:
        """
        cleaned = prompt.replace("'", "''")
        cmd = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('snowflake-arctic', '{cleaned}') as response"
        try:
            r = snowpark_session.sql(cmd).to_pandas()
            return r['RESPONSE'][0]
        except Exception as e:
            st.error(f"Error generating natural language answer with Cortex: {e}")
            return "Sorry, I had trouble interpreting the results."

    if q := st.chat_input("e.g. What's the most common churn reason for 'Inactive' customers aged under 30?"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Generating SQL..."):
            sql_query = generate_sql_query(q)
            st.session_state.messages.append({"role": "assistant", "sql": sql_query, "results": None, "summary": None, "error": None})
        st.rerun()

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if isinstance(message.get("content"), str):
                st.markdown(message["content"])
            else:
                if message.get("sql"):
                    st.markdown("Here is the SQL query I generated:")
                    st.code(message["sql"], language="sql")
                    if message.get("results") is None and message.get("error") is None:
                        if st.button("Execute Query", key=f"execute_{i}"):
                            with st.spinner("Running query..."):
                                try:
                                    results_df = snowpark_session.sql(message["sql"]).to_pandas()
                                    st.session_state.messages[i]["results"] = results_df
                                    if not results_df.empty:
                                        summary = get_natural_language_answer(st.session_state.messages[i-1]['content'], message["sql"], results_df)
                                        st.session_state.messages[i]["summary"] = summary
                                    else:
                                        st.session_state.messages[i]["summary"] = "The query ran successfully but returned no results."
                                except Exception as e:
                                    st.session_state.messages[i]["error"] = f"An error occurred: {e}"
                            st.rerun()
                if message.get("error"):
                    st.error(message["error"])
                if message.get("summary"):
                    st.subheader("Summary")
                    st.markdown(message["summary"])
                    with st.expander('See results'):
                        st.dataframe(message["results"])

# ---------- BI KPIs & Charts ----------
if filtered_df.empty:
    st.warning("No data matches the selected filters.")
else:
    # User-level view for business KPIs
    user_cols = ['USER_ID','AGE','CITY','COUNTRY','SIGNUP_DATE','SUBSCRIPTION_PLAN','SUBSCRIPTION_STATUS','MONTHLY_REVENUE','CHURN_REASON','SUPPORT_TICKETS_LAST_YEAR']
    present_user_cols = [c for c in user_cols if c in filtered_df.columns]
    users_df = filtered_df[present_user_cols].drop_duplicates(subset='USER_ID', keep='last').copy()
    users_df['SUBSCRIPTION_STATUS_NORM'] = users_df['SUBSCRIPTION_STATUS'].astype(str).str.lower()

   # --- Simplified KPI metrics with deltas ---
    if not filtered_df.empty:
        # st.markdown("<hr style='border: 0.5px solid #0169ca;'>", unsafe_allow_html=True)

        users_df = filtered_df.drop_duplicates(subset='USER_ID', keep='last').copy()
        users_df['SUBSCRIPTION_STATUS_NORM'] = users_df['SUBSCRIPTION_STATUS'].astype(str).str.lower()
        users_df['SIGNUP_DATE'] = pd.to_datetime(users_df['SIGNUP_DATE'], errors='coerce')

        # Basic metrics
        total_customers = users_df['USER_ID'].nunique()
        active_mask = users_df['SUBSCRIPTION_STATUS_NORM'].isin(['active', 'trialing', 'live'])
        active_subs = int(active_mask.sum())
        mrr = float(users_df.loc[active_mask, 'MONTHLY_REVENUE'].fillna(0).sum())
        arpu = mrr / active_subs if active_subs else 0.0

        # Simple deltas vs 30 days ago (based on signup date)
        baseline = pd.Timestamp.today() - pd.Timedelta(days=30)
        prev_df = users_df[users_df['SIGNUP_DATE'] <= baseline]
        prev_customers = prev_df['USER_ID'].nunique()
        prev_active = int(prev_df['SUBSCRIPTION_STATUS_NORM'].isin(['active','trialing','live']).sum())
        prev_mrr = float(prev_df.loc[prev_df['SUBSCRIPTION_STATUS_NORM'].isin(['active','trialing','live']), 'MONTHLY_REVENUE'].fillna(0).sum())
        prev_arpu = prev_mrr / prev_active if prev_active else 0.0

        def pct_delta(current, previous):
            if previous == 0:
                return None
            return f"{((current - previous) / previous) * 100:+.1f}%"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Customers", f"{total_customers:,}", pct_delta(total_customers, prev_customers))
        col2.metric("Active Subs", f"{active_subs:,}", pct_delta(active_subs, prev_active))
        col3.metric("MRR", f"${mrr:,.0f}", pct_delta(mrr, prev_mrr))
        col4.metric("ARPU", f"${arpu:,.2f}", pct_delta(arpu, prev_arpu))

        st.markdown("<hr style='border: 0.5px solid #0169ca;'>", unsafe_allow_html=True)


    # ---------- Charts ----------
     # (3) Support burden vs Churn status — BOX (insightful replacement for MRR by plan)
    st.subheader("Churn by Age Group")
    users_sankey = users_df[['USER_ID','AGE','SUBSCRIPTION_STATUS']].dropna().copy()
    users_sankey['STATUS_NORM'] = users_sankey['SUBSCRIPTION_STATUS'].astype(str).str.lower()
    users_sankey['STATUS_ROLLUP'] = np.where(
        users_sankey['STATUS_NORM'].isin(['churned','cancelled','canceled','inactive']),
        'Churned/Inactive',
        'Active/Trialing'
    )

    # Age groups
    def age_bucket(a: float) -> str:
        a = int(a)
        if a <= 25: return "18–25"
        if a <= 35: return "26–35"
        if a <= 45: return "36–45"
        if a <= 55: return "46–55"
        if a <= 65: return "56–65"
        return "66+"
    users_sankey['AGE_GROUP'] = users_sankey['AGE'].apply(age_bucket)

    flows = (
        users_sankey.groupby(['AGE_GROUP','STATUS_ROLLUP'])['USER_ID']
                    .nunique()
                    .reset_index(name='COUNT')
    )

    left_nodes  = ["18–25","26–35","36–45","46–55","56–65","66+"]
    left_nodes  = [g for g in left_nodes if g in flows['AGE_GROUP'].unique().tolist()]
    right_nodes = ["Active/Trialing","Churned/Inactive"]

    # Blue palette (dark → light)
    blue_palette = ["#08306B", "#08519C", "#2171B5", "#4292C6", "#6BAED6", "#C6DBEF"]

    # Build nodes with blue colors
    nodes = []
    for i, name in enumerate(left_nodes):
        nodes.append({"name": name, "itemStyle": {"color": blue_palette[i % len(blue_palette)]}})
    # Right side: distinct blues for consistency
    right_colors = ["#0B3C5D", "#60A3D9"]  # Active/Trialing, Churned/Inactive
    for i, name in enumerate(right_nodes):
        nodes.append({"name": name, "itemStyle": {"color": right_colors[i]}})

    links = [
        {"source": r["AGE_GROUP"], "target": r["STATUS_ROLLUP"], "value": int(r["COUNT"])}
        for _, r in flows.iterrows()
        if r["AGE_GROUP"] in left_nodes
    ]

    option = {
    "backgroundColor": "transparent",
    "textStyle": {"color": "#FFFFFF"},  # global default text color
    "tooltip": {
        "trigger": "item",
        "triggerOn": "mousemove",
        "backgroundColor": "rgba(0,0,0,0.85)",
        "textStyle": {"color": "#FFFFFF"},
        "borderColor": "#2171B5"
    },
    "series": [{
        "type": "sankey",
        "data": nodes,
        "links": links,
        "nodeAlign": "left",
        "label": {"color": "#FFFFFF", "fontSize": 12},          # node labels white
        "emphasis": {"focus": "adjacency", "label": {"color": "#FFFFFF"}},
        "lineStyle": {"color": "source", "opacity": 0.55, "curveness": 0.5}
        }],
    }


    st_echarts(option, height="460px", key="sankey_age_churn_blue")


    colA, colB = st.columns([2, 1])

    # (1) Signups Over Time — BAR
    with colA:
        st.subheader("Signups Over Time")
        if {'SIGNUP_DATE','GENDER'}.issubset(users_df.columns):
            signups_gender = (
                users_df.dropna(subset=['SIGNUP_DATE'])
                        .assign(SIGNUP_MONTH=lambda x: x['SIGNUP_DATE'].dt.to_period('M').dt.to_timestamp())
                        .groupby(['SIGNUP_MONTH','GENDER'])['USER_ID']
                        .nunique()
                        .reset_index(name='CUSTOMERS')
                        .sort_values('SIGNUP_MONTH')
            )
            if not signups_gender.empty:
                fig = px.bar(signups_gender, x='SIGNUP_MONTH', y='CUSTOMERS',
                            color='GENDER', barmode='group',
                            labels={'SIGNUP_MONTH':'Month','CUSTOMERS':'New Customers'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No signup data available.")
        else:
            st.info("SIGNUP_DATE or GENDER column not available.")

    # (2) Churn Reasons — PIE
    with colB:
    # --- Churn Reasons (Top 5) — blue palette, legend only ---
        st.subheader("Reasons for Customer Churn")
        needed = {'CHURN_REASON', 'SUBSCRIPTION_STATUS'}
        if needed.issubset(users_df.columns):
            status_norm = users_df['SUBSCRIPTION_STATUS'].astype(str).str.lower()
            churn_mask = status_norm.isin(['churned', 'cancelled', 'canceled', 'inactive'])

            reasons = (
                users_df.loc[churn_mask, 'CHURN_REASON']
                .astype(str).str.strip()
                .replace({'': 'Unknown', 'none': 'Unknown', 'None': 'Unknown'})
            )

            top5 = (
                reasons[reasons != 'Unknown']
                .value_counts()
                .head(5)
                .reset_index()
            )
            top5.columns = ['CHURN_REASON', 'COUNT']

            if not top5.empty:
                # shades of blue (dark → light)
                blue_palette = ["#0B3C5D", "#1D65A6", "#2E86DE", "#60A3D9", "#A7C7E7"]

                fig_cr = px.pie(
                    top5,
                    names="CHURN_REASON",
                    values="COUNT",
                    color="CHURN_REASON",
                    category_orders={"CHURN_REASON": top5["CHURN_REASON"].tolist()},
                    color_discrete_sequence=blue_palette[: len(top5)]
                )
                # legend only (no labels on slices)
                fig_cr.update_traces(textinfo="none", hovertemplate="%{label}: %{value} (%{percent})")
                fig_cr.update_layout(showlegend=True, legend_title_text="Churn Reason")
                st.plotly_chart(fig_cr, use_container_width=True)
            else:
                st.info("No churn reasons available in the current selection.")
        else:
            st.info("Required columns for churn reasons are not available.")
