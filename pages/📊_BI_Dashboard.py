from snowflake.snowpark import Session
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_echarts import st_echarts

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
st.title("Customer Insights Dashboard")
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
    st.header("Ask me anything")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "What do you want to know about your customers and subscriptions?"}
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

    if q := st.chat_input("e.g. What is churn rate by country?"):
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
        st.markdown("<hr style='border: 0.5px solid #0169ca;'>", unsafe_allow_html=True)

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
    st.subheader("Support Tickets vs Churn Status")
    if 'SUPPORT_TICKETS_LAST_YEAR' in users_df.columns:
        users_df['STATUS_ROLLUP'] = np.where(
            users_df['SUBSCRIPTION_STATUS_NORM'].isin(['churned', 'cancelled', 'canceled', 'inactive']),
            'Churned/Inactive',
            'Active'
        )

        tmp = users_df.dropna(subset=['SUPPORT_TICKETS_LAST_YEAR', 'STATUS_ROLLUP']).copy()
        tmp['SUPPORT_TICKETS_LAST_YEAR'] = pd.to_numeric(tmp['SUPPORT_TICKETS_LAST_YEAR'], errors='coerce')
        tmp = tmp.dropna(subset=['SUPPORT_TICKETS_LAST_YEAR'])

        # Bucketize ticket volume (adjust bins if you like)
        def bucketize(n: float) -> str:
            n = int(n)
            if n <= 1:  return "0–1"
            if n <= 3:  return "2–3"
            if n <= 6:  return "4–6"
            if n <= 10: return "7–10"
            if n <= 20: return "11–20"
            return "21+"

        tmp['TICKET_BUCKET'] = tmp['SUPPORT_TICKETS_LAST_YEAR'].apply(bucketize)

        # Aggregate to flows (unique users)
        flows = (
            tmp.groupby(['TICKET_BUCKET', 'STATUS_ROLLUP'])['USER_ID']
               .nunique()
               .reset_index(name='COUNT')
        )

        # Order nodes left (buckets) to right (statuses)
        bucket_order = ["0–1", "2–3", "4–6", "7–10", "11–20", "21+"]
        status_order = ["Active", "Churned/Inactive"]

        # Keep only buckets that exist in the data
        bucket_order = [b for b in bucket_order if b in flows['TICKET_BUCKET'].unique().tolist()]

        nodes = [{'name': n} for n in bucket_order + status_order]
        links = [
            {'source': r['TICKET_BUCKET'], 'target': r['STATUS_ROLLUP'], 'value': int(r['COUNT'])}
            for _, r in flows.iterrows()
        ]

        option = {
            "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
            "series": [{
                "type": "sankey",
                "data": nodes,
                "links": links,
                "nodeAlign": "left",
                "emphasis": {"focus": "adjacency"},
                "lineStyle": {"color": "source", "curveness": 0.5},
                "label": {"color": "#333", "fontSize": 12},
            }],
        }

        st_echarts(option, height="460px", key="sankey_support_churn")
    else:
        st.info("`SUPPORT_TICKETS_LAST_YEAR` not available.")
    colA, colB = st.columns([2, 1])

    # (1) Signups Over Time — BAR
    with colA:
        st.subheader("Signups Over Time")
        if 'SIGNUP_DATE' in users_df.columns:
            signups = (
                users_df
                .dropna(subset=['SIGNUP_DATE'])
                .assign(SIGNUP_MONTH=lambda x: x['SIGNUP_DATE'].dt.to_period('M').dt.to_timestamp())
                .groupby('SIGNUP_MONTH')['USER_ID']
                .nunique()
                .reset_index(name='CUSTOMERS')
                .sort_values('SIGNUP_MONTH')
            )
            if not signups.empty:
                fig_signup = px.bar(signups, x='SIGNUP_MONTH', y='CUSTOMERS')
                fig_signup.update_layout(xaxis_title='Month', yaxis_title='New Customers')
                st.plotly_chart(fig_signup, use_container_width=True)
            else:
                st.info("No signup dates available in the current filter.")
        else:
            st.info("SIGNUP_DATE column not available.")

    # (2) Churn Reasons — PIE
    with colB:
        st.subheader("Churn Reasons")
        if 'CHURN_REASON' in users_df.columns:
            cr = (
                users_df.loc[churn_mask, 'CHURN_REASON']
                .astype(str)
                .replace({'': 'Unknown', 'None': 'Unknown'})
                .value_counts()
                .reset_index()
            )
            cr.columns = ['CHURN_REASON','COUNT']
            cr = cr[cr['CHURN_REASON'].notna() & (cr['CHURN_REASON'] != 'Unknown')]
            if not cr.empty:
                fig_cr = px.pie(cr, names='CHURN_REASON', values='COUNT')
                st.plotly_chart(fig_cr, use_container_width=True)
            else:
                st.info("No churned/inactive customers in the current filter.")
        else:
            st.info("CHURN_REASON column not available.")