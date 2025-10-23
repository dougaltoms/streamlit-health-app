from snowflake.snowpark import Session
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_resource(ttl=1)
def get_session() -> Session:
    """
    Establishes a connection to Snowflake using credentials from Streamlit's secrets.
    The connection is cached to avoid reconnecting on every app rerun.
    Returns:
        Session: An active Snowpark session object.
    """
    try:
        # st.secrets accesses the secrets.toml file
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
        st.error(f"Failed to connect to Snowflake. Please check your credentials. Error: {e}")
        st.stop()


st.title("Customer Insights Dashboard")
st.write("This dashboard provides key insights into customer data and allows any user to ask questions in natural language")


# Initialize the Snowpark session
try:
    snowpark_session = get_session()
    TABLE_NAME = "ACTIVITY_DATA"
    
    # Check if the table exists
    try:
        # Use .to_pandas() for easier manipulation in Streamlit
        df = snowpark_session.table(TABLE_NAME).to_pandas()
        if df.empty:
             st.warning(
                f"The table `{TABLE_NAME}` appears to be empty. "
                f"Please follow the `setup_guide.md` to create and populate it."
            )
             st.stop()
        # Convert date columns to datetime objects and add age if available
        df['START_DATE_LOCAL'] = pd.to_datetime(df['START_DATE_LOCAL'])
        # A placeholder for age is used since it's not in the new dataset
        if 'AGE' not in df.columns:
            df['AGE'] = np.random.randint(18, 81, size=len(df))


    except Exception:
        st.warning(
            f"The table `{TABLE_NAME}` does not seem to exist. "
            f"Please follow the `setup_guide.md` to create and populate it."
        )
        st.stop()

except Exception as e:
    st.error(f"An error occurred during app initialization: {e}")
    st.stop()

# --- Sidebar Filters ---

st.sidebar.header("Dashboard Filters")

# Age Range Slider
min_age = int(df['AGE'].min())
max_age = int(df['AGE'].max())
age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

# Continent Multiselect
continent_map = {
    'USA': 'North America', 'UK': 'Europe', 'Canada': 'North America', 
    'Australia': 'Australia', 'France': 'Europe', 'Japan': 'Asia'
}
df['CONTINENT'] = df['COUNTRY'].map(continent_map)
all_continents = df['CONTINENT'].dropna().unique().tolist()
selected_continents = st.sidebar.segmented_control(
    "Select Continents",
    options=all_continents,
    selection_mode="multi",
    default=all_continents)

# Sport filters:
all_activity_types = df['ACTIVITY_TYPE'].dropna().unique().tolist()
selected_activities = st.sidebar.pills(
    'Sport Type',
    options = all_activity_types,
    selection_mode="multi",
    default=all_activity_types
)

# Apply filters
filtered_df = df[
    (df['AGE'] >= age_range[0]) &
    (df['AGE'] <= age_range[1]) &
    (df['CONTINENT'].isin(selected_continents)) &
    (df['ACTIVITY_TYPE'].isin(selected_activities))
]

# --- BI Dashboard Section ---
with st.expander('🤖 Chat to your data'):
    st.header("🤖 Ask me anything")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "What do you want to know about your customers?"}
        ]

    # --- Core Chatbot Functions ---

    @st.cache_data()
    def generate_sql_query(user_question: str):
        """
        Uses Cortex LLM to convert a user's natural language question into a SQL query.
        """

        prompt = f"""
        You are an expert SQL assistant specializing in fitness activity data. 
        Your task is to generate a single, valid SQL query for a Snowflake database.
        Do not provide any explanations or conversational text. Only output the SQL query.

        Here is the schema for the table '{TABLE_NAME}':
    
    create or replace TABLE ACTIVITY_DATA (
        ACTIVITY_ID NUMBER(38,0),
        USER_ID NUMBER(38,0),
        AGE NUMBER(38,0),
        ACTIVITY_TYPE VARCHAR(50),
        START_DATE_LOCAL TIMESTAMP_NTZ(9),
        DISTANCE_KM FLOAT,
        MOVING_TIME_SEC NUMBER(38,0),
        ELAPSED_TIME_SEC NUMBER(38,0),
        TOTAL_ELEVATION_GAIN_M FLOAT,
        AVG_SPEED_KPH FLOAT,
        HAS_HEARTRATE BOOLEAN,
        AVG_HEARTRATE NUMBER(38,0),
        MAX_HEARTRATE NUMBER(38,0),
        CALORIES_KJ FLOAT,
        KUDOS NUMBER(38,0),
        CITY VARCHAR(50),
        COUNTRY VARCHAR(50)
    );
        Based on this schema, answer the following user question:
        User Question: "{user_question}"

        SQL Query:
        """
        
        cleaned_prompt = prompt.replace("'", "''")
        command = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('snowflake-arctic', '{cleaned_prompt}') as response"
        
        try:
            result_df = snowpark_session.sql(command).to_pandas()
            sql_query = result_df['RESPONSE'][0].strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:-4].strip()
            return sql_query
        except Exception as e:
            st.error(f"Error generating SQL with Cortex: {e}")
            return None

    def get_natural_language_answer(user_question: str, sql_query: str, results_df: pd.DataFrame):
        """
        Uses Cortex LLM to convert query results into a natural language answer.
        """
        prompt = f"""
        You are a helpful fitness data assistant. Your job is to provide a clear, concise, and friendly answer to the user's question based on the provided data.

        The user asked: "{user_question}"

        The following SQL query was executed to find the answer:
        `{sql_query}`

        And here is the data returned:
        {results_df.to_string()}

        Based on this information, provide a natural language answer.
        """
        
        cleaned_prompt = prompt.replace("'", "''")
        command = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('snowflake-arctic', '{cleaned_prompt}') as response"
        
        try:
            result_df = snowpark_session.sql(command).to_pandas()
            return result_df['RESPONSE'][0]
        except Exception as e:
            st.error(f"Error generating natural language answer with Cortex: {e}")
            return "Sorry, I had trouble interpreting the results."

    # --- Main Chat and Display Loop ---

    # Handle new user input
    if prompt := st.chat_input("e.g. What are the most active cities in USA?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🔍 Generating SQL..."):
            sql_query = generate_sql_query(prompt)
            assistant_message = {
                "role": "assistant",
                "sql": sql_query,
                "results": None,
                "summary": None,
                "error": None
            }
            st.session_state.messages.append(assistant_message)
        st.rerun()

    # Display the chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if isinstance(message.get("content"), str):
                st.markdown(message["content"])
            else:
                if message.get("sql"):
                    st.markdown("Here is the SQL query I generated:")
                    st.code(message["sql"], language="sql")
                    
                    if message.get("results") is None and message.get("error") is None:
                        if st.button("✅ Execute Query", key=f"execute_{i}"):
                            with st.spinner("🏃 Running query..."):
                                try:
                                    results_df = snowpark_session.sql(message["sql"]).to_pandas()
                                    st.session_state.messages[i]["results"] = results_df
                                    
                                    if not results_df.empty:
                                        summary = get_natural_language_answer(
                                            st.session_state.messages[i-1]['content'], 
                                            message["sql"], 
                                            results_df
                                        )
                                        st.session_state.messages[i]["summary"] = summary
                                    else:
                                        st.session_state.messages[i]["summary"] = "The query ran successfully but returned no results."
                                except Exception as e:
                                    st.session_state.messages[i]["error"] = f"An error occurred: {e}"
                            st.rerun()

                if message.get("error"):
                    st.error(message["error"])

                # if message.get("results") is not None:
                #     if not message["results"].empty:
                #         st.subheader("Summary")
                #         st.dataframe(message["results"])

                if message.get("summary"):
                    # st.markdown("---")
                    st.subheader("Summary")
                    st.markdown(message["summary"])
                    with st.expander('See results'):
                        st.dataframe(message["results"])

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
else:
    st.markdown("<hr style='border: 0.5px solid #0169ca;'>", unsafe_allow_html=True)
    # Top-level metrics based on filtered data
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Activities", f"{filtered_df.shape[0]:,}")
    col2.metric("Total Distance", f"{filtered_df['DISTANCE_KM'].sum():,.1f} km")
    col3.metric("Total Elevation Gain", f"{filtered_df['TOTAL_ELEVATION_GAIN_M'].sum():,.0f} m")
    col4.metric("Total Kudos", f"{filtered_df['KUDOS'].sum():,}")

    st.markdown("<hr style='border: 0.5px solid #0169ca;'>", unsafe_allow_html=True)

    # Dashboard Charts
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Activities Over Time by Age Group")
        # Create age groups
        bins = [18, 30, 40, 50, 60, 70, 80, float('inf')]
        labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
        filtered_df['AGE_GROUP'] = pd.cut(filtered_df['AGE'], bins=bins, labels=labels, right=False)

        # Group by month and age group, then format for the line chart
        activities_by_age_group = filtered_df.groupby(
            [pd.Grouper(key='START_DATE_LOCAL', freq='ME'), 'AGE_GROUP']
        ).size().unstack(fill_value=0)
        
        st.line_chart(activities_by_age_group)

        st.subheader("Distance vs Elevation Gain")
        st.write("How activity distance relates to elevation.")
        st.scatter_chart(filtered_df, x='DISTANCE_KM', y='TOTAL_ELEVATION_GAIN_M', color='#0169ca')

    with col2:
        st.subheader("Activity Types")
        activity_dist = filtered_df['ACTIVITY_TYPE'].value_counts()
        st.bar_chart(activity_dist, color='#0169ca')
        
        st.subheader("Activities by Country")
        country_dist = filtered_df['COUNTRY'].value_counts().head(10)
        st.bar_chart(country_dist, color='#0169ca')
