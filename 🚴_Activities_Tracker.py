import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from datetime import timedelta
from streamlit_option_menu import option_menu
from streamlit_calendar import calendar

# --- Page Configuration ---
st.set_page_config(
    page_title="Personal Activity Dashboard",
    page_icon="🚴",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stMetric {
        border-radius: 10px;
        padding: 15px;
        background-color: #262730;
    }
    .summary-box {
        border-radius: 10px;
        padding: 15px 20px;
        background-color: #1e293b; /* Different dark blue/grey */
        border: 1px solid #4a4a5a;
    }
    .summary-box .activity-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #fafafa;
    }
    .summary-box .activity-date {
        color: #a0a0a0;
        font-style: italic;
    }
    .summary-box .activity-stats {
        margin-top: 10px;
        display: flex;
        justify-content: space-around;
    }
    /* Center the title */
    h1 {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """
    Loads and preprocesses the Strava activities data from a CSV file.
    Caches the data to avoid reloading on every interaction.
    """
    try:
        df = pd.read_csv(file_path)

        # Data Cleaning and Preprocessing
        df['Start Date Local'] = pd.to_datetime(df['Start Date Local'], errors='coerce')
        df.dropna(subset=['Start Date Local'], inplace=True)
        
        df.rename(columns={
            'Distance (km)': 'Distance_km',
            'Total Elevation Gain (m)': 'Elevation_Gain_m',
            'Moving Time (sec)': 'Moving_Time_sec',
            'AVG Speed (km/h)': 'Avg_Speed_kmh',
            'AVG Heartrate': 'Avg_Heartrate'
        }, inplace=True)
        
        # Clean up Kudos column
        df['Kudos'] = pd.to_numeric(df['Kudos'], errors='coerce').fillna(0).astype(int)

        # Extracting time components for filtering
        df['Year'] = df['Start Date Local'].dt.year
        df['Month'] = df['Start Date Local'].dt.month
        df['Hour'] = df['Start Date Local'].dt.hour

        # Infer Activity Type from Name for filtering (more granular with expanded keywords)
        df['Activity Type'] = 'Other'
        df.loc[df['Name'].str.contains('run', case=False, na=False), 'Activity Type'] = 'Run'
        df.loc[df['Name'].str.contains('ride|bike|cycling|peloton', case=False, na=False), 'Activity Type'] = 'Ride'
        df.loc[df['Name'].str.contains('swim', case=False, na=False), 'Activity Type'] = 'Swim'
        df.loc[df['Name'].str.contains('hike', case=False, na=False), 'Activity Type'] = 'Hike'
        df.loc[df['Name'].str.contains('walk', case=False, na=False), 'Activity Type'] = 'Walk'
        df.loc[df['Name'].str.contains('workout|gym|strength|training|crossfit|lifting', case=False, na=False), 'Activity Type'] = 'Workout'
        df.loc[df['Name'].str.contains('yoga', case=False, na=False), 'Activity Type'] = 'Yoga'
        df.loc[df['Name'].str.contains('ski|snowboard', case=False, na=False), 'Activity Type'] = 'Skiing'
        df.loc[df['Name'].str.contains('kayak|paddle|canoe|sup|rowing', case=False, na=False), 'Activity Type'] = 'Paddling'


        # Process coordinates
        df_coords = df['Start Coordinates'].astype(str).str.split(',', expand=True)
        df_coords.columns = ['start_lat', 'start_lon']
        df_coords = df_coords.apply(pd.to_numeric, errors='coerce')
        df = pd.concat([df, df_coords], axis=1)
        
        return df
    except FileNotFoundError:
        st.error("The file 'strava_activities.csv' was not found. Please make sure it's in the root directory.")
        return None

df_strava = load_data('strava_activities.csv')

if df_strava is not None:
    st.title(f"Personal Activity Dashboard")
    st.link_button("Go to Profile", st.secrets['strava_user'])

    # --- Main Page Filters in an Expander ---
    with st.expander("Filter Your Activities", expanded=True):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            years = sorted(df_strava['Year'].unique(), reverse=True)
            selected_year = st.selectbox("Select Year", options=years, index=0)
            
            month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            months_in_year = sorted(df_strava[df_strava['Year'] == selected_year]['Month'].unique())
            
            month_options = ["All"] + [month_map[m] for m in months_in_year]
            selected_month_name = st.pills("Select Month", options=month_options, default='All')

            if selected_month_name == "All":
                selected_months = months_in_year
            else:
                selected_months = [m for m, name in month_map.items() if name == selected_month_name]


        with filter_col2:
            activity_types = sorted(df_strava['Activity Type'].unique())
            activity_options = ["All"] + activity_types
            selected_activity = st.pills("Select Activity Type", options=activity_options, default='All')
        
        with filter_col3:
            min_dist, max_dist = 0.0, df_strava['Distance_km'].max()
            selected_distance = st.slider("Select Distance Range (km)", min_value=min_dist, max_value=max_dist, value=(min_dist, max_dist))

    # --- Filtering Logic ---
    current_mask = (
        (df_strava['Year'] == selected_year) &
        (df_strava['Month'].isin(selected_months)) &
        (df_strava['Distance_km'] >= selected_distance[0]) &
        (df_strava['Distance_km'] <= selected_distance[1])
    )
    previous_mask = (
        (df_strava['Year'] == selected_year - 1) &
        (df_strava['Month'].isin(selected_months))
    )

    if selected_activity != "All":
        current_mask &= (df_strava['Activity Type'] == selected_activity)
        previous_mask &= (df_strava['Activity Type'] == selected_activity)

    current_period_df = df_strava[current_mask]
    previous_period_df = df_strava[previous_mask]
    
    # --- Navigation Menu ---
    selected_tab = option_menu(
        menu_title=None,
        options=["Performance KPIs", "Geospatial Analysis", "Activity Calendar"],
        icons=["graph-up-arrow", "map-fill", "calendar-week"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#1c1c24"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#3a3a4a"},
            "nav-link-selected": {"background-color": "#0068c9"},
        }
    )

    # --- Tab Content ---
    if selected_tab == "Performance KPIs":
        if current_period_df.empty:
            st.warning("No activities found for the selected filters.")
        else:
            # Display Kudos Toast Notification
            total_kudos = current_period_df['Kudos'].sum()
            st.toast(f"🎉 You received a total of {total_kudos} kudos in this period!", icon="👏")
            
            def calculate_kpi(current_df, previous_df, metric_col):
                current_val = current_df[metric_col].sum() if metric_col else len(current_df)
                previous_val = previous_df[metric_col].sum() if metric_col else len(previous_df)
                delta = ((current_val - previous_val) / previous_val) * 100 if previous_val != 0 else 0
                return current_val, delta

            total_activities, activity_delta = calculate_kpi(current_period_df, previous_period_df, None)
            total_distance, distance_delta = calculate_kpi(current_period_df, previous_period_df, 'Distance_km')
            total_elevation, elevation_delta = calculate_kpi(current_period_df, previous_period_df, 'Elevation_Gain_m')
            total_moving_time_hr, time_delta = calculate_kpi(current_period_df, previous_period_df, 'Moving_Time_sec')
            total_moving_time_hr /= 3600

            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            kpi_col1.metric("Total Activities", f"{total_activities}", f"{activity_delta:.1f}%")
            kpi_col2.metric("Total Distance", f"{total_distance:,.1f} km", f"{distance_delta:.1f}%")
            kpi_col3.metric("Total Elevation Gain", f"{total_elevation:,.0f} m", f"{elevation_delta:.1f}%")
            kpi_col4.metric("Total Moving Time", f"{total_moving_time_hr:,.1f} hrs", f"{time_delta:.1f}%")
            
            st.markdown("<hr style='border: 0.5px solid red;'>", unsafe_allow_html=True)
            
            # Get the most recent activity from the filtered dataframe
            last_activity = current_period_df.sort_values(by='Start Date Local', ascending=False).iloc[0]
            
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                st.subheader("Weekly Activity Counts")
                weekly_counts = current_period_df.set_index('Start Date Local') \
                                                 .groupby([pd.Grouper(freq='W-MON'), 'Activity Type']) \
                                                 .size().reset_index(name='Count')
                
                if not weekly_counts.empty:
                    weekly_counts['Week Start'] = weekly_counts['Start Date Local'].dt.date
                    fig_weekly_activity = px.bar(weekly_counts, x='Week Start', y='Count', color='Activity Type', title="Activities per Week", labels={'Week Start': 'Week', 'Count': 'Number of Activities'})
                    fig_weekly_activity.update_layout(xaxis_title="Week", yaxis_title="Number of Activities", barmode='stack')
                    st.plotly_chart(fig_weekly_activity, use_container_width=True)
                else:
                    st.info("Not enough data to display weekly trends.")
                
            with col_perf2:
                scatter_data = current_period_df.dropna(subset=['Avg_Speed_kmh', 'Avg_Heartrate'])
                if not scatter_data.empty:
                    st.subheader("Performance Analysis")
                    fig_scatter = px.scatter(scatter_data, x='Avg_Speed_kmh', y='Avg_Heartrate', color='Activity Type', size='Distance_km', hover_name='Name', title='Avg Speed vs. Avg Heart Rate')
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Not enough heart rate data to display this chart.")

            # Always have the raw data expander available
        with st.expander("Show Raw Filtered Data"):
            st.dataframe(current_period_df)

    if selected_tab == "Geospatial Analysis":
        st.subheader("Activity Hotspots (All Time)")
        
        # Use all available data for the map
        map_data = df_strava[['start_lat', 'start_lon']].dropna()

        if not map_data.empty:
            view_state = pdk.ViewState(
                latitude=map_data['start_lat'].mean(), 
                longitude=map_data['start_lon'].mean(), 
                zoom=5, 
                pitch=45
            )
            
            layer = pdk.Layer(
                'ScreenGridLayer',
                data=map_data,
                get_position='[start_lon, start_lat]',
                cell_size_pixels=17, # Kept granular as requested
                color_range=[ # Green color range
                   [0, 255, 0, 90],
                    [0, 255, 0, 130],
                    [0, 255, 0, 170],
                    [0, 255, 0, 210],
                    [0, 255, 0, 240],
                    [0, 255, 0, 255]
                ],
                pickable=True,
            )

            st.pydeck_chart(pdk.Deck(
                # Setting map_style to None uses the default base map which doesn't require an API key
                map_style=None,
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": "Density of Activity Start Points"}
            ))
        else:
            st.info("No coordinate data to display map.")
    
    if selected_tab == "Activity Calendar":
        st.subheader("Your Activity Calendar")
        
        CALENDAR_COLOR_MAP = {"Ride": "#0068c9", "Run": "#ff4b4b", "Hike": "#2b83ba", "Walk": "#abddb4", "Swim": "#006d2c", "Workout": "#990099", "Yoga": "#ff69b4", "Skiing": "#add8e6", "Paddling": "#4daf4a", "Other": "#262730"}
        
        calendar_events = []
        for index, row in current_period_df.iterrows():
            calendar_events.append({
                "title": f"{row['Activity Type']}: {row['Name']}",
                "start": row['Start Date Local'].isoformat(),
                "end": (row['Start Date Local'] + timedelta(seconds=row['Moving_Time_sec'])).isoformat(),
                "color": CALENDAR_COLOR_MAP.get(row['Activity Type'], CALENDAR_COLOR_MAP['Other'])
            })
            
        cal = calendar(events=calendar_events, options={"headerToolbar": { "left": "prev,next today", "center": "title", "right": "dayGridMonth,timeGridWeek,timeGridDay"}, "initialView": "dayGridMonth"})
        
        if cal and 'event' in cal:
            st.markdown("<hr style='border: 0.5px solid red;'>", unsafe_allow_html=True)
            st.write(f"**Selected Activity:** {cal['event']['title']}")
        