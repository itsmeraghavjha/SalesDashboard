import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import urllib.parse # For URL encoding WhatsApp messages
import json # For displaying feedback as JSON

# Set page configuration for a wider layout and a dairy-themed icon
st.set_page_config(layout="wide", page_title="Dairy Sales Dashboard", page_icon="🥛")

# --- Main Title ---
st.title("🥛 Dairy & Dairy Products Sales Analysis Dashboard")
st.markdown("---") # Visual separator

# --- Dummy Data Generation ---
@st.cache_data
def generate_dummy_data(num_records=5000):
    np.random.seed(42) # for reproducibility to ensure data is consistent on refresh
    data = {
        'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_records, freq='D').tolist() * 2)[:num_records],
        'Sales_Office': np.random.choice(['North Office', 'South Office', 'East Office', 'West Office', 'Central Office'], num_records),
        'Product': np.random.choice(['Milk (L)', 'Curd (Kg)', 'Ghee (Kg)', 'Butter (Kg)', 'Paneer (Kg)', 'Cheese (Kg)', 'Ice Cream (Pcs)'], num_records),
        'Quantity_Sold': np.random.randint(50, 500, num_records),
        'Price_Per_Unit': np.round(np.random.uniform(20, 300, num_records), 2),
        'Channel_Partner': np.random.choice(['Supermarket A', 'Local Store B', 'Wholesaler C', 'Hotel D', 'Online Retailer E', 'Cafeteria F'], num_records),
        'Scheme_Name': np.random.choice(['No Scheme', 'Summer Discount (10% Off)', 'Bulk Buy Offer', 'Loyalty Bonus'], num_records, p=[0.5, 0.2, 0.2, 0.1]) # Introduce schemes
    }
    df = pd.DataFrame(data)

    # Initial Total_Sales calculation before scheme impact and seasonality
    df['Total_Sales'] = df['Quantity_Sold'] * df['Price_Per_Unit']

    # Apply scheme impact to Total_Sales for demonstration purposes
    df.loc[df['Scheme_Name'] == 'Summer Discount (10% Off)', 'Total_Sales'] *= 1.10 # Simulate higher sales due to discount attraction
    df.loc[df['Scheme_Name'] == 'Bulk Buy Offer', 'Total_Sales'] *= 1.15 # Simulate higher sales due to bulk purchases
    df.loc[df['Scheme_Name'] == 'Loyalty Bonus', 'Total_Sales'] *= 1.05 # Simulate slight boost from loyalty

    # Introduce some seasonality or trends (e.g., higher sales in certain months for ice cream)
    df['Month'] = df['Date'].dt.month
    df.loc[(df['Product'] == 'Ice Cream (Pcs)') & (df['Month'].isin([6, 7, 8])), 'Total_Sales'] *= 1.05 # Summer boost, cumulative
    df.loc[(df['Product'] == 'Ghee (Kg)') & (df['Month'].isin([10, 11, 12])), 'Total_Sales'] *= 1.03 # Festival boost, cumulative
    df['Total_Sales'] = np.round(df['Total_Sales'], 2)

    # Generate dummy Sales_Target after sales are finalized, for realistic target setting
    df['Sales_Target'] = df['Total_Sales'] * np.random.uniform(0.85, 1.20, num_records) # Targets +/- 15% of sales
    df['Sales_Target'] = np.round(df['Sales_Target'], 2)

    return df

df = generate_dummy_data(num_records=5000)

# Add Month and Year columns for time-based analysis
df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)
df['Year'] = df['Date'].dt.year
df_original = df.copy() # Keep a copy of the full dataset for overall comparisons and whitespace analysis

# --- Define Focus and Must-Sell SKUs (Products) ---
# These lists typically come from a business strategy or product management team
FOCUS_SKUS = ['Milk (L)', 'Cheese (Kg)']
MUST_SELL_SKUS = ['Curd (Kg)', 'Ghee (Kg)', 'Butter (Kg)']


# --- Sidebar Filters ---
st.sidebar.header("📊 Dashboard Filters")
st.sidebar.write("Use these options to customize the data displayed in the dashboard.")

# View selection radio button
view_type = st.sidebar.radio(
    "Select Your Dashboard View:",
    ('Management View', 'Sales Office View'),
    index=0, # Default to Management View
    help="Choose between an overall management perspective or a detailed view for a specific sales office."
)

st.sidebar.markdown("---")

# Date range slider
min_date = df['Date'].min().to_pydatetime()
max_date = df['Date'].max().to_pydatetime()

date_range = st.sidebar.date_input(
    "📆 Select Date Range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key='date_filter',
    help="Adjust the date range to analyze sales data over a specific period."
)

# Initialize filtered_df based on date range
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
else:
    filtered_df = df.copy()

# Sales Office Selection based on view_type
all_sales_offices = sorted(filtered_df['Sales_Office'].unique().tolist()) # Sort for consistent order
selected_offices = [] # Initialize for management view or single selection for sales office view

if view_type == 'Management View':
    selected_offices = st.sidebar.multiselect(
        "🏢 Select Sales Office(s):",
        options=all_sales_offices,
        default=all_sales_offices,
        key='management_office_select',
        help="Select one or more sales offices to include in the overall management analysis."
    )
    if selected_offices:
        filtered_df = filtered_df[filtered_df['Sales_Office'].isin(selected_offices)]
else: # Sales Office View
    selected_office_for_view = st.sidebar.selectbox(
        "🏢 Select Your Sales Office:",
        options=all_sales_offices,
        key='sales_office_view_select',
        help="Choose your specific sales office to see its dedicated performance metrics."
    )
    if selected_office_for_view:
        filtered_df = filtered_df[filtered_df['Sales_Office'] == selected_office_for_view]
        selected_offices = [selected_office_for_view] # For consistency in product/channel filter logic below
    else: # If no office is selected (e.g., initial load before selection)
        st.warning("Please select a Sales Office from the dropdown to view its performance.")
        st.stop() # Stop execution if no office is selected to prevent errors


# Product Multiselect
all_products_in_filtered_df = sorted(filtered_df['Product'].unique().tolist()) # Sort for consistent order
selected_products = st.sidebar.multiselect(
    "🍦 Select Product(s):",
    options=all_products_in_filtered_df,
    default=all_products_in_filtered_df,
    key='product_select',
    help="Filter the data by specific dairy products."
)
if selected_products:
    filtered_df = filtered_df[filtered_df['Product'].isin(selected_products)]

# Channel Partner Multiselect (Dynamic based on view type)
if view_type == 'Management View':
    all_channel_partners = sorted(filtered_df['Channel_Partner'].unique().tolist()) # Sort for consistent order
    selected_channel_partners = st.sidebar.multiselect(
        "🤝 Select Channel Partner(s):",
        options=all_channel_partners,
        default=all_channel_partners,
        key='management_channel_select',
        help="Filter data by specific channel partners across all selected offices."
    )
    if selected_channel_partners:
        filtered_df = filtered_df[filtered_df['Channel_Partner'].isin(selected_channel_partners)]
else: # Sales Office View - restrict channel partners to only those associated with the selected office
    # Get all channel partners that the selected_office_for_view has ever dealt with (within the date filter)
    relevant_channel_partners_for_office = sorted(df_original[
        (df_original['Sales_Office'] == selected_office_for_view) &
        (df_original['Date'] >= pd.to_datetime(start_date)) &
        (df_original['Date'] <= pd.to_datetime(end_date))
    ]['Channel_Partner'].unique().tolist())

    selected_channel_partners = st.sidebar.multiselect(
        "🤝 Select Channel Partner(s) (Your Office):",
        options=relevant_channel_partners_for_office,
        default=relevant_channel_partners_for_office,
        key='sales_office_channel_select',
        help="Filter data by channel partners relevant to your specific sales office."
    )
    if selected_channel_partners:
        filtered_df = filtered_df[filtered_df['Channel_Partner'].isin(selected_channel_partners)]


# New: Toggle to show charts as tables
st.sidebar.markdown("---")
show_as_table_toggle = st.sidebar.checkbox(
    "📋 Show charts as tables",
    value=False,
    help="Toggle to view all charts as raw data tables for detailed inspection."
)
st.sidebar.markdown("---")

# Check if filtered_df is empty after applying all filters
if filtered_df.empty:
    st.warning("🚫 No data available for the selected filters. Please adjust your selections in the sidebar.")
    st.stop() # Stop execution if no data is found to avoid further errors

# --- KPI Calculation Function ---
def display_kpis(df_to_analyze, title_prefix=""):
    """Displays key performance indicators (KPIs) in a clean, metric format."""
    col1, col2, col3, col4, col5 = st.columns(5) # Five columns for five KPIs

    total_sales = df_to_analyze['Total_Sales'].sum()
    total_target = df_to_analyze['Sales_Target'].sum()
    total_quantity_sold = df_to_analyze['Quantity_Sold'].sum()
    num_transactions = len(df_to_analyze)
    avg_transaction_value = total_sales / num_transactions if num_transactions > 0 else 0
    achievement_percentage = (total_sales / total_target) * 100 if total_target > 0 else 0

    with col1:
        st.metric(f"{title_prefix} Total Sales Revenue", f"₹ {total_sales:,.2f}")
    with col2:
        st.metric(f"{title_prefix} Total Sales Target", f"₹ {total_target:,.2f}")
    with col3:
        st.metric(f"{title_prefix} Total Quantity Sold", f"{total_quantity_sold:,.0f} Units")
    with col4:
        st.metric(f"{title_prefix} Number of Transactions", f"{num_transactions:,.0f}")
    with col5:
        # Using delta_color="normal" for automatic green/red based on positive/negative delta
        st.metric(
            f"{title_prefix} Target Achievement",
            f"{achievement_percentage:,.2f}%",
            delta=f"{achievement_percentage - 100:,.2f}%" if achievement_percentage != 0 else None,
            delta_color="normal", # Green for positive delta (over 100%), Red for negative delta (under 100%)
            help="Percentage of total sales revenue achieved against the set target. Green indicates target met or exceeded, red indicates lagging."
        )

# Define performance status for conditional coloring in charts
def get_performance_status(percentage):
    """Categorizes performance into 'Achieved Target', 'On Track', or 'Lagging Target'."""
    if percentage >= 100:
        return 'Achieved Target'
    elif percentage >= 95: # Within 5% of target
        return 'On Track (Near Target)' # Renamed for clearer understanding
    else:
        return 'Lagging Target'

# --- Dashboard Views ---
if view_type == 'Management View':
    st.header("🏢 Management View: Overall Sales Performance")
    st.markdown(f"Displaying data for: **{', '.join(selected_offices) if selected_offices else 'All Offices'}**")
    st.write("This view provides a high-level overview of sales performance across selected offices for management decision-making.")
    st.markdown("---")

    st.subheader("📊 Key Performance Indicators (Overall)")
    display_kpis(filtered_df)
    st.markdown("---")

    # Sales Trend Over Time
    st.subheader("📈 Sales Trend Over Time (Sales vs. Target)")
    st.write("Track how actual sales revenue compares against the set targets over time.")
    sales_by_month = filtered_df.groupby('Month_Year').agg(
        Total_Sales=('Total_Sales', 'sum'),
        Total_Target=('Sales_Target', 'sum')
    ).reset_index()
    sales_by_month['Month_Year'] = pd.to_datetime(sales_by_month['Month_Year']) # Convert back for proper sorting

    if show_as_table_toggle:
        st.markdown("##### Monthly Sales Revenue vs. Target (Table View)")
        st.dataframe(sales_by_month.sort_values('Month_Year').style.format({
            'Total_Sales': "₹ {:,.2f}",
            'Total_Target': "₹ {:,.2f}"
        }), use_container_width=True)
    else:
        fig_sales_trend = px.line(
            sales_by_month.sort_values('Month_Year'), # Ensure chronological order
            x='Month_Year',
            y=['Total_Sales', 'Total_Target'],
            title='Monthly Sales Revenue vs. Target (All Selected Offices)',
            labels={'Month_Year': 'Month', 'value': 'Amount (₹)', 'variable': 'Metric'},
            hover_data={'Total_Sales': ':.2f', 'Total_Target': ':.2f'},
            markers=True,
            color_discrete_map={'Total_Sales': '#28a745', 'Total_Target': '#dc3545'} # Green for Sales, Red for Target
        )
        fig_sales_trend.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig_sales_trend.update_xaxes(showgrid=False)
        fig_sales_trend.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
        st.plotly_chart(fig_sales_trend, use_container_width=True)

    st.markdown("---")

    # Sales by Sales Office and Product Distribution (side-by-side)
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("🏢 Sales Office Performance: Target Achievement")
        st.write("See which sales offices are meeting or falling short of their sales targets.")
        office_performance = filtered_df.groupby('Sales_Office').agg(
            Total_Sales=('Total_Sales', 'sum'),
            Total_Target=('Sales_Target', 'sum')
        ).reset_index()
        office_performance['Achievement_Percentage'] = (office_performance['Total_Sales'] / office_performance['Total_Target']) * 100
        office_performance['Performance_Status'] = office_performance['Achievement_Percentage'].apply(get_performance_status)

        if show_as_table_toggle:
            st.markdown("##### Sales Office Target Achievement (Table View)")
            st.dataframe(office_performance.sort_values(by='Achievement_Percentage', ascending=False).style.format({
                'Total_Sales': "₹ {:,.2f}",
                'Total_Target': "₹ {:,.2f}",
                'Achievement_Percentage': "{:,.2f}%"
            }), use_container_width=True)
        else:
            fig_office_achievement = px.bar(
                office_performance.sort_values(by='Achievement_Percentage', ascending=False),
                x='Sales_Office',
                y='Achievement_Percentage',
                title='Sales Office Target Achievement (%)',
                labels={'Sales_Office': 'Sales Office', 'Achievement_Percentage': 'Achievement (%)'},
                color='Performance_Status',
                color_discrete_map={
                    'Achieved Target': '#28a745',   # Green
                    'On Track (Near Target)': '#CCCCCC',         # Light Grey (near white)
                    'Lagging Target': '#dc3545'     # Red
                },
                hover_data={'Total_Sales': ':.2f', 'Total_Target': ':.2f', 'Achievement_Percentage': ':.2f'}
            )
            fig_office_achievement.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            fig_office_achievement.update_xaxes(showgrid=False)
            fig_office_achievement.update_yaxes(showgrid=True, gridcolor='#e0e0e0', range=[0, 120]) # Set a reasonable range
            st.plotly_chart(fig_office_achievement, use_container_width=True)

    with col6:
        st.subheader("🍦 Product Sales Distribution (Overall)")
        st.write("Understand the proportion of revenue generated by different products.")
        sales_by_product = filtered_df.groupby('Product')['Total_Sales'].sum().reset_index().sort_values(by='Total_Sales', ascending=False)
        
        if show_as_table_toggle:
            st.markdown("##### Product Sales Distribution (Table View)")
            st.dataframe(sales_by_product.style.format({
                'Total_Sales': "₹ {:,.2f}"
            }), use_container_width=True)
        else:
            fig_product_sales = px.pie(
                sales_by_product,
                values='Total_Sales',
                names='Product',
                title='Sales Revenue Distribution by Product (Overall)',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel # Use a diverse, yet subtle palette for distribution
            )
            fig_product_sales.update_traces(textposition='inside', textinfo='percent+label')
            fig_product_sales.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_product_sales, use_container_width=True)

    st.markdown("---")

    # Campaign & Scheme Performance (Management View)
    st.subheader("🎯 Campaign & Scheme Performance (Overall)")
    st.write("Analyze the overall revenue generated by different sales schemes and campaigns.")
    scheme_performance_overall = filtered_df.groupby('Scheme_Name').agg(
        Total_Sales=('Total_Sales', 'sum'),
        Number_of_Transactions=('Total_Sales', 'count'),
        Average_Sales_Per_Transaction=('Total_Sales', 'mean')
    ).reset_index().sort_values(by='Total_Sales', ascending=False)

    if show_as_table_toggle:
        st.markdown("##### Campaign & Scheme Performance (Table View)")
        st.dataframe(scheme_performance_overall.style.format({
            'Total_Sales': "₹ {:,.2f}",
            'Average_Sales_Per_Transaction': "₹ {:,.2f}"
        }), use_container_width=True)
    else:
        fig_scheme_sales_overall = px.bar(
            scheme_performance_overall,
            x='Scheme_Name',
            y='Total_Sales',
            title='Total Sales Revenue by Scheme Type (Overall)',
            labels={'Scheme_Name': 'Scheme Name', 'Total_Sales': 'Sales Revenue (₹)'},
            color='Scheme_Name',
            color_discrete_sequence=px.colors.qualitative.D3 # A good qualitative palette for distinct schemes
        )
        fig_scheme_sales_overall.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig_scheme_sales_overall.update_xaxes(showgrid=False)
        fig_scheme_sales_overall.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
        st.plotly_chart(fig_scheme_sales_overall, use_container_width=True)
    
    st.markdown("---") # Separator for detailed view
    st.subheader("Detailed Sales Performance by Office & Product")
    st.write("A comprehensive table showing sales and target achievement for each product within each selected sales office.")
    detailed_sales = filtered_df.groupby(['Sales_Office', 'Product']).agg(
        Total_Sales=('Total_Sales', 'sum'),
        Total_Target=('Sales_Target', 'sum'),
        Quantity_Sold=('Quantity_Sold', 'sum'),
        Avg_Price_Per_Unit=('Price_Per_Unit', 'mean')
    ).reset_index()
    detailed_sales['Achievement_Percentage'] = (detailed_sales['Total_Sales'] / detailed_sales['Total_Target']) * 100
    detailed_sales = detailed_sales.sort_values(by='Total_Sales', ascending=False)

    st.dataframe(detailed_sales.style.format({
        'Total_Sales': "₹ {:,.2f}",
        'Total_Target': "₹ {:,.2f}",
        'Quantity_Sold': "{:,.0f}",
        'Avg_Price_Per_Unit': "₹ {:,.2f}",
        'Achievement_Percentage': "{:,.2f}%"
    }), use_container_width=True, height=300)

    st.markdown("---")



else: # Sales Office View
    st.header(f"🎯 Sales Office View: Performance for {selected_office_for_view}")
    st.markdown(f"This view provides detailed insights for **{selected_office_for_view}** to help analyze its performance and identify opportunities.")
    st.markdown("---")

    # Use Streamlit tabs for better navigation within the Sales Office View
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "✉️ Send Offers to Retailers", "💡 Product Insights & Feedback"])

    with tab1: # Main Dashboard for Sales Office
        st.subheader("📈 Key Performance Indicators (Your Office)")
        display_kpis(filtered_df, title_prefix="Your Office")
        st.markdown("---")

        # Sales Trend Over Time for selected office
        st.subheader("📊 Your Office's Monthly Sales Trend (Sales vs. Target) **(Displaying Dummy Data)**")
        st.write(f"Track the sales performance of {selected_office_for_view} against its monthly targets.")
        sales_by_month_office = filtered_df.groupby('Month_Year').agg(
            Total_Sales=('Total_Sales', 'sum'),
            Total_Target=('Sales_Target', 'sum')
        ).reset_index()
        sales_by_month_office['Month_Year'] = pd.to_datetime(sales_by_month_office['Month_Year'])

        if show_as_table_toggle:
            st.markdown("##### Your Office's Monthly Sales Trend (Table View)")
            st.dataframe(sales_by_month_office.sort_values('Month_Year').style.format({
                'Total_Sales': "₹ {:,.2f}",
                'Total_Target': "₹ {:,.2f}"
            }), use_container_width=True)
        else:
            fig_office_sales_trend = px.line(
                sales_by_month_office.sort_values('Month_Year'),
                x='Month_Year',
                y=['Total_Sales', 'Total_Target'],
                title=f'Monthly Sales Revenue vs. Target for {selected_office_for_view}',
                labels={'Month_Year': 'Month', 'value': 'Amount (₹)', 'variable': 'Metric'},
                hover_data={'Total_Sales': ':.2f', 'Total_Target': ':.2f'},
                markers=True,
                color_discrete_map={'Total_Sales': '#28a745', 'Total_Target': '#dc3545'} # Green for Sales, Red for Target
            )
            fig_office_sales_trend.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            fig_office_sales_trend.update_xaxes(showgrid=False)
            fig_office_sales_trend.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
            st.plotly_chart(fig_office_sales_trend, use_container_width=True)

        st.markdown("---")

        # Product Sales Distribution for selected office
        col7, col8 = st.columns(2)

        with col7:
            st.subheader("🍦 Your Office's Product Performance: Target Achievement")
            st.write(f"See how {selected_office_for_view}'s sales are performing against targets for each product.")
            product_performance_office = filtered_df.groupby('Product').agg(
                Total_Sales=('Total_Sales', 'sum'),
                Total_Target=('Sales_Target', 'sum')
            ).reset_index()
            product_performance_office['Achievement_Percentage'] = (product_performance_office['Total_Sales'] / product_performance_office['Total_Target']) * 100
            product_performance_office['Performance_Status'] = product_performance_office['Achievement_Percentage'].apply(get_performance_status)

            if show_as_table_toggle:
                st.markdown("##### Product Target Achievement (Table View)")
                st.dataframe(product_performance_office.sort_values(by='Achievement_Percentage', ascending=False).style.format({
                    'Total_Sales': "₹ {:,.2f}",
                    'Total_Target': "₹ {:,.2f}",
                    'Achievement_Percentage': "{:,.2f}%"
                }), use_container_width=True)
            else:
                fig_product_achievement_office = px.bar(
                    product_performance_office.sort_values(by='Achievement_Percentage', ascending=False),
                    x='Product',
                    y='Achievement_Percentage',
                    title=f'Product Target Achievement for {selected_office_for_view} (%)',
                    labels={'Product': 'Product', 'Achievement_Percentage': 'Achievement (%)'},
                    color='Performance_Status',
                    color_discrete_map={
                        'Achieved Target': '#28a745',   # Green
                        'On Track (Near Target)': '#CCCCCC',         # Light Grey (near white)
                        'Lagging Target': '#dc3545'     # Red
                    },
                    hover_data={'Total_Sales': ':.2f', 'Total_Target': ':.2f', 'Achievement_Percentage': ':.2f'}
                )
                fig_product_achievement_office.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                fig_product_achievement_office.update_xaxes(showgrid=False)
                fig_product_achievement_office.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
                st.plotly_chart(fig_product_achievement_office, use_container_width=True)

        with col8:
            st.subheader("📈 Your Office vs. Overall Average Sales")
            st.write("Compare your office's average daily sales with the company's overall average.")
            # Calculate overall average (filtered by date and product)
            # Ensure 'selected_products' is defined from the main filter, even if in sales office view
            products_for_overall_avg = selected_products if selected_products else df_original['Product'].unique().tolist()
            
            overall_avg_sales_per_office_per_day = df_original[
                (df_original['Date'] >= pd.to_datetime(start_date)) &
                (df_original['Date'] <= pd.to_datetime(end_date)) &
                (df_original['Product'].isin(products_for_overall_avg))
            ].groupby(['Date', 'Sales_Office'])['Total_Sales'].sum().mean() # Average daily sales per office

            current_office_avg_sales_per_day = filtered_df.groupby('Date')['Total_Sales'].sum().mean()

            comparison_data = pd.DataFrame({
                'Category': [f'{selected_office_for_view} Average Daily Sales', 'Overall Average Daily Sales (All Offices)'],
                'Average Sales': [current_office_avg_sales_per_day, overall_avg_sales_per_office_per_day]
            })

            if show_as_table_toggle:
                st.markdown("##### Your Office vs. Overall Average Sales (Table View)")
                st.dataframe(comparison_data.style.format({
                    'Average Sales': "₹ {:,.2f}"
                }), use_container_width=True)
            else:
                fig_comparison = px.bar(
                    comparison_data,
                    x='Category',
                    y='Average Sales',
                    title='Your Office vs. Overall Average Daily Sales',
                    labels={'Category': '', 'Average Sales': 'Average Daily Sales (₹)'},
                    color='Category',
                    color_discrete_sequence=['#28a745', '#CCCCCC'] # Green for selected, Light Grey for overall
                )
                fig_comparison.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                fig_comparison.update_xaxes(showgrid=False)
                fig_comparison.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
                st.plotly_chart(fig_comparison, use_container_width=True)

        st.markdown("---")

        # Campaign & Scheme Performance (Sales Office View)
        st.subheader("🎯 Your Office's Campaign & Scheme Performance")
        st.write(f"Analyze the impact of different schemes run by {selected_office_for_view} on total sales revenue.")
        scheme_performance_office = filtered_df.groupby('Scheme_Name').agg(
            Total_Sales=('Total_Sales', 'sum'),
            Number_of_Transactions=('Total_Sales', 'count'),
            Average_Sales_Per_Transaction=('Total_Sales', 'mean')
        ).reset_index().sort_values(by='Total_Sales', ascending=False)

        if show_as_table_toggle:
            st.markdown("##### Your Office's Campaign & Scheme Performance (Table View)")
            st.dataframe(scheme_performance_office.style.format({
                'Total_Sales': "₹ {:,.2f}",
                'Average_Sales_Per_Transaction': "₹ {:,.2f}"
            }), use_container_width=True)
        else:
            fig_scheme_sales_office = px.bar(
                scheme_performance_office,
                x='Scheme_Name',
                y='Total_Sales',
                title=f'Total Sales Revenue by Scheme Type for {selected_office_for_view}',
                labels={'Scheme_Name': 'Scheme Name', 'Total_Sales': 'Sales Revenue (₹)'},
                color='Scheme_Name',
                color_discrete_sequence=px.colors.qualitative.D3 # A good qualitative palette for distinct schemes
            )
            fig_scheme_sales_office.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            fig_scheme_sales_office.update_xaxes(showgrid=False)
            fig_scheme_sales_office.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
            st.plotly_chart(fig_scheme_sales_office, use_container_width=True)

        st.markdown("---")

        # --- Channel Partner Performance for the selected sales office ---
        st.subheader("🤝 Your Office's Channel Partner Performance")
        st.write(f"Understand which channel partners are contributing most to {selected_office_for_view}'s sales.")
        channel_partner_performance_office = filtered_df.groupby('Channel_Partner').agg(
            Total_Sales=('Total_Sales', 'sum'),
            Number_of_Transactions=('Total_Sales', 'count'),
            Average_Sales_Per_Transaction=('Total_Sales', 'mean')
        ).reset_index().sort_values(by='Total_Sales', ascending=False)

        if show_as_table_toggle:
            st.markdown("##### Your Office's Channel Partner Performance (Table View)")
            st.dataframe(channel_partner_performance_office.style.format({
                'Total_Sales': "₹ {:,.2f}",
                'Average_Sales_Per_Transaction': "₹ {:,.2f}"
            }), use_container_width=True)
        else:
            fig_channel_partner_sales_office = px.bar(
                channel_partner_performance_office,
                x='Channel_Partner',
                y='Total_Sales',
                title=f'Total Sales Revenue by Channel Partner for {selected_office_for_view}',
                labels={'Channel_Partner': 'Channel Partner', 'Total_Sales': 'Sales Revenue (₹)'},
                color='Channel_Partner',
                color_discrete_sequence=px.colors.qualitative.Alphabet # Use a diverse palette for partners
            )
            fig_channel_partner_sales_office.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            fig_channel_partner_sales_office.update_xaxes(showgrid=False)
            fig_channel_partner_sales_office.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
            st.plotly_chart(fig_channel_partner_sales_office, use_container_width=True)

        st.markdown("---")
        # --- SKU Performance: Focus & Must-Sell Products ---
        st.subheader("📊 SKU Performance: Focus & Must-Sell Products")
        st.write("Review the sales performance of your designated Focus SKUs (high priority) and Must-Sell SKUs (core products).")

        # Filter for Focus SKUs
        focus_skus_df = filtered_df[filtered_df['Product'].isin(FOCUS_SKUS)]
        if not focus_skus_df.empty:
            focus_skus_performance = focus_skus_df.groupby('Product').agg(
                Total_Sales=('Total_Sales', 'sum'),
                Total_Target=('Sales_Target', 'sum'),
                Quantity_Sold=('Quantity_Sold', 'sum')
            ).reset_index()
            focus_skus_performance['Achievement_Percentage'] = (focus_skus_performance['Total_Sales'] / focus_skus_performance['Total_Target']) * 100
            st.markdown("##### ⭐ Focus SKUs Performance")
            st.dataframe(focus_skus_performance.style.format({
                'Total_Sales': "₹ {:,.2f}",
                'Total_Target': "₹ {:,.2f}",
                'Quantity_Sold': "{:,.0f}",
                'Achievement_Percentage': "{:,.2f}%"
            }), use_container_width=True, height=200)
        else:
            st.info("No data for Focus SKUs in the current selection for your office.")

        st.markdown("---")

        # Filter for Must-Sell SKUs
        must_sell_skus_df = filtered_df[filtered_df['Product'].isin(MUST_SELL_SKUS)]
        if not must_sell_skus_df.empty:
            must_sell_skus_performance = must_sell_skus_df.groupby('Product').agg(
                Total_Sales=('Total_Sales', 'sum'),
                Total_Target=('Sales_Target', 'sum'),
                Quantity_Sold=('Quantity_Sold', 'sum')
            ).reset_index()
            must_sell_skus_performance['Achievement_Percentage'] = (must_sell_skus_performance['Total_Sales'] / must_sell_skus_performance['Total_Target']) * 100
            st.markdown("##### 🛒 Must-Sell SKUs Performance")
            st.dataframe(must_sell_skus_performance.style.format({
                'Total_Sales': "₹ {:,.2f}",
                'Total_Target': "₹ {:,.2f}",
                'Quantity_Sold': "{:,.0f}",
                'Achievement_Percentage': "{:,.2f}%"
            }), use_container_width=True, height=200)
        else:
            st.info("No data for Must-Sell SKUs in the current selection for your office.")

        st.markdown("---")


        # --- Whitespace Analysis ---
        st.subheader("🔍 Whitespace Analysis: Product Opportunities with Existing Partners **(Analysis based on Dummy Data)**")
        st.write(f"Identify products that your office, **{selected_office_for_view}**, is not currently selling to its active channel partners, but which are sold by other offices.")
        
        # Get all unique Product-Channel_Partner combinations *for the selected office* within the filtered date range
        # These are the actual relationships the office has.
        office_actual_product_channel_combinations = filtered_df[
            filtered_df['Sales_Office'] == selected_office_for_view
        ][['Product', 'Channel_Partner', 'Total_Sales']].groupby(['Product', 'Channel_Partner']).sum().reset_index()
        office_actual_product_channel_combinations_set = set(tuple(row) for row in office_actual_product_channel_combinations[['Product', 'Channel_Partner']].values)

        # Get all unique products that the *entire company* sells (within the date filter)
        all_company_products_in_period = df_original[
            (df_original['Date'] >= pd.to_datetime(start_date)) & 
            (df_original['Date'] <= pd.to_datetime(end_date))
        ]['Product'].unique()

        # Get all channel partners that the *selected sales office* currently works with (within the date filter)
        # This ensures we are only looking at "their" channel partners
        current_office_channel_partners_for_whitespace = df_original[
            (df_original['Sales_Office'] == selected_office_for_view) &
            (df_original['Date'] >= pd.to_datetime(start_date)) & 
            (df_original['Date'] <= pd.to_datetime(end_date))
        ]['Channel_Partner'].unique()

        # Get all unique Product-Channel_Partner combinations available in the company for the selected date range (for checking existence)
        df_date_filtered_original_for_whitespace = df_original[
            (df_original['Date'] >= pd.to_datetime(start_date)) &
            (df_original['Date'] <= pd.to_datetime(end_date))
        ]
        all_company_product_channel_combinations_for_period = df_date_filtered_original_for_whitespace[['Product', 'Channel_Partner']].drop_duplicates()
        all_company_set_for_period = set(tuple(row) for row in all_company_product_channel_combinations_for_period.values)


        whitespace_opportunities_list = []

        # Iterate through each existing channel partner of the selected office
        for cp in current_office_channel_partners_for_whitespace:
            # For each partner, check all products the company sells
            for product in all_company_products_in_period:
                # If the (product, channel_partner) combination is NOT found in the office's actual sales
                # BUT it *is* found in the overall company data for the period
                if (product, cp) not in office_actual_product_channel_combinations_set and (product, cp) in all_company_set_for_period:
                    whitespace_opportunities_list.append({'Product': product, 'Potential Channel Partner': cp})

        # Remove duplicates (ensure uniqueness based on content)
        whitespace_opportunities_list_unique = [dict(t) for t in {tuple(sorted(d.items())) for d in whitespace_opportunities_list}]
        
        if whitespace_opportunities_list_unique:
            st.info(f"For **{selected_office_for_view}**, explore opportunities to sell the following products to your existing channel partners:")
            
            # Convert list of dicts to DataFrame for better display
            whitespace_df = pd.DataFrame(whitespace_opportunities_list_unique).sort_values(by=['Potential Channel Partner', 'Product'])
            st.dataframe(whitespace_df, use_container_width=True, height=250)
            st.write("These combinations indicate products that your office could potentially sell to channel partners you already work with, but haven't sold (or have very low sales) in the selected period.")
        else:
            st.success(f"{selected_office_for_view} is selling all available products to all its active channel partners (for the selected period)! Excellent market penetration and cross-selling!")

        st.markdown("---")


    with tab2: # Send Offers Tab
        st.subheader("✉️ Send Offers & Schemes to Retailers via WhatsApp")
        st.write("Easily generate WhatsApp messages to send offers and scheme details to your channel partners. You can also upload a poster image for visual communication.")
        st.warning("Important: This tool generates a WhatsApp Web link. You will need to click the link to open WhatsApp and manually send the message. If you upload a poster, you'll need to attach it separately in WhatsApp, as images cannot be directly embedded in the generated text message via this method.")
        st.markdown("---")

        # Get active channel partners for the current sales office for selection
        available_channel_partners_for_offers = df_original[
            (df_original['Sales_Office'] == selected_office_for_view) &
            (df_original['Date'] >= pd.to_datetime(start_date)) &
            (df_original['Date'] <= pd.to_datetime(end_date))
        ]['Channel_Partner'].unique().tolist()
        available_channel_partners_for_offers.insert(0, "-- Select Channel Partner --") # Add a default option

        selected_retailer = st.selectbox(
            "1. Select Channel Partner:",
            options=available_channel_partners_for_offers,
            key='offer_retailer_select',
            help="Choose the retailer you want to send the offer to."
        )

        # Get all products for selection
        all_products_for_offers = sorted(df_original['Product'].unique().tolist())
        all_products_for_offers.insert(0, "-- Select Product (Optional) --")

        selected_offer_product = st.selectbox(
            "2. Select Product for Offer (Optional):",
            options=all_products_for_offers,
            key='offer_product_select',
            help="If the offer is specific to a product, select it here."
        )

        st.markdown("---")
        st.subheader("3. Upload Offer Poster (Optional)")
        uploaded_file = st.file_uploader(
            "Upload an image for your offer poster (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            key='poster_uploader',
            help="Upload a visual poster to accompany your offer. Remember to send it manually in WhatsApp."
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Offer Poster Preview", use_column_width=True)
            st.info("Uploaded poster preview. This image will NOT be sent automatically via the WhatsApp link.")
        st.markdown("---")

        offer_details = st.text_area(
            "4. Enter Offer/Scheme Details:",
            "Dear [Channel Partner Name],\n\nWe have an exciting new offer for you!\n\nDetails: [Insert your offer details here, e.g., 'Get 15% off on Milk (L) for orders over 100 units this week!']\n\nBest regards,\nYour Sales Team at [Your Sales Office Name]",
            height=200,
            key='offer_details_text',
            help="Type your offer message here. [Channel Partner Name] and [Your Sales Office Name] will be replaced automatically."
        )

        st.markdown("---")
        if st.button("Generate WhatsApp Message Link", key='generate_whatsapp_button', help="Click to generate the unique WhatsApp link for sending your message."):
            if selected_retailer == "-- Select Channel Partner --":
                st.warning("Please select a Channel Partner to generate the WhatsApp message link.")
            else:
                # Customize the message placeholders
                final_message = offer_details.replace("[Channel Partner Name]", selected_retailer)
                final_message = final_message.replace("[Your Sales Office Name]", selected_office_for_view)
                if selected_offer_product != "-- Select Product (Optional) --":
                    # Replace placeholder only if a product is selected
                    final_message = final_message.replace(
                        "[Insert your offer details here, e.g., 'Get 15% off on Milk (L) for orders over 100 units this week!']",
                        f"Special offer on {selected_offer_product}: [Your specific offer details for this product here]."
                    )
                else:
                    # Keep original placeholder or a generic message if no specific product was selected
                    final_message = final_message.replace(
                        "[Insert your offer details here, e.g., 'Get 15% off on Milk (L) for orders over 100 units this week!']",
                        "Check out our latest exciting offers!"
                    )

                # Encode the message for URL to ensure special characters are handled correctly
                encoded_message = urllib.parse.quote(final_message)
                
                # Dummy phone number for demonstration. In a real application, this would come from a database.
                dummy_phone_number = "919876543210" # Example Indian mobile number

                whatsapp_link = f"https://wa.me/{dummy_phone_number}?text={encoded_message}"

                st.success("✅ WhatsApp message link successfully generated!")
                st.markdown(f"Click the link below to open WhatsApp with the pre-filled message for **{selected_retailer}**:")
                st.markdown(f"**[🔗 Send Offer via WhatsApp to {selected_retailer}]({whatsapp_link})**")
                st.info("Remember to click 'Send' in WhatsApp. If you uploaded a poster, attach it manually in the chat.")


    with tab3: # Product Insights & Feedback Tab
        st.subheader("💡 Product Insights & Feedback")
        st.write("Here you can find information about upcoming products and share your valuable feedback and suggestions with the product and marketing teams.")
        st.markdown("---")

        st.markdown("#### ✨ Upcoming / Latest Products")
        st.write("Stay informed about new products joining our portfolio:")
        # Dummy data for upcoming products
        upcoming_products = [
            {"Name": "Probiotic Fresh Curd (500g)", "Launch": "Q3 2024", "Features": "Enhanced probiotics for gut health, creamy texture, 10-day shelf life.", "Target Market": "Health-conscious consumers, urban families, fitness enthusiasts."},
            {"Name": "Spiced Buttermilk (200ml) - On-the-Go Pack", "Launch": "Q4 2024", "Features": "Ready-to-drink, traditional Indian flavors, natural ingredients, convenient packaging.", "Target Market": "Convenience seekers, office-goers, traditional beverage lovers, travel segments."},
            {"Name": "Vegan Cheese Alternative (Cheddar Block)", "Launch": "Q1 2025", "Features": "Plant-based (almond milk base), lactose-free, authentic cheddar taste, melts well.", "Target Market": "Vegan/vegetarian consumers, dairy-allergic individuals, health food stores, conscious consumers."},
        ]

        for product in upcoming_products:
            st.write(f"**🌟 {product['Name']}**")
            st.write(f"  - **Expected Launch:** {product['Launch']}")
            st.write(f"  - **Key Features:** {product['Features']}")
            st.write(f"  - **Target Market:** {product['Target Market']}")
            st.markdown("---")

        st.markdown("#### 💬 Share Your Feedback & Suggestions")
        st.write("Your insights are valuable! Please use the form below to share your thoughts on products, schemes, or market trends.")
        
        feedback_type = st.radio(
            "What kind of feedback do you have?",
            ("Product Suggestion", "Scheme Suggestion", "Market Insight", "Other"),
            key='feedback_type_radio',
            help="Select the category that best describes your feedback."
        )
        feedback_subject = st.text_input("Subject:", key='feedback_subject_input', help="A brief summary of your feedback.")
        feedback_message = st.text_area("Your detailed feedback/suggestion:", height=150, key='feedback_message_input', help="Provide detailed information, observations, or ideas.")

        if st.button("Submit Feedback", key='submit_feedback_button', help="Click to submit your feedback."):
            if not feedback_subject or not feedback_message:
                st.warning("Please fill in both the 'Subject' and 'Your detailed feedback/suggestion' fields before submitting.")
            else:
                feedback_data = {
                    "Sales Office": selected_office_for_view,
                    "Feedback Type": feedback_type,
                    "Subject": feedback_subject,
                    "Message": feedback_message,
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.success("🎉 Thank you for your valuable feedback! It has been submitted.")
                st.info("In a real application, this feedback would be saved to a database or sent to the relevant team for review.")
                st.json(feedback_data) # Displaying as JSON for demonstration purposes


st.markdown("---") # Final separator at the bottom of the page
st.caption("Dashboard created with Streamlit and Plotly by Raghav for a Dairy & Dairy Products Company.")
