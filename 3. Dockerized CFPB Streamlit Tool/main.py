import requests

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_nested_layout

import pandas as pd
import altair as alt
import numpy as np
import re

import plotly.express as px

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from bokeh.io import output_notebook

import nltk
from nltk.tokenize import RegexpTokenizer
import re
from datetime import datetime, timedelta


import psycopg2
from psycopg2 import extras


import matplotlib.pyplot as plt
import geopandas as gpd
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Dictionary to map original column names to more consistent, snake_case column names
rename_dict = {
    'Date received': 'Date_received',
    'Product': 'Product',
    'Sub-product': 'Sub_product',
    'Issue': 'Issue',
    'Sub-issue': 'Sub_issue',
    'Consumer complaint narrative': 'Consumer_complaint_narrative',
    'Company public response': 'Company_public_response',
    'Company': 'Company',
    'State': 'State',
    'ZIP code': 'ZIP_code',
    'Tags': 'Tags',
    'Consumer consent provided?': 'Consumer_consent_provided',
    'Submitted via': 'Submitted_via',
    'Date sent to company': 'Date_sent_to_company',
    'Company response to consumer': 'Company_response_to_consumer',
    'Timely response?': 'Timely_response',
    'Consumer disputed?': 'Consumer_disputed',
    'Complaint ID': 'Complaint_ID'
}
# Create a reversed dictionary for the opposite mapping
reversed_dict = {v: k for k, v in rename_dict.items()}

def create_connection():
    # Define connection parameters
    user="username"
    password="password"
    database = "cfpb"
    host = "server ip address"
    # Connect to your postgres server
    conn = psycopg2.connect(
        dbname=database,
        user=user,
        password=password,
        host=host
    )
    # Create and return both a cursor and connection to the database
    cur = conn.cursor()
    return cur,conn

def text_normalizer(text):
    # Function to process and normalize the text
    if text:
        # Use NLTK RegexpTokenizer for tokenization. 
        # This tokenizer splits the text by white space and also keeps tokens like "wasn't" and "don't".
        tokenizer = RegexpTokenizer(r'\b\w[\w\'-]*\w\b|\w')
        words = tokenizer.tokenize(text)

        # Clean up any token with repeating characters like '666', 'aaa', '!!!!!!', substitute them with empty string ''.
        # This includes 'XXXX' maskings in the text created by CFPB.
        words = [re.sub(r'(\w)\1{2,}', '', word) if re.search(r'(\w)\1{2,}', word) else word for word in words]

        # Convert to lowercase and remove punctuations.
        words = [word.lower().strip() for word in words]

        # Substitute the tokens with "" where they are just numbers.
        words = ['' if word.isdigit() else word for word in words]

        # Join the words back into a single string.
        text = ' '.join([word for word in words if word])
    
    return text
def ping_hello():
    # Function to initialize the complaint evaluation API
    response = requests.get('https://cfpb-eval-ltsyz5ld4a-uc.a.run.app/hello')
    return response.text

def get_prediction_json(text):
    # Get prediction results from the complaint evaluation API
    response = requests.get(f'https://cfpb-eval-ltsyz5ld4a-uc.a.run.app/complaint_eval/{text}')
    data = response.json()
    print(data)
    issue_df = pd.DataFrame.from_dict(data["Issue"], orient='index', columns=["Relevance"]).reset_index().rename(columns={"index": "Issue"})
    product_df = pd.DataFrame.from_dict(data["Product"], orient='index', columns=["Relevance"]).reset_index().rename(columns={"index": "Product"})
    return issue_df, product_df


def get_data_template():
    # Fetch a random sample of data from the database for template purposes
    cur, conn = create_connection()
    # define query
    query = f"""
        SELECT *
        FROM cfpb
        ORDER BY random()
        LIMIT 100;
        """
    df = pd.read_sql_query(query, conn)
    # Close the cursor and connection
    cur.close()
    conn.close()
    return df

@st.cache_data
def get_unique_products(start_date, end_date):
    # Query the database to retrieve unique products within a given date range
    cur, conn = create_connection()
    query = """
    SELECT 
        "Product", 
        COUNT("Product") AS ProductCount
    FROM 
        cfpb
    WHERE
        "Date_received" BETWEEN '{}' AND '{}'
    GROUP BY 
        "Product"
    ORDER BY 
        ProductCount DESC;
    """
    query = query.format(start_date, end_date)
    df = pd.read_sql_query(query, conn)
    cur.close()
    conn.close()
    return df.Product.to_list()

@st.cache_data
def get_unique_company_by_products(product_lst=None):
    # Query the database to retrieve companies based on given products
    cur, conn = create_connection()
    query = """
    SELECT 
        DISTINCT "Company"
    FROM 
        cfpb;
    """
    if product_lst:
        query = """
        SELECT 
            "Company", 
            COUNT("Company") AS CompanyCount
        FROM 
            cfpb
        WHERE 
            "Product" IN ({})
        GROUP BY 
            "Company"
        ORDER BY 
            CompanyCount DESC;
        """
        formatted_product_lst = ", ".join("'{}'".format(p) for p in product_lst)
        query = query.format(formatted_product_lst)
    df = pd.read_sql_query(query, conn)
    cur.close()
    conn.close()
    return df.Company.to_list()

@st.cache_data
def get_product_company_record_within_date_range(company_lst, start_date, end_date, product_lst=None):
    # Query the database to retrieve records based on provided companies and date range
    cur, conn = create_connection()
    query = """
    SELECT 
        "Date_received",
        "Product", 
        "Sub_product", 
        "Issue", 
        "Sub_issue",
        "Company_public_response", 
        "Company",
        "State", 
        "ZIP_code",
        "Company_response_to_consumer",
        "Timely_response", 
        "Consumer_disputed", 
        "Complaint_ID", 
        "Date_sent_to_company",
        "Tags"
    FROM 
        cfpb
    WHERE
        "Company" IN ({})
    AND
        "Date_received" BETWEEN '{}' AND '{}'
    ;
    """
    formatted_company_lst = ", ".join("'{}'".format(c) for c in company_lst)
    query = query.format(formatted_company_lst, start_date, end_date)
    if product_lst:
        query = """
        SELECT 
            "Date_received",
            "Product", 
            "Sub_product", 
            "Issue", 
            "Sub_issue",
            "Company_public_response", 
            "Company",
            "State", 
            "ZIP_code",
            "Company_response_to_consumer",
            "Timely_response", "Consumer_disputed", "Complaint_ID"
        FROM 
            cfpb
        WHERE 
            "Product" IN ({})
        AND 
            "Company" IN ({})
        AND
            "Date_received" BETWEEN '{}' AND '{}'
        ;
        """
        formatted_product_lst = ", ".join("'{}'".format(p) for p in product_lst)
        formatted_company_lst = ", ".join("'{}'".format(c) for c in company_lst)
        query = query.format(formatted_product_lst, formatted_company_lst, start_date, end_date)

    df = pd.read_sql_query(query, conn)
    cur.close()
    conn.close()
    return df


# Default date range settings
start_date = datetime.now() - timedelta(days=45)
end_date = datetime.now()
# Initialize session state with default dates if they haven't been set
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = start_date
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = end_date
def update_start_date():
    # Function to update the session's start date
    st.session_state.start_date = start_date
def update_end_date():
    # Function to update the session's end date
    st.session_state.end_date = end_date




def main():
    # Set the page layout to 'wide'
    st.set_page_config(layout='wide')

    # Create a sidebar
    with st.sidebar:
        # Create an option menu with icons and vertical orientation
        option = option_menu(menu_title="Main Menu", 
                             options=["Welcome!", 
                                      "Complaint Evaluation and Filing Assistant", 
                                      "Service Provider Comparison Tool",
                                      "Enterprise Customer Complaint Analysis Tool"],
                             default_index=0,
                             menu_icon='filter-square',
                             icons=['cast','file-earmark-bar-graph', 'file-earmark-text','layout-text-window-reverse'],
                             orientation="vertical",
                             styles=None)

        # Depending on the user's choice, perform different actions
        if option == "Complaint Evaluation and Filing Assistant":
            st.write("Warming up the API :)")
            st.write("This may take a a minute :)")
            # Ping an API
            ping_hello()

        elif option == "Service Provider Comparison Tool":
            # Ask the user to select a product
            product_selected = st.selectbox(label="Select A Product:", 
                                            options=get_unique_products(st.session_state['start_date'], st.session_state['end_date']),
                                            placeholder="Select A Product")
            # Ask the user to select up to 5 service providers
            company_selected = st.multiselect(label="Select Service Provider(s):",
                                              options=get_unique_company_by_products([product_selected]),
                                              placeholder="Select Up To 5 Company(s)",
                                              max_selections=5)
            # Provide an informative note to the user about company listing
            st.caption("Note: If a specific company isn't listed, it may be due to either of the following reasons: \
            1. No complaints have been filed against it in the Consumer Financial Protection Bureau (CFPB) database. \
            2. The company operates under a different business name.")
            
            # Ask the user to select a date range for searching
            start_date = st.date_input('Enter a start date', min_value=datetime.now() - timedelta(days=365), max_value=datetime.now(), value=datetime.now() - timedelta(days=7))
            end_date = st.date_input('Enter an end date', min_value=start_date, max_value=datetime.now(), value=datetime.now())
            
            # Create a button to start the search
            company_submitted = st.button('Start Search!')

        elif option == "Enterprise Customer Complaint Analysis Tool":
            # Ask the user to select up to 5 competitor companies
            company_names = st.multiselect(label="Select Competitor():",
                                           options=get_unique_company_by_products(),
                                           placeholder="Select Up To 5 Company(s)",
                                           max_selections=5)
            
            # Provide an informative note to the user about company listing
            st.caption("Note: If a specific company isn't listed, it may be due to either of the following reasons: \
            1. No complaints have been filed against it in the Consumer Financial Protection Bureau (CFPB) database. \
            2. The company operates under a different business name.")
            
            # Ask the user to select a longer date range for searching
            start_date = st.date_input('Enter a start date', min_value=datetime.now() - timedelta(days=1825), max_value=datetime.now(), value=datetime.now() - timedelta(days=60))
            end_date = st.date_input('Enter an end date', min_value=start_date, max_value=datetime.now(), value=datetime.now())
            
            # Ask the user to select a type of market segmentation for insights
            segmentation_option = st.sidebar.selectbox('Select Market Segmentation', 
                                                       ['Demographic Tags', 'State', 'ZIP code'],
                                                       key='demographic_insights_selectbox')
            
            # Create a button to start the search
            company_submitted = st.button('Start Search!')

    # Dash board's main panel is depend on the side bar selections
    if option == "Complaint Evaluation and Filing Assistant":
        # Set the title for the Streamlit application
        st.title(":writing_hand: Complaint Evaluation and Filing Assistant")
        # Prompt the user to enter their complaint
        st.markdown("### Enter your complaint here")
        # Create a text area for the user to input their complaint, with a height of 500 pixels and a max of 3000 characters
        text = st.text_area("Drag the lower right corner if you need more space.", height=500,max_chars=3000)
        # Check if the 'Submit' button is pressed by the user
        if st.button('Submit'):
            # Process the user's input, clean it, and then fetch relevant predictions regarding the complaint's product and issue
            issue_df, product_df = get_prediction_json(text_normalizer(re.sub('[^a-zA-Z ]', '', text)))
            # Create two columns on the Streamlit application for product and issue respectively
            product_col, issue_col = st.columns(2)
            # Process and display suggestions in the 'Product' column
            with product_col:
                # Calculate the mean and standard deviation of the Relevance scores for products
                product_df_mean = product_df['Relevance'].mean()
                product_df_std = product_df['Relevance'].std()
                # Filter out products that have a Relevance score greater than 0.5 standard deviations from the mean
                product_df_high_counts = product_df[product_df['Relevance'] > product_df_mean + product_df_std].sort_values(by='Relevance', ascending=False)

                # Convert the filtered data to a markdown format for display
                markdown_string = '## Suggested Product for Your Complaint\n\n'
                markdown_string += 'Based on our data analysis, here are the most relevant products for your complaint:\n'

                for index, row in product_df_high_counts.iterrows():
                    markdown_string += f'1. **{row["Product"]}**\n'

                # Display the product suggestions using markdown
                st.markdown(markdown_string)
                st.write("We suggest that you consider these issues while filing your complaint to ensure it's processed accurately and efficiently. If your product is not listed, feel free to use your best judgement.")
            
            # Process and display suggestions in the 'Issue' column
            with issue_col:
                # Calculate the mean and standard deviation of the Relevance scores for issues
                issue_df_mean = issue_df['Relevance'].mean()
                issue_df_std = issue_df['Relevance'].std()
                # Filter out issues that have a Relevance score greater than 0.5 standard deviations from the mean
                issue_df_high_counts = issue_df[issue_df['Relevance'] > issue_df_mean + issue_df_std].sort_values(by='Relevance', ascending=False)

                # Convert the filtered data to a markdown format for display
                markdown_string = '## Suggested Issues for Your Complaint\n\n'
                markdown_string += 'Based on our data analysis, here are the most relevant issues for your complaint:\n'

                for index, row in issue_df_high_counts.iterrows():
                    markdown_string += f'1. **{row["Issue"]}**\n'
                # Display the issue suggestions using markdown
                st.markdown(markdown_string)
                st.write("We suggest that you consider these issues while filing your complaint to ensure it's processed accurately and efficiently. If your product is not listed, feel free to use your best judgement.")
            
            # Create an expander section that contains visualizations of product and issue relevance
            with st.expander("##### Product & Issue Relevance Breakdown: \n\n (Click to expand/collapse)"):
                product_bkd, issue_bkd = st.columns(2, gap='large')
                with product_bkd:
                    # Plot a bar chart for product relevance predictions
                    product_chart = alt.Chart(product_df).mark_bar().encode(x='Relevance:Q',
                                                                            y=alt.Y('Product:O', sort='-x', axis=alt.Axis(labelLimit=400, 
                                                                                                                            title=None, 
                                                                                                                            orient='left',#labelFontSize=8
                                                                                                                            )),
                                                                            tooltip=['Relevance:Q', 'Product:O']
                                                                            ).properties(
                                                                            title="Products Prediction",
                                                                            )
                    st.altair_chart(product_chart, use_container_width=True)
                with issue_bkd:
                    # Plot a bar chart for issue relevance predictions
                    issue_chart = alt.Chart(issue_df).mark_bar().encode(x='Relevance:Q',
                                                                        y=alt.Y('Issue:O', sort='-x', axis=alt.Axis(labelLimit=400, 
                                                                                                                    title=None, 
                                                                                                                    orient='left'#labelFontSize=8
                                                                                                                            )),
                                                                        tooltip=['Relevance:Q', 'Issue:O']
                                                                        ).properties(
                                                                        title="Issues Prediction",
                                                                        )
                    st.altair_chart(issue_chart, use_container_width=True)

    elif option=="Service Provider Comparison Tool" and company_submitted and (product_selected and company_selected and start_date and end_date):
        # Setting the title for the Streamlit app
        st.title(":thinking_face: Service Provider Comparison Tool")
        # Fetch data based on user's selected parameters
        df = get_product_company_record_within_date_range(company_selected, start_date, end_date, [product_selected])
        # Commented out - showing the dataframe can be useful for debugging
        # st.dataframe(df)


        #### First Section
        st.markdown(f"### :mag: **Quick Stats & Trend**")
        # Grouping data by Date and Company to get counts
        L0_company_break_down_df = df.groupby(['Date_received', 'Company']).size().reset_index(name='Counts')
        # Converting the date to pandas datetime format for easier manipulation
        L0_company_break_down_df['Date_received'] = pd.to_datetime(L0_company_break_down_df['Date_received'])
        # Sorting values by Company and Date
        L0_company_break_down_df = L0_company_break_down_df.sort_values(['Company', 'Date_received']).reset_index(drop=True)
        # Splitting the page into columns for visualization
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("#### :bar_chart: Quick Break Down")
            # Explanation for the users
            st.write("""Please note that the number of complaints registered against a company can often be proportionate to the size or scale of the company. 
            A larger number of complaints does not necessarily indicate poor service quality. 
            We recommend examining the trend in daily complaint counts. 
            A noticeable increase could suggest a recent surge in issues reported by customers related to your selected product. These surges tend to be smoothed out in the plot on the right.
            """)
            # Getting the maximum daily complaints for setting chart scales
            daily_max = int(L0_company_break_down_df.Counts.max())
            # This step is to pad the dataframe for data editor view
            # Find the min and max dates in the original data frame
            min_date = L0_company_break_down_df['Date_received'].min()
            max_date = L0_company_break_down_df['Date_received'].max()
            # Create a date range from min to max date
            dates = pd.date_range(start=min_date, end=max_date)
            # Get a list of unique companies
            companies = L0_company_break_down_df['Company'].unique()
            # Create a new data frame with all combinations of dates and companies
            new_df = pd.DataFrame(index=pd.MultiIndex.from_product([dates, companies], names=['Date_received', 'Company'])).reset_index()

            # Merge the new data frame with the original one to get the counts
            L0_company_break_down_df_pretty = pd.merge(new_df, L0_company_break_down_df, on=['Date_received', 'Company'], how='left')
            # Fill NA values in Counts with 0
            L0_company_break_down_df_pretty['Counts'] = L0_company_break_down_df_pretty['Counts'].fillna(0)


            # Group by 'Company' and aggregate 'Counts' into a list and also calculate the sum
            L0_company_break_down_df_pretty = L0_company_break_down_df_pretty.groupby('Company').agg({'Counts': [list, sum]}).reset_index()
            # Flatten the MultiIndex in columns
            L0_company_break_down_df_pretty.columns = L0_company_break_down_df_pretty.columns.get_level_values(0) + '_' + L0_company_break_down_df_pretty.columns.get_level_values(1)
            # Manually rename the columns
            L0_company_break_down_df_pretty.rename(columns={'Company_':'Company', 'Counts_list':'Daily_Comp_Count', 'Counts_sum':'Total_Comp_Count'}, inplace=True)
            L0_company_break_down_df_pretty = L0_company_break_down_df_pretty.sort_values(by='Total_Comp_Count', ascending=False)
            # visualize the data
            st.data_editor(
                data = L0_company_break_down_df_pretty,
                column_config={
                    "Total_Comp_Count": st.column_config.ProgressColumn(
                                                                "Total Complaints",
                                                                help = f"Total number of CFPB complaints filed against the institution within selected date range.",
                                                                format="%f",
                                                                min_value=0,
                                                                max_value=int(L0_company_break_down_df_pretty.Total_Comp_Count.max()),
                                                                ),
                    "Daily_Comp_Count": st.column_config.LineChartColumn(
                                                        "Daily Complaint Trend",
                                                        help=f"Trend of daily CFPB complaint number filed against the institution within selected date range.",
                                                        y_min=0,
                                                        y_max=daily_max,
                                                        ),
                                },
                hide_index=True,
                use_container_width = True
            )
        with col2:
            st.markdown("#### :chart_with_downwards_trend: Complaint Counts Over Time")
            # Start by defining a base chart, which includes the data points plotted as circles
            y_max = np.percentile(L0_company_break_down_df['Counts'], 99)

            base = alt.Chart(L0_company_break_down_df).mark_circle(opacity=0.25, clip=True).encode(
                # For the x-axis, use the Date_received field, formatted as a date, and name the axis "Date"
                alt.X('Date_received:T', title='Date', axis=alt.Axis(format="%m/%d/%y")),
                
                # For the y-axis, use the Counts field, formatted as an integer, and name the axis "Complaint Counts"
                alt.Y('Counts:Q', title='Complaint Counts', axis=alt.Axis(format='d'), scale=alt.Scale(domain=(0, y_max))),
                
                # For the color encoding, use the Company field, and place the legend at the bottom
                alt.Color('Company:N', legend=alt.Legend(orient='bottom'))
            )

            # Create a smoothed line for each Company, using the transform_loess function
            loess = base.transform_loess('Date_received', 'Counts', groupby=['Company']).mark_line(size=4)

            # Combine the base chart with the smoothed line chart
            # Uncomment the next line if you want to allow legend labels to use as much space as they need
            # chart = (base + loess).configure_legend(labelLimit=0)
            chart = (base + loess)#.properties(title='Complaint Counts Over Time')

            # Display the chart using Streamlit, and allow it to use the full width of the container
            st.altair_chart(chart, use_container_width=True)
            st.markdown(":warning: Please be aware the trend line tend to smooth out the surges.")
        
        #### Second Section
        st.markdown(f"### :male-detective: **Biggest Pain Points Breakdown**")
        companies = df.Company.unique()
        cols = st.columns(len(companies),)
        for i in range(len(companies)):
            company = companies[i]
            df_company = df[df.Company==company].copy()
            total_complaint_counts = len(df_company)
            timely_response_count = len(df_company[df_company['Timely_response']=='Yes'])

            df_company['Consumer_disputed'] = df_company.apply(lambda row: 'Not yet' if (row['Consumer_disputed']=='NaN' and 'In progress' in row['Company_response_to_consumer']) else 'No',
                                                axis=1)

            with cols[i]:
                with st.expander(f"##### :classical_building: **{company}** \n :clipboard: **{total_complaint_counts}** complaints  :stopwatch: Timely response rate: **{round(timely_response_count/total_complaint_counts,2):.2%}**"):

                    # Extracting company-specific data and aggregating it for visualization
                    agg_data = df_company.groupby(['Company_response_to_consumer', 'Consumer_disputed']).size().reset_index(name='counts')

                    # Pivot the data to have 'Consumer_disputed' values as columns
                    pivoted_data = agg_data.pivot_table(index='Company_response_to_consumer', columns='Consumer_disputed', values='counts', fill_value=0).reset_index()
                    pivoted_data = pivoted_data.melt(id_vars='Company_response_to_consumer', value_name='counts')
                    # Visualizing complaint counts over time using Altair charts
                    chart = alt.Chart(pivoted_data).mark_bar().encode(
                        y=alt.Y('Company_response_to_consumer:N', axis=alt.Axis(labelLimit=400,title='Company Response')),
                        x=alt.X('sum(counts):Q', title='Count of Responses'),
                        color=alt.Color('Consumer_disputed:N', legend=alt.Legend(title="Consumer Disputed", orient="bottom")),
                        order=alt.Order(
                        # Sort the segments of the bars by this field
                        'Consumer_disputed:N',
                        sort='ascending'
                        )
                    ).properties(title="Distribution of Consumer Disputes by Company Response")


                    st.altair_chart(chart, use_container_width=True)

                    grouped = df_company.groupby(['Issue', 'Sub_issue']).size().reset_index(name='Counts')
                    # Find combinations that have significantly higher counts
                    # This could be done in a variety of ways, but for this example, let's consider
                    # combinations with counts higher than one standard deviation from the mean to be significant
                    mean = grouped['Counts'].mean()
                    std = grouped['Counts'].std()

                    # Displaying issues and sub-issues for each company
                    high_counts = grouped[grouped['Counts'] > mean + 0.5*std].sort_values(by='Counts', ascending=False)
                    # Print combinations in Streamlit markdown
                    st.divider()
                    for index, row in high_counts.iterrows():
                        st.markdown(f"* **{row['Issue']}**  \n   * {row['Sub_issue']}")


                        st.caption(f"{row['Counts']} / {total_complaint_counts} ({round(row['Counts']/total_complaint_counts,2):.1%}) total complaints.")
        st.markdown("### A More Comprehensive Breakdown...")
        # Using a Streamlit expander for optional detailed visualization
        with st.expander("Click for drop down interactive visualization"):
            st.write("Use the expand button at the upper right of the visualization to inspect more details. You can also click on the boxes to zoom in. To zoom out, you'll need to click the top/left section.")
            # Calculate counts of subissues under each issue for each company
            df_counts = df.groupby(['Company', 'Issue', 'Sub_issue']).size().reset_index(name='counts')
            df_counts['root'] = 'Breakdown'
            # Calculate total
            total = df_counts['counts'].sum()
            # Calculate percentage and add as new column
            df_counts['percentage'] = round(df_counts['counts'] / total * 100, 2)
            # Creating icicle and treemap visualizations using Plotly
            fig = px.icicle(df_counts,
                            path=['root','Company', 'Issue', 'Sub_issue'],
                            values='counts',
                            )

            fig.update_traces(textfont_size=20,
                            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[1]:.2f}%", 
                            selector=dict(type='icicle'),
                            root_color="#333333") # text color set to black

            fig.update_layout(margin=dict(t=25, l=25, r=25, b=25), 
                            coloraxis_colorbar=dict(title="Counts"),
                            uniformtext_minsize=15,
                            # uniformtext_mode='hide'
                            )

            st.plotly_chart(fig, use_container_width=True, sharing='streamlit', theme='streamlit')
            

            # Plotly Treemap chart
            fig = px.treemap(df_counts, 
                            path=['root','Company', 'Issue', 'Sub_issue'],
                            values='counts',
                            hover_data=['counts', 'percentage'],
                            color_continuous_scale='RdBu',)

            fig.update_traces(textfont_size=20,
                            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[1]:.2f}%", 
                            selector=dict(type='treemap'),
                            # textfont=dict(color='black'),
                            root_color="#333333") #"whitesmoke"

            fig.update_layout(margin=dict(t=25, l=25, r=25, b=25), 
                            coloraxis_colorbar=dict(title="Counts"),
                            uniformtext_minsize=15,
                            # uniformtext_mode='hide'# text color set to black 
                            )

            st.plotly_chart(fig, use_container_width=True, sharing='streamlit', theme='streamlit')

    # Yang Cui's Design
    elif option=="Enterprise Customer Complaint Analysis Tool" and company_submitted and (company_names and start_date and end_date):
        #show report details of selected report ID
        def show_detailed_report(ID):
            ID_data = data[data['Complaint ID'] == ID]
            st.markdown(f"## Detailed Report of Complaint ID **:red[{ID}]**:")
            dic = ID_data.T[ID_data.index.values[0]].to_dict()
            with st.expander('Detailed Report'):
                for inx, item in enumerate(dic.keys()):
                    st.markdown(f"**:blue[{item}]**: **{dic[item]}**")
            st.markdown('----')
            
        # generate summary insights of each selected financial institution over selected period of time
        def generate_insights_summary(company_names):
            with col2:
                st.header('Insights Summary')
                for company_name in company_names:
                    try:
                        st.write(f"**{company_name}**")
                        col12, col13 = st.columns([5, 5])
                        with col12:
                            #visual 1
                            with st.expander('Geographical Distribution'):
                                fig, ax = plt.subplots(figsize=(16, 16))
                                company_data = data[data['Company'] == company_name]

                                # compute number of complaints by state
                                complaint_counts = company_data['State'].value_counts()

                                # load US map
                                us_states = gpd.read_file('States_shapefile.shp')

                                # merge Map data with complaint data
                                merged_data = us_states.merge(complaint_counts, left_on='State_Code', right_index=True)

                                # plot geographical distribution
                                merged_data.plot(column='State_Code', cmap='Blues', ax=ax, linewidth=0.8, edgecolor='0.8',
                                                legend=True)
                                ax.set_aspect('auto')  # adjust ratio
                                ax.set_title('Number of Complaints by State', size=24)
                                ax.tick_params(axis='both',
                                            labelsize=18, bottom=False, top=False, left=False, right=False,
                                            labelbottom=False, labelleft=False)
                                st.pyplot(fig, use_container_width=True)# show plot in streamlit
                            #visual 3
                            with st.expander('Products'):
                                fig, ax = plt.subplots(figsize=(16, 16))
                                # compute quantity of complaints by product
                                product_counts = company_data['Product'].value_counts()
                                product_counts.plot(kind='barh', stacked=False, ax=ax)
                                ax.set_xlabel('Number of Complaints')
                                ax.set_ylabel('Product Type')
                                ax.set_title('Number of Products Complained by Customers', size=24)
                                ax.tick_params(axis='x',
                                            labelsize=18)
                            
                                st.pyplot(fig, use_container_width=True)
                        
                        with col13:
                            #visual 2
                            with st.expander('Trend'):
                                fig, ax = plt.subplots(figsize=(16, 16))
                                # convert datatype to datatime
                                company_data['Date received'] = pd.to_datetime(company_data['Date received'], format='%Y-%m-%d')

                                # compute the total number of complaints by date
                                complaints_by_date = company_data.groupby(company_data['Date received'].dt.date)[
                                    'Company'].count().reset_index()

                                # plot line chart of number of complaints change over time
                                ax.plot(complaints_by_date['Date received'], complaints_by_date['Company'])
                                ax.set_xlabel('Date')
                                ax.set_ylabel('Total Number of Complaints')
                                ax.set_title('Total Number of Complaints Change over Time', size=24)
                                ax.tick_params(axis='x',
                                            labelsize=18, rotation=45)
                                ax.tick_params(axis='y',
                                            labelsize=18)
                                plt.legend(fontsize=10)
                                st.pyplot(fig, use_container_width=True)
                            #visual 4
                            with st.expander('Issues'):
                                fig, ax = plt.subplots(figsize=(16, 16))
                                # Compute number of complained issues per product type
                                issue_counts = company_data.groupby(['Product', 'Issue']).size().unstack()
                                # plot bar chart
                                issue_counts.plot(kind='barh', stacked=True, ax=ax)
                                ax.set_xlabel('Number of Complaints')
                                ax.set_ylabel('Product Type')
                                ax.set_title('Number of Issues per Product Type Complained by Consumers ', size=24)
                                ax.tick_params(axis='x',
                                            labelsize=18)
                                ax.legend(fontsize=8)
                                plt.legend(fontsize=8)
                                st.pyplot(fig, use_container_width=True)
                    except:
                        st.write(f"No compliant data found for **{company_name}** during selected time period.")
                st.markdown('----')

        # generate complaint analyses comparison of selected companies over selected period of time
        def competitor_analysis(company_names):
            col4, col5 = st.columns([5, 5])
            with col4:
                #visual 1
                with st.expander('Processing Time Comparison'):
                    competitor_data = data[data['Company'].isin(company_names)]

                    competitor_data['Date received'] = pd.to_datetime(competitor_data['Date received'], format='%Y-%m-%d')
                    competitor_data['Date sent to company'] = pd.to_datetime(competitor_data['Date sent to company'],
                                                                            format='%Y-%m-%d')
                    competitor_data['Processing Time'] = (competitor_data['Date sent to company'] - competitor_data[
                        'Date received']).dt.total_seconds() / 3600

                    if competitor_data.empty:
                        st.write("Please Select Companies for Comparison")
                        return

                    # compute processing delays for each company
                    avg_processing_time = competitor_data.groupby('Company')['Processing Time'].mean().reset_index()
                    avg_processing_time = avg_processing_time.sort_values(by='Processing Time')

                    
                
                    avg_processing_time.set_index('Company').plot(kind='barh', figsize=(10, 6))
                    #plot bar chart
                    plt.ylabel('Company')
                    plt.xlabel('Avg Processing Time (hr)', size=18)
                    plt.title('Complaint Processing Delay Comparison', size=20)
                    plt.xticks(size=18)
                    plt.tight_layout()
                    st.pyplot(plt, use_container_width=True)  
                #visual 3
                with st.expander('Top 2 Products Comparison'):
                    # identify the top 2 most complained product types of the whole dataset across all institutions
                    top_products = data['Product'].value_counts().index[:2]

                    # compute number of top 2 most complained products by company
                    product_data = data[data['Company'].isin(company_names) & data['Product'].isin(top_products)]
                    product_counts = product_data.groupby(['Company', 'Product']).size().unstack()

                    # plot bar chart
                    product_counts.plot(kind='barh', stacked=False, figsize=(10, 6))
                    plt.xlabel('Number of Complaints')
                    plt.ylabel('Company', size=18)
                    plt.title('Quantity of Top 2 Most Complained Products by Company', size=20)
                    plt.xticks(size=18)
                    plt.legend(fontsize=10)
                    st.pyplot(plt, use_container_width=True) 
                
            with col5:
                #visual 2
                with st.expander('Company Response Comparison'):
                    X = []
                    labels = []
                    for company in company_names:
                        response_counts = competitor_data[competitor_data['Company'] == company][
                            'Company response to consumer'].value_counts()
                        X.append(response_counts.values)
                        labels.append(response_counts.index)
                        #plot pie chart
                        plt.figure(figsize=(10, 10))
                        plt.pie(response_counts.values, labels=response_counts.index,
                                autopct='%1.1f%%', startangle=90,
                                textprops={'fontsize': 20,  
                                        })
                        plt.title(f'{company}', fontsize=20, fontweight="bold")
                        st.pyplot(plt, use_container_width=True)
                #visual 4
                with st.expander('Top 2 Issues Comparison'):
                    # identify the top 2 most complained issue types of the whole dataset across all institutions
                    top_issues = data['Issue'].value_counts().index[:2]

                    # Number of top 2 most complained issues by company
                    issue_data = data[data['Company'].isin(company_names) & data['Issue'].isin(top_issues)]
                    issue_counts = issue_data.groupby(['Company', 'Issue']).size().unstack()

                    # plot bar chart
                    issue_counts.plot(kind='barh', figsize=(10, 6))
                    plt.xlabel('Number of Complaints')
                    plt.ylabel('Company', size=18)
                    plt.title('Quantity of Top 2 Most Complained Issues by Company', size=20)
                    plt.xticks(size=18)
                    plt.legend(fontsize=10)
                    st.pyplot(plt, use_container_width=True)
                
            st.markdown('----')

        #generating insights into segmentation by demographic tag, state, and zip of selected companies over selected period of time
        def consumer_segmentation(company_names, segmentation_option):
            col7, col8 = st.columns([5,5])
            company_data = data[data['Company'].isin(company_names)]
            
            if segmentation_option == 'Demographic Tags':
                with col7:
                    #Visual 1
                    with st.expander('Number of Complaints'):
                        tags_counts = company_data['Tags'].value_counts()
                        #plot bar chart
                        plt.figure(figsize=(10, 6))
                        plt.bar(tags_counts.index, tags_counts.values)
                        plt.xlabel('Tag', size=18)
                        plt.ylabel('Number of Complaints', size=18)
                        plt.title('Number of Complaints by Demographic Tags', size=22)
                        plt.xticks(rotation=30, size=18)
                        st.pyplot(plt, use_container_width=True)
                    #Visual 3
                    with st.expander('Products'):
                        fig, ax= plt.subplots(figsize= (16,16))
                        product_tag = company_data.groupby(['Tags', 'Product']).size().unstack()
                        #plot bar chart
                        product_tag.plot(kind='bar', stacked=True, ax=ax)
                        plt.xlabel('Tag', size=18)
                        plt.ylabel('Number of Complaints', size=18)
                        plt.title('Number of Complained Product Types by Demographic Tags', size=22)
                        plt.xticks(rotation=30, size=18)
                        ax.legend(fontsize=8)
                        plt.legend(fontsize=8)
                        st.pyplot(plt, use_container_width=True)
                with col8:
                    #Visual 2
                    with st.expander('Company Response'):
                        response_tag = company_data.groupby(['Tags', 'Company response to consumer']).size().unstack()

                        #Plot bar chart
                        response_tag.plot(kind='bar', stacked=True, figsize=(10, 6))
                        plt.xlabel('Tag', size=18)
                        plt.ylabel('Number of Complaints', size=18)
                        plt.title('Company Response to Consumers by Demongraphic Tag', size=22)
                        plt.xticks(rotation=30, size=18)
                        plt.tight_layout()
                        plt.legend(fontsize=10)
                        st.pyplot(plt, use_container_width=True)
                    #Visual 4
                    with st.expander('Issues'):
                        fig, ax= plt.subplots(figsize= (16,16))
                        issue_tag = company_data.groupby(['Tags', 'Issue']).size().unstack()
                        
                        #Plot bar chart
                        issue_tag.plot(kind='bar', stacked=True, ax=ax)
                        plt.xlabel('Tag', size=18)
                        plt.ylabel('Number of Complaints', size=18)
                        plt.title('Number of Complained Issue Types by Demographic Tags', size=22)
                        plt.xticks(rotation=30, size=18)
                        ax.legend(fontsize=8)
                        plt.legend(fontsize=8)
                        st.pyplot(plt, use_container_width=True)
                    

            elif segmentation_option == 'State':
                with col7:
                    #Visual 1
                    with st.expander('Number of Complaints'):
                        state_counts = company_data['State'].value_counts()
                        #plot bar chart
                        plt.figure(figsize=(10, 6))
                        plt.bar(state_counts.index, state_counts.values)
                        plt.xlabel('State', size=18)
                        plt.ylabel('Number of Complaints', size=18)
                        plt.title('Number of Complaints by State', size=22)
                        plt.xticks(rotation=90, size=8)
                        st.pyplot(plt, use_container_width=True)

                    #Visual 3
                    with st.expander('Products'):
                        fig, ax= plt.subplots(figsize= (16,16))
                        product_state = company_data.groupby(['State', 'Product']).size().unstack()
                        #plot bar chart
                        product_state.plot(kind='barh', stacked=True, ax=ax)
                        ax.set_xlabel('State', size=18)
                        ax.set_ylabel('Number of Complaints', size=18)
                        ax.set_title('Number of Complained Product Types by States', size=22)
                        ax.tick_params(axis='x', labelsize=18)
                        ax.legend(fontsize=8)
                        plt.legend(fontsize=8)
                        st.pyplot(plt, use_container_width=True)
                                    
                with col8:
                    #Visual 2
                    with st.expander('Company Response'):
                        response_state = company_data.groupby(['State', 'Company response to consumer']).size().unstack()

                        #Plot bar chart
                        response_state.plot(kind='bar', stacked=True, figsize=(10, 6))
                        plt.xlabel('Number of Complaints', size=18)
                        plt.ylabel('State', size=18)
                        plt.title('Company Response to Consumers by State', size=22)
                        plt.xticks(rotation=30, size=8)
                        plt.tight_layout()
                        plt.legend(fontsize=10)
                        st.pyplot(plt, use_container_width=True)
                    #Visual 4
                    with st.expander('Issues'):
                        fig, ax= plt.subplots(figsize= (16,16))
                        issue_state = company_data.groupby(['State', 'Issue']).size().unstack()
                        
                        #plot bar chart
                        issue_state.plot(kind='barh', stacked=True, ax=ax)
                        ax.set_xlabel('Number of Complaints', size=18)
                        ax.set_ylabel('State', size=18)
                        ax.set_title('Number of Complained Issue Types by States', size=22)
                        ax.tick_params(axis='x', labelsize=18)
                        ax.legend(fontsize=8)
                        plt.legend(fontsize=8)
                        st.pyplot(plt, use_container_width=True)
                    
            elif segmentation_option == 'ZIP code':
                with col7:
                    #Visual 1
                    with st.expander('Number of Complaints'):
                        zipcode_counts = company_data['ZIP code'].value_counts().head(20)
                        zipcode_counts.index = zipcode_counts.index.astype('object').astype('str').str[:-2]
                        x_ticks = range(len(zipcode_counts))
                        
                        #plot bar chart
                        plt.figure(figsize=(10, 6))
                        plt.bar(x_ticks, zipcode_counts.values)
                        plt.xlabel('ZipCode', size=18)
                        plt.ylabel('Number of Complaints', size=18)
                        plt.title('Number of Complaints by ZipCode (Top20) ', size=22)
                        plt.xticks(x_ticks, zipcode_counts.index,rotation=30, size=8)
                        st.pyplot(plt, use_container_width=True)
                    #Visual 3
                    with st.expander('Products'):
                        fig, ax= plt.subplots(figsize= (16,16))
                        value_counts = company_data['ZIP code'].value_counts()
                        # Get the top 20 values based on counts
                        top_20 = value_counts.head(20).index.tolist()
                        # Filter the DataFrame based on the top values
                        filtered_df = company_data[company_data['ZIP code'].isin(top_20)]
                        filtered_df['ZIP code']=filtered_df['ZIP code'].astype('object').astype('str').str[:-2]
                        product_zip = filtered_df.groupby(['ZIP code', 'Product']).size().unstack()
                        
                        #plot bar chart
                        product_zip.plot(kind='barh', stacked=True, ax=ax)
                        ax.set_xlabel('Number of Complaints', size=18)
                        ax.set_ylabel('ZipCode', size=18)
                        ax.set_title('Number of Complained Product Types ZipCode (Top20)', size=22)
                        ax.tick_params(axis='x', labelsize=18)
                        ax.legend(fontsize=8)
                        plt.legend(fontsize=8)
                        st.pyplot(plt, use_container_width=True)
                        
                with col8:
                    #Visual 2
                    with st.expander('Company Response'):
                        response_state = filtered_df.groupby(['ZIP code', 'Company response to consumer']).size().unstack()

                        #Plot bar chart
                        response_state.plot(kind='bar', stacked=True, figsize=(10, 6))
                        plt.xlabel('ZipCode', size=18)
                        plt.ylabel('Number of Complaints', size=18)
                        plt.title('Company Response to Consumers by ZipCode (Top20)', size=22)
                        plt.xticks(rotation=30, size=8)
                        plt.tight_layout()
                        plt.legend(fontsize=10)
                        st.pyplot(plt, use_container_width=True)
                    #Visual 4
                    with st.expander('Issues'):
                        fig, ax= plt.subplots(figsize= (16,16))
                        issue_state = filtered_df.groupby(['ZIP code', 'Issue']).size().unstack()
                        
                        #plot bar chart
                        issue_state.plot(kind='barh', stacked=True, ax=ax)
                        ax.set_xlabel('Number of Complaints', size=18)
                        ax.set_ylabel('ZipCode', size=18)
                        ax.set_title('Number of Complained Issue Types by ZipCode (Top20)', size=22)
                        ax.tick_params(axis='x', labelsize=18)
                        ax.legend(fontsize=8)
                        plt.legend(fontsize=8)
                        st.pyplot(plt, use_container_width=True)
          
            st.markdown('----')
        
        data = get_product_company_record_within_date_range(company_names, start_date, end_date)
        data.rename(columns=reversed_dict, inplace=True)

        col1, col2 = st.columns([1, 5])
        with col1:
            ID = st.selectbox('Please Select a Complaint ID to Show Detailed Report',
                      data['Complaint ID'].unique())
            show_detailed_report(ID)

        if company_submitted:
            with st.spinner('Generating Analyses，Please wait...'):
                generate_insights_summary(company_names)
                st.header('Competitor Analysis')
                competitor_analysis(company_names)
                st.header('Consumer Segmentation')
                consumer_segmentation(company_names,segmentation_option)

    disclaimer = """
    ### Disclaimer
    Please note that this application does not store any data input by the user nor any cookie data. It solely serves to make requests to specified URLs and display the returned results.
    """
    st.markdown(disclaimer)

if __name__ == "__main__":
    main()
