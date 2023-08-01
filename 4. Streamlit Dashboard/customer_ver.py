import requests
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
import re

import psycopg2
from psycopg2 import extras

def create_connection():
    user="xxxxx"
    password="xxxxx"
    database = "cfpb"
    host = "xx.xx.xx.xx"
    # Connect to your postgres server
    conn = psycopg2.connect(
        dbname=database,
        user=user,
        password=password,
        host=host
    )
    # Create a cursor object
    cur = conn.cursor()
    return cur,conn

def text_normalizer(text):
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
    response = requests.get('https://cfpb-eval-ltsyz5ld4a-uc.a.run.app/hello')
    return response.text
def get_prediction_json(text):
    response = requests.get(f'https://cfpb-eval-ltsyz5ld4a-uc.a.run.app/complaint_eval/{text}')
    data = response.json()
    issue_df = pd.DataFrame.from_dict(data["Issue"], orient='index', columns=["Value"]).reset_index().rename(columns={"index": "Issue"})
    product_df = pd.DataFrame.from_dict(data["Product"], orient='index', columns=["Value"]).reset_index().rename(columns={"index": "Product"})
    return issue_df, product_df

def visualize_data(df, title, x, y):
    chart = alt.Chart(df).mark_bar().encode(
        x=x,
        y=alt.Y(y, sort='-x'),
        tooltip=[x, y]
    ).properties(
        title=title
    )
    return chart

def get_data_template():
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
def get_unique_products(days_of_hist):
    cur, conn = create_connection()
    query = """
    SELECT 
        "Product", 
        COUNT("Product") AS ProductCount
    FROM 
        cfpb
    WHERE
        "Date_received" >= CURRENT_DATE - INTERVAL '{} days'
    GROUP BY 
        "Product"
    ORDER BY 
        ProductCount DESC;
    """
    query = query.format(days_of_hist)
    df = pd.read_sql_query(query, conn)
    cur.close()
    conn.close()
    return df.Product.to_list()
@st.cache_data
def get_uniuqe_company_by_products_within_x_days(product_lst,days_of_hist):
    cur, conn = create_connection()
    #Prepare the SQL query
    query = """
    SELECT 
        "Company", 
        COUNT("Company") AS CompanyCount
    FROM 
        cfpb
    WHERE 
        "Product" IN ({})
    AND
        "Date_received" >= CURRENT_DATE - INTERVAL '{} days'
    GROUP BY 
        "Company"
    ORDER BY 
        CompanyCount DESC;
    """
    # Format the query string with the product list
    # Convert each product in product_lst into a string enclosed by single quotes
    formatted_product_lst = ", ".join("'{}'".format(p) for p in product_lst)
    query = query.format(formatted_product_lst,days_of_hist)
    # Execute the query and get the data
    df = pd.read_sql_query(query, conn)
    cur.close()
    conn.close()
    # Return the company list
    return df.Company.to_list()
@st.cache_data
def get_product_company_record_within_x_days(product_lst, company_lst, days_of_hist):
    cur, conn = create_connection()
    # Prepare the SQL query
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
        "Consumer_consent_provided",
        "Company_response_to_consumer",
        "Timely_response", "Consumer_disputed", "Complaint_ID"
    FROM 
        cfpb
    WHERE 
        "Product" IN ({})
    AND 
        "Company" IN ({})
    AND
        "Date_received" >= CURRENT_DATE - INTERVAL '{} days'
    ;
    """
    # Format the query string with the product and company list
    formatted_product_lst = ", ".join("'{}'".format(p) for p in product_lst)
    formatted_company_lst = ", ".join("'{}'".format(c) for c in company_lst)
    query = query.format(formatted_product_lst, formatted_company_lst, days_of_hist)

    # Execute the query and get the data
    df = pd.read_sql_query(query, conn)
    cur.close()
    conn.close()
    return df



# Initialize a new session state if it doesn't exist
if 'days_history' not in st.session_state:
    st.session_state['days_history'] = 60
days_history = 60
def update_days_history():
    st.session_state.days_history = days_history
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
def main():
    st.set_page_config(layout='wide')

    with st.sidebar:
        option = st.selectbox("Select an Option:", ["Service Provider Comparison Tool", "Complaint Evaluation and Filing Assistant"])
        if option == "Complaint Evaluation and Filing Assistant":
            st.write("Warming up the API :)")
            ping_hello()
        elif option == "Service Provider Comparison Tool":
            product_selected = st.selectbox(label = "Select A Product:", 
                                              options = get_unique_products(st.session_state['days_history']),
                                              placeholder = "Select A Product")

            company_selected = st.multiselect(label = "Select Service Provider(s):",
                                              options = get_uniuqe_company_by_products_within_x_days([product_selected], st.session_state['days_history']),
                                              placeholder = "Select Up To 5 Company(s)",
                                              max_selections=5)
            days_history = st.sidebar.number_input("Select days history:", min_value=30, max_value=180, step=30, value=st.session_state['days_history'], on_change=update_days_history)

            company_submitted = st.button('Submit')

    
    ########################################################################
    ########################################################################
    ########################################################################
    if option == "Complaint Evaluation and Filing Assistant":
        text = st.text_area('Enter your complaint here')
        if st.button('Submit'):
            issue_df, product_df = get_prediction_json(text_normalizer(text))
            st.dataframe(issue_df)
            st.dataframe(product_df)
            issue_chart = visualize_data(issue_df, "Issues Prediction", "Value:Q", "Issue:O")
            product_chart = visualize_data(product_df, "Products Prediction", "Value:Q", "Product:O")
            st.altair_chart(issue_chart)
            st.altair_chart(product_chart)
    elif option=="Service Provider Comparison Tool" and company_submitted and (product_selected and company_selected and days_history):
            df = get_product_company_record_within_x_days([product_selected], company_selected, days_history)
            col1, col2 = st.columns([3, 2])
            # First Group By Company and Date
            L0_company_break_down_df = df.groupby(['Date_received', 'Company']).size().reset_index(name='Counts')
            # Convert 'Date_received' in L0_company_break_down_df to datetime
            L0_company_break_down_df['Date_received'] = pd.to_datetime(L0_company_break_down_df['Date_received'])
            L0_company_break_down_df = L0_company_break_down_df.sort_values(['Company', 'Date_received']).reset_index(drop=True)
            st.dataframe(L0_company_break_down_df)

            with col1:
                st.markdown("Quick Break Down")
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

                # visualize the data
                st.data_editor(
                    data = L0_company_break_down_df_pretty,
                    column_config={
                        "Total_Comp_Count": st.column_config.ProgressColumn(
                                                                    "Total Complaints",
                                                                    help = f"Total number of CFPB complaints filed against the institution in the last {days_history} days.",
                                                                    format="%f",
                                                                    min_value=0,
                                                                    max_value=int(L0_company_break_down_df_pretty.Total_Comp_Count.max()),
                                                                    ),
                        "Daily_Comp_Count": st.column_config.LineChartColumn(
                                                            "Daily Complaint Trend",
                                                            help=f"Trend of daily CFPB complaint number filed against the institution in the last {days_history} days.",
                                                            y_min=0,
                                                            y_max=daily_max,
                                                            ),
                                    },
                    hide_index=True,
                    use_container_width = True
                )
            with col2:
                st.markdown("Complaint Counts Over Time")
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
        


    disclaimer = """
    ### Disclaimer
    Please note that this application does not store any data input by the user nor any cookie data. It solely serves to make requests to specified URLs and display the returned results.
    """
    st.markdown(disclaimer)

if __name__ == "__main__":
    main()
