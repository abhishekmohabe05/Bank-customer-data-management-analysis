# import streamlit as st
# import pandas as pd
# import os
# from main import load_data, clean_data, analyze_data, analyze_preferences, export_report, generate_charts, embed_charts_in_excel


# def search_applicant(df_app_data, application_no):
#     """Search for an applicant by application number."""
#     # Try different column names for application number
#     app_no_cols = ['Application_Number', 'org_id', 'Application No', 'Application_No']
#     app_no_col = None
    
#     for col in app_no_cols:
#         if col in df_app_data.columns:
#             app_no_col = col
#             break
    
#     if app_no_col is None:
#         return None
    
#     # Convert application_no to string for comparison
#     application_no = str(application_no).strip()
    
#     # Search for the applicant
#     result = df_app_data[df_app_data[app_no_col].astype(str).str.strip() == application_no]
    
#     if not result.empty:
#         return result.iloc[0]
#     else:
#         return None


# def main():
#     st.set_page_config(page_title="Applicant Data Management System", layout="wide")
    
#     st.title("🏠 Applicant Data Management and Preference Analysis System")
#     st.markdown("---")
    
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.selectbox("Choose a page", ["Search Applicant", "Generate Analysis Report", "View Charts"])
    
#     # Load data
#     if 'app_data' not in st.session_state or 'app_pref' not in st.session_state:
#         with st.spinner("Loading data..."):
#             try:
#                 app_data, app_pref = load_data('All_Applicant_Total_Data.xlsx')
#                 app_data_clean, app_pref_clean = clean_data(app_data, app_pref)
#                 st.session_state.app_data = app_data_clean
#                 st.session_state.app_pref = app_pref_clean
#                 st.success("Data loaded successfully!")
#             except Exception as e:
#                 st.error(f"Error loading data: {str(e)}")
#                 return
    
#     app_data = st.session_state.app_data
#     app_pref = st.session_state.app_pref
    
#     if page == "Search Applicant":
#         st.header("🔍 Search Applicant Details")
        
#         # Search input
#         application_no = st.text_input("Enter Application Number:", placeholder="e.g., 3220007102")
        
#         if st.button("Search"):
#             if application_no:
#                 result = search_applicant(app_data, application_no)
                
#                 if result is not None:
#                     st.success(f"Applicant found!")
                    
#                     # Display applicant details in a nice format
#                     col1, col2 = st.columns(2)
                    
#                     with col1:
#                         st.subheader("Personal Information")
#                         if 'Full_Name' in result.index:
#                             st.write(f"**Name:** {result['Full_Name']}")
#                         if 'Gender' in result.index:
#                             st.write(f"**Gender:** {result['Gender']}")
#                         if 'DOB' in result.index:
#                             st.write(f"**Date of Birth:** {result['DOB']}")
#                         if 'Category' in result.index:
#                             st.write(f"**Category:** {result['Category']}")
#                         if 'Income' in result.index:
#                             st.write(f"**Income:** ₹{result['Income']:,}")
                    
#                     with col2:
#                         st.subheader("Contact & Location")
#                         if 'Mobile' in result.index:
#                             st.write(f"**Mobile:** {result['Mobile']}")
#                         if 'email' in result.index:
#                             st.write(f"**Email:** {result['email']}")
#                         if 'Taluka' in result.index:
#                             st.write(f"**Taluka:** {result['Taluka']}")
#                         if 'Village' in result.index:
#                             st.write(f"**Village:** {result['Village']}")
#                         if 'Status' in result.index:
#                             st.write(f"**Status:** {result['Status']}")
                    
#                     # Show validation status
#                     st.subheader("Validation Status")
#                     col3, col4 = st.columns(2)
#                     with col3:
#                         if 'Aadhaar_Valid' in result.index:
#                             status = "✅ Valid" if result['Aadhaar_Valid'] else "❌ Invalid"
#                             st.write(f"**Aadhaar:** {status}")
#                     with col4:
#                         if 'IFSC_Valid' in result.index:
#                             status = "✅ Valid" if result['IFSC_Valid'] else "❌ Invalid"
#                             st.write(f"**IFSC:** {status}")
                    
#                     # Show full details in expandable section
#                     with st.expander("View All Details"):
#                         st.dataframe(result.to_frame().T)
                
#                 else:
#                     st.error("Applicant not found. Please check the application number.")
#             else:
#                 st.warning("Please enter an application number.")
    
#     elif page == "Generate Analysis Report":
#         st.header("📊 Generate Analysis Report")
        
#         if st.button("Generate Complete Analysis Report"):
#             with st.spinner("Generating analysis report..."):
#                 try:
#                     # Perform analysis
#                     gender_dist, category_dist, income_stats, taluka_village_app, status_app = analyze_data(app_data)
#                     project_prefs, most_popular_proj, applicant_pref_order = analyze_preferences(app_pref)
                    
#                     # Export report
#                     output_excel = 'analysis_report.xlsx'
#                     export_report(
#                         output_excel,
#                         gender_dist,
#                         category_dist,
#                         income_stats,
#                         status_app,
#                         project_prefs,
#                         taluka_village_app,
#                     )
                    
#                     # Generate charts
#                     charts_dir = 'charts'
#                     chart_paths = generate_charts(app_data, charts_dir)
#                     embed_charts_in_excel(output_excel, chart_paths)
                    
#                     st.success("Analysis report generated successfully!")
                    
#                     # Display summary
#                     st.subheader("Analysis Summary")
#                     col1, col2, col3 = st.columns(3)
                    
#                     with col1:
#                         st.metric("Total Applicants", len(app_data))
#                         st.metric("Most Popular Project", most_popular_proj)
                    
#                     with col2:
#                         if not gender_dist.empty:
#                             st.metric("Male Applicants", gender_dist.get('Male', 0))
#                             st.metric("Female Applicants", gender_dist.get('Female', 0))
                    
#                     with col3:
#                         st.metric("Average Income", f"₹{income_stats['Average_Income']:,.0f}")
#                         st.metric("Max Income", f"₹{income_stats['Maximum_Income']:,.0f}")
                    
#                     # Provide download link
#                     if os.path.exists(output_excel):
#                         with open(output_excel, 'rb') as f:
#                             st.download_button(
#                                 label="📥 Download Analysis Report (Excel)",
#                                 data=f.read(),
#                                 file_name=output_excel,
#                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                             )
                
#                 except Exception as e:
#                     st.error(f"Error generating report: {str(e)}")
    
#     elif page == "View Charts":
#         st.header("📈 Data Visualization Charts")
        
#         # Check if charts exist
#         charts_dir = 'charts'
#         chart_files = {
#             'Gender Distribution (Pie Chart)': 'gender_distribution_pie.png',
#             'Category Distribution (Bar Chart)': 'category_wise_bar.png',
#             'Income Distribution (Histogram)': 'income_distribution_hist.png'
#         }
        
#         if st.button("Generate Charts"):
#             with st.spinner("Generating charts..."):
#                 try:
#                     chart_paths = generate_charts(app_data, charts_dir)
#                     st.success("Charts generated successfully!")
#                 except Exception as e:
#                     st.error(f"Error generating charts: {str(e)}")
        
#         # Display charts if they exist
#         for chart_name, chart_file in chart_files.items():
#             chart_path = os.path.join(charts_dir, chart_file)
#             if os.path.exists(chart_path):
#                 st.subheader(chart_name)
#                 st.image(chart_path, use_column_width=True)
#             else:
#                 st.info(f"{chart_name} not found. Click 'Generate Charts' to create it.")


# if __name__ == "__main__":
#     main()


import streamlit as st
from main import (
    load_data,
    check_missing_values, fill_missing_values, drop_missing_values,
    remove_duplicates, rename_columns, change_column_dtype, clean_text_column,
    calculate_average, calculate_max, calculate_min, group_by_column, value_counts,
    filter_data, sort_data, generate_charts, export_cleaned_data
)

def main():
    st.set_page_config(page_title="Bank Customer Data Management", layout="wide")
    st.title("🏦 Bank Customer Data Management System")
    st.markdown("---")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Data Cleaning & Analysis",
        "Search Customer",
        "Generate Analysis Report",
        "View Charts"
    ])

    # Load Data
    if "app_data" not in st.session_state:
        try:
            app_data, app_pref = load_data("All_Applicant_Total_Data.xlsx")
            st.session_state.app_data = app_data
            st.session_state.app_pref = app_pref
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    df = st.session_state.app_data

    # ------------------- Data Cleaning & Analysis -------------------
    if page == "Data Cleaning & Analysis":
        st.header("🧹 Data Cleaning & Analysis")
        st.subheader("1. Check Missing Values")
        st.write(check_missing_values(df))

        st.subheader("2. Cleaning Options")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Fill Missing with 0"):
                df = fill_missing_values(df)
                st.session_state.app_data = df
                st.success("Missing values filled with 0.")

            if st.button("Drop Missing Rows"):
                df = drop_missing_values(df)
                st.session_state.app_data = df
                st.success("Rows with missing values dropped.")

        with col2:
            if st.button("Remove Duplicates"):
                df = remove_duplicates(df)
                st.session_state.app_data = df
                st.success("Duplicates removed.")

            if st.button("Rename Columns"):
                df = rename_columns(df)
                st.session_state.app_data = df
                st.success("Column names formatted (lowercase, stripped).")

        st.subheader("3. Analysis Options")
        col3, col4 = st.columns(2)

        with col3:
            if "avg_income" in df.columns:
                st.write("**Average Income:**", calculate_average(df, "avg_income"))
                st.write("**Max Income:**", calculate_max(df, "avg_income"))
                st.write("**Min Income:**", calculate_min(df, "avg_income"))

            if "gender" in df.columns:
                st.write("**Gender Counts:**")
                st.write(value_counts(df, "gender"))

        with col4:
            if "category" in df.columns and "avg_income" in df.columns:
                st.write("**Average Income by Category:**")
                st.write(group_by_column(df, "category", "avg_income"))

            if "avg_income" in df.columns:
                threshold = st.slider("Filter customers with income >", 0, int(df["avg_income"].max()), 10000)
                st.write(filter_data(df, "avg_income", threshold))

        st.subheader("4. Sorting")
        if "avg_income" in df.columns:
            sort_order = st.radio("Sort by average income:", ["Descending", "Ascending"])
            sorted_df = sort_data(df, "avg_income", ascending=(sort_order == "Ascending"))
            st.dataframe(sorted_df.head(10))

        st.subheader("5. Export Cleaned Data")
        if st.button("Export Cleaned Data"):
            export_cleaned_data(df)
            st.success("✅ Cleaned data exported to 'cleaned_data.xlsx'")

    # ------------------- Search Customer -------------------
    elif page == "Search Customer":
        st.header("🔍 Search Customer")
        org_id = st.text_input("Enter Org ID").strip()
        if st.button("Search"):
            if org_id:
                df["org_id"] = df["org_id"].astype(str).str.strip().str.split(".").str[0]
                result = df[df["org_id"] == org_id]
                if not result.empty:
                    st.success(f"✅ Found {len(result)} record(s)")
                    st.dataframe(result)
                else:
                    st.error("❌ No record found")
            else:
                st.warning("⚠️ Please enter an Org ID.")

    # ------------------- Other Pages -------------------
    elif page == "Generate Analysis Report":
        st.info("📊 Generate Analysis Report page is available in future updates.")

    elif page == "View Charts":
        st.subheader("📈 Charts")
        if st.button("Generate Charts"):
            charts = generate_charts(df)
            st.success("Charts generated!")
            for name, path in charts.items():
                st.image(path, caption=name, use_column_width=True)

if __name__ == "__main__":
    main()

