# import os
# import re
# from typing import Tuple, Dict, Any, List

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from openpyxl import load_workbook
# from openpyxl.drawing.image import Image as XLImage


# # 1) Excel File Reading

# def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Read App_data and App_Pref sheets from the Excel file into DataFrames.
#     App_Pref may have a merged title row; detect header row dynamically.
#     """
#     # App_data is clean with headers
#     app_data_df = pd.read_excel(file_path, sheet_name='App_data')

#     # App_Pref may have a title row; read without header and detect
#     raw_pref = pd.read_excel(file_path, sheet_name='App_Pref', header=None)

#     header_row_idx = None
#     for i in range(min(5, len(raw_pref))):
#         row_vals = raw_pref.iloc[i].astype(str).str.strip().tolist()
#         # Look for 'Application Number' or similar to identify the header row
#         if any(re.search(r'[Aa]pplication [Nn]umber', v) for v in row_vals):
#             header_row_idx = i
#             break

#     if header_row_idx is None:
#         # Fallback: assume the first row after the title row is header (index 1)
#         header_row_idx = 1 if len(raw_pref) > 1 else 0

#     # Extract column names from the detected header row
#     new_cols = raw_pref.iloc[header_row_idx].tolist()
#     # Clean column names: replace NaN with empty string, strip whitespace
#     new_cols = [str(c).strip() if pd.notna(c) else f'Unnamed_{idx}' for idx, c in enumerate(new_cols)]

#     # The actual data starts from the row after the header
#     app_pref_df = raw_pref.iloc[header_row_idx + 1:].copy()
#     app_pref_df.columns = new_cols

#     # Drop fully empty rows
#     app_pref_df = app_pref_df.dropna(how='all')

#     return app_data_df, app_pref_df


# # 2) Data Cleaning

# def _aadhaar_is_valid(value: Any) -> bool:
#     """Validate Aadhaar last-4 digits per requirement (should be 4 digits).
#     Accept also full 12-digit Aadhaar if present.
#     """
#     s = str(value).strip()
#     if not s or s.lower() == 'nan':
#         return False
#     if s.isdigit() and len(s) == 12:
#         return True
#     return s.isdigit() and len(s) == 4


# def _ifsc_is_valid(value: Any) -> bool:
#     s = str(value).strip()
#     return len(s) == 11


# def normalize_app_data_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """Rename App_data columns to a standard schema used downstream."""
#     col_map = {
#         'org_id': 'Application_Number',
#         'Full Name': 'Full_Name',
#         'date_of_birth': 'DOB',
#         'gender': 'Gender',
#         'avg_income': 'Income',
#         'aadhar_last_for_digits': 'Aadhaar',
#         'aadhar_last_four_digits': 'Aadhaar',
#         'ifsc_code': 'IFSC',
#         'status': 'Status',
#         'category': 'Category',
#         'Taluka': 'Taluka',
#         'Village': 'Village',
#     }
#     # Apply mapping where possible, leave others as-is
#     renamed = {}
#     for c in df.columns:
#         renamed[c] = col_map.get(c, c)
#     df = df.rename(columns=renamed)
#     return df


# def clean_data(df_app_data: pd.DataFrame, df_app_pref: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     df_app_data = normalize_app_data_columns(df_app_data)

#     # Fill missing values with defaults
#     df_app_data = df_app_data.fillna({
#         'Full_Name': 'Unknown',
#         'DOB': 'Unknown',
#         'Gender': 'Unknown',
#         'Income': 0,
#         'Aadhaar': 'Unknown',
#         'IFSC': 'Unknown',
#         'Status': 'Unknown',
#         'Category': 'Unknown',
#         'Taluka': 'Unknown',
#         'Village': 'Unknown',
#     })
#     df_app_pref = df_app_pref.fillna('Unknown')

#     # Remove duplicate rows
#     df_app_data = df_app_data.drop_duplicates()
#     df_app_pref = df_app_pref.drop_duplicates()

#     # Validation flags
#     if 'Aadhaar' in df_app_data.columns:
#         df_app_data['Aadhaar_Valid'] = df_app_data['Aadhaar'].apply(_aadhaar_is_valid)
#     else:
#         df_app_data['Aadhaar_Valid'] = False

#     if 'IFSC' in df_app_data.columns:
#         df_app_data['IFSC_Valid'] = df_app_data['IFSC'].apply(_ifsc_is_valid)
#     else:
#         df_app_data['IFSC_Valid'] = False

#     return df_app_data, df_app_pref


# # 3) Data Analysis

# def analyze_data(df_app_data: pd.DataFrame):
#     df = df_app_data.copy()
#     # Ensure Income numeric
#     if 'Income' in df.columns:
#         df['Income'] = pd.to_numeric(df['Income'], errors='coerce').fillna(0)
#     else:
#         df['Income'] = 0

#     gender_distribution = df['Gender'].value_counts().sort_index() if 'Gender' in df.columns else pd.Series()
#     category_distribution = df['Category'].value_counts().sort_index() if 'Category' in df.columns else pd.Series()

#     income_statistics = pd.Series({
#         'Average_Income': float(df['Income'].mean()),
#         'Minimum_Income': float(df['Income'].min()),
#         'Maximum_Income': float(df['Income'].max()),
#     })

#     if 'Taluka' in df.columns and 'Village' in df.columns:
#         taluka_village_applicants = (
#             df.groupby(['Taluka', 'Village']).size().reset_index(name='Count').sort_values(['Taluka', 'Village'])
#         )
#     else:
#         taluka_village_applicants = pd.DataFrame(columns=['Taluka', 'Village', 'Count'])

#     status_applicants = df['Status'].value_counts().sort_index() if 'Status' in df.columns else pd.Series()

#     return gender_distribution, category_distribution, income_statistics, taluka_village_applicants, status_applicants


# # 4) Preference Analysis

# def analyze_preferences(df_app_pref: pd.DataFrame):
#     # Try to find application number column name variant
#     app_no_col = None
#     for cand in ['Application Number', 'Application_Number', 'Application No', 'Application_No', 'org_id']:
#         if cand in df_app_pref.columns:
#             app_no_col = cand
#             break

#     # Identify preference columns by name pattern or by being numeric and not the app_no_col
#     # Assuming preference columns are those that contain project names (strings) and are not the Application Number column
#     pref_cols: List[str] = []
#     for col in df_app_pref.columns:
#         if col == app_no_col:
#             continue
#         # Check if the column contains mostly non-numeric values (likely project names)
#         # Take a sample to avoid iterating over large datasets
#         sample_values = df_app_pref[col].dropna().head(5).astype(str).str.strip()
#         if not sample_values.empty and all(not s.isdigit() for s in sample_values):
#             pref_cols.append(col)

#     # Project-wise preference count
#     project_preferences = pd.Series(dtype='int')
#     for col in pref_cols:
#         project_preferences = pd.concat([project_preferences, df_app_pref[col].value_counts()])
#     if not project_preferences.empty:
#         project_preferences = project_preferences.groupby(level=0).sum().sort_values(ascending=False)
#     else:
#         project_preferences = pd.Series([], dtype='int')

#     most_popular_project = project_preferences.index[0] if not project_preferences.empty else 'N/A'

#     # Applicant preference order analysis: how many preferences filled per row
#     def count_filled(row):
#         return row[pref_cols].replace({'Unknown': pd.NA}).count()

#     if pref_cols:
#         df_tmp = df_app_pref.copy()
#         df_tmp['Preferences_Filled'] = df_tmp.apply(count_filled, axis=1)
#         applicant_preference_order = df_tmp['Preferences_Filled'].value_counts().sort_index()
#     else:
#         applicant_preference_order = pd.Series([], dtype='int')

#     return project_preferences, most_popular_project, applicant_preference_order


# # 5) Report Generation (Excel)

# def export_report(
#     output_path: str,
#     gender_distribution: pd.Series,
#     category_distribution: pd.Series,
#     income_statistics: pd.Series,
#     status_applicants: pd.Series,
#     project_preferences: pd.Series,
#     taluka_village_applicants: pd.DataFrame,
# ):
#     with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
#         gender_distribution.rename_axis('Gender').reset_index(name='Count').to_excel(writer, sheet_name='Gender distribution', index=False)
#         category_distribution.rename_axis('Category').reset_index(name='Count').to_excel(writer, sheet_name='Category distribution', index=False)
#         income_statistics.rename('Value').reset_index().rename(columns={'index': 'Metric'}).to_excel(writer, sheet_name='Income stats', index=False)
#         status_applicants.rename_axis('Status').reset_index(name='Count').to_excel(writer, sheet_name='Status count', index=False)
#         project_preferences.rename_axis('Project').reset_index(name='Preference Count').to_excel(writer, sheet_name='Project preferences', index=False)
#         taluka_village_applicants.to_excel(writer, sheet_name='Taluka_Village', index=False)
#         # Create a Charts sheet placeholder
#         pd.DataFrame({'Charts': ['See embedded images']}).to_excel(writer, sheet_name='charts', index=False)


# # 6) Graphs & Visualization + embed into Excel

# def generate_charts(df_app_data: pd.DataFrame, charts_dir: str) -> Dict[str, str]:
#     os.makedirs(charts_dir, exist_ok=True)
#     paths: Dict[str, str] = {}
#     sns.set_theme(style='whitegrid')

#     # Pie: Gender Distribution
#     if 'Gender' in df_app_data.columns and not df_app_data['Gender'].empty:
#         plt.figure(figsize=(6, 6))
#         df_app_data['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, pctdistance=0.85)
#         plt.title('Gender Distribution')
#         centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#         fig = plt.gcf()
#         fig.gca().add_artist(centre_circle)
#         plt.ylabel('')
#         gender_path = os.path.join(charts_dir, 'gender_distribution_pie.png')
#         plt.tight_layout()
#         plt.savefig(gender_path, dpi=150)
#         plt.close()
#         paths['gender_pie'] = gender_path

#     # Bar: Category-wise Applicants
#     if 'Category' in df_app_data.columns and not df_app_data['Category'].empty:
#         plt.figure(figsize=(8, 5))
#         counts = df_app_data['Category'].value_counts().sort_index()
#         sns.barplot(x=counts.index, y=counts.values, palette='Blues_d')
#         plt.title('Category-wise Applicants')
#         plt.xlabel('Category')
#         plt.ylabel('Applicants')
#         cat_path = os.path.join(charts_dir, 'category_wise_bar.png')
#         plt.tight_layout()
#         plt.savefig(cat_path, dpi=150)
#         plt.close()
#         paths['category_bar'] = cat_path

#     # Histogram: Income Distribution
#     if 'Income' in df_app_data.columns and not df_app_data['Income'].empty:
#         plt.figure(figsize=(8, 5))
#         income_series = pd.to_numeric(df_app_data['Income'], errors='coerce').fillna(0)
#         sns.histplot(income_series, bins=20, kde=True, color='#4C72B0')
#         plt.title('Income Distribution')
#         plt.xlabel('Income')
#         plt.ylabel('Frequency')
#         hist_path = os.path.join(charts_dir, 'income_distribution_hist.png')
#         plt.tight_layout()
#         plt.savefig(hist_path, dpi=150)
#         plt.close()
#         paths['income_hist'] = hist_path

#     return paths


# def embed_charts_in_excel(excel_path: str, chart_paths: Dict[str, str]) -> None:
#     wb = load_workbook(excel_path)
#     if 'Charts' not in wb.sheetnames:
#         wb.create_sheet('Charts')
#     ws = wb['Charts']

#     row = 1
#     for label, path in chart_paths.items():
#         if os.path.exists(path):
#             img = XLImage(path)
#             img.width = img.width  # keep original
#             img.height = img.height
#             cell = f'A{row}'
#             ws.add_image(img, cell)
#             row += 20  # spacing between images
#     wb.save(excel_path)


# # 7) Orchestration

# def main():
#     source_file = 'All_Applicant_Total_Data.xlsx'
#     output_excel = 'analysis_report.xlsx'
#     charts_dir = 'charts'

#     # Load
#     app_data, app_pref = load_data(source_file)

#     # Clean
#     app_data_clean, app_pref_clean = clean_data(app_data, app_pref)

#     # Analyze
#     gender_dist, category_dist, income_stats, taluka_village_app, status_app = analyze_data(app_data_clean)
#     project_prefs, most_popular_proj, applicant_pref_order = analyze_preferences(app_pref_clean)

#     # Export report
#     export_report(
#         output_excel,
#         gender_dist,
#         category_dist,
#         income_stats,
#         status_app,
#         project_prefs,
#         taluka_village_app,
#     )

#     # Charts
#     chart_paths = generate_charts(app_data_clean, charts_dir)
#     embed_charts_in_excel(output_excel, chart_paths)

#     # Console summary
#     print('\nMost popular project:', most_popular_proj)
#     print('\nApplicant preferences filled (distribution):')
#     print(applicant_pref_order)
#     print('\nCharts saved:')
#     for k, v in chart_paths.items():
#         print(f' - {k}: {v}')
#     print(f"\nReport written to: {output_excel}")


# if __name__ == '__main__':
#     main()


import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ------------------- Data Loading -------------------
def load_data(file_path: str):
    try:
        app_data = pd.read_excel(file_path, sheet_name=0)
        app_pref = None
        return app_data, app_pref
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

# ------------------- Cleaning Functions -------------------
def check_missing_values(df):
    return df.isnull().sum()

def fill_missing_values(df, value=0):
    return df.fillna(value)

def drop_missing_values(df):
    return df.dropna()

def remove_duplicates(df):
    return df.drop_duplicates(subset=['org_id'])

def rename_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

def change_column_dtype(df, column, dtype):
    try:
        df[column] = df[column].astype(dtype)
    except Exception as e:
        print(f"Error converting {column}: {e}")
    return df

def clean_text_column(df, column):
    if column in df.columns:
        df[column] = df[column].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    return df

# ------------------- Analysis Functions -------------------
def calculate_average(df, column):
    return df[column].mean()

def calculate_max(df, column):
    return df[column].max()

def calculate_min(df, column):
    return df[column].min()

def group_by_column(df, group_col, agg_col):
    return df.groupby(group_col)[agg_col].mean()

def value_counts(df, column):
    return df[column].value_counts()

def filter_data(df, column, threshold):
    return df[df[column] > threshold]

def sort_data(df, column, ascending=False):
    return df.sort_values(by=column, ascending=ascending)

# ------------------- Chart Functions -------------------

def generate_charts(df, charts_dir="charts"):
    os.makedirs(charts_dir, exist_ok=True)
    chart_paths = {}

    # ------------------- 1. Age Distribution (Histogram) -------------------
    if "date_of_birth" in df.columns:
        df['age'] = df['date_of_birth'].apply(lambda x: (datetime.now() - pd.to_datetime(x)).days // 365)
        df['age'].plot(kind='hist', bins=20)
        plt.title("Age Distribution")
        plt.xlabel("Age")
        age_hist = os.path.join(charts_dir, "age_distribution.png")
        plt.savefig(age_hist)
        plt.close()
        chart_paths["age_distribution"] = age_hist

    # ------------------- 2. Gender vs Account Type (Grouped Bar Chart) -------------------
    if "gender" in df.columns and "account_type" in df.columns:
        grouped = df.groupby(['gender', 'account_type']).size().unstack()
        grouped.plot(kind='bar')
        plt.title("Gender vs Account Type")
        plt.ylabel("Number of Customers")
        gender_account_chart = os.path.join(charts_dir, "gender_vs_account_type.png")
        plt.savefig(gender_account_chart)
        plt.close()
        chart_paths["gender_vs_account_type"] = gender_account_chart

    # ------------------- 3. Average Balance by Account Type (Bar Chart) -------------------
    if "account_type" in df.columns and "balance" in df.columns:
        avg_balance = df.groupby('account_type')['balance'].mean()
        avg_balance.plot(kind='bar')
        plt.title("Average Balance by Account Type")
        plt.ylabel("Average Balance")
        avg_balance_chart = os.path.join(charts_dir, "avg_balance_by_account_type.png")
        plt.savefig(avg_balance_chart)
        plt.close()
        chart_paths["avg_balance_by_account_type"] = avg_balance_chart

    # ------------------- 4. Loan Amount Distribution (Histogram) -------------------
    if "balance" in df.columns and "loan_status" in df.columns:
        df[df['loan_status'].notna()]['balance'].plot(kind='hist', bins=20)
        plt.title("Loan Amount Distribution")
        plt.xlabel("Loan Amount")
        loan_hist = os.path.join(charts_dir, "loan_amount_distribution.png")
        plt.savefig(loan_hist)
        plt.close()
        chart_paths["loan_amount_distribution"] = loan_hist

    # ------------------- 5. Loan Status (Pie Chart) -------------------
    if "loan_status" in df.columns:
        df['loan_status'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title("Loan Status Distribution")
        loan_status_pie = os.path.join(charts_dir, "loan_status_pie.png")
        plt.savefig(loan_status_pie)
        plt.close()
        chart_paths["loan_status_pie"] = loan_status_pie

    # ------------------- 6. Credit Score vs Balance (Scatter Plot) -------------------
    if "credit_score" in df.columns and "balance" in df.columns:
        plt.scatter(df['credit_score'], df['balance'])
        plt.xlabel("Credit Score")
        plt.ylabel("Balance")
        plt.title("Credit Score vs Balance")
        credit_balance_scatter = os.path.join(charts_dir, "credit_vs_balance.png")
        plt.savefig(credit_balance_scatter)
        plt.close()
        chart_paths["credit_vs_balance"] = credit_balance_scatter

    # ------------------- 7. Transactions by City/Branch (Bar Chart) -------------------
    if "taluka" in df.columns:
         plt.figure(figsize=(24,12))
        #  df['taluka'].value_counts().head(20).plot(kind='bar')
         plt.title("Transactions by City/Branch")
         plt.ylabel("Number of Transactions")
    #  
         df['taluka'].value_counts().plot(kind='bar')
         plt.xticks(rotation=45)
         plt.tight_layout()
         transactions_chart = os.path.join(charts_dir, "transactions_by_city.png")
         plt.savefig(transactions_chart)
         plt.close()
         chart_paths["transactions_by_city"] = transactions_chart



        # df['taluka'].value_counts().head(20).plot(kind='bar')
        # plt.title("Transactions by City/Branch")
        # plt.ylabel("Number of Transactions")
        # transactions_chart = os.path.join(charts_dir, "transactions_by_city.png")
        # plt.savefig(transactions_chart)
        # plt.close()
        # chart_paths["transactions_by_city"] = transactions_chart

    return chart_paths


# ------------------- Export Cleaned Data -------------------
def export_cleaned_data(df, file_name="cleaned_data.xlsx"):
    df.to_excel(file_name, index=False)


