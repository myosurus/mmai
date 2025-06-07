import pandas as pd

pd.set_option('display.max_columns', None)  # Ensures all columns are displayed on the screen
pd.set_option('display.max_rows', None)  # Ensures all rows are displayed on the screen
pd.set_option('display.max_colwidth', None)  # Remove column width limits


# Load data
file_path = 'clean_sales_dataset.csv'
data = pd.read_csv(file_path)

# See first 5 lines
data_info = {
    "head": data.head(),
    "info": data.info(),
    "columns": data.columns.tolist()
}


print("\n\nLoading up dataset...\n\n")
print(data_info)
print("\n\nDone.\n\n")


# InvoiceDate to datetime
print("Processing datetime...")
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
# New datetimes
data['DayOfWeek'] = data['InvoiceDate'].dt.day_name()
data['Hour'] = data['InvoiceDate'].dt.hour

# New features
print("Adding and rounding revenue...")
data['TotalPrice'] = (data['Quantity'] * data['UnitPrice']) - data['Discount']
data['TotalPrice'] = data['TotalPrice'].round(2)
print("Adding high priority flag...")
data['IsHighPriority'] = (data['OrderPriority'] == 'High').astype(int)

# See start of new columns
new_features_preview = data[['InvoiceDate', 'DayOfWeek', 'Hour', 'TotalPrice', 'IsHighPriority']].head()
print(new_features_preview)
print("\n\nDone.\n\n")


# Categorical data
print("Encoding categorical fields...")
categorical_columns = [
    'DayOfWeek', 'Country', 'PaymentMethod', 'Category', 
    'SalesChannel', 'ReturnStatus', 'ShipmentProvider', 'WarehouseLocation', 'OrderPriority'
]

# One-hot encoding categorical data
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# See start of encoded data
encoded_data_preview = data_encoded.head()
print(encoded_data_preview)
print("\n\nDone.\n\n")


data_encoded.to_csv("ready_sales_dataset.csv", sep=',', encoding='utf-8', index=False, header=True)
print("\n\n>Data processing complete. Data recorded under 'ready_sales_dataset.csv'.")
input("\n\nWaiting to exit on button push...")
