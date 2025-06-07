import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns 
import missingno as msno 
import plotly.express as px  
import folium  
from folium import plugins 
from skimpy import skim  
import warnings 
from matplotlib.colors import LinearSegmentedColormap  

plt.rcParams["figure.figsize"] = (6,4)  # Sets the default size for the plots
warnings.filterwarnings("ignore")  # Ignores warning messages
pd.set_option('display.max_columns', None)  # Ensures all columns are displayed on the screen
pd.set_option('display.max_rows', None)  # Ensures all rows are displayed on the screen
pd.set_option('display.max_colwidth', None)  # Remove column width limit


print(">Loading dataset...")
df_original = pd.read_csv("online_sales_dataset.csv")
df = df_original.copy()
print("First 5 entries:")
print(df.head(5))

print("\n\nInfo:")
skim(df)
df.info()
print(msno.bar(df))
print(df.describe().T)

print("\n\nDataset loaded.")


print("\n\n>Cleaning up data...\n\n")

print(">Counting negative price entries... ")
print(df[df["UnitPrice"] < 0]["UnitPrice"].count())
print("Replacing entries with absolute...")
df["UnitPrice"] = df["UnitPrice"].abs()
print("Negative entries left...")
print(df[df["UnitPrice"] < 0]["UnitPrice"].count())
print("Done.\n\n")

print(">Counting negative quantity entries... ")
print(df[df["Quantity"] < 0]["Quantity"].count())
print("Replacing entries with absolute...")
df["Quantity"] = df["Quantity"].abs()
print("Negative entries left...")
print(df[df["Quantity"] < 0]["Quantity"].count())
print("Done.\n\n")

print(">Listed payment methods: ")
print(df["PaymentMethod"].value_counts())
print("Replacing 'paypall' with 'PayPal'... ")
df["PaymentMethod"].replace("paypall", "PayPal", inplace=True)
print("Resulting payment methods: ")
print(df["PaymentMethod"].value_counts())
print("Done.\n\n")

print(">Stripping prefix from stock code, transforming to integer... ")
df["StockCode"] = df["StockCode"].str.lstrip("SKU_").astype("int")
df["StockCode"].info()
print("Done.\n\n")

print(">Customer IDs:")
print(df["CustomerID"].head(5))
print("Transforming to non-integer type...")
df["CustomerID"] = df["CustomerID"].astype("object")
print("Filling empty entries with 'Unknown'...")
df["CustomerID"].fillna("Unknown", inplace=True)
print("Cleaning up extra digits...")
df["CustomerID"] = df["CustomerID"].astype("str").str.rstrip("0").str.rstrip(".")
print("Resulting Customer IDs:")
print(df["CustomerID"].head(5))
print("Done.\n\n")

print(">Rounding up discount entries...")
df["Discount"] = df["Discount"].round(2)
print("Discounts: ")
print(df["Discount"].head())
print("Done.\n\n")

print(">Checking for missing entries...")
print(df.isnull().sum())
print("Done.\n\n")

print(">Calculating shipping cost mean... ")
print(df["ShippingCost"].mean())
print("Calculating shipping cost mean (by shipment provider)... ")
print(df.groupby("ShipmentProvider")["ShippingCost"].mean().mean())
print("Calculating shipping cost mean (by warehouse location)... ")
print(df.groupby("WarehouseLocation")["ShippingCost"].mean().mean())
print("Negligible difference. Filling empty entries with mean... ")
df["ShippingCost"].fillna(17.49, inplace=True)
print("Done.\n\n")

print(">Counting empty warehouse location entries...")
print(df["WarehouseLocation"].value_counts())
print("Dropping records...")
df = df.dropna()
print("Done.\n\n")

print("\n\n>Resulting data:")
df.info()
skim(df)


df.to_csv("clean_sales_dataset.csv", sep=',', encoding='utf-8', index=False, header=True)
print("\n\n>Data cleanup complete. Data recorded under 'clean_sales_dataset.csv'.")

input("\n\nWaiting to exit on button push...")





