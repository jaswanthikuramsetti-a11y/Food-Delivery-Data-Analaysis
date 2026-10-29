import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

# STEP 1: LOAD DATASET
uploaded = files.upload()
df = pd.read_csv("food_delivery_dataset_50000.csv")
print("Original Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# STEP 2: CLEAN DATA
df["Order Value"] = df["Order Value"].fillna(df["Order Value"].median())
df["Delivery Time"] = df["Delivery Time"].fillna(df["Delivery Time"].median())
df["Customer Rating"] = df["Customer Rating"].fillna(df["Customer Rating"].median())
df["Distance"] = df["Distance"].fillna(df["Distance"].median())
df["Customer Age"] = df["Customer Age"].fillna(df["Customer Age"].median())

df["Cuisine"] = df["Cuisine"].fillna(df["Cuisine"].mode()[0])
df["City"] = df["City"].fillna(df["City"].mode()[0])
df["Payment Method"] = df["Payment Method"].fillna(df["Payment Method"].mode()[0])

df = df.drop_duplicates()

print("\nShape After Cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

# STEP 3: BAR CHARTS
df["Cuisine"].value_counts().plot(kind="bar", color="blue")
plt.title("Orders by Cuisine")
plt.xlabel("Cuisine")
plt.ylabel("Number of Orders")
plt.savefig("bar_cuisine.png")
plt.show()

df["City"].value_counts().plot(kind="bar", color="green")
plt.title("Orders by City")
plt.xlabel("City")
plt.ylabel("Number of Orders")
plt.savefig("bar_city.png")
plt.show()

df["Restaurant"].value_counts().plot(kind="bar", color="orange")
plt.title("Orders by Restaurant")
plt.xlabel("Restaurant")
plt.ylabel("Number of Orders")
plt.savefig("bar_restaurant.png")
plt.show()

df["Order Status"].value_counts().plot(kind="bar", color="purple")
plt.title("Order Status")
plt.xlabel("Status")
plt.ylabel("Number of Orders")
plt.savefig("bar_status.png")
plt.show()

df.groupby("Cuisine")["Delivery Time"].mean().plot(kind="bar", color="red")
plt.title("Average Delivery Time by Cuisine")
plt.xlabel("Cuisine")
plt.ylabel("Minutes")
plt.savefig("bar_delivery_time.png")
plt.show()

# STEP 4: PIE CHARTS
df["Payment Method"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Payment Method Distribution")
plt.ylabel("")
plt.savefig("pie_payment.png")
plt.show()

df["Order Status"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Order Status Share")
plt.ylabel("")
plt.savefig("pie_status.png")
plt.show()

df["Cuisine"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Cuisine Share")
plt.ylabel("")
plt.savefig("pie_cuisine.png")
plt.show()
