import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

df.head()
df.info()
df.describe()

print(df[df["total_bill"] > 30])
print(df[df["sex"] == "Female"])
print(df.sort_values("tip", ascending=False).head(5))

df.groupby("day")["total_bill"].mean()
df.groupby("sex")["tip"].mean()
df.groupby(["day","sex"])["total_bill"].mean()

print(df.groupby("day")["total_bill"].agg(["mean","std","count"]))

df["tip_percentage"] = df["tip"] / df["total_bill"]


print(df.groupby("day")["tip_percentage"].mean())


import matplotlib.pyplot as plt


plt.figure(figsize=(8,5))
plt.scatter(df["total_bill"], df["tip"], alpha=0.6, c='green')
plt.title("Total Bill vs Tip")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.hist(df["total_bill"], bins=20, edgecolor='black', color='orange')
plt.title("Distribution of Total Bill")
plt.xlabel("Total Bill ($)")
plt.ylabel("Frequency")
plt.show()

avg_bill = df.groupby("day")["total_bill"].mean()
plt.figure(figsize=(8,5))
plt.bar(avg_bill.index, avg_bill.values, color='purple', alpha=0.7)
plt.title("Average Total Bill per Day")
plt.xlabel("Day")
plt.ylabel("Average Bill ($)")
plt.show()

plt.figure(figsize=(8,5))
df.boxplot(column="total_bill", by="day")
plt.title("Boxplot of Total Bill per Day")
plt.suptitle("")
plt.xlabel("Day")
plt.ylabel("Total Bill ($)")
plt.show()