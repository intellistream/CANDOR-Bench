import pandas as pd
import matplotlib.pyplot as plt

file1 = "48core_128dim_sift10m_base.csv"
file2 = "48core_128dim_sift10m_v1.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

if "Parameter" not in df1.columns or "Queries per second" not in df1.columns:
    print("Error: 'Parameter' or 'Queries per second' column not found in the first CSV file.")
if "Parameter" not in df2.columns or "Queries per second" not in df2.columns:
    print("Error: 'Parameter' or 'Queries per second' column not found in the second CSV file.")

plt.figure(figsize=(10, 6))
plt.plot(df1["Parameter"].to_numpy(), df1["Queries per second"].to_numpy(), marker="o", label="BASE QPS")
plt.plot(df2["Parameter"].to_numpy(), df2["Queries per second"].to_numpy(), marker="s", label="V1 QPS")
plt.xlabel("Parameter")
plt.ylabel("Queries Per Second (QPS)")
plt.title("QPS Comparison")
plt.legend()
plt.grid(True)
plt.xscale("log") 
output_path = "qps_comparison_48core_16dim.png"  
plt.savefig(output_path, dpi=300, bbox_inches="tight")  
plt.close()  
