
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

cluster = LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1, memory_limit="3GB")
client = Client(cluster)

print("Dask LocalCluster initialized successfully.")
print(f"Monitor cluster activity via the Dask Dashboard: {client.dashboard_link}")

data_path = r"C:\Users\s6584\Downloads\archive (2)\DelayedFlights.csv"

print(f"Attempting to load dataset from: {data_path}")
try:
    
    df = dd.read_csv(data_path)
    print("Dataset loaded into Dask DataFrame.")
    print(f"Initial number of Dask partitions: {df.npartitions}")
except FileNotFoundError:
    print(f"Error: The data file was not found at {data_path}. Please verify the path.")
    
    client.close()
    cluster.close()
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    client.close()
    cluster.close()
    exit()

print("\n--- Initial Data Exploration ---")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDescriptive statistics for numerical columns:")
print(df.describe().compute())

print(f"\nTotal number of flight records: {len(df).compute()}") 

print("\nNull values check for key columns:")
print(f"'DepDelay' null count: {df['DepDelay'].isnull().sum().compute()}")
print(f"'UniqueCarrier' null count: {df['UniqueCarrier'].isnull().sum().compute()}")

print("\n--- Data Preprocessing Steps ---")

initial_rows = len(df) 
df_cleaned = df.dropna(subset=['DepDelay', 'UniqueCarrier'])
rows_after_dropping_nulls = len(df_cleaned).compute()
print(f"Original row count: {initial_rows.compute()}")
print(f"Rows after removing nulls in 'DepDelay' or 'UniqueCarrier': {rows_after_dropping_nulls}")

print("\n--- Performing Key Data Analysis ---")

print("\nAggregating average departure delay by airline carrier:")
start_time_agg = time.time()
avg_delay_by_carrier = df_cleaned.groupby("UniqueCarrier")['DepDelay'].mean().compute()
end_time_agg = time.time()
print("Top 10 carriers by average delay:")
print(avg_delay_by_carrier.sort_values(ascending=False).head(10))
print(f"Time taken for average delay aggregation: {end_time_agg - start_time_agg:.2f} seconds")

# Analysis 2: Identifying the volume of flights experiencing significant delays (over 1 hour).
print("\nFiltering flights with departure delays exceeding 60 minutes:")
start_time_filter = time.time()
high_delay_flights = df_cleaned[df_cleaned['DepDelay'] > 60]
num_high_delay_flights = len(high_delay_flights).compute()
end_time_filter = time.time()
print(f"Total flights with > 60 minute delay: {num_high_delay_flights}")
print(f"Time taken for significant delay filtering: {end_time_filter - start_time_filter:.2f} seconds")

print("\n--- Generating Visualizations of Analysis ---")

plt.figure(figsize=(12, 7))
avg_delay_by_carrier.sort_values(ascending=False).head(10).plot(kind='bar', color=sns.color_palette("pastel")[0])
plt.title('Top 10 Unique Carriers by Average Departure Delay (Minutes)', fontsize=16)
plt.xlabel('Unique Carrier', fontsize=12)
plt.ylabel('Average Departure Delay (Minutes)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n--- Scalability Demonstration ---")

partition_counts = [2, os.cpu_count(), os.cpu_count() * 2]

scalability_results = []
for n_parts in partition_counts:
    print(f"\nRe-running aggregation with {n_parts} partitions...")
    
    repartitioned_df = df_cleaned.repartition(npartitions=n_parts)
    print(f"DataFrame repartitioned to {repartitioned_df.npartitions} Dask partitions.")

    start_time_scalable = time.time()
    
    current_avg_delay = repartitioned_df.groupby("UniqueCarrier")['DepDelay'].mean().compute()
    end_time_scalable = time.time()
    elapsed_time = end_time_scalable - start_time_scalable

    print(f"Execution time with {n_parts} partitions: {elapsed_time:.2f} seconds")
    scalability_results.append({
        "Number of Partitions": n_parts,
        "Time (seconds)": elapsed_time
    })

scalability_df_results = pd.DataFrame(scalability_results)
print("\n--- Scalability Test Results ---")
print(scalability_df_results.to_markdown(index=False))

print("\n--- Visualizing Scalability Performance ---")

plt.figure(figsize=(10, 6))
plt.plot(scalability_df_results['Number of Partitions'], scalability_df_results['Time (seconds)'],
         marker='o', linestyle='-', color='teal', markersize=8, linewidth=2)
plt.title('Dask Aggregation Time vs. Number of Partitions (Scalability Test)', fontsize=16)
plt.xlabel('Number of Partitions', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.xticks(scalability_df_results['Number of Partitions'], fontsize=10) # Ensure ticks are at tested partition counts
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
 
output_dir = "dask_flight_insights"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path_agg = os.path.join(output_dir, "avg_delay_by_carrier.csv")
try:
    # Saving as a single file, which is suitable for this relatively small aggregated result.
    avg_delay_by_carrier.to_csv(output_path_agg, single_file=True, header=True)
    print(f"\nAverage delay by carrier results successfully saved to: {output_path_agg}")
except Exception as e:
    print(f"Error saving aggregated output: {e}")
client.close()
cluster.close()
print("\nDask Client and LocalCluster have been gracefully stopped.")
