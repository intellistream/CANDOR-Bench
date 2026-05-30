import subprocess
import re
import statistics
import csv

params = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2000000, 5000000]
repeats = 5
output_file = "48core-sift10m.csv"

qps_pattern = re.compile(r"Queries\s+per\s+second:\s+(\d+\.?\d*)")
recall_pattern = re.compile(r"Recall:\s+(\d+\.?\d*)")

with open(output_file, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Parameter", "Queries per second", "Recall"])

    for param in params:
        qps_values = []
        recall_values = []

        for _ in range(repeats):
            command = f"../build/bench 48 {param} 1 ../data/sift10m/sift10m_base.fvecs"
            try:
                process = subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                for line in process.stdout:
                    print(line.strip())
                    qps_match = qps_pattern.search(line)
                    recall_match = recall_pattern.search(line)

                    if qps_match:
                        qps_values.append(float(qps_match.group(1)))
                    if recall_match:
                        recall_values.append(float(recall_match.group(1)))

                process.wait()

                if process.returncode != 0:
                    print(f"Error: Command failed with return code {process.returncode}")
                    error_output = process.stderr.read()
                    print(f"Stderr: {error_output}")

            except Exception as e:
                print(f"Error executing command: {command}\n{e}")

        if qps_values and recall_values:
            avg_qps = statistics.mean(qps_values)
            avg_recall = statistics.mean(recall_values)
            csvwriter.writerow([param, f"{avg_qps:.2f}", f"{avg_recall:.5f}"])
            print(f"Parameter: {param}, Avg QPS: {avg_qps:.2f}, Avg Recall: {avg_recall:.5f}")
        else:
            print(f"Skipping parameter {param} due to missing data.")
