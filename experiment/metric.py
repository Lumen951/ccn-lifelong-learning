import csv
import os


class Metric:
    # CSV-backed replacement for the MariaDB metric tables.
    # Rows are buffered in memory and flushed on commit_values().

    def __init__(self, database_name, table_name, keys, types, index):
        out_dir = os.path.join("results", database_name)
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, table_name + ".csv")
        self.db_columns = keys
        self.data_vector = []
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(keys)

    def record_value(self, values):
        self.data_vector.append(values)

    def commit_values(self):
        if self.data_vector:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(self.data_vector)
            self.data_vector.clear()
