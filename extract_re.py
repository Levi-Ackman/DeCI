#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

def extract_metrics(line):
    parts = line.split(',')
    values = []
    for part in parts:
        if ':' in part:
            value_str = part.split(':')[-1].strip()
            try:
                value = float(value_str)
                values.append(value)
            except Exception as e:
                print(f"False value in line '{line}': {e}")
    return values

base_dir = './logs'
first_var_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for first_dir in first_var_dirs:
    first_dir_path = os.path.join(base_dir, first_dir)
    total_log_path = os.path.join(first_dir_path, 'total.log')

    with open(total_log_path, 'w') as total_log_file:
        second_var_dirs = [d for d in os.listdir(first_dir_path) if os.path.isdir(os.path.join(first_dir_path, d))]
        second_var_dirs.sort()

        for model in second_var_dirs:
            model_path = os.path.join(first_dir_path, model)
            log_files = [f for f in os.listdir(model_path) if f.endswith('.log')]

            best_mean_accuracy = -float('inf')
            best_means = None
            best_stds = None
            best_log_filename = None  # New: store the name of the best log file

            for log_file in log_files:
                log_file_path = os.path.join(model_path, log_file)
                try:
                    with open(log_file_path, 'r') as f:
                        lines = f.readlines()
                except Exception as e:
                    print(f"Failed to read file {log_file_path}: {e}")
                    continue

                # Extract lines containing evaluation metrics
                mean_line = None
                std_line = None
                for line in lines:
                    if "Mean accuracy:" in line:
                        mean_line = line.strip()
                    elif "Std accuracy:" in line:
                        std_line = line.strip()
                    if mean_line and std_line:
                        break

                if not mean_line or not std_line:
                    print(f"Metrics not found in file {log_file_path}.")
                    continue

                means = extract_metrics(mean_line)
                stds = extract_metrics(std_line)

                if len(means) < 5 or len(stds) < 5:
                    print(f"Incomplete metric extraction in file {log_file_path}: means {means}, stds {stds}")
                    continue

                current_accuracy = means[0]
                if current_accuracy > best_mean_accuracy:
                    best_mean_accuracy = current_accuracy
                    best_means = means[:5]
                    best_stds = stds[:5]
                    best_log_filename = log_file  # Save the name of the best log file

            # Write the best result for each model
            if best_means is not None and best_stds is not None:
                means_str = ','.join(f"{x * 100:.2f}" for x in best_means)
                stds_str = ','.join(f"{x * 100:.2f}" for x in best_stds)

                total_log_file.write(f"{model}:\n{means_str}\n{stds_str}\nbest log: {best_log_filename}\n\n")

    print(f"Generated summary log: {total_log_path}")
