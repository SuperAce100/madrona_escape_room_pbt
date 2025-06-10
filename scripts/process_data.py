#!/usr/bin/env python3
"""
Convert TensorBoard .tfevents files to CSV format recursively.

This script walks through a directory tree, finds all .tfevents files,
and converts them to CSV files - much more compact than JSON.
"""

import os
import csv
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

try:
    from tensorboard.backend.event_processing import event_accumulator
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: TensorBoard is required. Install with: pip install tensorboard")
    exit(1)


def extract_experiment_info(directory_path: str) -> Dict[str, Any]:
    """
    Extract experiment information from directory name.
    Based on the naming convention from the training script.

    Args:
        directory_path: Path to the experiment directory

    Returns:
        Dictionary with parsed experiment parameters
    """
    dir_name = Path(directory_path).name

    # Initialize with defaults
    info = {
        "experiment_name": dir_name,
        "experiment_type": "escape_room",
        "pbt_enabled": False,
        "population_size": None,
        "elite_fraction": None,
        "mutation_rate": None,
        "learning_rate": None,
        "gamma": None,
        "entropy_coef": None,
        "value_coef": None,
        "num_channels": None,
        "separate_value": False,
        "num_worlds": None,
        "num_updates": None,
        "steps_per_update": None,
        "fp16": False,
        "timestamp": None,
    }

    # Split by underscores and parse each part
    parts = dir_name.split("_")

    for part in parts:
        # PBT parameters
        if part.startswith("pbt") and part[3:].isdigit():
            info["pbt_enabled"] = True
            info["population_size"] = int(part[3:])
        elif part.startswith("elite") and part[5:].isdigit():
            info["elite_fraction"] = int(part[5:]) / 100.0
        elif part.startswith("mut") and part[3:].isdigit():
            info["mutation_rate"] = int(part[3:]) / 100.0

        # Training hyperparameters
        elif part.startswith("lr") and "e" in part:
            try:
                info["learning_rate"] = float(part[2:])
            except ValueError:
                pass
        elif part.startswith("gamma"):
            try:
                info["gamma"] = float(part[5:])
            except ValueError:
                pass
        elif part.startswith("ent"):
            try:
                info["entropy_coef"] = float(part[3:])
            except ValueError:
                pass
        elif part.startswith("val"):
            try:
                info["value_coef"] = float(part[3:])
            except ValueError:
                pass

        # Model architecture
        elif part.startswith("ch") and part[2:].isdigit():
            info["num_channels"] = int(part[2:])
        elif part == "sep":
            info["separate_value"] = True

        # Training setup
        elif part.startswith("w") and part[1:].isdigit():
            info["num_worlds"] = int(part[1:])
        elif part.startswith("u") and part[1:].isdigit():
            info["num_updates"] = int(part[1:])
        elif part.startswith("s") and part[1:].isdigit():
            info["steps_per_update"] = int(part[1:])

        # Other flags
        elif part == "fp16":
            info["fp16"] = True

        # Timestamp (8 digits followed by underscore and 6 digits)
        elif len(part) == 8 and part.isdigit():
            idx = parts.index(part)
            if (
                idx + 1 < len(parts)
                and len(parts[idx + 1]) == 6
                and parts[idx + 1].isdigit()
            ):
                info["timestamp"] = f"{part}_{parts[idx + 1]}"

    return info


def extract_scalars_to_dataframe(
    tfevents_path: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract scalar data from tfevents file and return as DataFrame.

    Args:
        tfevents_path: Path to the .tfevents file

    Returns:
        Tuple of (DataFrame with all scalar data, experiment info dict)
    """
    # Load the TensorBoard event file
    ea = EventAccumulator(tfevents_path)
    ea.Reload()

    # Get experiment info
    experiment_info = extract_experiment_info(str(Path(tfevents_path).parent))

    # Extract all scalar data
    tags = ea.Tags()["scalars"]

    if not tags:
        return pd.DataFrame(), experiment_info

    # Collect all data points
    all_data = []

    for tag in tags:
        scalar_events = ea.Scalars(tag)
        for event in scalar_events:
            all_data.append(
                {
                    "experiment_name": experiment_info["experiment_name"],
                    "metric": tag,
                    "step": event.step,
                    "value": event.value,
                    "wall_time": event.wall_time,
                    # Add experiment info as columns for easy filtering
                    "pbt_enabled": experiment_info["pbt_enabled"],
                    "population_size": experiment_info["population_size"],
                    "learning_rate": experiment_info["learning_rate"],
                    "gamma": experiment_info["gamma"],
                    "entropy_coef": experiment_info["entropy_coef"],
                    "value_coef": experiment_info["value_coef"],
                    "num_channels": experiment_info["num_channels"],
                    "separate_value": experiment_info["separate_value"],
                    "num_worlds": experiment_info["num_worlds"],
                    "num_updates": experiment_info["num_updates"],
                    "steps_per_update": experiment_info["steps_per_update"],
                    "fp16": experiment_info["fp16"],
                    "timestamp": experiment_info["timestamp"],
                }
            )

    df = pd.DataFrame(all_data)
    return df, experiment_info


def find_tfevents_files(root_dir: str) -> List[str]:
    """
    Recursively find all .tfevents files in a directory tree.

    Args:
        root_dir: Root directory to search

    Returns:
        List of paths to .tfevents files
    """
    tfevents_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if "tfevents" in file and file.startswith("events.out.tfevents"):
                tfevents_files.append(os.path.join(root, file))

    return tfevents_files


def create_unified_csv(
    tfevents_files: List[str], output_path: str, verbose: bool = False
) -> str:
    """
    Convert all tfevents files to a single unified CSV file.

    Args:
        tfevents_files: List of paths to .tfevents files
        output_path: Path for the unified CSV output
        verbose: Whether to print verbose output

    Returns:
        Path to the created unified CSV file
    """
    all_dataframes = []
    experiment_summaries = []

    for i, tfevents_path in enumerate(tfevents_files):
        if verbose:
            print(
                f"Processing {i + 1}/{len(tfevents_files)}: {os.path.basename(tfevents_path)}"
            )

        try:
            df, exp_info = extract_scalars_to_dataframe(tfevents_path)

            if not df.empty and len(df) > 2:
                all_dataframes.append(df)

                # Create experiment summary
                summary = {
                    "experiment_name": exp_info["experiment_name"],
                    "source_file": tfevents_path,
                    "total_data_points": len(df),
                    "unique_metrics": df["metric"].nunique() if not df.empty else 0,
                    "max_step": df["step"].max() if not df.empty else 0,
                    "min_step": df["step"].min() if not df.empty else 0,
                    **{k: v for k, v in exp_info.items() if k != "experiment_name"},
                }
                experiment_summaries.append(summary)

                if verbose:
                    print(
                        f"  -> {len(df)} data points, {df['metric'].nunique()} unique metrics"
                    )

        except Exception as e:
            print(f"Error processing {tfevents_path}: {e}")
            continue

    if not all_dataframes:
        print("No data extracted from any files")
        return None

    # Combine all dataframes
    if verbose:
        print("Combining all data...")

    unified_df = pd.concat(all_dataframes, ignore_index=True)

    # Sort by experiment, metric, and step for better organization
    unified_df = unified_df.sort_values(["experiment_name", "metric", "step"])

    # Save to CSV
    unified_df.to_csv(output_path, index=False)

    # Also create a summary CSV
    summary_path = output_path.replace(".csv", "_summary.csv")
    if experiment_summaries:
        summary_df = pd.DataFrame(experiment_summaries)
        summary_df.to_csv(summary_path, index=False)

        if verbose:
            print(f"Created summary file: {summary_path}")

    return output_path


def create_separate_csvs(
    tfevents_files: List[str], output_dir: str, verbose: bool = False
) -> List[str]:
    """
    Create separate CSV files for each experiment.

    Args:
        tfevents_files: List of paths to .tfevents files
        output_dir: Output directory for CSV files
        verbose: Whether to print verbose output

    Returns:
        List of created CSV file paths
    """
    created_files = []

    for i, tfevents_path in enumerate(tfevents_files):
        try:
            df, exp_info = extract_scalars_to_dataframe(tfevents_path)

            if df.empty:
                if verbose:
                    print(f"No data in {tfevents_path}")
                continue

            # Generate output filename
            exp_name = exp_info["experiment_name"]
            csv_filename = f"{exp_name}.csv"
            output_path = os.path.join(output_dir or ".", csv_filename)

            # Save to CSV
            df.to_csv(output_path, index=False)
            created_files.append(output_path)

            if verbose:
                print(
                    f"Created: {output_path} ({len(df)} data points, {df['metric'].nunique()} metrics)"
                )
            else:
                print(f"Converted: {os.path.basename(tfevents_path)} -> {csv_filename}")

        except Exception as e:
            print(f"Error converting {tfevents_path}: {e}")
            continue

    return created_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert TensorBoard .tfevents files to CSV format recursively"
    )
    parser.add_argument(
        "input_dir", help="Root directory to search for .tfevents files"
    )
    parser.add_argument(
        "--output-file",
        default="training_data.csv",
        help="Output filename for unified CSV (default: training_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for CSV files (default: current directory)",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Create separate CSV files instead of unified",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        if args.verbose:
            print(f"Created output directory: {args.output_dir}")

    # Find all .tfevents files
    tfevents_files = find_tfevents_files(args.input_dir)

    if not tfevents_files:
        print(f"No .tfevents files found in '{args.input_dir}'")
        return 0

    print(f"Found {len(tfevents_files)} .tfevents file(s)")

    if args.dry_run:
        print("Dry run - showing what would be processed:")
        total_estimated_rows = 0

        for tfevents_path in tfevents_files:
            try:
                df, exp_info = extract_scalars_to_dataframe(tfevents_path)
                estimated_rows = len(df)
                total_estimated_rows += estimated_rows

                print(f"  {exp_info['experiment_name']}: ~{estimated_rows} data points")
            except Exception as e:
                print(f"  {Path(tfevents_path).parent.name}: Error - {e}")

        if not args.separate:
            output_path = os.path.join(args.output_dir or ".", args.output_file)
            print(f"\nWould create unified CSV: {output_path}")
            print(f"Estimated total rows: {total_estimated_rows}")
            # Rough size estimate (assuming ~50 bytes per row average)
            estimated_size_mb = (total_estimated_rows * 50) / (1024 * 1024)
            print(f"Estimated file size: ~{estimated_size_mb:.1f} MB")

        return 0

    if args.separate:
        # Create separate CSV files
        created_files = create_separate_csvs(
            tfevents_files, args.output_dir, args.verbose
        )
        print(f"Conversion complete. Created {len(created_files)} CSV files")

    else:
        # Create unified CSV file
        output_path = os.path.join(args.output_dir or ".", args.output_file)

        print(f"Creating unified CSV file: {output_path}")

        try:
            actual_output = create_unified_csv(
                tfevents_files, output_path, args.verbose
            )

            if actual_output:
                # Show file size and summary
                file_size = os.path.getsize(actual_output) / (1024 * 1024)  # MB

                # Load and show summary
                df = pd.read_csv(actual_output)
                unique_experiments = df["experiment_name"].nunique()
                unique_metrics = df["metric"].nunique()
                total_rows = len(df)

                print(f"\nUnified CSV created: {actual_output}")
                print(f"File size: {file_size:.1f} MB")
                print(f"Total rows: {total_rows:,}")
                print(f"Experiments: {unique_experiments}")
                print(f"Unique metrics: {unique_metrics}")

                if args.verbose:
                    print("\nTop metrics by frequency:")
                    metric_counts = df["metric"].value_counts().head(10)
                    for metric, count in metric_counts.items():
                        print(f"  {metric}: {count:,} data points")

                # Check for summary file
                summary_path = output_path.replace(".csv", "_summary.csv")
                if os.path.exists(summary_path):
                    summary_size = os.path.getsize(summary_path) / 1024  # KB
                    print(f"Summary file: {summary_path} ({summary_size:.1f} KB)")
            else:
                print("No output file was created")
                return 1

        except Exception as e:
            print(f"Error creating unified CSV: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
