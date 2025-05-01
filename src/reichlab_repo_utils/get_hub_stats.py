"""Get row counts for hub model-output and target-data files.

This script loops through a list of GitHub-based Hubverse repositories and counts the
number of lines/rows for each model-output and target data file. It saves the each
hub's information in data/hub_stats directory as a parquet file in the form
{hub_name}.parquet.

Once it has retrieved row counts for all hubs on the list, the script c reates
a .csv file with the combined data in the hub_stats directory.
The .csv is recreated each time the script is run and will include data from
all .parquet files in hub_stats (in other words, data from prior script runs
will be included).

Notes
-----
To run this script, you will need a personal GitHub token with read access
to public repositories. The token should be stored in an environment variable
named GITHUB_TOKEN.

The script makes several assumptions:

1. Hub repositories are public and hosted on GitHub
2. All hubs use a directory named "model-output" for model output files
3. Hubs with target data use a directory named "target-data"
4. Files are either CSV or parquet format
5. Parquet files have a .parquet extension
6. No two hubs have the same repository name
7. Users have a machine with a reasonable amount of memory (for expediency, the
script pulls entire .csv files into memory to count the rows instead of chunking
them)

Example
-------

To use this script:

1. Install uv on your machine: https://docs.astral.sh/uv/getting-started/installation/
2. Clone this repo
3. Modify the HUB_REPO_LIST variable in this script. Items should be in the format "owner/repo".
4. From the root of the repo, run the script:
`uv run src/reichlab_repo_utils/get_hub_stats.py`
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb",
#   "polars",
#   "requests",
#   "rich",
# ]
# ///

import concurrent.futures
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import duckdb
import polars as pl
import requests
from rich import print
from rich.console import Console
from rich.table import Table

###########################################################
# Location of hubs.json file                              #
###########################################################
try:
    HUB_JSON_PATH = Path(importlib.util.find_spec("reichlab_repo_utils").origin).parent  # type: ignore
except Exception:
    HUB_JSON_PATH = Path(__file__).parent / "hubs.json"

###########################################################
# A bunch of init stuff lazily stored in global variables #
###########################################################
try:
    token = os.environ["GITHUB_TOKEN"]
except KeyError:
    raise ValueError("GITHUB_TOKEN environment variable is required")
session = requests.Session()
session.headers.update({"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"})
pl.Config(tbl_rows=500)
pl.Config.set_fmt_str_lengths(100)
# For testing, limit number of files to process. Set to 0 for no limit
FILE_COUNT = 0


###########################################################
# Actual work starts here                                 #
###########################################################
def main(owner: str, repo: str, data_dir: str | None) -> Path:
    if data_dir is None:
        output_dir = Path.cwd() / "hub_stats"
    else:
        output_dir = Path(data_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Getting stats for [italic green]{owner}/{repo}[/italic green]")


    repo_line_counts = pl.DataFrame()
    for directory in ["model-output", "target-data"]:
        files = list_files_in_directory(owner, repo, directory)
        if len(files) == 0:
            continue

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(count_rows, file) for file in files]

        all_results = [f.result() for f in futures]
        all_counts = defaultdict(int)  # type: ignore
        for result in all_results:
            all_counts[result[0]] += result[1]

        count_df = pl.DataFrame({"file": list(all_counts.keys()), "row_count": list(all_counts.values())})

        if len(count_df) > 0:
            count_df = count_df.with_columns(
                pl.lit(directory).alias("dir"),
                pl.lit(f"{owner}/{repo}").alias("repo"),
            )
            # extract model_id from file name
            count_df = count_df.with_columns(
                pl.when(pl.col("dir") == "model-output").then(
                    pl.col("file")
                    .str.slice(11)
                    .str.strip_chars_start("_-")
                    .str.splitn(".", 2)
                    .struct.rename_fields(["model_id", "file_type"])
                    .struct.field("model_id")
                )
            )
            repo_line_counts = pl.concat([repo_line_counts, count_df])

    parquet_file = output_dir / f"{repo}.parquet"
    repo_line_counts.write_parquet(parquet_file)
    return parquet_file


def count_rows(file_url) -> tuple[str, int]:
    """Returns a dataframe with a line count for each file a list."""

    file_path = Path(urlsplit(file_url).path)
    file_name = file_path.name
    file_type = file_path.suffix

    try:
        if file_type == ".csv":
            count = count_rows_csv(file_url)
        else:
            count = count_rows_parquet(file_url)
    except Exception as e:
        print(f"Error processing {file_url}")
        print(e)
        count = 0
    finally:
        line_counts = (file_name, count)

    return line_counts


def count_rows_parquet(file_url: str) -> int:
    """Use duckdb to get row count from parquet file metadata."""
    # duckdb is a handy way to access parquet file metadata
    # (we're not using it to store any output data)
    with duckdb.connect() as con:
        con.sql("INSTALL httpfs;")
        sql = f"SELECT num_rows FROM parquet_file_metadata('{file_url}');"
        if (query_result := con.sql(sql).fetchone()) is not None:
            count = query_result[0]
        else:
            print(f"Unable to access parquet metadata for {file_url}")
            count = 0

    return count


def count_rows_csv(file_url: str) -> int:
    """Get .csv row count by requesting the file and counting lines."""
    response = session.get(file_url)
    response.raise_for_status()
    count = len(response.text.splitlines())

    return count


def list_files_in_directory(owner, repo, directory) -> list[str]:
    """Use GitHub API to get a list of files in a Hub's directory."""
    url: str | None = f"https://api.github.com/repos/{owner}/{repo}/contents/{directory}"

    files = []

    while url:
        response = session.get(url)
        # some hubs don't have a target-data directory, so 404 is a-ok
        if response.status_code == 404:
            print(f"[yellow]URL {url} not found[/yellow]")
            break
        response.raise_for_status()
        data = response.json()

        for item in data:
            if item["type"] == "file" and item["download_url"].endswith((".csv", ".parquet")):
                files.append(item["download_url"])
            elif item["type"] == "dir":
                files.extend(list_files_in_directory(owner, repo, item["path"]))
            if FILE_COUNT > 0 and len(files) >= FILE_COUNT:
                # file count is constrained for testing
                break

        # Check if there is a next page
        if "next" in response.links:
            url = response.links["next"]["url"]
        else:
            url = None

    return files


def write_csv(output_dir: Path):
    """Write output of all hub stats .parquet files to .csv."""
    parquet_files = f"{str(output_dir)}/*.parquet"
    try:
        # save detailed hub stats as .csv
        hub_stats = pl.read_parquet(parquet_files)
        csv_file = output_dir / "hub_stats.csv"
        hub_stats.write_csv(csv_file)

        # save a summarized version of hub stats
        hub_stats_summary = hub_stats.sql("""
            SELECT repo, dir, SUM(row_count) as row_count
            FROM self
            GROUP BY repo, dir
            ORDER BY repo, dir
        """)
        summary_csv_file = output_dir / "hub_stats_summary.csv"
        hub_stats_summary.write_csv(summary_csv_file)
        print(f"Saved hub summaries: {summary_csv_file}")

        # display summarized data on console
        console = Console()
        table = Table(title="Hub Stats Summary")

        table.add_column("Repo", justify="left", style="cyan", no_wrap=True)
        table.add_column("Dir", justify="left", style="magenta")
        table.add_column("Row Count", justify="right", style="green")

        for row in hub_stats_summary.iter_rows(named=True):
            formatted_count = f"{row['row_count']:,}"
            table.add_row(row["repo"], row["dir"], formatted_count)

        console.print(table)

    except pl.exceptions.ComputeError:
        print(f"Cannot create .csv: no parquet files found in {output_dir}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data" / "hub_stats"
    if FILE_COUNT > 0:
        print(f"Limiting file count to {FILE_COUNT} for testing")
    with open(HUB_JSON_PATH, "r") as file:
        hubs = json.load(file)
    hub_list = hubs.get("hubs")
    for hub in hub_list:
        owner = hub["org"]
        repo = hub["repo"]
        main(owner, repo, str(data_dir))
    write_csv(data_dir)
