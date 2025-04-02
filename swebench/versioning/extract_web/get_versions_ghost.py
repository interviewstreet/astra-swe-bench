import os
import json
import requests
import pandas as pd
import datetime
from bs4 import BeautifulSoup
from swebench.versioning.utils import get_instances

PATH_TASKS_GHOST = "../../collect/ghost/ghost-tasks.jsonl.all"
PATH_TO_SAVE = "../../collect/ghost"
NODE_VERSION_URL = "https://ghost.org/docs/faq/node-versions/"

# Get raw ghost dataset
data_tasks = get_instances(PATH_TASKS_GHOST)


# Get all versions from the Ghost Node Compatibility Matrix
def get_node_compatibility_matrix(url):
    # Download the page content
    response = requests.get(url)
    response.raise_for_status()  # raise an error for bad status codes

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table element with class 'gh-table' just after <h1> with id 'node-compatibility-matrix'
    table = soup.find('h1', id='node-compatibility-matrix').find_next('table', class_='gh-table')
    if not table:
        raise ValueError("No table found after the specified header.")

    # Extract table headers
    # The headers are usually in the <thead> section
    headers = []
    thead = table.find('thead')
    if thead:
        header_row = thead.find('tr')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
    else:
        # If no <thead>, try to get headers from the first row
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(['td', 'th'])]
    if not headers:
        raise ValueError("No headers found in the table.")
    # Print headers for debugging

    # Extract table rows
    rows = []
    for tr in table.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        if cells:
            row = [cell.get_text(strip=True) for cell in cells]
            rows.append(row)

    # Use the first row as header if available
    if headers:
        df = pd.DataFrame(rows[1:], columns=headers)
    else:
        # Fallback if no headers were found
        df = pd.DataFrame(rows)

    return df


version_df = get_node_compatibility_matrix(NODE_VERSION_URL)


def parse_version(version):
    # Remove any leading ≥ sign
    version = version.lstrip("≥")
    return version


version_df["Version"] = version_df["Version"].apply(parse_version)
version_df["Released"] = pd.to_datetime(version_df["Released"], format="%Y/%m/%d")

# Set the first version from the dataframe as the default version of the tasks
for task in data_tasks:
    task["version"] = version_df.iloc[0]["Version"]

for task in data_tasks:
    created_at = datetime.datetime.strptime(task["created_at"].split("T")[0], "%Y-%m-%d")
    for _, row in version_df.iterrows():
        if row["Released"] <= created_at:
            task['version'] = row["Version"]
        else:
            break
# print task id and version
# for task in data_tasks:
#     print(f"Task ID: {task['instance_id']}, Created At: {task['created_at']} Version: {task['version']}")

versioned_path = "ghost-task-instances-versioned.json"
with open(
        os.path.join(PATH_TO_SAVE, versioned_path),
        "w",
) as f:
    json.dump(data_tasks, fp=f)

# Print all versions
versioned = json.load(open(os.path.join(PATH_TO_SAVE, versioned_path)))
print(sorted(list({t["version"] for t in versioned if t["version"] is not None})))
