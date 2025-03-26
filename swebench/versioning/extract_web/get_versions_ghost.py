import json
import os
import re
import sys
from ghapi.core import GhApi

sys.path.append("../../harness")
from swebench.versioning.utils import get_instances

GITHUB_TOKEN = "<your GitHub token>"
PATH_TASKS_GHOST = "<path to sqlfluff task instances>"
PATH_TO_SAVE = "<path to save versioned task instances to>"

# Get raw ghost dataset
data_tasks = get_instances(PATH_TASKS_GHOST)

# Get all GitHub releases
api = GhApi(owner="TryGhost", repo="ghost", token=GITHUB_TOKEN)

releases, i = [], 0
while True:
    temp = api.repos.list_releases("TryGhost", "Ghost", 100, page=i + 1)
    releases.extend(temp)
    print(len(releases))
    if len(releases) > 300:
        break
    i += 1
pairs = [(x["name"], x["published_at"]) for x in releases]


def process(x):
    """Extract version number from name"""
    if x.startswith("Ghost "):
        x = x[len("Ghost "):]
    pattern = re.compile(r"(\d+)\.\d+\.\d+")
    matches = pattern.findall(x)
    if len(matches) > 0:
        ver = matches[0]
        return ver, None

    return None, None


# Collect version/date pairs
version_date_map = {}
for pair in pairs:
    pair_rv = process(pair[0])
    if pair_rv[0] is None:
        continue
    version = pair_rv[0]
    if version.startswith("Bugfix Release "):
        version = version[len("Bugfix Release "):]
    date = pair[1] if pair_rv[1] is None else pair_rv[1]
    if version in version_date_map:
        version_date_map[version] = max(version_date_map[version], date)
    else:
        version_date_map[version] = date

print(version_date_map)
# Get (date, version) pairs
times = [(v, k) for k, v in version_date_map.items()]
times = sorted(times, key=lambda x: x[0])[::-1]

# Iterate through data_tasks and assign versions
for task in data_tasks:
    created_at = task["created_at"].split("T")[0]
    set_version = False
    for t in times:
        if t[0] < created_at:
            task["version"] = t[1]
            set_version = True
            break
    if not set_version:
        task["version"] = None

# Save sqlfluff versioned data to repository
versioned_path = "ghost-task-instances-versioned.jsonl"
with open(
        os.path.join(PATH_TO_SAVE, versioned_path),
        "w",
) as f:
    json.dump(data_tasks, fp=f)

# Print all versions
versioned = json.load(open(os.path.join(PATH_TO_SAVE, versioned_path)))
print(sorted(list({t["version"] for t in versioned if t["version"] is not None})))
