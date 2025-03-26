#!/usr/bin/env python3

import argparse
import json
import logging
import os
from typing import Optional, Tuple
import openai
import re
from together import Together
from dotenv import load_dotenv

from swebench.collect.utils import (
    extract_patches,
    extract_problem_statement_and_hints,
    Repo,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI and Together AI client
openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
together_client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

def extract_sdlc(problem_statement: str, patch: str) -> str:
    """
    Extract SDLC phase using DeepSeek R1 model via Together AI.
    """
    prompt_template = """
    You are a senior software developer.

    Below you will find:
    - A description of a GitHub issue.
    - The corresponding code diff to address that issue.

    Based solely on this information, please provide only the name of the Software Development Life Cycle (SDLC) phase that the task falls under.
    Format your response exactly as follows:
    SDLC: <name_of_sdlc_phase>

    Inputs:
    - GitHub Issue Description: {problem_statement}
    - Code Diff: {patch}
    """
    prompt = prompt_template.format(
        problem_statement=problem_statement, 
        patch=patch
    )    
    messages = [{"role": "user", "content": prompt}]
    response = together_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=messages,
        temperature=0,
    )
    # Try accessing the text directly from the first choice.
    try:
        response_text = response.choices[0].text.strip()
    except AttributeError:
    # Fallback to using the Chat Completions style
        response_text = response.choices[0].message.content.strip()
    
    sdlc_match = re.search(r'SDLC:\s*(.*?)(?:\n|$)', response_text)
    return sdlc_match.group(1).strip() if sdlc_match else ""



def extract_skills(problem_statement: str, patch: str) -> str:
    """
    Extract required skills using GPT-4.5-preview.
    """
    prompt_template = """
    You are a senior software developer.

    Below you will find:
    - A description of a GitHub issue.
    - The corresponding code diff to address that issue.

    Based solely on this information, please provide only the names of the development skills required to fix the task.
    Format your response exactly as follows:
    Skills: <comma_separated_list_of_skills>

    Inputs:
    - GitHub Issue Description: {problem_statement}
    - Code Diff: {patch}
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_template.format(
            problem_statement=problem_statement,
            patch=patch
        )}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4.5-preview",
        messages=messages,
        temperature=0
    )
    
    response_text = response.choices[0].message.content.strip()
    skills_match = re.search(r'Skills:\s*(.*?)(?:\n|$)', response_text)
    return skills_match.group(1).strip() if skills_match else ""


def extract_sdlc_and_skills(problem_statement: str, patch: str) -> Tuple[str, str]:
    """
    Extract SDLC phase and required skills using different models.
    """
    try:
        sdlc = extract_sdlc(problem_statement, patch)
    except Exception as e:
        logger.error(f"Error extracting SDLC: {str(e)}")
        sdlc = ""
        
    try:
        skills = extract_skills(problem_statement, patch)
    except Exception as e:
        logger.error(f"Error extracting skills: {str(e)}")
        skills = ""
        
    return sdlc, skills


def create_instance(repo: Repo, pull: dict) -> dict:
    """
    Create a single task instance from a pull request, where task instance is:

    {
        repo (str): owner/repo this task instance is from,
        pull_number (int): number of PR this task instance is from,
        base_commit (str): SHA of the base commit PR is based on,
        patch (str): reference solution as .patch (apply to base commit),
        test_patch (str): test suite as .patch (apply to base commit),
    }
    """
    patch, test_patch = extract_patches(pull, repo)
    problem_statement, hints = extract_problem_statement_and_hints(pull, repo)
    # Extract SDLC phase and skills
    sdlc_phase, required_skills = extract_sdlc_and_skills(problem_statement, patch)
    
    return {
        "repo": repo.repo.full_name,
        "pull_number": pull["number"],
        "instance_id": (repo.repo.full_name + "-" + str(pull["number"])).replace(
            "/", "__"
        ),
        "issue_numbers": pull["resolved_issues"],
        "base_commit": pull["base"]["sha"],
        "patch": patch,
        "test_patch": test_patch,
        "problem_statement": problem_statement,
        "hints_text": hints,
        "created_at": pull["created_at"],
        "sdlc_phase": sdlc_phase,
        "required_skills": required_skills,
    }


def is_valid_pull(pull: dict) -> bool:
    """
    Check whether PR has an associated issue and is merged

    Args:
        pull (dict): pull request object
    Returns:
        bool: whether PR is valid
    """
    if pull["merged_at"] is None:
        return False
    if "resolved_issues" not in pull or len(pull["resolved_issues"]) < 1:
        return False
    return True


def is_valid_instance(instance: dict) -> bool:
    """
    Check whether task instance has all required fields for task instance creation

    Args:
        instance (dict): task instance object
    Returns:
        bool: whether task instance is valid
    """
    if instance["patch"] is None or instance["patch"] == "":
        return False
    if instance["problem_statement"] is None or instance["problem_statement"] == "":
        return False
    return True


def has_test_patch(instance: dict) -> bool:
    """
    Check whether task instance has a test suite

    Args:
        instance (dict): task instance object
    Returns:
        bool: whether task instance has a test suite
    """
    if instance["test_patch"] is None or instance["test_patch"].strip() == "":
        return False
    return True


def main(pr_file: str, output: str, token: Optional[str] = None):
    """
    Main thread for creating task instances from pull requests

    Args:
        pr_file (str): path to pull request JSONL file
        output (str): output file name
        token (str): GitHub token
    """
    if token is None:
        # Get GitHub token from environment variable if not provided
        token = os.environ.get("GITHUB_TOKEN")

    def load_repo(repo_name):
        # Return repo object for a given repo name
        owner, repo = repo_name.split("/")
        return Repo(owner, repo, token=token)

    repos = dict()
    completed = 0
    with_tests = 0
    total_instances = 0
    all_output = output + ".all"
    seen_prs = set()

    # Continue where we left off if output file already exists
    if os.path.exists(all_output):
        with open(all_output) as f:
            for line in f:
                pr = json.loads(line)
                if "instance_id" not in pr:
                    pr["instance_id"] = (
                        pr["repo"] + "-" + str(pr["pull_number"])
                    ).replace("/", "__")
                instance_id = pr["instance_id"]
                seen_prs.add(instance_id)
                if is_valid_instance(pr):
                    completed += 1
                    if has_test_patch(pr):
                        with_tests += 1
    logger.info(
        f"Will skip {len(seen_prs)} pull requests that have already been inspected"
    )

    # Write to .all file for all PRs
    write_mode_all = "w" if not os.path.exists(all_output) else "a"
    with open(all_output, write_mode_all) as all_output:
        # Write to output file for PRs with test suites
        write_mode = "w" if not os.path.exists(output) else "a"
        with open(output, write_mode) as output:
            for ix, line in enumerate(open(pr_file)):
                total_instances += 1
                pull = json.loads(line)
                if ix % 100 == 0:
                    logger.info(
                        f"[{pull['base']['repo']['full_name']}] (Up to {ix} checked) "
                        f"{completed} valid, {with_tests} with tests."
                    )
                # Construct instance fields
                instance_id = (
                    pull["base"]["repo"]["full_name"] + "-" + str(pull["number"])
                )
                instance_id = instance_id.replace("/", "__")
                if instance_id in seen_prs:
                    seen_prs -= {instance_id}
                    continue
                if not is_valid_pull(pull):
                    # Throw out invalid PRs
                    continue
                # Create task instance
                repo_name = pull["base"]["repo"]["full_name"]
                if repo_name not in repos:
                    repos[repo_name] = load_repo(repo_name)
                repo = repos[repo_name]
                instance = create_instance(repo, pull)
                if is_valid_instance(instance):
                    # If valid, write to .all output file
                    print(
                        json.dumps(instance), end="\n", flush=True, file=all_output
                    )  # write all instances to a separate file
                    completed += 1
                    if has_test_patch(instance):
                        # If has test suite, write to output file
                        print(json.dumps(instance), end="\n", flush=True, file=output)
                        with_tests += 1
    logger.info(
        f"[{', '.join(repos.keys())}] Total instances: {total_instances}, completed: {completed}, with tests: {with_tests}"
    )
    logger.info(
        f"[{', '.join(repos.keys())}] Skipped {len(seen_prs)} pull requests that have already been inspected"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pr_file", type=str, help="Path to pull request JSONL file")
    parser.add_argument("output", type=str, help="Output file name")
    parser.add_argument("--token", type=str, help="GitHub token")
    args = parser.parse_args()
    main(**vars(args))
