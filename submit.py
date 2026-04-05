#!/usr/bin/env python3
"""
clawRxiv submission script for "The Replication Trap".

Usage:
    # Step 1: Register to get an API key (run once)
    python3 submit.py register --name "YourAgentName"

    # Step 2: Submit the paper
    python3 submit.py submit --api-key oc_your_key_here

    # Or using the CLAWRXIV_API_KEY environment variable:
    CLAWRXIV_API_KEY=oc_... python3 submit.py submit

    # Dry run (print payload without submitting):
    python3 submit.py submit --dry-run

Do NOT submit before the deadline (April 5, 2026).
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://www.clawrxiv.io/api"

TITLE = (
    "The Replication Trap: Precision Failures in LLM Scrutiny "
    "of Flawed Statistical Workflows"
)

TAGS = [
    "benchmarking",
    "peer-review",
    "replication-crisis",
    "methodology",
    "agent-evaluation",
]


def load_content():
    content_path = Path("submission_content.md")
    if not content_path.exists():
        sys.exit("submission_content.md not found. Run from the repo root.")
    return content_path.read_text()


def load_skill():
    skill_path = Path("SKILL.md")
    if not skill_path.exists():
        sys.exit("SKILL.md not found. Run from the repo root.")
    return skill_path.read_text()


def extract_abstract(content: str) -> str:
    """Extract the abstract section from submission_content.md."""
    lines = content.split("\n")
    in_abstract = False
    abstract_lines = []
    for line in lines:
        if line.strip() == "## Abstract":
            in_abstract = True
            continue
        if in_abstract:
            if line.startswith("## ") or line.strip() == "---":
                break
            abstract_lines.append(line)
    abstract = "\n".join(abstract_lines).strip()
    if not abstract:
        sys.exit("Could not extract abstract from submission_content.md")
    return abstract


def register(name: str):
    """Register an agent name and return the API key."""
    payload = json.dumps({"claw_name": name}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/auth/register",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            print(f"Registered successfully!")
            print(f"API key: {data.get('api_key', data)}")
            print(f"\nSet it as an environment variable:")
            print(f"  export CLAWRXIV_API_KEY={data.get('api_key', '<key>')}")
            return data.get("api_key")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        sys.exit(f"Registration failed ({e.code}): {body}")


def submit(api_key: str, dry_run: bool = False):
    """Submit the paper to clawRxiv."""
    content = load_content()
    skill = load_skill()
    abstract = extract_abstract(content)

    payload = {
        "title": TITLE,
        "abstract": abstract,
        "content": content,
        "tags": TAGS,
        "human_collaborators": ["Jeff Heuer"],
        "skill": skill,
    }

    if dry_run:
        print("=== DRY RUN — not submitting ===")
        print(f"Title: {payload['title']}")
        print(f"Tags: {payload['tags']}")
        print(f"Abstract ({len(abstract)} chars):\n{abstract[:500]}...")
        print(f"Content length: {len(content)} chars")
        print(f"Skill file: {len(skill)} chars (SKILL.md)")
        print("\nPayload keys:", list(payload.keys()))
        return

    payload_bytes = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/posts",
        data=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            print("Submission successful!")
            print(json.dumps(data, indent=2))
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        sys.exit(f"Submission failed ({e.code}): {body}")


def main():
    parser = argparse.ArgumentParser(description="clawRxiv submission tool")
    sub = parser.add_subparsers(dest="command", required=True)

    reg_parser = sub.add_parser("register", help="Register agent name")
    reg_parser.add_argument("--name", required=True, help="Agent name (2-64 chars)")

    sub_parser = sub.add_parser("submit", help="Submit the paper")
    sub_parser.add_argument("--api-key", help="clawRxiv API key (or set CLAWRXIV_API_KEY)")
    sub_parser.add_argument("--dry-run", action="store_true", help="Print payload without submitting")

    args = parser.parse_args()

    if args.command == "register":
        register(args.name)

    elif args.command == "submit":
        if args.dry_run:
            submit(api_key="", dry_run=True)
        else:
            api_key = args.api_key or os.environ.get("CLAWRXIV_API_KEY")
            if not api_key:
                sys.exit(
                    "Provide --api-key or set CLAWRXIV_API_KEY environment variable.\n"
                    "Get a key by running: python3 submit.py register --name YourName"
                )
            submit(api_key, dry_run=False)


if __name__ == "__main__":
    main()
