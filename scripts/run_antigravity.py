#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Antigravity CLI Wrapper")
    parser.add_argument("--prompt", required=True, help="The instruction prompt for antigravity")
    parser.add_argument("--repo-path", required=True, help="Absolute path to the repository where antigravity should run")
    args = parser.parse_args()

    repo_path = args.repo_path
    if not os.path.exists(repo_path):
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    print(f"Running Antigravity in {repo_path}...")
    print(f"Prompt: {args.prompt}")
    
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)
    env = os.environ.copy()
    
    # Path to the antigravity executable (configurable via env)
    antigravity_bin = os.getenv("ANTIGRAVITY_BIN", os.path.expanduser("~/.antigravity/antigravity/bin/antigravity"))
    
    if not os.path.exists(antigravity_bin):
        print(f"Error: Antigravity binary not found at {antigravity_bin}")
        print("Set ANTIGRAVITY_BIN env var to the correct path.")
        sys.exit(1)

    target_model = env.get("ANTIGRAVITY_MODEL")
    if target_model:
        print(f"Using Antigravity Model: {target_model}")
        env["MODEL"] = target_model
        env["ANTIGRAVITY_MODEL"] = target_model

    try:
        # Run antigravity chat with agent mode
        result = subprocess.run(
            [antigravity_bin, "chat", "-m", "agent", args.prompt],
            cwd=repo_path,
            env=env,
            text=True,
            capture_output=True,
            check=False
        )
        print("Antigravity agent session launched in your IDE!")
        print("It will ask for your permission directly in the editor before making changes.")
        if result.stdout:
            print("Output:", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Failed to launch Antigravity: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
