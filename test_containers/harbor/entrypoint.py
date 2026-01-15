import argparse
import json
import subprocess
import sys
import os
import select
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Garak test container")
    parser.add_argument(
        "--systems-params", required=True, help="Systems parameters as JSON string"
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )

    args = parser.parse_args()

    try:
        # Parse inputs
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)

        # Extract system_under_test
        sut_params = systems_params.get("system_under_test", {})
        if not sut_params:
            raise ValueError("Missing system_under_test in systems_params")

        # Validate SUT type
        sut_type = sut_params.get("type")
        if sut_type not in ["agent_cli"]:
            raise ValueError(f"Unsupported system_under_test type: {sut_type}")
        
        result = run_harbor(sut_params=sut_params, test_params=test_params)
        
        success = True
        test_result = {
            "success": success,
            "agent_config": systems_params,
            "test_params": test_params,
        }

        print(json.dumps(test_result, indent=2))

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    
    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "error": f"Invalid JSON in arguments: {e}",
            "score": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except subprocess.TimeoutExpired:
        error_result = {
            "success": False,
            "error": "Harbor execution timed out",
            "score": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "score": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def run_harbor(sut_params, test_params):
    # Build harbor command
    harbor_cmd = ["harbor", "run"]
    provider = sut_params.get("provider", None)
    model = sut_params.get("model", None)
    api_key = sut_params.get("api_key", None)

    if not provider:
        raise ValueError("Missing provider in systems_params")
    if not model:
        raise ValueError("Missing model in systems_params")
    if not api_key:
        raise ValueError("Missing api_key in systems_params")
    
    harbor_cmd.extend(["--agent", provider, "--model", model])
    harbor_cmd.extend(["--env", "docker"])
    
    match provider:
        case "codex":
            API_ENV = "OPENAI_API_KEY"
        case "claude-code":
            API_ENV = "ANTHROPIC_API_KEY"
        case _:
            raise ValueError(f"Unsupported provider: {provider}")
         
    # Add test parameters
    dataset = test_params.get("dataset", "hello-world@1.0")
    harbor_cmd.extend(["--dataset", dataset])

    # Configure output directory
    # For Docker-in-Docker: Harbor needs to use the host path (not container path)
    # so the host Docker daemon can mount it into nested containers
    host_output_path = os.environ.get("HOST_OUTPUT_PATH")
    if host_output_path:
        # Use host path for Docker-in-Docker support
        job_dir = str(Path(host_output_path) / "harbor")
        os.makedirs(job_dir, exist_ok=True)
        print(f"DEBUG: Using host output path for Docker-in-Docker: {job_dir}", file=sys.stderr)
    else:
        # Fallback to local directory (will be copied to output mount later)
        job_dir = str(Path("/app/jobs"))
        os.makedirs(job_dir, exist_ok=True)
        
    harbor_cmd.extend(["--jobs-dir", job_dir])

    # Use current environment API keys
    env = os.environ.copy()

    # Set appropriate API key environment variable based on model type
    env[API_ENV] = api_key

    print(
        f"DEBUG: About to run command: {' '.join(harbor_cmd)}", file=sys.stderr
    )

    # --- run command: start ---

    # Run harbor with real-time output and capture
    process = subprocess.Popen(
        harbor_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=0,  # Unbuffered for real-time character output
        universal_newlines=True,
    )

    stdout_content = []
    stderr_content = []

    # handle streaming progress bar
    def handle_output():
        while process.poll() is None:
            ready, _, _ = select.select(
                [process.stdout, process.stderr], [], [], 0.1
            )

            for stream in ready:
                if stream == process.stdout and process.stdout is not None:
                    char = process.stdout.read(1)
                    if char:
                        stdout_content.append(char)
                        sys.stdout.write(char)
                        sys.stdout.flush()
                elif stream == process.stderr and process.stderr is not None:
                    char = process.stderr.read(1)
                    if char:
                        stderr_content.append(char)
                        sys.stderr.write(char)
                        sys.stderr.flush()

        # Read any remaining output
        remaining_stdout = (
            process.stdout.read() if process.stdout is not None else ""
        )
        remaining_stderr = (
            process.stderr.read() if process.stderr is not None else ""
        )

        if remaining_stdout:
            stdout_content.append(remaining_stdout)
            sys.stdout.write(remaining_stdout)
            sys.stdout.flush()

        if remaining_stderr:
            stderr_content.append(remaining_stderr)
            sys.stderr.write(remaining_stderr)
            sys.stderr.flush()

    handle_output()
    process.wait()
    result = subprocess.CompletedProcess(
        args=harbor_cmd,
        returncode=process.returncode,
        stdout="".join(stdout_content),
        stderr="".join(stderr_content),
    )

    # --- run command: end ---

    # Copy results to output mount only if not using Docker-in-Docker
    # (with Docker-in-Docker, results are already in the host path)
    if not host_output_path and Path("/output").exists() and Path("/app/jobs").exists():
        dest_root = Path("/output") / "harbor"
        print(f"DEBUG: Copying results from /app/jobs to {dest_root}", file=sys.stderr)
        try:
            shutil.copytree(Path("/app/jobs"), dest_root, dirs_exist_ok=True)
        except Exception as e:
            print(f"WARNING: Failed to copy results: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()