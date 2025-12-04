#!/usr/bin/env python3
"""
Script to connect via SSH, download a GitHub repository, and run train.py
"""

import paramiko
import sys
import time
from datetime import datetime, timedelta

# SSH Configuration
SSH_HOST = "78.128.134.167"
SSH_USER = "adam"
SSH_PASSWORD = "adam123"
SSH_PORT = 22

# Repository Configuration
GITHUB_REPO = input("Enter GitHub repository URL (e.g., https://github.com/user/repo.git): ").strip()
REPO_NAME = GITHUB_REPO.split("/")[-1].replace(".git", "")
WORK_DIR = f"/home/{SSH_USER}/{REPO_NAME}"

def execute_command(ssh_client, command, description="", capture_output=False):
    """Execute a command via SSH and print output."""
    print(f"\n{'='*60}")
    if description:
        print(f"[{description}]")
    print(f"Executing: {command}")
    print('='*60)

    stdin, stdout, stderr = ssh_client.exec_command(command)

    # Capture output
    output_lines = []
    for line in stdout:
        line_stripped = line.strip()
        if not capture_output:
            print(line_stripped)
        output_lines.append(line_stripped)

    # Check for errors
    error_output = stderr.read().decode()
    if error_output and not capture_output:
        print("STDERR:", error_output)

    exit_status = stdout.channel.recv_exit_status()
    if not capture_output:
        print(f"Exit status: {exit_status}")

    if capture_output:
        return exit_status, output_lines
    return exit_status

def get_directory_info(ssh_client, directory):
    """Get information about files in a directory."""
    _, files = execute_command(
        ssh_client,
        f"find {directory} -type f -printf '%s %p\\n' 2>/dev/null | sort -rn | head -20",
        capture_output=True
    )
    return files

def format_size(bytes_size):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def print_summary_report(ssh_client, work_dir, duration, training_exit_status):
    """Print a comprehensive summary report after training."""
    print("\n" + "="*70)
    print(" " * 20 + "TRAINING SUMMARY REPORT")
    print("="*70)

    # Training status
    print(f"\n{'Training Status:':<25} ", end="")
    if training_exit_status == 0:
        print("✓ SUCCESS")
    else:
        print(f"✗ FAILED (exit code: {training_exit_status})")

    # Duration
    print(f"{'Training Duration:':<25} {duration}")
    print(f"{'Completed at:':<25} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Generated files
    print(f"\n{'Generated Files:':<25}")
    print("-" * 70)

    # Get files created/modified in the last hour
    _, recent_files = execute_command(
        ssh_client,
        f"find {work_dir} -type f -mmin -60 -printf '%T@ %s %p\\n' 2>/dev/null | sort -rn",
        capture_output=True
    )

    if recent_files:
        print(f"{'Size':<12} {'Modified':<20} {'File'}")
        print("-" * 70)
        for file_info in recent_files[:15]:  # Show top 15 files
            parts = file_info.split(maxsplit=2)
            if len(parts) >= 3:
                timestamp, size, filepath = parts
                try:
                    mod_time = datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
                    size_formatted = format_size(int(size))
                    filename = filepath.replace(work_dir + '/', '')
                    print(f"{size_formatted:<12} {mod_time:<20} {filename}")
                except (ValueError, IndexError):
                    continue
    else:
        print("No files were created or modified during training.")

    # Model files specifically
    print(f"\n{'Model Files Found:':<25}")
    print("-" * 70)
    _, model_files = execute_command(
        ssh_client,
        f"find {work_dir} -type f \\( -name '*.pkl' -o -name '*.h5' -o -name '*.pt' -o -name '*.pth' -o -name '*.joblib' -o -name '*.model' -o -name '*.weights' \\) -printf '%s %p\\n' 2>/dev/null",
        capture_output=True
    )

    if model_files:
        for file_info in model_files:
            parts = file_info.split(maxsplit=1)
            if len(parts) == 2:
                size, filepath = parts
                size_formatted = format_size(int(size))
                filename = filepath.replace(work_dir + '/', '')
                print(f"  • {filename} ({size_formatted})")
    else:
        print("  No model files detected (.pkl, .h5, .pt, .pth, .joblib, .model, .weights)")

    # Log files
    print(f"\n{'Log Files Found:':<25}")
    print("-" * 70)
    _, log_files = execute_command(
        ssh_client,
        f"find {work_dir} -type f \\( -name '*.log' -o -name '*.txt' \\) -mmin -60 -printf '%s %p\\n' 2>/dev/null",
        capture_output=True
    )

    if log_files:
        for file_info in log_files:
            parts = file_info.split(maxsplit=1)
            if len(parts) == 2:
                size, filepath = parts
                size_formatted = format_size(int(size))
                filename = filepath.replace(work_dir + '/', '')
                print(f"  • {filename} ({size_formatted})")
    else:
        print("  No log files found")

    # Directory size
    _, du_output = execute_command(
        ssh_client,
        f"du -sh {work_dir} 2>/dev/null",
        capture_output=True
    )
    if du_output:
        print(f"\n{'Total Directory Size:':<25} {du_output[0].split()[0]}")

    print("\n" + "="*70)
    if training_exit_status == 0:
        print(" " * 22 + "✓ TRAINING COMPLETED")
    else:
        print(" " * 22 + "✗ TRAINING FAILED")
    print("="*70 + "\n")

def main():
    """Main execution function."""
    print("Starting SSH deployment script...")
    print(f"Target: {SSH_USER}@{SSH_HOST}")

    # Create SSH client
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to SSH server
        print(f"\nConnecting to {SSH_HOST}...")
        ssh_client.connect(
            hostname=SSH_HOST,
            port=SSH_PORT,
            username=SSH_USER,
            password=SSH_PASSWORD,
            timeout=10
        )
        print("✓ SSH connection established successfully!")

        # Check if git is installed
        execute_command(ssh_client, "git --version", "Checking Git installation")

        # Remove existing directory if it exists and clone repository
        commands = [
            (f"rm -rf {WORK_DIR}", "Cleaning up existing directory"),
            (f"git clone {GITHUB_REPO} {WORK_DIR}", "Cloning GitHub repository"),
            (f"cd {WORK_DIR} && ls -la", "Listing repository contents"),
        ]

        for cmd, desc in commands:
            execute_command(ssh_client, cmd, desc)

        # Check if train.py exists
        exit_status = execute_command(
            ssh_client,
            f"test -f {WORK_DIR}/train.py && echo 'train.py found' || echo 'train.py NOT found'",
            "Checking for train.py"
        )

        # Install requirements if requirements.txt exists
        execute_command(
            ssh_client,
            f"cd {WORK_DIR} && if [ -f requirements.txt ]; then pip3 install -r requirements.txt; else echo 'No requirements.txt found'; fi",
            "Installing dependencies"
        )

        # Run train.py and track duration
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)

        start_time = time.time()
        training_exit_status = execute_command(
            ssh_client,
            f"cd {WORK_DIR} && python3 train.py",
            "Running train.py"
        )
        end_time = time.time()

        # Calculate duration
        duration_seconds = int(end_time - start_time)
        duration = str(timedelta(seconds=duration_seconds))

        # Print comprehensive summary report
        print_summary_report(ssh_client, WORK_DIR, duration, training_exit_status)

    except paramiko.AuthenticationException:
        print("✗ Authentication failed. Please check credentials.")
        sys.exit(1)
    except paramiko.SSHException as e:
        print(f"✗ SSH connection error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ An error occurred: {e}")
        sys.exit(1)
    finally:
        ssh_client.close()
        print("\nSSH connection closed.")

if __name__ == "__main__":
    main()
