#!/usr/bin/env python3
"""
Deploy terminal for the robot dog: syncs the workspace to the Pi and runs commands there.

    pixi run deploy
    robotdog> pixi run build

Every remote pixi command first rsyncs the workspace (respecting .gitignore). Calibration
updates printed on the Pi are applied to the laptop's robot_dog.yaml, which the next deploy
pushes back.
"""

import hashlib
import json
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

SCRIPT_DIR     = Path(__file__).parent
CACHE_FILE     = SCRIPT_DIR / ".deploy_config"
HISTORY_FILE   = SCRIPT_DIR / ".deploy_history"
WORKSPACE_ROOT = SCRIPT_DIR.parent  # Project root (contains pixi.toml)

DEFAULT_USERNAME   = "pi"
DEFAULT_HOST       = "robotdog.local"
DEFAULT_DEPLOY_DIR = "robot_dog_ws"  # relative to the home directory on the Pi

COMMANDS = {
    "deploy": "Sync workspace to Pi using rsync",
    "remote": "SSH into Pi (no args) or run a command remotely (with args)",
    "pixi":   "Deploy then run a pixi command on Pi (e.g. pixi run build)",
    "scp":    "Copy files between local and Pi (scp pull <path> | scp push <path>)",
    "ping":   "Ping Pi and show response time",
    "config": "Show or update connection settings (username, host, deploy dir)",
    "help":   "Show this help message",
    "service": "Manage remote services and processes", 
    "exit":   "Exit the terminal",
}

# Fallback task list for autocomplete when pixi.toml can't be parsed
PIXI_TASKS_FALLBACK = ["build", "clean"]

# Directories with at most this many files are compared per file by MD5
DIR_CHECKSUM_THRESHOLD = 50


def get_pixi_tasks() -> list[str]:
    """Parse pixi.toml and return list of task names (for autocomplete)."""
    pixi_toml = WORKSPACE_ROOT / "pixi.toml"
    if not pixi_toml.exists():
        return []
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return []
    try:
        with open(pixi_toml, "rb") as f:
            data = tomllib.load(f)
        return list(data.get("tasks", {}).keys())
    except Exception:
        return []


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(cfg: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        console.print(f"[yellow]Warning: could not save config: {e}[/yellow]")


def get_connection() -> tuple[str, str, str]:
    cfg = load_config()
    username   = cfg.get("username",   DEFAULT_USERNAME)
    host       = cfg.get("host",       DEFAULT_HOST)
    deploy_dir = cfg.get("deploy_dir", DEFAULT_DEPLOY_DIR)
    return username, host, deploy_dir


# ── SSH key helpers ───────────────────────────────────────────────────────────

def ssh_key_path() -> Path:
    return Path.home() / ".ssh" / "id_ed25519_robotdog"


def ensure_ssh_key() -> Optional[Path]:
    """Ensure the dedicated SSH key exists, generating it if needed.

    Returns the key path, or None if generation failed (callers should abort).
    """
    ssh_dir = Path.home() / ".ssh"
    try:
        ssh_dir.mkdir(mode=0o700, exist_ok=True)
    except Exception as e:
        console.print(f"[red]Could not create {ssh_dir}: {e}[/red]")
        return None

    key = ssh_key_path()
    if key.exists():
        return key

    console.print(f"[yellow]Generating SSH key: {key.name}[/yellow]")
    try:
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(key), "-N", "", "-C", "robotdog-deploy"],
            check=True,
        )
        console.print(f"[green]SSH key generated: {key}[/green]")
        return key
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[red]Failed to generate SSH key: {e}[/red]")
        return None


def ssh_cmd(username: str, host: str) -> list[str]:
    key = ssh_key_path()
    return ["ssh", "-i", str(key), f"{username}@{host}"]


def check_ssh_authorized(username: str, host: str) -> bool:
    key = ssh_key_path()
    if not key.exists():
        return False
    try:
        result = subprocess.run(
            ["ssh", "-i", str(key),
             "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=5",
             "-o", "StrictHostKeyChecking=accept-new",
             f"{username}@{host}", "echo ok"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


def copy_ssh_key(username: str, host: str) -> bool:
    key = ssh_key_path()
    pub = Path(str(key) + ".pub")
    if not pub.exists():
        console.print("[red]Public key not found.[/red]")
        return False
    console.print(f"[cyan]Copying SSH key to {username}@{host} (you'll be prompted for password once)...[/cyan]")
    try:
        result = subprocess.run(["ssh-copy-id", "-i", str(key), f"{username}@{host}"], check=False)
        if result.returncode == 0:
            console.print("[green]SSH key copied. Password-less login enabled.[/green]")
            return True
        console.print("[yellow]ssh-copy-id failed, trying manual method...[/yellow]")
    except FileNotFoundError:
        console.print("[yellow]ssh-copy-id not found, trying manual method...[/yellow]")
    # Manual fallback
    pub_key = pub.read_text().strip()
    remote_cmd = (
        f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        f"echo '{pub_key}' >> ~/.ssh/authorized_keys && "
        f"chmod 600 ~/.ssh/authorized_keys"
    )
    result = subprocess.run(["ssh", f"{username}@{host}", remote_cmd], check=False)
    if result.returncode == 0:
        console.print("[green]SSH key copied. Password-less login enabled.[/green]")
        return True
    console.print("[red]Failed to copy SSH key.[/red]")
    return False


# ── Remote probe helpers ──────────────────────────────────────────────────────

def remote_path_type(full_remote: str, base_ssh_cmd: list[str]) -> Optional[str]:
    """Return 'dir' or 'file' for an existing/assumed remote path, or
    None if the SSH probe itself failed (key not authorized, host down, etc.).
    """
    q = shlex.quote(full_remote)
    cmd = base_ssh_cmd + ["-o", "BatchMode=yes",
                          f"if [ -d {q} ]; then echo dir; else echo file; fi"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        out = result.stdout.strip()
        if result.returncode == 0 and out in ("dir", "file"):
            return out
    except Exception:
        pass
    return None


def count_remote_files(remote_dir: str, base_ssh_cmd: list[str]) -> Optional[int]:
    """Count regular files under remote_dir, or None on failure."""
    q = shlex.quote(remote_dir)
    cmd = base_ssh_cmd + ["-o", "BatchMode=yes",
                          f"find {q} -type f 2>/dev/null | wc -l"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return None
        return int(result.stdout.strip())
    except Exception:
        return None


def remote_mkdir_p(remote_dir: str, base_ssh_cmd: list[str]) -> None:
    """Best-effort 'mkdir -p' on the remote."""
    q = shlex.quote(remote_dir)
    try:
        subprocess.run(base_ssh_cmd + ["-o", "BatchMode=yes", f"mkdir -p {q}"],
                       capture_output=True, timeout=10)
    except Exception:
        pass


# ── MD5 helpers (for smart scp) ───────────────────────────────────────────────

def get_local_md5(local_path: Path) -> Optional[str]:
    """Return MD5 hex digest of a local file, or None on failure."""
    try:
        h = hashlib.md5()
        with open(local_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def get_remote_md5(remote_path: str, base_ssh_cmd: list[str]) -> Optional[str]:
    """Return MD5 hex digest of a remote file, or None on failure."""
    q = shlex.quote(remote_path)
    try:
        result = subprocess.run(
            base_ssh_cmd + ["-o", "BatchMode=yes", f"md5sum {q} 2>/dev/null"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.split()[0]
    except Exception:
        pass
    return None


def get_dir_md5s_local(local_dir: Path) -> dict[str, str]:
    """Return {relative_path: md5} for all files under local_dir."""
    md5s = {}
    for f in local_dir.rglob("*"):
        if f.is_file():
            md5 = get_local_md5(f)
            if md5:
                md5s[str(f.relative_to(local_dir))] = md5
    return md5s


def get_dir_md5s_remote(remote_dir: str, base_ssh_cmd: list[str]) -> Optional[dict[str, str]]:
    """Return {relative_path: md5} for all files under remote_dir, or None on failure."""
    q = shlex.quote(remote_dir)
    cmd = base_ssh_cmd + ["-o", "BatchMode=yes",
                          f"find {q} -type f -exec md5sum {{}} \\; 2>/dev/null"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 and not result.stdout.strip():
            return None
        md5s = {}
        prefix = remote_dir.rstrip("/") + "/"
        for line in result.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                md5, path = parts
                rel = path[len(prefix):] if path.startswith(prefix) else path
                md5s[rel] = md5
        return md5s
    except Exception:
        return None


# ── SCP helpers ───────────────────────────────────────────────────────────────

def _rel_under_deploy_dir(full_remote: str, deploy_dir: str) -> str:
    """Map a remote path to a workspace-relative path."""
    base = deploy_dir.rstrip("/")
    if full_remote.rstrip("/") == base:
        return ""
    prefix = base + "/"
    if full_remote.startswith(prefix):
        return full_remote[len(prefix):]
    return Path(full_remote).name


def scp_pull(remote_path_str: str, deploy_dir: str, username: str, host: str,
             key: Path, base_ssh_cmd: list[str]):
    """Copy a file or directory from Pi to local workspace."""
    if remote_path_str.startswith("/"):
        full_remote = remote_path_str
    else:
        full_remote = f"{deploy_dir}/{remote_path_str}"

    ptype = remote_path_type(full_remote, base_ssh_cmd)
    if ptype is None:
        console.print("[yellow]Warning: SSH probe failed (is the key authorized? Run 'config'). Assuming single file.[/yellow]")
    is_dir = ptype == "dir"

    rel = _rel_under_deploy_dir(full_remote, deploy_dir)
    local_path = WORKSPACE_ROOT / rel if rel else WORKSPACE_ROOT

    if is_dir:
        console.print(f"[cyan]Pulling directory: {full_remote} → {local_path}[/cyan]")
        file_count = count_remote_files(full_remote, base_ssh_cmd)

        if file_count is not None and file_count <= DIR_CHECKSUM_THRESHOLD:
            console.print(f"[dim]Checking checksums for {file_count} file(s)...[/dim]")
            remote_files = get_dir_md5s_remote(full_remote, base_ssh_cmd)
            if remote_files is not None:
                local_files = get_dir_md5s_local(local_path) if local_path.exists() else {}
                to_transfer = [r for r, md5 in remote_files.items() if local_files.get(r) != md5]
                skipped = len(remote_files) - len(to_transfer)
                if not to_transfer:
                    console.print(f"[green]All {len(remote_files)} file(s) identical, nothing to transfer.[/green]")
                    return
                if skipped:
                    console.print(f"[dim]Skipping {skipped} identical, transferring {len(to_transfer)}[/dim]\n")
                errors = 0
                for r in to_transfer:
                    local_file = local_path / r
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    remote_file = f"{full_remote}/{r}"
                    cmd = ["scp", "-i", str(key), f"{username}@{host}:{remote_file}", str(local_file)]
                    console.print(f"[dim]  ← {r}[/dim]")
                    res = subprocess.run(cmd, check=False)
                    if res.returncode != 0:
                        console.print(f"[red]  Failed: {r}[/red]")
                        errors += 1
                console.print("\n[bold green]Done![/bold green]" if errors == 0
                              else f"\n[yellow]Done with {errors} error(s).[/yellow]")
                return
            console.print("[dim]Remote checksum fetch failed, falling back to full transfer[/dim]")
        elif file_count is not None:
            console.print(f"[dim]{file_count} files exceeds threshold ({DIR_CHECKSUM_THRESHOLD}), skipping checksum check[/dim]")
        else:
            console.print("[dim]Could not count remote files, falling back to full transfer[/dim]")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        scp_dest = str(local_path.parent) + "/" if local_path.exists() else str(local_path)
        cmd = ["scp", "-i", str(key), "-r", f"{username}@{host}:{full_remote}", scp_dest]
        console.print(f"[dim]{' '.join(cmd)}[/dim]\n")
        try:
            result = subprocess.run(cmd, check=False)
            console.print("\n[bold green]Done![/bold green]" if result.returncode == 0
                          else f"\n[red]SCP failed (exit code {result.returncode})[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
        return

    if local_path.exists():
        remote_md5 = get_remote_md5(full_remote, base_ssh_cmd)
        local_md5  = get_local_md5(local_path)
        if remote_md5 and local_md5 and remote_md5 == local_md5:
            console.print(f"[green]Skipping (identical): {full_remote}[/green]")
            return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Pulling file: {full_remote} → {local_path}[/cyan]")
    cmd = ["scp", "-i", str(key), f"{username}@{host}:{full_remote}", str(local_path)]
    console.print(f"[dim]{' '.join(cmd)}[/dim]\n")
    try:
        result = subprocess.run(cmd, check=False)
        console.print("\n[bold green]Done![/bold green]" if result.returncode == 0
                      else f"\n[red]SCP failed (exit code {result.returncode})[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")


def scp_push(local_path_str: str, deploy_dir: str, username: str, host: str,
             key: Path, base_ssh_cmd: list[str]):
    """Copy a file or directory from local workspace to Pi."""
    local_path = (Path(local_path_str) if local_path_str.startswith("/")
                  else WORKSPACE_ROOT / local_path_str)

    if not local_path.exists():
        console.print(f"[red]Local path not found: {local_path}[/red]")
        return

    is_dir = local_path.is_dir()

    try:
        rel = local_path.relative_to(WORKSPACE_ROOT)
        full_remote = f"{deploy_dir}/{rel}"
    except ValueError:
        full_remote = f"{deploy_dir}/{local_path.name}"

    if not is_dir:
        remote_md5 = get_remote_md5(full_remote, base_ssh_cmd)
        local_md5  = get_local_md5(local_path)
        if remote_md5 and local_md5 and remote_md5 == local_md5:
            console.print(f"[green]Skipping (identical): {local_path}[/green]")
            return

    if is_dir:
        console.print(f"[cyan]Pushing directory: {local_path} → {full_remote}[/cyan]")
        local_files = get_dir_md5s_local(local_path)
        file_count  = len(local_files)

        if file_count <= DIR_CHECKSUM_THRESHOLD:
            console.print(f"[dim]Checking checksums for {file_count} file(s)...[/dim]")
            remote_files = get_dir_md5s_remote(full_remote, base_ssh_cmd)
            if remote_files is not None:
                to_transfer = [r for r, md5 in local_files.items() if remote_files.get(r) != md5]
                skipped = file_count - len(to_transfer)
                if not to_transfer:
                    console.print(f"[green]All {file_count} file(s) identical, nothing to transfer.[/green]")
                    return
                if skipped:
                    console.print(f"[dim]Skipping {skipped} identical, transferring {len(to_transfer)}[/dim]\n")
                errors = 0
                for r in to_transfer:
                    remote_file   = f"{full_remote}/{r}"
                    remote_parent = str(Path(remote_file).parent)
                    remote_mkdir_p(remote_parent, base_ssh_cmd)
                    cmd = ["scp", "-i", str(key), str(local_path / r),
                           f"{username}@{host}:{remote_file}"]
                    console.print(f"[dim]  → {r}[/dim]")
                    res = subprocess.run(cmd, check=False)
                    if res.returncode != 0:
                        console.print(f"[red]  Failed: {r}[/red]")
                        errors += 1
                console.print("\n[bold green]Done![/bold green]" if errors == 0
                              else f"\n[yellow]Done with {errors} error(s).[/yellow]")
                return
            console.print("[dim]Remote checksum fetch failed, falling back to full transfer[/dim]")
        else:
            console.print(f"[dim]{file_count} files exceeds threshold ({DIR_CHECKSUM_THRESHOLD}), skipping checksum check[/dim]")

        ptype = remote_path_type(full_remote, base_ssh_cmd)
        if ptype is None:
            console.print("[yellow]Warning: SSH probe failed (is the key authorized? Run 'config').[/yellow]")
        remote_dir_exists = ptype == "dir"
        remote_mkdir_p(str(Path(full_remote).parent), base_ssh_cmd)
        scp_dest = (f"{username}@{host}:{str(Path(full_remote).parent)}/"
                    if remote_dir_exists else f"{username}@{host}:{full_remote}")
        cmd = ["scp", "-i", str(key), "-r", str(local_path), scp_dest]
        console.print(f"[dim]{' '.join(cmd)}[/dim]\n")
        try:
            result = subprocess.run(cmd, check=False)
            console.print("\n[bold green]Done![/bold green]" if result.returncode == 0
                          else f"\n[red]SCP failed (exit code {result.returncode})[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
        return

    remote_mkdir_p(str(Path(full_remote).parent), base_ssh_cmd)
    console.print(f"[cyan]Pushing file: {local_path} → {full_remote}[/cyan]")
    cmd = ["scp", "-i", str(key), str(local_path), f"{username}@{host}:{full_remote}"]
    console.print(f"[dim]{' '.join(cmd)}[/dim]\n")
    try:
        result = subprocess.run(cmd, check=False)
        console.print("\n[bold green]Done![/bold green]" if result.returncode == 0
                      else f"\n[red]SCP failed (exit code {result.returncode})[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")


# ── Remote Lifecycle Wrap Helper ──────────────────────────────────────────────

def wrap_remote_longlived(inner: str) -> str:
    """Wrap a long-running remote command so a dropped SSH connection or Ctrl+C kills its whole
    remote process group."""
    return (
        'cleanup() { trap "" INT TERM HUP; '
        'kill -TERM 0 2>/dev/null; sleep 1; kill -KILL 0 2>/dev/null; }; '
        'trap cleanup INT TERM HUP; '
        f'{inner} & wait'
    )


# ── Remote output streaming (config updates from the calibration tools) ───────

def _config_updater():
    """update_calibration + parse_config_update from the workspace (no build needed)."""
    try:
        from robot_dog_hw.calibration import parse_config_update, update_calibration
    except ImportError:
        sys.path.insert(0, str(WORKSPACE_ROOT / "src" / "robot_dog_hw"))
        from robot_dog_hw.calibration import parse_config_update, update_calibration
    return parse_config_update, update_calibration


def run_remote_streaming(cmd: list[str]) -> int:
    """Run an ssh command, echo its output live, and apply every ROBOT_DOG_CONFIG_UPDATE
    line from the calibration tools to the laptop's robot_dog.yaml."""
    parse_update, apply_update = _config_updater()
    # Binary pass-through of the output.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    applied = 0
    out = sys.stdout.buffer
    try:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            out.write(raw)
            out.flush()
            upd = parse_update(raw.decode("utf-8", errors="replace"))
            if upd is None:
                continue
            try:
                path = apply_update(upd["node"], upd["values"], announce=False)
                applied += 1
                console.print(f"[green]laptop {path.name} updated: {upd['node']} "
                              f"{upd['values']}[/green]")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]could not apply config update locally: {exc}[/red]")
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
        console.print("\n[yellow]Interrupted.[/yellow]")
    rc = proc.wait()
    if applied:
        console.print(f"[bold green]{applied} config update(s) written to "
                      f"src/robot_dog_bringup/config/robot_dog.yaml - commit it; the next "
                      f"deploy pushes it to the Pi.[/bold green]")
    return rc


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_help():
    table = Table(title="Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="green")
    table.add_column("Description")
    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)
    console.print("\n[dim]Tip: Tab for autocomplete, Up/Down for history[/dim]")


def cmd_ping():
    _, host, _ = get_connection()
    console.print(f"[cyan]Pinging {host}...[/cyan]\n")
    try:
        result = subprocess.run(["ping", "-c", "4", host], timeout=30)
        if result.returncode == 0:
            console.print("\n[green]Ping successful![/green]")
        else:
            console.print("\n[red]Ping failed.[/red]")
    except subprocess.TimeoutExpired:
        console.print("\n[red]Ping timed out.[/red]")


def cmd_config(args: str = ""):
    cfg = load_config()
    username   = cfg.get("username",   DEFAULT_USERNAME)
    host       = cfg.get("host",       DEFAULT_HOST)
    deploy_dir = cfg.get("deploy_dir", DEFAULT_DEPLOY_DIR)

    table = Table(show_header=False, box=None)
    table.add_row("Username",   username)
    table.add_row("Host",       host)
    table.add_row("Deploy dir", deploy_dir)
    console.print("\n[cyan]Current connection settings:[/cyan]")
    console.print(table)

    try:
        if not Confirm.ask("\nEdit settings?", default=False):
            return
        username   = Prompt.ask("Username",   default=username)
        host       = Prompt.ask("Host",       default=host)
        deploy_dir = Prompt.ask("Deploy dir", default=deploy_dir)
        save_config({"username": username, "host": host, "deploy_dir": deploy_dir})
        console.print("[green]Settings saved.[/green]")

        if ensure_ssh_key() is None:
            return
        console.print(f"\n[cyan]Checking SSH key on {username}@{host}...[/cyan]")
        if check_ssh_authorized(username, host):
            console.print("[green]SSH key already authorized.[/green]")
        else:
            console.print("[yellow]SSH key not authorized.[/yellow]")
            if Confirm.ask("Copy SSH key to Pi?", default=True):
                copy_ssh_key(username, host)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")


def cmd_deploy() -> bool:
    username, host, deploy_dir = get_connection()

    if not (WORKSPACE_ROOT / "pixi.toml").exists():
        console.print(f"[bold red]Safety check failed: no pixi.toml in {WORKSPACE_ROOT}.[/bold red]")
        console.print("[red]Refusing to deploy — is this script in the right place (one level below the project root)?[/red]")
        return False

    key = ensure_ssh_key()
    if key is None:
        return False

    rsync_cmd = [
        "rsync", "-avz", "--delete",
        "-e", f"ssh -i {shlex.quote(str(key))}",
        "--filter=:- .gitignore",
        "--exclude", ".git/",
        "--exclude", ".deploy_config",
        "--exclude", ".deploy_history",
        "./",
        f"{username}@{host}:{deploy_dir}/",
    ]

    console.print(f"[cyan]Deploying to {username}@{host}:{deploy_dir}[/cyan]")
    console.print(f"[dim]{' '.join(rsync_cmd)}[/dim]\n")

    try:
        subprocess.run(rsync_cmd, cwd=WORKSPACE_ROOT, check=True)
        console.print("\n[bold green]Deploy complete![/bold green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"\n[bold red]Deploy failed (exit code {e.returncode})[/bold red]")
        return False
    except FileNotFoundError:
        console.print("[bold red]rsync not found.[/bold red]")
        return False
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return False


def cmd_remote(args: str = ""):
    username, host, deploy_dir = get_connection()
    if ensure_ssh_key() is None:
        return
    base_cmd = ssh_cmd(username, host)

    if args.strip():
        console.print(f"[cyan]Running on {username}@{host}: {args}[/cyan]")
        remote = (
            f'export PATH="$HOME/.pixi/bin:$PATH"; '
            f'cd {shlex.quote(deploy_dir)} || exit 1; '
            f'{wrap_remote_longlived(args)}'
        )
        run_remote_streaming(base_cmd + ["-tt", remote])
    else:
        console.print(f"[cyan]SSH → {username}@{host}:{deploy_dir}[/cyan]")
        console.print("[dim]Type 'exit' to return.[/dim]\n")
        try:
            subprocess.run(base_cmd + ["-t", f"cd {shlex.quote(deploy_dir)} && exec $SHELL -l"], check=False)
        except KeyboardInterrupt:
            pass
        console.print()


def cmd_pixi(args: str = ""):
    if not args.strip():
        console.print("[red]Usage: pixi <command>  e.g. pixi run build[/red]")
        return
    if not cmd_deploy():
        return
    console.print()
    username, host, deploy_dir = get_connection()
    if ensure_ssh_key() is None:
        return
    base_cmd = ssh_cmd(username, host)
    console.print(f"[cyan]Running: pixi {args}[/cyan]")
    
    remote = (
        f'export PATH="$HOME/.pixi/bin:$PATH"; '
        f'cd {shlex.quote(deploy_dir)} || exit 1; '
        f'{wrap_remote_longlived(f"pixi {args}")}'
    )
    run_remote_streaming(base_cmd + ["-tt", remote])

def cmd_scp(args: str = ""):
    if not args.strip():
        console.print("[red]Usage: scp pull <path> | scp push <path>[/red]")
        return

    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        console.print("[red]Usage: scp pull <path> | scp push <path>[/red]")
        return

    direction, path_arg = parts[0], parts[1]

    username, host, deploy_dir = get_connection()
    key = ensure_ssh_key()
    if key is None:
        return
    base_ssh_cmd = ssh_cmd(username, host)

    if direction == "pull":
        scp_pull(path_arg, deploy_dir, username, host, key, base_ssh_cmd)
    elif direction == "push":
        scp_push(path_arg, deploy_dir, username, host, key, base_ssh_cmd)
    else:
        console.print(f"[red]Unknown direction: {direction}. Use 'pull' or 'push'.[/red]")

def cmd_service(args: str = ""):
    subcommand = args.strip().lower() if args.strip() else "status"

    if subcommand == "kill":
        username, host, deploy_dir = get_connection()
        if ensure_ssh_key() is None:
            return
        base_cmd = ssh_cmd(username, host)

        console.print(f"[cyan]Killing all active processes under {deploy_dir} on Pi...[/cyan]")
        
        kill_cmd_str = (
            f"pkill -f '{deploy_dir}' || true; "
            f"pkill -f 'foxglove_bridge' || true"
        )
        
        try:
            subprocess.run(base_cmd + ["-t", kill_cmd_str], check=False)
            console.print("[bold green]Kill commands executed successfully![/bold green]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            
    elif subcommand == "status":
        console.print("[yellow]Status subcommand is not fully implemented yet.[/yellow]")
        console.print(f"[dim]Usage: service kill[/dim]")
    else:
        console.print(f"[red]Unknown service subcommand: {subcommand}[/red]")
        console.print("[dim]Usage: service kill[/dim]")


# ── Autocomplete ──────────────────────────────────────────────────────────────

class DeployCompleter(Completer):
    def get_completions(self, document, complete_event):
        text  = document.text_before_cursor
        words = text.split()

        if not words or (len(words) == 1 and not text.endswith(" ")):
            word = words[0] if words else ""
            for cmd, desc in COMMANDS.items():
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word), display_meta=desc)

        elif words[0] == "pixi":
            tasks = get_pixi_tasks() or PIXI_TASKS_FALLBACK
            if len(words) == 1 and text.endswith(" "):
                for sub in ["run", "install", "shell"]:
                    yield Completion(sub)
            elif len(words) == 2 and words[1] == "run" and text.endswith(" "):
                for task in tasks:
                    yield Completion(task, display_meta="task")
            elif len(words) == 3 and words[1] == "run" and not text.endswith(" "):
                partial = words[2]
                for task in tasks:
                    if task.startswith(partial):
                        yield Completion(task, start_position=-len(partial), display_meta="task")

        elif words[0] == "scp":
            if len(words) == 1 and text.endswith(" "):
                for sub, desc in [("pull", "Pull from Pi"), ("push", "Push to Pi")]:
                    yield Completion(sub, display_meta=desc)
            elif len(words) == 2 and not text.endswith(" "):
                partial = words[1]
                for sub in ["pull", "push"]:
                    if sub.startswith(partial):
                        yield Completion(sub, start_position=-len(partial))


# ── Main loop ─────────────────────────────────────────────────────────────────

def process(user_input: str) -> bool:
    user_input = user_input.strip()
    if not user_input:
        return True

    parts = user_input.split(maxsplit=1)
    cmd  = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "exit":
        console.print("[dim]Goodbye![/dim]")
        return False
    elif cmd == "help":   cmd_help()
    elif cmd == "ping":   cmd_ping()
    elif cmd == "config": cmd_config(args)
    elif cmd == "deploy": cmd_deploy()
    elif cmd == "remote": cmd_remote(args)
    elif cmd == "pixi":   cmd_pixi(args)
    elif cmd == "scp":    cmd_scp(args)
    elif cmd == "service": cmd_service(args)
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]  Type 'help' for options.")

    return True


def main():
    console.print(
        Panel.fit(
            "[bold cyan]Robot Dog[/bold cyan]  Deploy Terminal\n"
            "[dim]Tab for autocomplete · 'help' for commands[/dim]",
        )
    )

    username, host, deploy_dir = get_connection()
    console.print(f"[dim]Target: {username}@{host}:{deploy_dir}[/dim]")

    if not load_config():
        console.print("[yellow]No config found. Run 'config' to set up connection.[/yellow]")

    console.print()

    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        completer=DeployCompleter(),
        style=Style.from_dict({"prompt": "ansicyan bold"}),
        complete_while_typing=True,
    )

    while True:
        try:
            user_input = session.prompt("robotdog> ")
            if not process(user_input):
                break
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Unexpected error: {e}[/bold red]")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())