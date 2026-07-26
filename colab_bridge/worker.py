from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import requests


OWNER = "sunil123456897"
REPO = "SEF-GRAM"
QUEUE_REF = "infra/colab-bridge"
API_ROOT = "https://api.github.com"
DEFAULT_POLL_SECONDS = 15
MAX_TEXT_ARTIFACT_BYTES = 1_500_000
MAX_TIMEOUT_SECONDS = 6 * 60 * 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_repo_path(value: str) -> str:
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe repository path: {value!r}")
    return p.as_posix()


def run_capture(cmd: list[str], cwd: Path | None = None, timeout: int = 20) -> str:
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return cp.stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


@dataclass
class GitHubContents:
    token: str
    owner: str = OWNER
    repo: str = REPO
    ref: str = QUEUE_REF

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "sef-gram-colab-worker/1",
            }
        )

    def _url(self, path: str) -> str:
        path = safe_repo_path(path)
        return f"{API_ROOT}/repos/{self.owner}/{self.repo}/contents/{path}"

    def list_dir(self, path: str) -> list[dict[str, Any]]:
        r = self.session.get(self._url(path), params={"ref": self.ref}, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else [data]

    def get_file(self, path: str) -> tuple[bytes, str] | None:
        r = self.session.get(self._url(path), params={"ref": self.ref}, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        raw = base64.b64decode(data["content"])
        return raw, data["sha"]

    def exists(self, path: str) -> bool:
        return self.get_file(path) is not None

    def put_bytes(self, path: str, content: bytes, message: str) -> str:
        current = self.get_file(path)
        body: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content).decode("ascii"),
            "branch": self.ref,
        }
        if current is not None:
            body["sha"] = current[1]
        r = self.session.put(self._url(path), json=body, timeout=60)
        r.raise_for_status()
        return r.json()["commit"]["sha"]

    def put_json(self, path: str, value: Any, message: str) -> str:
        payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        return self.put_bytes(path, payload, message)

    def delete(self, path: str, message: str) -> None:
        current = self.get_file(path)
        if current is None:
            return
        r = self.session.delete(
            self._url(path),
            json={"message": message, "sha": current[1], "branch": self.ref},
            timeout=60,
        )
        r.raise_for_status()


class ColabWorker:
    def __init__(
        self,
        token: str,
        owner: str = OWNER,
        repo: str = REPO,
        queue_ref: str = QUEUE_REF,
        poll_seconds: int = DEFAULT_POLL_SECONDS,
        work_root: str = "/content/colab_bridge_runs",
    ) -> None:
        self.gh = GitHubContents(token=token, owner=owner, repo=repo, ref=queue_ref)
        self.owner = owner
        self.repo = repo
        self.queue_ref = queue_ref
        self.poll_seconds = max(5, int(poll_seconds))
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)

    def runtime_info(self) -> dict[str, Any]:
        torch_info: dict[str, Any] = {}
        try:
            import torch

            torch_info = {
                "version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception as exc:
            torch_info = {"error": f"{type(exc).__name__}: {exc}"}

        return {
            "captured_at": utc_now(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "nvidia_smi": run_capture(["nvidia-smi", "-L"]),
            "torch": torch_info,
        }

    def validate_job(self, job: dict[str, Any], filename: str) -> dict[str, Any]:
        version = int(job.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported job version: {version}")

        job_id = str(job.get("id") or Path(filename).stem).strip()
        if not job_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for ch in job_id):
            raise ValueError("Job id may contain only letters, digits, dash, underscore and dot")

        entrypoint = safe_repo_path(str(job["entrypoint"]))
        if not entrypoint.endswith(".py"):
            raise ValueError("entrypoint must be a .py file inside the repository")

        ref = str(job.get("ref", "main")).strip() or "main"
        args = job.get("args", [])
        if not isinstance(args, list) or not all(isinstance(x, (str, int, float, bool)) for x in args):
            raise ValueError("args must be a list of scalar values")

        env = job.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, (str, int, float, bool)) for k, v in env.items()):
            raise ValueError("env must be an object containing scalar values")

        pip_packages = job.get("pip_packages", [])
        if not isinstance(pip_packages, list) or not all(isinstance(x, str) and x.strip() for x in pip_packages):
            raise ValueError("pip_packages must be a list of package specifiers")

        requirements_file = job.get("requirements_file")
        if requirements_file is not None:
            requirements_file = safe_repo_path(str(requirements_file))

        result_files = job.get("result_files", [])
        if not isinstance(result_files, list) or not all(isinstance(x, str) and x.strip() for x in result_files):
            raise ValueError("result_files must be a list of glob patterns")

        timeout_seconds = int(job.get("timeout_seconds", 3600))
        if not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout_seconds must be between 1 and {MAX_TIMEOUT_SECONDS}")

        return {
            **job,
            "version": 1,
            "id": job_id,
            "ref": ref,
            "entrypoint": entrypoint,
            "args": [str(x) for x in args],
            "env": {str(k): str(v) for k, v in env.items()},
            "pip_packages": pip_packages,
            "requirements_file": requirements_file,
            "result_files": result_files,
            "timeout_seconds": timeout_seconds,
        }

    def pending_jobs(self) -> list[dict[str, Any]]:
        items = self.gh.list_dir("jobs/pending")
        return sorted(
            [x for x in items if x.get("type") == "file" and str(x.get("name", "")).endswith(".json")],
            key=lambda x: x["name"],
        )

    def claim(self, item: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
        pending_path = item["path"]
        raw_and_sha = self.gh.get_file(pending_path)
        if raw_and_sha is None:
            return None
        raw, _ = raw_and_sha
        job = self.validate_job(json.loads(raw.decode("utf-8")), item["name"])
        job_id = job["id"]
        running_path = f"jobs/running/{job_id}.json"
        terminal_paths = [f"jobs/completed/{job_id}.json", f"jobs/failed/{job_id}.json"]

        if self.gh.exists(running_path) or any(self.gh.exists(p) for p in terminal_paths):
            self.gh.delete(pending_path, f"bridge: remove duplicate pending job {job_id}")
            return None

        running = {**job, "bridge_status": "running", "claimed_at": utc_now()}
        self.gh.put_json(running_path, running, f"bridge: claim job {job_id}")
        self.gh.delete(pending_path, f"bridge: dequeue job {job_id}")
        return running, running_path

    def prepare_checkout(self, job: dict[str, Any], run_dir: Path) -> Path:
        repo_dir = run_dir / "repo"
        clone_url = f"https://github.com/{self.owner}/{self.repo}.git"
        subprocess.run(["git", "clone", "--quiet", "--no-tags", clone_url, str(repo_dir)], check=True, timeout=300)
        subprocess.run(["git", "fetch", "--quiet", "origin", job["ref"]], cwd=repo_dir, check=True, timeout=300)
        subprocess.run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=repo_dir, check=True, timeout=120)
        return repo_dir

    def install_dependencies(self, job: dict[str, Any], repo_dir: Path, stdout_path: Path, stderr_path: Path) -> None:
        commands: list[list[str]] = []
        if job.get("requirements_file"):
            req = repo_dir / job["requirements_file"]
            if not req.is_file():
                raise FileNotFoundError(f"requirements_file not found: {job['requirements_file']}")
            commands.append([sys.executable, "-m", "pip", "install", "-r", str(req)])
        if job.get("pip_packages"):
            commands.append([sys.executable, "-m", "pip", "install", *job["pip_packages"]])

        for cmd in commands:
            with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open("a", encoding="utf-8") as err:
                out.write("\n$ " + " ".join(cmd) + "\n")
                out.flush()
                subprocess.run(cmd, cwd=repo_dir, stdout=out, stderr=err, check=True, timeout=1800)

    def execute(self, job: dict[str, Any], repo_dir: Path, stdout_path: Path, stderr_path: Path) -> tuple[int, bool, float]:
        entrypoint = repo_dir / job["entrypoint"]
        if not entrypoint.is_file():
            raise FileNotFoundError(f"entrypoint not found at ref {job['ref']}: {job['entrypoint']}")

        cmd = [sys.executable, str(entrypoint), *job["args"]]
        env = os.environ.copy()
        env.update(job["env"])
        env.update({"COLAB_BRIDGE_JOB_ID": job["id"], "PYTHONUNBUFFERED": "1"})

        start = time.monotonic()
        timed_out = False
        with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open("a", encoding="utf-8") as err:
            out.write("$ " + " ".join(cmd) + "\n")
            out.flush()
            proc = subprocess.Popen(cmd, cwd=repo_dir, env=env, stdout=out, stderr=err, text=True)
            try:
                code = proc.wait(timeout=job["timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                code = proc.wait(timeout=30)
        return int(code), timed_out, time.monotonic() - start

    def collect_artifacts(self, job: dict[str, Any], repo_dir: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        uploaded: set[str] = set()
        for pattern in job.get("result_files", []):
            for found in glob.glob(str(repo_dir / pattern), recursive=True):
                path = Path(found)
                if not path.is_file():
                    continue
                rel = path.relative_to(repo_dir).as_posix()
                if rel in uploaded:
                    continue
                uploaded.add(rel)
                size = path.stat().st_size
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                record = {"path": rel, "size_bytes": size, "sha256": digest, "uploaded": False}
                if size <= MAX_TEXT_ARTIFACT_BYTES:
                    try:
                        data = path.read_bytes()
                        data.decode("utf-8")
                        safe_name = rel.replace("/", "__")
                        self.gh.put_bytes(
                            f"results/{job['id']}/artifacts/{safe_name}",
                            data,
                            f"bridge: upload artifact for {job['id']}: {rel}",
                        )
                        record["uploaded"] = True
                    except UnicodeDecodeError:
                        record["reason"] = "binary artifact not uploaded"
                    except Exception as exc:
                        record["reason"] = f"upload failed: {type(exc).__name__}: {exc}"
                else:
                    record["reason"] = f"larger than {MAX_TEXT_ARTIFACT_BYTES} byte bridge limit"
                records.append(record)
        return records

    def finalize(self, job: dict[str, Any], running_path: str, status: str, summary: dict[str, Any], stdout_path: Path, stderr_path: Path) -> None:
        job_id = job["id"]
        result_root = f"results/{job_id}"
        for local, name in [(stdout_path, "stdout.txt"), (stderr_path, "stderr.txt")]:
            if local.exists():
                data = local.read_bytes()
                if len(data) > MAX_TEXT_ARTIFACT_BYTES:
                    data = data[-MAX_TEXT_ARTIFACT_BYTES:]
                self.gh.put_bytes(f"{result_root}/{name}", data, f"bridge: save {name} for {job_id}")
        self.gh.put_json(f"{result_root}/summary.json", summary, f"bridge: save summary for {job_id}")
        terminal = {**job, "bridge_status": status, "finished_at": utc_now(), "result": f"{result_root}/summary.json"}
        self.gh.put_json(f"jobs/{status}/{job_id}.json", terminal, f"bridge: mark job {job_id} {status}")
        self.gh.delete(running_path, f"bridge: clear running job {job_id}")

    def run_job(self, job: dict[str, Any], running_path: str) -> None:
        job_id = job["id"]
        run_dir = self.work_root / job_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        summary: dict[str, Any] = {
            "job_id": job_id,
            "started_at": utc_now(),
            "runtime": self.runtime_info(),
            "job": job,
        }
        status = "failed"
        try:
            repo_dir = self.prepare_checkout(job, run_dir)
            summary["resolved_commit"] = run_capture(["git", "rev-parse", "HEAD"], cwd=repo_dir)
            self.install_dependencies(job, repo_dir, stdout_path, stderr_path)
            code, timed_out, elapsed = self.execute(job, repo_dir, stdout_path, stderr_path)
            summary.update({"exit_code": code, "timed_out": timed_out, "elapsed_seconds": elapsed})
            summary["artifacts"] = self.collect_artifacts(job, repo_dir)
            status = "completed" if code == 0 and not timed_out else "failed"
        except Exception as exc:
            summary["exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            stderr_path.write_text(summary["exception"]["traceback"], encoding="utf-8")
        finally:
            summary["finished_at"] = utc_now()
            summary["status"] = status
            self.finalize(job, running_path, status, summary, stdout_path, stderr_path)

    def run_once(self) -> int:
        jobs = self.pending_jobs()
        if not jobs:
            return 0
        claimed = self.claim(jobs[0])
        if claimed is None:
            return 0
        job, running_path = claimed
        print(f"[{utc_now()}] running {job['id']} ({job['ref']}:{job['entrypoint']})", flush=True)
        self.run_job(job, running_path)
        print(f"[{utc_now()}] finished {job['id']}", flush=True)
        return 1

    def run_forever(self) -> None:
        print(f"Colab bridge worker: {self.owner}/{self.repo}@{self.queue_ref}", flush=True)
        print(json.dumps(self.runtime_info(), indent=2), flush=True)
        while True:
            try:
                if self.run_once() == 0:
                    time.sleep(self.poll_seconds)
            except KeyboardInterrupt:
                raise
            except Exception:
                traceback.print_exc()
                time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="SEF-GRAM GitHub ↔ Colab GPU worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--queue-ref", default=QUEUE_REF)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit("GITHUB_TOKEN is not set. In Colab, load it from google.colab.userdata first.")

    worker = ColabWorker(
        token=token,
        owner=args.owner,
        repo=args.repo,
        queue_ref=args.queue_ref,
        poll_seconds=args.poll_seconds,
    )
    if args.once:
        worker.run_once()
    else:
        worker.run_forever()


if __name__ == "__main__":
    main()
