import argparse
import collections
import re
import os
import subprocess
import uuid
import signal
import time
import sys

import torch

from meshfn.launcher.multinode import PDSHLauncher


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hostfile", type=str, default=None)
    parser.add_argument("--master_addr", type=str, default=None)
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--launcher", type=str, default="pdsh")
    parser.add_argument("--launcher_args", type=str, default="")
    parser.add_argument("script", type=str)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)

    return parser.parse_args()


def _parse_hostfile(hostfile_lines):
    # Regex matches one or more non-whitespace characters (\S+) at the start of
    # the line, followed by one or more whitespace characters (\s+), followed
    # by the string "slots=", followed by one or more digits (\d+).
    pattern = r"^(\S+)\s+slots=(\d+)"

    resource_pool = collections.OrderedDict()

    for line in hostfile_lines:
        line = line.strip()
        match = re.search(pattern, line)
        if line.startswith("#") or line == "":
            # hostfile comment or empty line, ignore
            continue
        elif match:
            host = match.group(1)
            num_slots = int(match.group(2))
            if host in resource_pool:
                raise ValueError(
                    f"Hostfile contains multiple entries for {host}, unable to proceed with launching"
                )
            resource_pool[host] = num_slots
        else:
            raise ValueError(
                "Hostfile contains a bad entry: {line}, unable to proceed with launching"
            )

    if len(resource_pool) == 0:
        raise ValueError(
            "Hostfile is empty or not formatted correctly, unable to proceed with launching."
        )

    return resource_pool


def fetch_hostfile(hostfile_path):
    # e.g., worker-0 slots=16
    with open(hostfile_path, "r") as fd:
        hostfile_text = fd.readlines()

    return _parse_hostfile(hostfile_text)


def main():
    args = parse_args()

    if args.hostfile is not None:
        resources = fetch_hostfile(args.hostfile)

        if args.master_addr is None:
            first_host = next(iter(resources.keys()))
            hostname_cmd = [f"ssh {first_host} hostname -I"]

            try:
                result = subprocess.check_output(hostname_cmd, shell=True)

            except subprocess.CalledProcessError:
                raise SystemError("unabled to detect master address via `hostname -I`")

            args.master_addr = result.decode("utf-8").split()[0]

        multinode = True

    else:
        resources = {"localhost": torch.cuda.device_count()}
        args.master_addr = "127.0.0.1"
        multinode = False

    env = os.environ.copy()

    env["OMP_NUM_THREADS"] = "1"

    run_id = str(uuid.uuid4())[:8]

    if multinode:
        if args.launcher == "pdsh":
            launcher = PDSHLauncher(
                args.launcher_args,
                args.master_addr,
                args.master_port,
                args.script,
                args.script_args,
            )

        curr_path = os.path.abspath(".")

        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{curr_path}:{env['PYTHONPATH']}"

        else:
            env["PYTHONPATH"] = curr_path

        for var in env.keys():
            if any([var.startswith(name) for name in ("NCCL", "PYTHON", "OMP")]):
                launcher.add_env(var, env[var])

        cmd, kill_cmd = launcher.get_cmd(run_id, env, resources)

    else:
        launch = [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--node_rank=0",
            f"--rdzv-id={run_id}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint={args.master_addr}:{args.master_port}",
            f"--nnodes=1",
            "--nproc_per_node=gpu",
        ]
        cmd = launch + [args.script] + args.script_args

    result = subprocess.Popen(cmd, env=env)

    def sigkill_handler(signum, frame):
        result.send_signal(signal.SIGINT)
        time.sleep(0.1)
        result.send_signal(signal.SIGTERM)
        result_kill = subprocess.Popen(kill_cmd, env=env)
        result_kill.wait()
        time.sleep(1)
        sys.exit(1)

    if args.launcher == "pdsh":
        signal.signal(signal.SIGINT, sigkill_handler)

    result.wait()

    if result.returncode > 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
