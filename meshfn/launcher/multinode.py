import os
import shutil
from shlex import split
import sys

from meshfn.logging import logger

PDSH_MAX_FAN_OUT = 1024


class PDSHLauncher:
    def __init__(self, launcher_args, master_addr, master_port, script, script_args):
        self.launcher_args = launcher_args
        self.master_addr = master_addr
        self.master_port = master_port
        self.script = script
        self.script_args = script_args
        self.envs = {}

    def backend_available(self):
        return shutil.which("pdsh")

    def add_env(self, key, val):
        self.envs[key.strip()] = val.strip()

    def get_cmd(self, id, env, resources):
        env["PDSH_RCMD_TYPE"] = "ssh"

        workers = ",".join(resources.keys())

        logger.info(f"launch on workers: {workers}")

        pdsh_cmd_args = [
            "pdsh",
            "-S",
            "-f",
            str(PDSH_MAX_FAN_OUT),
            "-w",
            workers,
        ] + split(self.launcher_args)

        exports = ""
        for key, val in self.envs.items():
            exports += f"export {key}={val};"

        launch = [
            exports,
            f"cd {os.path.abspath('.')};",
            sys.executable,
            "-u" "-m",
            "torch.distributed.run",
            "--node_rank=%n",
            f"--rdzv-id={id}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint={self.master_addr}:{self.master_port}",
            f"--nnodes={len(resources)}",
            "--nproc_per_node=gpu",
        ]

        cmd_to_search = [i + "\\" for i in launch[2:6]]
        kill_cmd = pdsh_cmd_args + ["pkill -f ", " ".join(cmd_to_search)[:-2]]

        return pdsh_cmd_args + launch + [self.script] + self.script_args, kill_cmd
