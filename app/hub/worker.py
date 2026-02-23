import asyncio
import os
import subprocess
import datetime
import uuid
import logging
from typing import Dict, Any

from arq import create_pool
from arq.connections import RedisSettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hub.worker")

async def run_cli_command(ctx: Dict[Any, Any], cmd: list[str], env: Dict[str, str], cwd: str, log_path: str) -> Dict[str, Any]:
    """Execute a CLI command as a background subprocess, capturing logs."""
    logger.info(f"Starting background task cmd: {' '.join(cmd)}")
    try:
        with open(log_path, "w") as log_f:
            log_f.write(f"[hub] cmd: {' '.join(cmd)}\n\n")
            log_f.flush()
            
            # Use asyncio.create_subprocess_exec instead of subprocess.Popen 
            # so the async worker isn't blocked by the OS process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_f,
                stderr=log_f,
                start_new_session=True
            )
            
            # Wait for the process to finish
            await process.wait()
            
            with open(log_path, "a") as log_end:
                log_end.write(f"\n[hub] Process exited with code {process.returncode}\n")
                
            return {
                "status": "completed" if process.returncode == 0 else "failed",
                "exit_code": process.returncode,
                "log_path": log_path
            }
            
    except Exception as e:
        logger.exception(f"Failed to execute command: {e}")
        return {
            "status": "error",
            "error": str(e),
            "log_path": log_path
        }


class WorkerSettings:
    functions = [run_cli_command]
    redis_settings = RedisSettings()
    
# ARQ requires a module-level variable for the worker settings
# Usage: arq app.hub.worker.WorkerSettings
