"""
environment_logger.py
=====================
Automatically logs hardware, OS, GPU, runtime, and software dependencies
for reproducible scientific reporting in compliance with IEEE reproducibility standards.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
import psutil

def get_cpu_info() -> dict:
    info = {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_threads": psutil.cpu_count(logical=True),
        "name": platform.processor()
    }
    # On Windows, query CIM for precise marketing brand string
    if platform.system() == "Windows":
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0 and res.stdout.strip():
                info["name"] = res.stdout.strip().splitlines()[0]
        except Exception:
            pass
    return info

def get_gpu_info() -> list:
    gpus = []
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0:
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "name": parts[0],
                        "driver_version": parts[1],
                        "total_vram_mb": float(parts[2]),
                        "free_vram_mb": float(parts[3])
                    })
    except Exception:
        pass
    return gpus

def get_ollama_info() -> dict:
    info = {
        "installed": False,
        "version": None,
        "daemon_reachable": False,
        "available_models": []
    }
    try:
        ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=3)
        if ver.returncode == 0:
            info["installed"] = True
            info["version"] = ver.stdout.strip() or ver.stderr.strip()
    except Exception:
        pass

    # Check daemon reachability via HTTP or ollama python client
    try:
        import urllib.request
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        if req.status == 200:
            info["daemon_reachable"] = True
            payload = json.loads(req.read().decode())
            info["available_models"] = [m.get("name") for m in payload.get("models", [])]
    except Exception:
        info["daemon_reachable"] = False

    return info

def collect_telemetry() -> dict:
    vm = psutil.virtual_memory()
    telemetry = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform()
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable
        },
        "cpu": get_cpu_info(),
        "host_ram": {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2)
        },
        "gpu": get_gpu_info(),
        "ollama": get_ollama_info()
    }
    return telemetry

def main():
    data = collect_telemetry()
    out_file = "environment_telemetry.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Logged system environment to {out_file}:")
    print(f"  CPU: {data['cpu']['name']} ({data['cpu']['physical_cores']}C / {data['cpu']['logical_threads']}T)")
    print(f"  RAM: {data['host_ram']['total_gb']} GB total")
    if data['gpu']:
        for g in data['gpu']:
            print(f"  GPU: {g['name']} ({g['total_vram_mb']} MB VRAM, Driver {g['driver_version']})")
    else:
        print("  GPU: None detected")
    print(f"  Ollama Daemon: {'Reachable' if data['ollama']['daemon_reachable'] else 'Unreachable / Offline'}")

if __name__ == "__main__":
    main()
