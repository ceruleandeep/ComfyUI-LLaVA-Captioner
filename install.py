import locale
import os
import subprocess
import sys
import threading
import traceback

import torch

# For the painful details of installing llama-cpp-python see:
# https://github.com/abetlen/llama-cpp-python

# Performance for LLaVA will be horrible unless the layers are offloaded to GPU
# which for most people means cuBLAS or MPS.

# We forcibly reinstall llama-cpp-python to ensure that the build flags
# are actually applied. Otherwise the non-accelerated version will just be
# reinstalled from the pip cache.


# bits of this looted from https://github.com/ltdrdata/ComfyUI-Impact-Pack


if sys.argv[0] == "install.py":
    sys.path.append(".")  # for portable version

impact_path = os.path.join(os.path.dirname(__file__), "modules")
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.append(impact_path)
sys.path.append(comfy_path)

args = {
    "default": "",
    "cublas": "-DLLAMA_CUBLAS=on",
    "openblas": "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS",
    "mps": "-DLLAMA_METAL=on",
    "clblast": "-DLLAMA_CLBLAST=on",
    "hipblas": "-DLLAMA_HIPBLAS=on",
}


def cmake_args(force=None):
    if force:
        return args[force]

    try:
        if torch.cuda.is_available():
            return args["cublas"]
    except:
        pass

    if sys.platform == "darwin":
        return args["metal"]

    return args["default"]


def handle_stream(stream, is_stdout):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors="replace")

    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else:
            print(msg, end="", file=sys.stderr)


def process_wrap(cmd_str, cwd=None, handler=None, env_vars=None):
    if env_vars:
        env = dict(os.environ, **env_vars)
    else:
        env = None

    print(
        f"[ComfyUI LLaVA Captioner] running {' '.join(cmd_str)} with {env_vars}"
    )
    process = subprocess.Popen(
        cmd_str,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    if handler is None:
        handler = handle_stream

    stdout_thread = threading.Thread(target=handler, args=(process.stdout, True))
    stderr_thread = threading.Thread(target=handler, args=(process.stderr, False))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()


try:
    import platform
    import folder_paths

    print("[ComfyUI LLaVA Captioner] Installing llama-cpp-python")

    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        pip_install = [sys.executable, "-s", "-m", "pip", "install"]
    else:
        pip_install = [sys.executable, "-m", "pip", "install"]

    env_vars = {"CMAKE_ARGS": cmake_args()}

    # would rather avoid this in case it updates pytorch etc
    # to a version that comfyui/other custom nodes hate
    # force_reinstall = ["--upgrade", "--force-reinstall", "--no-cache-dir"]

    process_wrap("pip uninstall -y llama-cpp-python".split(" "), env_vars=env_vars)
    process_wrap(pip_install + ["llama-cpp-python", "--no-cache-dir"], env_vars=env_vars)

except Exception as e:
    print(
        "[ComfyUI LLaVA Captioner] Dependency installation has failed. Please install manually."
    )
    traceback.print_exc()
