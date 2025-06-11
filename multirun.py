"""
Simple script that handles taking a bunch of hydra configs and 
running them across a set of GPUs - making sure not to double up
"""
import os
import concurrent.futures
import threading
from queue import Queue

models = [
    # "model=pns memory=exact-replay",
    # "model=pns memory=task-aware",
    # "model=pns memory=naive",
    # "dataset=continual_small version=0 train=False",
    # "dataset=continual_small version=1 train=False",
    # "dataset=continual_small version=2 train=False",
    # "dataset=continual_small version=3 train=False",
    # "dataset=continual_small version=4 train=False",
    
    "limited_samples=3 dataset=continual_small version=0 train=False",
    "limited_samples=4 dataset=continual_small version=0 train=False",
    "limited_samples=6 dataset=continual_small version=0 train=False",
    "limited_samples=8 dataset=continual_small version=0 train=False",
    "limited_samples=10 dataset=continual_small version=0 train=False",
    
    # "model=feedforward-mask memory=exact-replay",
    # "model=feedforward-mask memory=task-aware",
    # "model=feedforward-mask memory=naive",
    
    # "model=maml memory=exact-replay",
    # "model=maml memory=task-aware",
    # "model=maml memory=naive",
]

seeds = [
    1111,
    2222,
    3333,
    4444,
    5555,
    # 6666,
    # 7777
    # 8888,
    # 9999
]

commands = []
for model in models:
    for seed in seeds:    
        commands.append(model + f" seed={seed}")
print(commands)


class GPUManager:
    def __init__(self, devices):
        self.devices = devices
        self.available_devices = Queue()
        self.lock = threading.Lock()
        
        # Initialize available devices
        for device in devices:
            self.available_devices.put(device)

    def get_device(self):
        return self.available_devices.get()

    def release_device(self, device):
        self.available_devices.put(device)


def run_command(command, device, gpu_manager):
    try:
        # Run the command with the device flag
        # full_command = f"python3 main_continual.py {command} devices=[{device}]"
        full_command = f"python3 main_pretrained.py {command} devices=[{device}]"
        print(f"Running: {full_command}")
        exit_code = os.system(full_command)
        return exit_code, device
    finally:
        # Always release the device back to the pool
        gpu_manager.release_device(device)


def run_commands_in_parallel(commands, max_workers, devices):
    gpu_manager = GPUManager(devices)
    command_queue = Queue()
    for cmd in commands:
        command_queue.put(cmd)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        
        while not command_queue.empty() or futures:
            # Submit new tasks if devices are available and commands exist
            while not command_queue.empty() and len(futures) < max_workers:
                try:
                    device = gpu_manager.get_device()
                    command = command_queue.get()
                    future = executor.submit(run_command, command, device, gpu_manager)
                    futures.add(future)
                except Queue.Empty:
                    break

            # Wait for any task to complete
            done, futures = concurrent.futures.wait(
                futures, 
                timeout=None,
                return_when=concurrent.futures.FIRST_COMPLETED
            )

            # Process completed tasks
            for future in done:
                try:
                    exit_code, device = future.result()
                    print(f"Task finished with exit code {exit_code} on device {device}")
                except Exception as e:
                    print(f"Task generated an exception: {e}")

# Assume you have 2 GPUs: devices 0 and 1
devices = [3, 4, 5, 6, 7, 8, 9]

# Set the number of parallel workers based on the number of GPUs
max_workers = len(devices)

run_commands_in_parallel(commands, max_workers, devices)
