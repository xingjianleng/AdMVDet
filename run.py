import sys
import docker
import os
import queue
import subprocess
import time


def run_carla(gpu=2):
    client = docker.from_env()

    env = {}  # Environment variables to set in the container

    command = "/bin/bash ./CarlaUE4.sh -RenderOffScreen"
    volumes = {"/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"}}

    if gpu == 2:
        pass
    elif gpu == 3:
        command += " -world-port=3000 -carla-rpc-port=4000"
    else:
        raise Exception("Invalid GPU ID")

    container = None
    try:
        container = client.containers.run("carlasim/carla:0.9.14", 
                                          command=command, 
                                          detach=True, 
                                          privileged=True, 
                                          network_mode="host", 
                                          environment=env, 
                                          volumes=volumes, 
                                          device_requests=[docker.types.DeviceRequest(device_ids=[str(gpu)], capabilities=[['gpu']])])
        # wait for carla to start
        while container.status == "created":
            container.reload()
            time.sleep(2)
        # print(container.logs().decode('utf-8'))
    except Exception as e:
        print(f"Error executing command: {e}")

    return container


def main():
    # Change scripts here
    scripts = sys.argv[1]

    processes = []
    containers = {0: None, 1: None}

    gpu_queue = queue.Queue()
    for i in range(2):
        gpu_queue.put(i)

    scripts_lst = scripts.split(" ")
    
    # run 5 times
    script_arg_queue = queue.Queue()
    for i in range(5):
        script_arg_queue.put(i)
    
    while not script_arg_queue.empty():
        # Only start a new process if there is a GPU available
        if gpu_queue.empty():
            # Check if any process has finished, and if so, free up the GPU and stop the associated container
            for process in processes:
                if process.poll() is not None:  # A None value indicates that the process is still running
                    processes.remove(process)

                    container = containers[process.gpu_id]
                    container.stop()
                    containers[process.gpu_id] = None

                    gpu_queue.put(process.gpu_id)
                    time.sleep(30)  # wait for clean up
                    break

            else:
                time.sleep(2)  # Polling interval
                continue


        gpu_id = gpu_queue.get()
        script_arg_queue.get()

        # Start the Docker container associated with this GPU until running
        container = run_carla(gpu=gpu_id+2)
        time.sleep(30)  # wait for stablization

        containers[gpu_id] = container

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        process = subprocess.Popen(
            ["python"] + (scripts_lst if gpu_id == 0 else scripts_lst + ['--port', '3000', '--tm_port', '5000']), env=env)
        process.gpu_id = gpu_id
        processes.append(process)

    while processes:  # while there are still running processes
        for process in processes:  # Iterate over a slice copy of processes so we can modify the list inside the loop
            if process.poll() is not None:  # If the process has finished
                processes.remove(process)

                container = containers[process.gpu_id]
                container.stop()
                containers[process.gpu_id] = None

                gpu_queue.put(process.gpu_id)
                time.sleep(30)  # wait for clean up
                break

        time.sleep(2)  # Polling interval


if __name__ == "__main__":
    main()
