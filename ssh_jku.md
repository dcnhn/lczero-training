WORKING ON THE SERVERS (student edition)
======================

These are just some guidelines for working with the servers we have at the institute.
In case of doubt, or if you run into any problems, 
please contact your supervisor.


Insitute Hardware
-----------------

All of our servers run Linux and are reachable via SSH. 
However, the default SSH port is 5792 (instead of the default: 22). 
See the section about 'ssh config' for more details.

### Server List

Except for *tiger*, all of our servers have GPUs attached to them. 
The current list of computing hardware (sorted by IP) is:

### Student Servers

The current list of student hardware (sorted by name) is:

| Server    | IP-address    | RAM (GB) | CPU cores | CPU type              | GPUs                                  |
| --------- | ------------- | --------:| --------- | ---------------------:| ------------------------------------- |
| student01 | 140.78.121.41 |       64 | 1x 6      |    i7-6800K @ 3.40GHz | 4x GTX 1080Ti                         |
| student02 | 140.78.121.42 |      128 | 1x 24     |    TR 3960X @ 3.80GHz | 3x RTX 2080Ti <br/> 1x GTX 1080Ti     |
| student03 | 140.78.121.43 |      128 | 1x 10     |  E5-2630 v4 @ 2.20GHz | 4x Titan X P                          |
| student04 | 140.78.121.44 |      128 | 1x 10     |  E5-2630 v4 @ 2.20GHz | 4x GTX 1080Ti                         |
| student05 | 140.78.121.45 |      128 | 1x 10     |  E5-2630 v4 @ 2.20GHz | 4x Titan X P                          |
| student06 | 140.78.121.46 |      128 | 1x 12     |    TR 1920X @ 3.50GHz | 4x GTX 2080Ti                         |
| student07 | 140.78.121.47 |      384 | 2x 12     |    6136 CPU @ 3.00GHz | 3x RTX 2080Ti <br/> 2x GV100          |
| student08 | 140.78.121.48 |      512 | 2x 12     |  E5-2687 v4 @ 3.00GHz | 4x GTX 1080Ti                         |
| student09 | 140.78.121.49 |      384 | 2x 18     |    6154 CPU @ 3.00GHz | 5x GTX 1080Ti                         |
| student10 | 140.78.90.40  |      384 | 2x 18     |    6154 CPU @ 3.00GHz | 2x P40 <br/> 1x V100 <br/> 2x Titan V |

**Notes:**
  * CPU core count is given as #CPUs x #cores/CPU and does not include hyperthreading
  * A VPN connection will be needed to use these servers off-campus!

### Graphics Cards

The specifics for each graphics card are listed (sorted by speed) in the table below:

| GPU        | CUDA cores | Tensor cores | Memory (GB) | double (GFLOPS) | single (GFLOPS) | half (GFLOPS) | CUDA Capability |
| ---------- | ----------:| ------------:| -----------:| ---------------:| ---------------:| -------------:| ---------------:|
| Titan X P  |       3584 |         N.A. |          12 |             343 |           10970 |           172 |             6.1 |
| GTX 1080Ti |       3584 |         N.A. |          11 |             354 |           11340 |           177 |             6.1 |
| P40        |       3840 |         N.A. |          24 |             367 |           11760 |           184 |             6.1 |
| RTX 2080Ti |       4352 |          544 |          11 |             420 |           13450 |         26900 |             7.5 |
| V100       |       5120 |          640 |          16 |            7066 |           14130 |         28260 |             7.0 |
| Titan V    |       5120 |          640 |          12 |            7450 |           14900 |         29800 |             7.0 |
| GV100      |       5120 |          640 |          32 |            8330 |           16660 |         33320 |             7.0 |

**Notes:**
  * *double*, *single* and *half* list the performance for the respective floating point precision.
  * speed differences are most noticeable for CNNs.
  * CUDA Capability refers to the 
    [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus#compute) of the GPU.

### GPU Scheduling

To use GPUs on our servers, please use the following Google Docs spreadsheet:

  https://docs.google.com/spreadsheets/d/1jr8rV7bWiDWaY-rEvGh2Kqp1BVqXGCzEqEwK6i33vlk

The idea of the spreadsheet is to give an overview of who is using what GPUs until when. If you need some computing power, the spreadsheet should tell you which GPUs should be free. The ideal process to acquire a GPU goes as follows:

  1. If one or more GPUs are free, enter your name and the date of request. Now you can compute the rest of the day (and probably also over night) on these GPUs.
  2. If no(t enough) GPUs are available, it might be that someone is already ready, but did not update the spreadsheet yet. Ask your colleagues if there are any free GPUs.
  3. If there are no free GPUs and you really need the compute, you will have to convince one of your colleagues to stop (one of) their runs for you. 
  4. If none of the above works, your supervisor might be able to help you out.

The spreadsheet should be updated daily (or at least once per week) or you will be the first target when someone needs a GPU. If you do not need your GPU anymore, update the spreadsheet to avoid annoyances. If you do not need the entire power of a single GPU, you can annotate this in the spreadsheet by inserting `can share` in the comment field. This notices your colleagues that they can start additional jobs on this GPU (if they fit in).


JKU wide servers
----------------

There are a few compute servers available to all users at JKU, which have a ton of CPU cores. These machines require a login for the Central Compute Servers (k-Number). Ask your supervisor for details.


Monitoring Resources
--------------------

Please be careful with the resources you use. You have different options to
monitor the server resources:

### GPUs

You can use `nvidia-smi` to check if a GPU is free or idle (and how much free
RAM it has): Note: if too many processes run on the same GPU, it will slow down
tremendously, so try to use a GPU that is not used by anyone else. 

### CPU and RAM

Use `top` to check how many CPUs / how much memory is currently in use, and to
keep an eye on your processes. NOTE: `top` can slow down the whole system
because it constantly makes system-calls to gather statistics, so don't let it
run all the time (if you do, use "s" to set the default update-delay to
something sensible, e.g. 5 or 10 seconds).
You can also use `top -n N` to make sure that `top` quits after `N` frames.


Setup
-----

### ssh config and passwordless login
typing  stuff like `ssh -p 5792 <username>@student01.ai-lab.jku.at` and your password each time you want to connect to a server is no fun. To set up a key-based
(i.e., password-less) login use following commands:

```bash
ssh-keygen
ssh-copy-id "<username>@student01.ai-lab.jku.at -p 5792"
```

Attached to this guide you'll find a ssh config. 
If you copy this into `~/.ssh/config` and enter your own username. 
Then you can login to the servers using:

```bash
  ssh student01
```

### .bashrc
The ~/.bashrc file is a script that is executed each time you log into the servers. 
It is used to set up some filepaths and activate some of our installed software. 
It's a good idea to add the following instructions towards the end of the file:


    # options for MKL (just trust me, these make sense)
    export OMP_NUM_THREADS=1
    export MKL_THREADING_LAYER=SEQUENTIAL

    # setup your conda environment
    source /system/apps/studentenv/miniconda3/bashrc


### Directories

You can use the following directories:

- `/system/user/studentwork/<username>` to store your data and code (this directory is limited to 200GB).
- `/system/user/publicdata/` contains datasets you might need.
- `/system/user/<username>` is your home directory. This directory is supposed to be small. Therefore you should never store datasets here! It is limited to 20GB.

### Software Installations
The directories `/sytem/apps/biosoft` and `/system/apps/mlsoft` contain shared software installations. 
Each subdirectory typically contains a `bashrc` file that you can use to setup the installation, like:

    source /system/apps/biosoft/python-365/bashrc

Either `source` these manually or add the source command to your ~/.bashrc if you always need that command.

### terminal multiplexers
If you work on our servers, it's a good idea to use a terminal multiplexer such as `screen` or `tmux`: these programs allow you to keep a session/computation running even after you log out. It doesn't really matter what you use, as both are well supported. screen is older (so it's often pre-installed on older servers), while tmux is a bit easier to use (it's easier to configure and has more utility functions). Attached to this guide you'll find my tmux.conf file that has some handy features that you can copy to ~/.tmux.conf (afterwards tmux has the same shortcuts as screen).
Once you've decided on which one to use, maybe look up a short introduction or ask someone at the institute to show you the most important features.


Random Notes
------------

### finding your processes
using `ps aux` you can show a list of all running processes. To find out your own processes, use `ps aux | grep <username>`. (or in top, use 'u')


### Killing a process
If your program locks up and you want to kill it, use `kill <PID>`, where <PID> is the process ID that you can find out e.g. using `ps aux` or `top`. Sometimes this isn't enough and doesn't work, then you can use `kill -9 <PID>`  (or use 'k' in top).


### Ports for services
Sometimes you'll have to run applications on the servers that communicate via the network. These usually require an IP (e.g. Ipython/jupyter notebooks, tensorboard, ...). It's a good idea to change the default port to something personalized to avoid surprises (e.g. someone else using your jupyter server because it runs on the default port). Ask your supervisor to check if a certain port is available / which ports to use.

### Tensorflow uses too much GPU
By default, TensorFlow reserves ALL GPU memory on ALL the GPUs in the machine. This is almost never what you want. To stop TF from reserving all GPUs, set the "CUDA_VISIBLE_DEVICES" environmental variable to the ID of the GPU you really want to use BEFORE you start your process. You can also do this from inside Python like this:

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"   # process will onw only use GPU 2
    import tensorflow as tf

To stop TF from pre-allocating all available space, start your tf.Session as follows so it will only allocate on demand:


    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ....

### Useful tools to operate on remote machine

* Mounting your server files onto your local machine.
    * use the command `sshfs`. Here is a [tutorial](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh)
* synchronize server and local files without mounting and copy paste
    * use `rsync`. Here is a [tutorial](https://linuxize.com/post/how-to-transfer-files-with-rsync-over-ssh/)
* synchronize server to other cloud storages (nextcloud/google drive/dropbox/...)
    * use `rclone`. Follow installation _Linux installation from precompiled binary_ as described
      [here](https://rclone.org/install/)
    * then add the binary to your `.bashrc`
* run Jupyter notebook on server and use local browser to access the instance
    * use an SSH tunnel to forward the port which Jupyter uses to your local machine as described
      [here](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh)
* write and debug your code directly on the server:
    * either use jupyter notebook + tunneling or for more complex code use Visual Studio Code Remote
      Development plugin. Tutorial is [here](https://code.visualstudio.com/docs/remote/ssh)