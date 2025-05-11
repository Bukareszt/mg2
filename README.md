# SFT repository to train on H100
This repository contains code and scripts to run SFT on H100 partition in the Bem 2 (the cluster).

## Limitations
Implemented at the moment:
 - multi-GPU single-node training
 - full finetuning (SFT)
 - connection to S3

Future work:
 - multi-node training
 - LoRA finetuning
 - QLoRA finetuning

## Instruction (Bem 2)

To run this code on the cluster with H100 you need to take following steps:


### 0. First login to the cluster

This step is not obligatory, and can be taken after the [1.](#1.-build-an-apptainer-image).

To login to Bem 2 you need to have granted permissions (ask for them on Discord server).

Your account for Bem 2 has the same username and password as for [e-science](https://users.e-science.pl/).

Yo can log in to the cluster using ssh:

    ssh <username>@ui.wcss.pl

After that you are probably in `/` directory instead of `~/` directory. If so you can set `HOME` variable to the correct one by:

    export HOME=/home/$USER

and change directory by `cd ~/` or `cd /home/$USER`. 

In the future logins variable `HOME` should be set to the correct one, but sometimes environment variables are reset and you should set it again. To fix that home-problem work is in progress, but we do not know when it will be completed.


### 1. Build an Apptainer image

To build Apptainer image you need to build docker image **locally** using `Dockerfile`: 

    docker build -t <image-name> .

Next step is to convert the docker image to `.sif` file (https://github.com/kaczmarj/apptainer-in-docker):

    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/work \
        kaczmarj/apptainer build <image-name>.sif docker-daemon://<image-name>

At the end you need to copy your `.sif` file to the cluster:

    scp <image-name>.sif <username>@ui.wcss.pl:/home/$USER

### 2. Prepare environment

To get the code on the cluster, download it or simply clone the repository:

    git clone https://gitlab.clarin-pl.eu/pllum/action-3/supervised-finetuning-h100.git

After that you should set following environment variables:

    export HOME=/home/$USER                     
    export HF_HOME=$TMPDIR/huggingface     
    export HF_TOKEN=<your huggingface token>     
    export WANDB_API_KEY=<your W&B api key>
    export WANDB_DIR=$TMPDIR
    export WANDB_CACHE_DIR=$TMPDIR/.cache/wandb
    export WANDB_CONFIG_DIR=$TMPDIR/.config/wandb



 - `HOME` - if you didn't do it in the [0.](#0-first-login-to-the-cluster) step 
 - `HF_TOKEN` - models or datasets could require permissions to huggingface
 - `WANDB_API_KEY` - default logging is set to W&B so you can change it or set your api key
 - `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_CONFIG_DIR` - change dirs for W&B to `$TMPDIR` because defaults are set to `$HOME`
 - `HF_HOME` - set `$TMPDIR` as cache for huggingface to not download models/datasets to `$HOME`.

> [!WARNING]  
> Remember that we want to reduce I/O operations in `$HOME` as much as we can. That's why we are setting `.cache`, `.config`, etc. to `$TMPDIR` and we do not use `$HOME` as the storage for data.

### 3. SBATCH file

File that is executed by the `sbatch` command is a common bash script with one specific feature. After the `#!/bin/bash` line there are defined resources for slurm that we will request. All resources are defined by the line: 

    #SBATCH <flag>=<argument>

The most important ones:

    #SBATCH --nodes=<number of nodes>
    #SBATCH --cpus-per-task=<number of CPUs per MPI task>
    #SBATCH --time=<requested time> # in format h:m:s or hh:mm:ss
    #SBATCH --mem=<amount of RAM> # i.e. 16gb 
    #SBATCH -p H100 # partition - in our case always H100
    #SBATCH --gpus-per-node=<GPUs per node> # on H100 partition max 4

> [!IMPORTANT]
> If something force us to keep data in our `$HOME` before running the program we should copy them to the `$TMPDIR`. If there are a lot of files we should keep them in `$HOME` as archives and we can extract them only in `$TMPDIR`. 
>
> In opposite way is the same. If the program produces any files they should be written into the `$TMPDIR` (**If S3 is not available**). If you want to keep them after the end of the job you should copy them into the `$HOME`. If there are a lot of files you should  first pack them into an archive, and only then copy them into the `$HOME`.

<!-- #### Exporting environmental variables

At the beggining of the script after defining resources to request you should export the environmental variables the same as before, to set them inside a slurm job 
> [!NOTE]
> some of them can be inherited from the outside, but I prefer to export all of them to be sure

    export HOME=/home/$USER                      
    export HF_HOME=$TMPDIR/huggingface     
    export HF_TOKEN=<your huggingface token>     
    export WANDB_API_KEY=<your W&B api key>
    export WANDB_DIR=$TMPDIR
    export WANDB_CACHE_DIR=$TMPDIR/.cache/wandb
    export WANDB_CONFIG_DIR=$TMPDIR/.config/wandb -->

#### Mounting and unmounting S3

Firstly you should create directory where you want to mount the S3 bucket

    mkdir -p ~/s3 

After that you want to mount a bucket to the created directory: 

    /opt/goofys/v0.24.0/bin/goofys --endpoint https://s3min2.e-science.pl/ <bucket's name> ~/s3


To check if the S3 has mounted correctly you could list the interior of the directory:

    ls ~/s3

At the end of your job (*after all what you wanted to run*) you should unmount the S3:

    fusermount -u ~/s3

#### Run the program

Inside a slurm's job we should run our programs inside the apptainer's containers. Firstly you should set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` to RAM or `$TMPDIR`. You could export them or simply put them right before the `apptainer run` command.
 
    APPTAINER_TMPDIR=/dev/shm/$SLURM_TASK_PID \
    APPTAINER_CACHEDIR=/dev/shm/$SLURM_TASK_PID

After that you run the program inside the apptainer's container:

    apptainer run --nv \
    --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
    ~/<image name>.sif \
    <program's command>

- `--nv` - allows the container to have access to GPUs.
- `--mount type=bind,src=$TMPDIR,dst=$TMPDIR` - mounts `$TMPDIR` inside the container to have access
- `<program's command>` - command to run your program i.e. `accelerate launch main.py --arg1 val1 ...`

> [!IMPORTANT]
> If you are using hydra framework for configs you should add `hydra.run.dir=$TMPDIR` as an argument for your program to not create log files in `$HOME`. 

### 4. Run the job

You can run job using `sbatch` command or you can do it in interactive way.

#### 4.1 An interactive way

You can run the job interactively using `srun` command.

    srun <flag1> <arg1> ... <flagN> <argN> --pty bash

Flags that you want to use are the same flags as in [the script](#3.-SBATCH-file) followed `#SBATCH `.  

When you will get resources you will be attached to a bash shell (as requested `--pty bash`). 

Firstly you should export [environmental variables](#2.-Prepare-environment). Subsequently you should mount the [S3](#mounting-and-unmounting-s3). To run the program you should execute the same command as in the [sbatch script](#run-the-program).

> [!IMPORTANT]
> Remember to [unmount](#mounting-and-unmounting-s3) S3 at the end of the job. Just before exit from the shell.

#### 4.2 A scripted way

At the end to finally run the training you have to simply run the following command:

    sbatch run_sbatch.sh

Your job will be put in the queue for the resources and you will see in the console the id of your job.

To check status of your job you can simply run:

    squeue -u <username>

And in [docs](https://man.e-science.pl/pl/kdm/slurm/zarzadzanie-zadaniami) you can check what status it is. 

To cancel the job you can simply run:

    scancel <your's job id>

Output of the job is in the file `slurm-<job's id>.out` and it is updating each few seconds (don't panic if nothing is changing out there in the 1 or 2 second period).

> [!TIP]
If you want to monitor your job by `nvidia-smi`, `top` or `dcgmi` you can attach interactive session to your current job (It will attach to the first node)
>
> >srun --pty --overlap --jobid <your's job id> bash
>


