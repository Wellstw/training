**Setting Up a VLLM Environment on Ubuntu Server 22.04 LTS**

In this guide, we will walk you through the process of setting up a VLLM (Visual Large Language Model) environment on an Ubuntu Server 22.04 LTS machine. This setup includes installing the necessary software, configuring the system for GPU acceleration, and deploying a pre-trained Llama-3.2-90B-Vision-Instruct-FP8-dynamic model.

**System Requirements**

* Ubuntu Server 22.04 LTS
* RAM: 256GB DDR5 ECC-5600
* Storage: SSD (256GB) + HDD (3T)
* NVIDIA A800 40GB Active X4 (for Cuda Computing)
* NVIDIA T1000 8GB x1 (For system Display)

**Step 1: Install GPU Headless Driver**

To enable GPU acceleration, we need to install the NVIDIA headless driver. Run the following commands:
```bash
sudo apt install nvidia-headless-550-server --no-install-recommends --no-install-suggests
sudo apt install nvidia-utils-550-server --no-install-recommends --no-install-suggests
```
**Step 2: Check GPU Status**

To verify that the GPUs are recognized by the system, run:
```bash
nvidia-smi
```
~~~
Mon Nov 25 08:37:31 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A800 40GB Active        Off |   00000000:27:00.0 Off |                    0 |
| 30%   58C    P0             55W /  240W |       1MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A800 40GB Active        Off |   00000000:38:00.0 Off |                    0 |
| 30%   58C    P0             49W /  240W |       1MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA T1000 8GB               Off |   00000000:81:00.0 Off |                  N/A |
| 40%   56C    P0             N/A /   50W |       1MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A800 40GB Active        Off |   00000000:98:00.0 Off |                    0 |
| 30%   57C    P0             51W /  240W |       1MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA A800 40GB Active        Off |   00000000:C8:00.0 Off |                    0 |
| 30%   56C    P0             44W /  240W |       1MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
~~~
This should display information about the available GPUs.

**Step 3: Install Miniconda and Conda Environment**

We will use Miniconda to manage our conda environment. Download the Miniconda installer and make it executable:
```bash
chmod +x ./Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
```
Create a new conda environment named `vllm` and clone the base environment:
```bash
conda create --name vllm --clone base
```
Activate the `vllm` environment:
```bash
conda activate vllm
```
**Step 4: Install PyTorch and Dependencies**

Install PyTorch, torchvision, torchaudio, and pytorch-cuda:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install the VLLM library and llmcompressor:
```bash
pip install vllm
pip install llmcompressor
```
**Step 5: Download Pre-trained Model**

Download the pre-trained Llama-3.2-90B-Vision-Instruct-FP8-dynamic model using Hugging Face CLI:
```bash
huggingface-cli download neuralmagic/Llama-3.2-90B-Vision-Instruct-FP8-dynamic
```
**Step 6: Configure Environment Variables**

Set the following environment variables to enable offline mode and specify the GPU devices:
```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,3,4 # Four GPUs
```
**Step 7: Serve the Model**

Finally, serve the pre-trained model using VLLM:
```bash
vllm serve neuralmagic/Llama-3.2-90B-Vision-Instruct-FP8-dynamic --enforce-eager --max-num-seqs 1 --trust_remote_code --max_model_len 16384 --gpu_memory_utilization 0.9 --tensor-parallel-size 4
```
This should start the VLLM server, and you can use it to generate text or perform other tasks using the pre-trained model.

Note: This guide assumes that you have a basic understanding of Linux and conda environments. If you encounter any issues during the setup process, please refer to the official documentation for more information.