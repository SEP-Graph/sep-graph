# SEG-Graph: Finding Shortest Exection Paths for Graph Processing under a Hybrid Framework on GPU #

## 1. Introduction ##
This repo contains all the source code and scripts to compile SEP-Graph and reproduce the experiments in
PPoPP 2019 paper.

## 2. Installation ##

### 2.1 Requirements ###

#### 2.1.1 Software ####
* CUDA == 9.x
* GCC == 5.x.0
* CMake >= 2.8
* Python == 3.X
* Linux/Unix

#### 2.1.2 Hardware ####

* Intel/AMD X64 CPUs
* 32GB RAM (at least)
* NVIDIA GTX 1080 or NVIDIA P100 or NVIDIA V100
* 50GB Storage space (at least)

### 2.2 Setup ###
1. Download

    git clone https://github.com/pwrliang/SEP-Graph
    
2. Build
* Atomically build
  
  The scripts **"setup_gunrock.py, setup_groute.py,setup_sep.py"** under the
   **SEP-Graph/evaluation/bin** will setup Gunrock, Groute and SEP-Graph separately

* Mannually build

For build gunrock and groute:

    https://github.com/gunrock
    https://github.com/groute

For build SEP-Graph:
  - cd SEP-Graph
  - mkdir build && cd build
  - cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda-9.1 -DCMAKE_C_COMPILER=/opt/gcc-5.3.0/bin/gcc -DCMAKE_CXX_COMPILER=/opt/gcc-5.3.0/bin/g++
  - make -j 8

### 2.3 Evaluation ###
1. Download the dataset by running `SEP-Graph/evaluation/bin/download.py`
2. Evaluate the Gunrock, Groute and SEP-Graph by running `SEP-Graph/evaluation/bin/run_all.py`
3. Convert the result to CSV format by running `SEP-Graph/evaluation/bin/parse_all.py [algo_name]`, the parameter of algo_name can be `bc, bfs, sssp and pr`

**Note: `run_all.py` will evaluate 5 times. `parse_all.py` will calculate the average running time (the maximum and minimum value are excluded)** 

## Contact ##

For the technical questions, please contact: **pwrliang@gmail.com**
 
For the questions about the paper, please contact: **hwang121@gmail.com**
