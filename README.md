# SEG-Graph: Finding Shortest Exection Paths for Graph Processing under a Hybrid Framework on GPU #

## 1. Introduction ##
This repo contains all the source code to build SEP-Graph.

## 2. Installation ##

#### 2.1 Software Requirements ####
* CUDA == 9.x
* GCC == 5.x.0
* CMake >= 2.8
* Linux/Unix

#### 2.2 Hardware Requirements ####

* Intel/AMD X64 CPUs
* 32GB RAM (at least)
* NVIDIA GTX 1080 or NVIDIA P100 or NVIDIA V100
* 50GB Storage space (at least)

### 2.3 Setup ###
1. Download

    git clone --recursive https://github.com/SEP-Graph/sep-graph.git
    
2. Build

  - cd sep-graph
  - mkdir build && cd build
  - cmake .. -DCUDA_TOOLKIT_ROOT_DIR=**CUDA_ROOT** -DCMAKE_C_COMPILER=**GCC_PATH** -DCMAKE_CXX_COMPILER=**G++_PATH**
  - make -j 8

## Contact ##

For the technical questions, please contact: **pwrliang@gmail.com**
 
For the questions about the paper, please contact: **hwang121@gmail.com**
