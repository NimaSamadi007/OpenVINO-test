# OpenVINO Framework
In this repository I will test and document the OpenVINO Framework. According to [OpenVINO](https://docs.openvino.ai/latest/index.html):
> OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference.

So, the purpose of OpenVINO is to enhance the performance of AI inference. It is currently support Intel hardware (also ARM CPU is supported but not officially as of now). Since Intel CPU is quite widespread and widely used, we choose it as one of the main implementation hardware. 

## Install
The installation is tested with Ubuntu 22.04 and Python3.9. Install Python3.9 manually as the default version of Python in Ubuntu 22.04 is 3.10. I suggest creating a virtual environment and installing OpenVINO in it.

OpenVINO comes in two flavors: 1) OpenVINO developement tools, 2) OpenVINO runtime 

**OpenVINO developement** tools is used to download, convert, optimize and tune pre-trained deep learning models. For this purpose, I used installed developement tool in Python. The following instruction are based on [this](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html). To install OpenVINO developement tools on Ubuntu 22.04 you need to install Python3.9 which is not the default version. So, add this ppa to your repository and install Python3.9.

```bash
$ sudo add-apt-repository ppa:deadsnakes/ppa 
$ sudo apt install python3.9 python3.9-venv
# create a virtual environment
$ python3.9 -m venv <path to virtualenv/openvino_venv>
# activate the virtual environment
$ source openvino_venv/bin/activate
# upgrade pip
$ python -m pip install --upgrade pip
# install OpenVINO developement tools
$ python -m pip install openvino-dev 
```
Test installation with `mo -h` command.

**OpenVINO runtime** is used to deploy model on various devices. To install it on Python, run the following command:

```bash
$ pip install openvino
```
To test the installation, run the following command:

```bash
(openvino_env) ➜  ~ python
Python 3.9.13 (main, May 23 2022, 21:57:12) 
[GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from openvino.runtime import Core
>>>
```
However, we're not going to use OpenVINO runtime in Python and only will use developement tools. To install OpenVINO runtime for C++, you need to build it from source as the installer and apt doesn't work for Ubuntu 22.04. To build from source, first clone git repository (based on [this](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode) link):

```bash
$ git clone https://github.com/openvinotoolkit/openvino.git
$ cd openvino
$ git submodule update --init --recursive
```
Then install dependencies:

```bash
$ chmod +x install_build_dependencies.sh
$ ./install_build_dependencies.sh
```
Then create build folder and run CMake:

```bash
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make --jobs=$(nproc --all)
```
Then wait till the build is finished. After that, install OpenVINO runtime in a custom folder:

```bash
$ cmake --install <BUILDDIR> --prefix <INSTALLDIR>
```
To test the installation, build code samples:

```bash
$ cd <INSTALLDIR>/samples/cpp
$ ./build_samples.sh
```
If everything goes well, you can use OpenVINO runtime in C++. 


