# OpenVINO Installtion
You can install release packages or build from source. As of writing this document, there is no release package for Ubuntu 22.04. Plus, to quantize dynamic models, you need to have at least 2022.2.0 version of OpenVINO. So, this document will only cover the installation from source. For other installation methods, refer to [OpenVINO documentation](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html).

You need to install both OpenVINO dev tools and runtime. Dev tools provide the tools for quantizing, optimizing, and converting models. Runtime provides the libraries for running the models on Python, C, and C++. This document will cover both of them.

The installation is tested on Ubuntu 22.04 and Python3.9. Install Python3.9 manually as the default version of Python in Ubuntu 22.04 is 3.10. Only CPU installation is done as we're only interested in optimizing model inference on CPU.

The following steps are taken from [OpenVINO build from source](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode) and only some minor changes are made.

## Steps to compile OpenVINO from source
1. Add deadsnakes PPA to install Python3.9:
    ```bash
    sudo add-apt-repository ppa:deadsnakes/ppa 
    sudo apt install python3.9 python3.9-venv python3.9-dev
    ```
    And create a virtual environment and activate it (make sure all of the following steps are done in the virtual environment):
    ```bash
    python3.9 -m venv <path to virtualenv/openvino_venv>
    source openvino_venv/bin/activate
    python -m pip install --upgrade pip
    ```
2. Clone OpenVINO repository:
    ```bash
    git clone https://github.com/openvinotoolkit/openvino.git
    cd openvino
    git submodule update --init --recursive
    ```
3. Install build dependencies:
    ```bash
    chmod +x ./install_build_dependencies.sh
    ./install_openvino_dependencies.sh
    ```
    And Python dependencies (in openvino folder):
    ```bash
    cd /src/bindings/python/src/compatibility/openvino
    pip install -r requirements-dev.txt
    cd /src/bindings/python/wheel
    pip install -r requirements-dev.txt
    ```
4. Create build folder in OpenVINO folder and run CMake:
    ```bash
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_INTEL_GPU=OFF -DENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=`which python` -DENABLE_WHEEL=ON ..
    ```
    **Note:** If you have `numeric_limits is not a member of std` error, just disable building GNA plugin by adding `-DENABLE_GNA=OFF` to the CMake command. 
5. Build OpenVINO:
    ```bash
    make 
    ```
6. After the build is done, install OpenVINO runtime and dev Python wheels (in `build/wheels` folder)
    ```bash
    pip install openvino-2022.3.0-000-cp39-cp39-manylinux_2_35_x86_64.whl openvino_dev-2022.3.0-000-py3-none-any.whl
    ```
    Note that the wheel names might be different.
7. Finally, install OpenVINO (I suggest creating a directory named `intel` in you home directory and install OpenVINO there):
    ```bash
    cmake --install openvino/build/ --prefix $HOME/intel/
    ```

If you've made so far, you've successfully installed OpenVINO. Now, you can test it by building C++ examples in `<INSTALL DIR>/samples/cpp`:

```bash
cd <INSTALL DIR>/samples/cpp
./build_samples.sh
```
Then cd into `openvino_._samples_build/intel64/Release/` and run `hello_query_device` to test if OpenVINO is working. 

You can also check `mo` and other dev tools by running `mo -h` in terminal.
