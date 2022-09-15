# OpenVINO Optimization Guide

Valuable tools are provided within OpenVINO environment to help data scientists and developers to optimize their models. In this guide, I summarize these techniques and investigate their benefits. All information gathered in this guide are derived from the [OpenVINO documentation](https://docs.openvino.ai/latest/documentation.html). For more information, please refer to the official documentation.

**Contents:**

+ [Model Optimizer](#model-optimizer)


## Model Optimizer
Model optimizer is used to convert the AI models to OpenVINO IR format. IR stands for intermediate representation. Every converted model is consisted of (at least) two files: 1) `.xml` file which contains the network topology and 2) `.bin` file which contains the weights of the network. Later, OpenVINO runtime loades the IR files and executes the model. 

**Note:** Although the OpenVINO runtime can run `.onnx` models directly, but it is recommended to convert the models to IR format as some further optimizations are applied to the IR models.

The details of how to convert the models with model optimizer are not covered in this guide. However, I would mention some important parameters and tips that could help optimizing models further.

### Tips and Notes:
1. **Static shape:** 

    It's better to use static shapes instead of dynamic shapes. If the input shape is not going to change in consecutive runs, then it's better to use static shapes. The reason is that static shapes helps using the same allocated memory multiple times and also simple operators will be used in OpenVINO runtime engine (instead of very general operatorss).
    Even if the input shape changes but there is a lower and upper bound on the dimmensions, it is recommended to provide the ranges (especially upper bound).

    There are two methods to make inputs of the model static: 1) with model optimizer (`mo`), 2) with OpenVINO runtime. 
    + model optimizer: 
        ```bash
        $ mo --input_model face-det.onnx --input images --input_shape [1,3,640,480]
        ```
    + OpenVINO runtime:
        ```cpp
        ov::Core core;
        auto m = core.read_model("model.xml");    
        m->reshape({1, 3, 640, 480});        
        ```
    Note that when you are making the inputs static using model optimizer, you won't be able to change the input shape later. However, OpenVINO runtime allows you to change the input shape before compiling it for a specific device.
