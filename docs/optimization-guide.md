# OpenVINO Optimization Guide

Valuable tools are provided within OpenVINO environment to help data scientists and developers to optimize their models. In this guide, I summarize these techniques and investigate their benefits. All information gathered in this guide are derived from the [OpenVINO documentation](https://docs.openvino.ai/latest/documentation.html). For more information, please refer to the official documentation.

**Contents:**

+ [Model Optimizer](#model-optimizer)
+ [Deploying with OpenVINO Runtime](#deploying-with-openvino-runtime)
+ [Working with Devices](#working-with-devices)
+ [Hardware Support](#hardware-support)

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
2. **Embedding Preprocessing Computation:**

    Almost all AI models require some pre- and post-processings. For example, in image classification models, the input image is resized and normalized before feeding it to the model. However, these routines are not a part of the model itself. So, if you convert a Pytorch model and later you wanna use it in OpenVINO runtime in C++, you should implement the preprocessing routines in C++ as well. It might not be a big deal but suppose you need to run the processing routines on GPU. Then you should rewrite whole functions. 

    By embedding the preprocessing routines into the model, you can avoid this problem. Plus, you can benefit other optimizations as well. For example, hybrid runtime would help you to run part of preprocessing on other devices. So, it is recommended to embedd preprocessing routines into the model. The preprocessing API will not be covered in detail in this guide. So, refer to [this](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Additional_Optimization_Use_Cases.html) for more information.

3. **Compression of a Model to FP16:**

    By compressing the model to FP16 you can reduce the model size. This compression would not affect the accuracy of the model that much and is recommended before quantizing model to INT8. Note that the inference will not be in FP16 format unless you have a device that supports bfloat16 format. For more information about hardware support, refer to [this](#hardware-support) section of this document.


## Deploying with OpenVINO Runtime
This section covers the deployment of the models with OpenVINO runtime. My focus is on the C++ API of OpenVINO runtime. However, the Python API is very similar and will not be discussed here. Plus, I only mention tips that will help to improve the performance of the model. For more information, refer to the [official documentation](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Integrate_OV_with_your_application.html). 

### Tips and Notes:
1. **Inference request:**

    After compiling model for a specific hardware, an "inference request" is created and then it will be used for running inference and getting results. Basically the inference request is run in two mode: 1) Synchrounous, 2) Asynchronous. In synchronous mode, the inference request will wait until the inference is done and then return the results. In asynchronous mode, the inference request will return immediately and the results will be available later. You can choose to wait for asynchronous inference or set a callback that will be run when the inference is done. Following snippet shows the usage of synchronous and asynchronous mode:
    ```cpp
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    // 1) synchronous execution
    infer_request.infer();
    // 2-1) asynchronous with callback
    infer_request.set_callback([&](std::exception_ptr ex_ptr){
        if(!ex_ptr){
            // inference is done and output data can be processed
            std::cout << "Inference is done!\n";
        } else {
            std::cout << "Something went wrong!\n";
        }
    });
    // 2-2) asynchronous with waiting
    infer_request.start_async();
    infer_request.wait();
    ```
    You can also set the exact waiting time with `wait_for` function. The duration specifies the blocking time of `infer_request` method. Using asynchronous mode is recommended and you can run other tasks while the inference is running. 


## Working with Devices

## Hardware Support
To be able to get the best performance, you should know your CPU features and hardware supports. According to [OpenVINO](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html), 