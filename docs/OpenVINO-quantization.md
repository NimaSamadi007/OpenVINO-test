# OpenVINO Quantization Details

This document describes the inner workings of the OpenVINO quantization tool. You might not need to read this document if you are just using the tool. However, if you are interested in the details, this document is for you. 

First the math behind the quantization will be introduced. Then actual implementation details will be discussed. Also several references will be provided throughout the document for further reading. 

Note that only `DefaultQuantization` and `AccuracyAwareQuantization` are discussed in this document.


**Contents:**

+ [Introduction](#introduction)
+ [Math behind the quantization](#math-behind-the-quantization)
    + [Symmetric quantization](#symmetric-quantization)
    + [Asymmetric quantization](#asymmetric-quantization)
+ [Implementation details](#implementation-details)
___

## Introduction
In quantization we are interested in representing a real number (FP32 in computer) with a finite (smaller) number of bits. The purpose of quantization is to reduce the number of bits that are used for representing numbers. This helps reducing the memory footprint of the model and helps increasing the inference speed. For example, in extreme case, instead of multiplying two 32 bit floating point numbers, we can multiply two 8 bit integers. The integer multiplicatio is way more efficient and faster than the floating point multiplication. In general, quantization can be done during model training and during model inference. In this document we are interested in quantizing the model during inference. For more information on quantization techniques and methods, please refer to [this whitepaper](https://www.intel.com/content/www/us/en/developer/articles/technical/lower-numerical-precision-deep-learning-inference-and-training.html). 

Note that by quantizing the model, we are trading off accuracy for speed and memory footprint. So, it's essential to find the right balance between accuracy and speed/memory footprint. 

Quantization techniques have been widely used in other fields such as signal processing and image processing. In this document we are interested in quantizing the weights and activations of the neural network. Plus, this document is focused on OpenVINO framework and its quantization tool. 

## Math behind the quantization
In this section we will discuss the math behind the quantization. We will start with symmetric quantization and then move to asymmetric quantization. This section is based on [OpenVINO doc](https://docs.openvino.ai/2020.4/pot_compression_algorithms_quantization_README.html).

In quantization algorithms, a specific range of input values ($[a, b]$) is mapped to a predefined range of output values ($[c, d]$). The mapping is usually linear but can be non-linear as well. Consider the following example:
+ **example:** We have 8 bits to represent a number (random variable) that can take values between -1 and 2. We want to map this range to the range of 8-bit unsigned integers (0-255).

The example shows the purpose of quantization. There are many ways to map the input range to the output range. The most common way is to use linear mapping. In linear mapping, the input range is divided into $2^n$ equal intervals. Then for each interval, a number is assigned to it. In OpenVINO, the mapping is done as follows:

$$
\begin{align*}
&\mathrm{output} = \frac{\mathrm{round}((\mathrm{clamp}(input, input\_low, input\_high) - input\_low)*s) }{s} + input\_low \\
&\mathrm{clamp}(input, input\_low, input\_high) = min(max(input, input\_low), input\_high) \\
&s = \frac{\mathrm{levels}-1}{input\_high - input\_low} \\
\end{align*}
$$
And $\mathrm{levels}$ specifies the number of quantization levels that can be represented. For example, if we have 8 bits, $\mathrm{levels} = 2^8 = 256$. 

OpenVINO supports two types of quantization modes: symmetric and asymmetric. In symmetric mode, the floating-point zero is quantized to the integer zero. However, in asymmetric mode, the floating-point zero is not necessarily quantized to the integer zero and it can be any integer number. But, in both modes, the floating-point zero is mapped directly to the quant witout any rounding.

### Symmetric quantization
In this method, $input\_low$ and $input\_high$ are computed as follows:
$$
\begin{align*}
input\_low &= scale * \frac{level\_low}{level\_high} \\
input\_high &= scale \\
\end{align*}
$$
$scale$ parameters is tuned during the quantization process. $level\_low$ and $level\_high$ are the lowest and highest quantization levels respectively. So, for quantizing neural network parameters the following values are used:
+ For weights:
    $$
    \begin{align*}
    level\_low &= -2^{bits-1}+1 \\
    level\_high &= 2^{bits-1}-1 \\
    levels &= 2^{bits}-1
    \end{align*}
    $$
+ For unsigned activations:
    $$
    \begin{align*}
    level\_low &= 0 \\
    level\_high &= 2^{bits}-1 \\
    levels &= 2^{bits}
    \end{align*}
    $$
+ For signed activations:
    $$
    \begin{align*}
    level\_low &= -2^{bits-1} \\
    level\_high &= 2^{bits-1}-1 \\
    levels &= 2^{bits}
    \end{align*}
    $$
Which $bits$ is the number of quantization bits (e.g. 8 bits).

### Asymmetric quantization
In asymmetric quantization, $input\_low$ and $input\_range$ are tuned druing the quantization process. For 8 bits quantization, the following values are used:
$$
\begin{align*}
input\_high &= input\_low + input\_range \\
levels &= 256
\end{align*}
$$
And for weights and activations, the following values are used:
$$
\begin{align*}
input\_low' &= min(input\_low, 0) \\
input\_high' &= max(input\_high, 0) \\
ZP &= \mathrm{round}\left(\frac{-input\_low' * (levels-1)}{input\_high' - input\_low'}\right) \\
input\_high'' &= \frac{ZP-levels+1}{ZP} * input\_low' \\
input\_low'' &= \frac{ZP}{ZP-levels+1} * input\_high' \\
input\_low, input\_high &= \begin{cases}
input\_low', input\_high', \quad ZP \in \{0, levels-1\} \\
input\_low', input\_high'', \quad input\_high''-input\_low' > input\_high'-input\_low'' \\
input\_low'', input\_high', \quad input\_high'' - input\_low' \leq input\_high'-input\_low'' \\
\end{cases}
\end{align*}
$$
## Implementation details
