#include <iostream>
#include <openvino/openvino.hpp>

int main(){
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model("/home/nima/Documents/w/retina/openvino-test/b12_fp32/b12.xml", "AUTO");
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    float memory_ptr[1*3*200*300];
    auto input_port = compiled_model.input();
    ov::Tensor input_tensor(input_port.get_element_type(),
                            input_port.get_shape(),
                            memory_ptr);
    infer_request.set_input_tensor(input_tensor);
    infer_request.start_async();
    infer_request.wait();
}
