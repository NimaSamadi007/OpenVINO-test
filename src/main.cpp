#include <iostream>
#include <openvino/openvino.hpp>

int main(){
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model("/home/nima/Documents/w/retina/openvino-test/b12_fp32/b12.xml", "AUTO");
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    float* memory_ptr = new float[1*3*640*480];
    auto input_port = compiled_model.input();
    auto output_port = compiled_model.output();
    ov::Tensor input_tensor = ov::Tensor(input_port.get_element_type(), {1, 3, 640, 480}, memory_ptr);
    infer_request.set_input_tensor(input_tensor);
    infer_request.start_async();
    infer_request.wait();
    auto output = infer_request.get_tensor(output_port.get_any_name());
    float* output_buffer = output.data<float>();
    ov::Shape shape = output.get_shape();
    for(int i = 0; i < shape.size(); i++)
        std::cout << shape[i] << ", ";
    std::cout << std::endl << output.get_size() << std::endl;

    delete[] memory_ptr;
}
