#include <iostream>
#include <memory>
#include <chrono>
#include <vector>
#include <openvino/openvino.hpp>

int main(){
    ov::Core core;
    /* using ov::Model */
    // std::shared_ptr<ov::Model> m = core.read_model("/home/nima/Documents/w/retina/openvino-test/b12_fp16_2/b12.xml");    
    // m->reshape({1, 3, 640, 480});
    // ov::CompiledModel compiled_model = core.compile_model(m, "AUTO");

    ov::CompiledModel compiled_model = core.compile_model("/home/nima/Documents/w/retina/openvino-test/b12_fp16_2/b12.xml", "AUTO");
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    // while(1){
    std::vector<float> input_tensor_values(1*3*640*480);
    ov::Output<const ov::Node> input_port = compiled_model.input();
    std::vector<ov::Output<const ov::Node>> output_ports = compiled_model.outputs();
    std::cout << "Output names: \n";
    for(const auto& o_port : output_ports){
        std::cout << o_port.get_any_name() << "\n";
    }
    std::cout << "===========\n";
    std::vector<int> input_shapes = {1, 3, 640, 480};
    ov::Shape in_shapes = std::vector<size_t>(input_shapes.begin(), input_shapes.end());
    ov::Tensor input_tensor = ov::Tensor(input_port.get_element_type(), in_shapes, input_tensor_values.data());
    infer_request.set_input_tensor(input_tensor);

    // auto start = std::chrono::high_resolution_clock::now();
    infer_request.start_async();
    infer_request.wait();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    // printf("FPS: %f\n", (1000000000.0 * BATCH_SIZE) / elapsed.count());

    // std::string names = output_port.get_any_name();
    // std::cout << "Names: " << names << std::endl;
    // ov::Output<ov::Node> output_tensors;
    std::vector<ov::Tensor> output_tensors;
    for(const auto& o_port : output_ports){
        output_tensors.push_back(infer_request.get_tensor(o_port));
        std::cout << o_port.get_any_name() << " : " << output_tensors[output_tensors.size()-1].get_size() << std::endl;
    }
    
    std::cout << "Inference done! \n";
    // float* output_buffer = output.data<float>();
    // ov::Shape shape = output.get_shape();
    // for(int i = 0; i < shape.size(); i++)
    //     std::cout << shape[i] << ", ";
    // std::cout << std::endl;
    // }
}
