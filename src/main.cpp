#include <iostream>
#include <memory>
#include <chrono>
#include <vector>
#include <openvino/openvino.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/visualize_tree.hpp>

bool end_exec = false;

void visualize_example(const std::shared_ptr<ov::Model>& m){
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::VisualizeTree>("test.svg");
    manager.run_passes(m);    
}

int main(){
    ov::Core core;
    // std::vector<std::string> devices = core.get_available_devices();
    // for(const auto& dev : devices)
    //     std::cout << dev << ", ";
    // std::cout << std::endl;
    /* using ov::Model */
    std::shared_ptr<ov::Model> m = core.read_model("/home/nima/Documents/w/retina/openvino-test/b12_fp16_2/b12.xml");    
    m->reshape({1, 3, 640, 480});
    // visualize_example(m);
    auto supp_pro = core.get_property("CPU", ov::supported_properties);
    for(const auto& name : supp_pro){
        std::cout << name << ", ";
    }
    std::cout << std::endl;
    auto cpuOptimizationCapabilities = core.get_property("CPU", ov::device::capabilities);
    std::cout << "CPU optimization cap: \n";
    for(const auto& name : cpuOptimizationCapabilities){
        std::cout << name << ", ";
    }
    std::cout << std::endl;
    ov::CompiledModel compiled_model = core.compile_model(m, "CPU",
                                                          ov::enable_profiling(true),
                                                          ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                                          ov::hint::inference_precision(ov::element::f32),
                                                          ov::inference_num_threads(2));
    std::cout << "Num of inference threads: " << compiled_model.get_property(ov::inference_num_threads) << std::endl;
    std::cout << "Inference precision: " <<  compiled_model.get_property(ov::hint::inference_precision) << std::endl;
    std::cout << "Num of streams: " <<  compiled_model.get_property(ov::num_streams) << std::endl;
    // ov::CompiledModel compiled_model = core.compile_model("/home/nima/Documents/w/retina/openvino-test/b12_fp16_2/b12.xml", "AUTO");

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

    // 1) synchronous execution
    // infer_request.infer();

    // 2) Asynchronous with callback mechanism
    // infer_request.set_callback([&](std::exception_ptr ex_ptr){
    //     if(!ex_ptr){
    //         // inference is done and output data can be processed
    //         std::cout << "Inference is done!\n";
    //         end_exec = true;
    //     } else {
    //         std::cout << "Something went wrong!\n";
    //     }
    // });
    // infer_request.start_async();
    // 3) Asynchronous with wait
    infer_request.start_async();
    infer_request.wait();

    // std::string names = output_port.get_any_name();
    // std::cout << "Names: " << names << std::endl;
    std::vector<ov::Tensor> output_tensors;
    for(const auto& o_port : output_ports){
        // you can use get_output_tensor with the index of desireable output
        output_tensors.push_back(infer_request.get_tensor(o_port));
        std::cout << o_port.get_any_name() << " : " << output_tensors[output_tensors.size()-1].get_size() << std::endl;
    }
    
    // std::cout << "Inference done! \n";
    // float* output_buffer = output.data<float>();
    // ov::Shape shape = output.get_shape();
    // for(int i = 0; i < shape.size(); i++)
    //     std::cout << shape[i] << ", ";
    // std::cout << std::endl;
    // }

    // while(!end_exec){
    //     std::cout << "Waiting...\n";
    // }

   return 0;
}
