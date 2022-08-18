#include <iostream>
#include <chrono>
#include <openvino/openvino.hpp>

#define BATCH_SIZE 1

int main(){
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model("/home/nima/Documents/w/retina/openvino-test/b12_fp16/b12.xml", "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    while(1){
        float* memory_ptr = new float[BATCH_SIZE*3*640*480];
        auto input_port = compiled_model.input();
        auto output_port = compiled_model.output();
        ov::Tensor input_tensor = ov::Tensor(input_port.get_element_type(), {BATCH_SIZE, 3, 640, 480}, memory_ptr);
        infer_request.set_input_tensor(input_tensor);

        auto start = std::chrono::high_resolution_clock::now();
        infer_request.start_async();
        infer_request.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("FPS: %f\n", (1000000000.0 * BATCH_SIZE) / elapsed.count());

        auto output = infer_request.get_tensor(output_port.get_any_name());
        float* output_buffer = output.data<float>();
        ov::Shape shape = output.get_shape();
        for(int i = 0; i < shape.size(); i++)
            std::cout << shape[i] << ", ";
        delete[] memory_ptr;
    }
}
