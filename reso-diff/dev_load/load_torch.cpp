#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

torch::Tensor filterOutput(torch::Tensor &eLibInputTensor, torch::Tensor &edgeList) {
    torch::Device device(torch::kCUDA, 0);
    std::vector<torch::jit::IValue> fInputTensorJit;
    fInputTensorJit.push_back(eLibInputTensor.to(device));
    fInputTensorJit.push_back(edgeList.to(device));

    
    torch::jit::script::Module f_model = torch::jit::load("/workspace/exatrkx-acts-demonstrator/models/smeared_hits/filter.pt", device);
    f_model.eval();
    return f_model.forward(fInputTensorJit).toTensor();
}

int main() {
    // 设置设备（如果您有 CUDA 支持的 GPU，可以改用 torch::kCUDA）
    torch::Device device(torch::kCPU);

    // 创建一个张量来存储加载的数据
    torch::Tensor eLibInputTensor_exatrkx;
    torch::load(eLibInputTensor_exatrkx, "/workspace/acts-aas/reso-diff/test_output/eLibInputTensor_exatrkx.pt");
    torch::Tensor edgeList_exatrkx; 
    torch::load(edgeList_exatrkx, "/workspace/acts-aas/reso-diff/test_output/edgeList_exatrkx.pt");
 
    torch::Tensor eLibInputTensor_acts;
    torch::load(eLibInputTensor_acts, "/workspace/acts-aas/reso-diff/test_output/eLibInputTensor_acts.pt");
    torch::Tensor edgeList_acts; 
    torch::load(edgeList_acts, "/workspace/acts-aas/reso-diff/test_output/edgeList_acts.pt");

    torch::Tensor fOutput = filterOutput(eLibInputTensor_acts, edgeList_acts);
    fOutput.squeeze_();
    fOutput.sigmoid_();

    std::cout << "fOutput sizes" << fOutput.sizes() << std::endl;
    auto sorted = fOutput.sort(/* dim= */0, /* descending= */true);
    auto sorted_output = std::get<0>(sorted);
    auto first_ten = sorted_output.slice(0, 0, 10); 
    auto last_ten = sorted_output.slice(0, -10, torch::indexing::None); 
    std::cout << "After Filtering, the top ten: \n" << first_ten << std::endl;
    std::cout << "After Filtering, the last ten: \n" << last_ten << std::endl;
    // std::cout << loadedTensor.sizes() << std::endl;
    // std::cout << loadedTensor2.sizes() << std::endl;

    // std::cout << loadedTensor.slice(0, 0, 10).slice(1, 0, 3) << std::endl;
    // std::cout << loadedTensor2.slice(0, 0, 10).slice(1, 0, 3) << std::endl;
    // // 加载张量
    // try {
    //     torch::load(loadedTensor, "/workspace/acts-aas/reso-diff/test_output/eLibInputTensor_exatrkx.pt");
    //     loadedTensor = loadedTensor.to(device);
    // } catch (const c10::Error& e) {
    //     std::cerr << "无法加载张量: " << e.what() << std::endl;
    //     return -1;
    // }

    // 输出张量以确认加载成功
    // std::cout << "Is equal? " << torch::equal(loadedTensor, loadedTensor2) << std::endl;

    return 0;
}
