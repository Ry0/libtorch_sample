#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    // モデルのロード
    std::string model_path = "my_model.pt";
    torch::jit::script::Module module;
    try {
        // モデルをロード
        module = torch::jit::load(model_path);
        module.eval();  // 推論モードに設定
    }
    catch (const c10::Error& e) {
        std::cerr << "モデルのロードに失敗しました: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "モデルが正常にロードされました!" << std::endl;

    // 入力テンソルの作成
    torch::Tensor x = torch::ones({1, 10});
    torch::Tensor feature = torch::ones({1, 5});

    // 1. featureを使用するケース
    try {
        std::vector<torch::jit::IValue> inputs_with_feature;
        inputs_with_feature.push_back(x);
        inputs_with_feature.push_back(feature);
        
        // torch::Tensor output = module.forward(inputs_with_feature).toTensor();
        auto output = module.run_method("forward_inference", x, feature).toTensor();
        std::cout << "With feature output: " << output << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "推論実行エラー (with feature): " << e.what() << std::endl;
    }

    // 2. featureなしのケース (None/nullopt)
    try {
        std::vector<torch::jit::IValue> inputs_without_feature;

        inputs_without_feature.push_back(x);
        inputs_without_feature.push_back(torch::nullopt);
        
        // torch::Tensor output = module.forward(inputs_without_feature).toTensor();
        auto output = module.run_method("forward_inference", x, torch::nullopt).toTensor();

        std::cout << "Without feature output: " << output << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "推論実行エラー (without feature): " << e.what() << std::endl;
    }

    std::cout << "処理が完了しました" << std::endl;
    return 0;
}