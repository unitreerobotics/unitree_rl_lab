// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <filesystem>
#include <iostream>
#include <mutex>
#ifdef USE_EXECUTORCH
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#endif

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Get output shape
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());

        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};

#ifdef USE_EXECUTORCH
class ExecuTorchRunner : public Algorithms
{
public:
    ExecuTorchRunner(std::string model_path)
    : module(model_path)
    {
        // Load and check the program's structure now: most malformed files then fail here with an error
        // instead of crashing, and the first step of the control loop does not pay for loading.
        if (module.load(executorch::runtime::Program::Verification::InternalConsistency) != executorch::runtime::Error::Ok
            || module.load_forward() != executorch::runtime::Error::Ok) {
            throw std::runtime_error("Failed to load " + model_path);
        }

        // The exporter stores the input names in the program, in the order forward() takes them.
        auto names = module.execute("input_names");
        if (!names.ok()) {
            throw std::runtime_error("Failed to read input_names from " + model_path);
        }
        auto method_meta = module.method_meta("forward");
        if (names->size() != method_meta->num_inputs()) {
            throw std::runtime_error("input_names in " + model_path + " must have one name per input");
        }
        for (const auto& name : *names) {
            if (!name.isString()) {
                throw std::runtime_error("input_names in " + model_path + " must be strings");
            }
            input_names.emplace_back(name.toString());
        }

        for (size_t i = 0; i < method_meta->num_inputs(); ++i) {
            auto input = method_meta->input_tensor_meta(i);
            if (!input.ok()) {
                throw std::runtime_error("Input " + input_names[i] + " of " + model_path + " must be a tensor");
            }
            input_shapes.emplace_back(input->sizes().begin(), input->sizes().end());
            size_t size = 1;
            for (auto dim : input->sizes()) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        auto output = method_meta->output_tensor_meta(0);
        if (!output.ok() || output->sizes().size() != 2 || output->scalar_type() != executorch::aten::ScalarType::Float) {
            throw std::runtime_error("The first output of " + model_path + " must be a 2-D float tensor");
        }
        action.resize(output->sizes()[1]);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        std::vector<executorch::extension::TensorPtr> input_tensors;
        std::vector<executorch::runtime::EValue> inputs;
        for (size_t i = 0; i < input_names.size(); ++i) {
            auto it = obs.find(input_names[i]);
            if (it == obs.end()) {
                throw std::runtime_error("Input name " + input_names[i] + " not found in observations.");
            }
            if (it->second.size() != input_sizes[i]) {
                throw std::runtime_error("Input " + input_names[i] + " has " + std::to_string(it->second.size())
                                         + " values, the policy expects " + std::to_string(input_sizes[i]));
            }
            input_tensors.push_back(executorch::extension::from_blob(it->second.data(), input_shapes[i]));
            inputs.emplace_back(*input_tensors.back());
        }

        auto outputs = module.forward(inputs);
        if (!outputs.ok()) {
            throw std::runtime_error("Failed to run the policy.");
        }
        const auto& output = outputs->at(0).toTensor();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), output.const_data_ptr<float>(), action.size() * sizeof(float));
        return action;
    }

private:
    executorch::extension::Module module;
    std::vector<std::string> input_names;
    std::vector<std::vector<executorch::aten::SizesType>> input_shapes;
    std::vector<size_t> input_sizes;
};
#endif

inline std::unique_ptr<Algorithms> make_runner(const std::filesystem::path& exported_dir)
{
#ifdef USE_EXECUTORCH
    return std::make_unique<ExecuTorchRunner>(exported_dir / "policy.pte");
#else
    return std::make_unique<OrtRunner>(exported_dir / "policy.onnx");
#endif
}
};