#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/extension.h>

namespace rtp_llm {

// Torch-native debug string utilities (no Buffer dependency)
inline std::string tensorDebugString(const torch::Tensor& t) {
    if (!t.defined())
        return "(undefined)";
    std::string s = "Tensor(dtype=" + std::string(c10::toString(t.scalar_type())) + ", shape=[";
    for (int64_t i = 0; i < t.dim(); i++) {
        if (i)
            s += ", ";
        s += std::to_string(t.size(i));
    }
    s += "], device=" + t.device().str() + ")";
    return s;
}

template<typename T>
std::string tensorDebugStringWithData(const torch::Tensor& t, size_t count = 0) {
    if (!t.defined())
        return "(undefined)";
    auto meta = tensorDebugString(t);
    if (t.is_cuda())
        return meta + ", Device tensor data can NOT be dumped";
    auto cpu_t = t.contiguous();
    auto base  = cpu_t.data_ptr<T>();
    auto total = static_cast<size_t>(cpu_t.numel());
    if (count == 0)
        count = total;
    auto               data_size = std::min(count, total);
    std::ostringstream oss;
    for (size_t i = 0; i < data_size; i++)
        oss << base[i] << ", ";
    if (data_size != total) {
        oss << "...... ";
        for (size_t i = total - data_size; i < total; i++)
            oss << base[i] << ", ";
    }
    return meta + ", Data(" + oss.str() + ")";
}

inline void printTensorInfo(const std::string& name, const torch::Tensor& tensor, int max_print_size = 20) {
    std::cout << "  " << name << ": defined=" << tensor.defined();
    if (tensor.defined()) {
        std::cout << ", shape=[";
        for (int i = 0; i < tensor.dim(); i++) {
            std::cout << tensor.size(i);
            if (i < tensor.dim() - 1)
                std::cout << ", ";
        }
        std::cout << "], is_cuda=" << tensor.is_cuda();
        if (!tensor.is_cuda()) {
            std::cout << ", is_pinned=" << tensor.is_pinned();
        }
        if (tensor.numel() > 0) {
            auto cpu_tensor = tensor.cpu();
            int  print_size = std::min(static_cast<int>(cpu_tensor.numel()), max_print_size);
            std::cout << ", data=[";
            auto dtype = cpu_tensor.scalar_type();
            for (int i = 0; i < print_size; i++) {
                if (dtype == torch::kInt32 || dtype == torch::kInt) {
                    std::cout << cpu_tensor.data_ptr<int>()[i];
                } else if (dtype == torch::kInt64 || dtype == torch::kLong) {
                    std::cout << cpu_tensor.data_ptr<int64_t>()[i];
                } else if (dtype == torch::kFloat32 || dtype == torch::kFloat) {
                    std::cout << cpu_tensor.data_ptr<float>()[i];
                } else if (dtype == torch::kFloat16 || dtype == torch::kHalf) {
                    std::cout << static_cast<float>(cpu_tensor.data_ptr<at::Half>()[i]);
                } else {
                    std::cout << "?";
                }
                if (i < print_size - 1)
                    std::cout << ", ";
            }
            if (cpu_tensor.numel() > print_size)
                std::cout << ", ...";
            std::cout << "]";
        }
    }
    std::cout << std::endl;
}

}  // namespace rtp_llm
