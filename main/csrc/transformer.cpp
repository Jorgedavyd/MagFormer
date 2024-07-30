#include <torch/extension.h>
#include <cudnn.h>
#include <cuda.h>

torch::Tensor MultiHeadAttention (torch::Tensor query, torch::Tensor key, torch::Tensor value);
torch::Tensor MultiQueryAttention (torch::Tensor query, torch::Tensor key, torch::Tensor value);
torch::Tensor GroupedQueryAttention (torch::Tensor query, torch::Tensor key, torch::Tensor value);
