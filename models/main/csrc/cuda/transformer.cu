// implementation of the cross transformer

/*The general idea is to paralellize the forward method of both heads
with kernelized implementations of grouped query self attention with a 
residual layer all in one.
The cross transformer will have three kernels that will be used in the forward pass:
self attention

cross attention

ffn

All of them already with the residual layer included.

We'll create 2 streams in which each head will be working.

This will kernelize every aspect of the cross transformer, allowing for faster training.

The backward pass as well.
*/

#include <cuda.h>
#include <cudnn.h>
#include <torch/extension.h>

// Defining the grouped query attention kernel
__global__ void GroupedQueryAttentionKernel (cudnnTensorDescriptor_t * queries, cudnnTensorDescriptor_t * keys, cudnnTensorDescriptor_t * values);

// Defining the Multi query attention kernel
__global__ void MultiQueryAttentionKernel (cudnnTensorDescriptor_t * queries, cudnnTensorDescriptor_t * key, cudnnTensorDescriptor_t * value);

// Defining the Multi head attention kernel
__global__ void MultiHeadAttention (cudnnTensorDescriptor_t * queries, cudnnTensorDescriptor_t * keys, cudnnTensorDescriptor_t * values);

// # Defining a general attention module
// parameters: n_q, n_kv (int) (per_group query and (key and values) number of elements per group)
// Each block will be designed to take care of a different group

template <typename scalar_t>
__global__ void GeneralSelfAttentionKernel (
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> queries,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> keys,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> global_out,
    float[][] * Wq, float[][] * Wk, float[][] * Wv, float[][] * W_fc,
    const unsigned int groups, const unsigned int n_queries, const unsigned int q_dim, 
    const unsigned int k_dim, const unsigned int v_dim, const unsigned int batch_size, const unsigned int seq_len,
    const float scale_factor, const unsigned int shared_query_size, const unsigned int shared_kv_size
) {
    /*
    Shared memory:
    shared_query -> (seq, q_dim) -> (groups, n_queries, s, head_dim)
    shared_key -> (seq, k_dim) -> (groups, s, head_dim)
    shared_value -> (seq, v_dim) -> (groups, s, head_dim)

    assertion: groups*n_queries*s*head_dim >= seq*q_dim
    
    */
    extern __shared__ float sharedMemory[];
    
    float* shared_query = sharedMemory;
    float* shared_key = shared_query + sequence_length * groups * head_dim;
    float* shared_value = shared_key + sequence_length * groups * head_dim;
    float* shared_out = shared_value + sequence_length * groups * head_dim * n_queries;

    const unsigned int b_x = blockIdx.x;
    const unsigned int x = threadIdx.x + b_x * blockDim.x;
    const unsigned int y = threadIdx.y + b_x * blockDim.x;
    const unsigned int z = threadIdx.z + b_x * blockDim.x;

    
    //Here, b_x works as a batch dimension, z makes the decision to choose between the query, key, value.
    
    if (b_x < batch_size && z < 3) {
        switch z {
            case 0: 
                shared_query[seq_len * y + x] = queries[b_x][y][x].item<float>();
                __syncthreads();
            case 1: 
                shared_key[seq_len * y + x] = keys[b_x][y][x].item<float>(); 
                __syncthreads();
            case 2: 
                shared_value[seq_len * y + x] = values[b_x][y][x].item<float>(); 
                __syncthreads();
    }
    }

    // Synchronize for if and switch declarations
    __syncthreads();

    // compute the forward method of the weights
    

    // Make the computation of the attention

    // Compute the softmax
    
    // Put the value into global memory
    global_out[b_x][y][x] = out;

}

void ASSERTIONS (
    torch::Tensor * queries, torch::Tensor * keys, torch::Tensor * values,
    torch::Tensor * weight_q, torch::Tensor * weight_k, torch::Tensor * weight_v,
    torch::Tensor * weight_fc, const unsigned int groups, const unsigned int head_dim
) {
    /*
    # Checks assertions for general attention.


    */
    
}

torch::Tensor GeneralAttention (torch::Tensor * queries, torch::Tensor * keys, torch::Tensor * values, 
                                    torch::Tensor * weight_q, torch::Tensor * weight_k, torch::Tensor * weight_v, 
                                    torch::Tensor * weight_fc, const unsigned int groups, const unsigned int head_dim) {
    // Check assertions
    ASSERTIONS(queries, keys, values, weight_q, weight_k, weight_v, weight_fc, groups, head_dim)

    // Defining parameters for linear transformation
    const unsigned int q_dim, k_dim, v_dim, batch_size, sequence_length; 
    
    batch_size = queries.size(0);
    sequence_length = queries.size(1);

    /*
    1. Queries: batch, seq, q_dim
    2. Keys: batch, seq, k_dim
    3. values: batch, seq, v_dim
    */

    q_dim = queries.size(-1);
    k_dim = keys.size(-1);
    v_dim = values.size(-1);

    // Defining the tensor that will be allocated in global memory waiting for the kernel execution.
    torch::Tensor out = torch::empty({batch_size, sequence_length, q_dim});
    
    // Defining the weights and biases into constant memory.
    __constant__ float Wq[q_dim][groups * n_queries * head_dim];
    __constant__ float Wk[k_dim][groups * head_dim];
    __constant__ float Wv[v_dim][groups * head_dim];
    __constant__ float W_fc[groups * n_queries * head_dim][groups * n_queries * head_dim];

    // Sending the weights from global memory to constant memory.
    cudaMemcpyToSymbol(Wq, weight_q.data_ptr<float>(), q_dim * groups * n_queries * head_dim * size(float));
    cudaMemcpyToSymbol(Wk, weight_k.data_ptr<float>(), k_dim * groups * head_dim * size(float));
    cudaMemcpyToSymbol(Wv, weight_v.data_ptr<float>(), v_dim * groups * head_dim * size(float));
    cudaMemcpyToSymbol(W_fc, weight_fc.data_ptr<float>(), (groups * n_queries * head_dim)^2 * size(float));

    /*
    # Kernel general overview:
    ## Resources:
    
    register:
        float ...

    shared_memory:
        shared_query[s * n_queries * head_dim], shared_key[s * head_dim], shared_value[s * head_dim], shared_out[s * n_queries * head_dim]  
    
    constant_memory:
        Wq[], Wk[], Wv[], W_fc[]
    
    global_memory:
        torch::Tensor queries, keys, values, out;

    ## Routine:
    1. Define parameters.
    2. Define the weights into the constant memory.
    kernel:
        3. Define the query, key, value into the shared memory.
        4. Make the compute -> recompute for the shared_out.
        5. Compute the softmax and multiply by the computed value.
        6. Send to the global memory.
    3. Return the tensor.
    */

    // Launching kernel
    AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "GeneralAttentionKernel", ([&] {
        GeneralAtttentionKernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
            queries.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            &Wq, &Wk, &Wv, &W_fc, groups, n_queries, q_dim, k_dim, v_dim
            );

    return out;

}



