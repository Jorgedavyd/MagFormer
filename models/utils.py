
# General utils
from torch import nn
import torch
import torch.nn.functional as F
from datetime import timedelta
from math import sqrt

"""
Deep Neural Network
"""
# Base Deep Neural Network
def  SingularLayer(input_size, output, activation):
	out = nn.Sequential(
		nn.Linear(input_size, output),
		activation()
	)
	return out

class DeepNeuralNetwork(nn.Module):
	def __init__(self, input_size, output_size, architecture,hidden_activations, out_activation=None):
		super(DeepNeuralNetwork, self).__init__()
		assert (len(hidden_activations) == len(architecture)), 'Must have activation for each layer, if not, put None as activation'
		self.overall_structure = nn.Sequential()
		#Model input and hidden layer
		for num, (output, activation) in enumerate(zip(architecture, hidden_activations)):
			self.overall_structure.add_module(name = f'layer_{num+1}', module = SingularLayer(input_size, output, activation))
			input_size = output

		#Model output layer
		self.output_layer = nn.Sequential(nn.Linear(input_size, output_size))
		if out_activation is not None:
			self.output_layer.add_module(name = 'fc_layer', module = out_activation)
	def forward(self, xb):
		out = self.overall_structure(xb)
		out = self.output_layer(out)
		return out

# Utilities for the transformer architecture
"""
Root Mean Squared Normalization (https://arxiv.org/pdf/1910.07467.pdf)

from summary:

Extensive experiments on several tasks using diverse network architectures 
show that RMSNorm achieves comparable performanceagainst LayerNorm but reduces 
the running time by 7%-64% on different models
"""
class RootMeanSquaredNormalization(nn.Module):
	def __init__(
			self,
			dim: int,
			eps: float = 1e-6
	):
		super(RootMeanSquaredNormalization, self).__init__()
		self.eps = eps
		self.g_i = nn.Parameter(torch.ones(dim))

	def forward(
			self,
			x_n
	):
		# RMSN(x_i) = g_i*(x_i/(RMSE(x_i) + eps))
		return self.g_i*(x_n * torch.rsqrt(x_n.pow(2).mean(-1, keepdim = True) + self.eps))

"""
Attention Module:

FlashAttention Implementation: (https://arxiv.org/pdf/2205.14135.pdf)

attention: Normal attention

"""   
class Attention(nn.Module):
	def __init__(
			self,
	):
		super().__init__()
	@staticmethod
	def flashattention(queries, keys,values, mask = None, dropout = 0, scale: float = 1):
		with torch.backends.cuda.sdp_kernel(
			enable_flash=True, 
			enable_math=False, 
			enable_mem_efficient=False
		):
			out = F.scaled_dot_product_attention(
				queries,
				keys,
				values,
				mask,
				dropout,
				scale = scale
			)
		return out
	
	@staticmethod
	def attention(queries, keys, values, mask = None, dropout = 0, scale: float = 1):
		F.scaled_dot_product_attention(queries, keys, values, mask, dropout, scale = scale)

"""
Pre normalized residual connection
"""

class PreNormResidualConnection(nn.Module):
  def __init__(
		  self,
		  norm,
		  dropout: float = 0.,
  ):
	  super().__init__()
	  self.dropout = nn.Dropout(dropout)
	  self.layer_norm = norm
  def forward(self, out, sublayer):
	  return out + self.dropout(sublayer(self.layer_norm(out)))


"""
RNN build in modules
LSTM: Long short Term Memory
GRU:Gated Recurrent Unit
"""
	
class LSTM(nn.LSTM):
	def __init__(self, *args, **kwargs):
		super(LSTM, self).__init__(*args, **kwargs)
	def forward(self, x, hn = None, cn = None):
		out, (_,_) = super().forward(out, (hn, cn))
		return out

class GRU(nn.GRU):
	def __init__(self, *args, **kwargs):
		super(GRU, self).__init__(*args, **kwargs)
	def forward(self, x, hn = None):
		out, _ = super().forward(x, hn)
		return out

"""
Monte Carlo Sampling with fc layer
"""
class MonteCarloFC(nn.Module):
	def __init__(
			self,
			fc_layer,
			dropout: float = 0.2,
			n_sampling: int = 5
	):
		super().__init__()
		self.n_sampling = n_sampling
		self.fc = fc_layer
		self.dropout = lambda x: F.dropout(x, dropout, True)
	def forward(self, x):
		outputs = []
		for _ in range(self.n_sampling):
			x = self.dropout(x)
			outputs.append(self.fc(x))
		out = torch.mean(torch.stack(outputs, dim = 0), dim = 0)
		return out
	
"""
Rotary Positional Encoder [source](https://arxiv.org/pdf/2104.09864.pdf)

RoFormer (applied to transformer architecture to enhance performance)
Llama (applied to keys and queries before MultiQueryAttention)

I'll probably make my own implementation for manual headed attention with
directed variable attention analysis
"""
class RotaryPositionalEncoding(nn.Module):
	def __init__(
			self,
			d_model: int,
			seq_len: int,
			theta: int = 10000,
	):
		super(RotaryPositionalEncoding, self).__init__()
		"""
		Creating rotary transformation matrix

		Given the embedding space V, a linear space in R^n , there is the finite sucession {x_n}_{n=1}^N where N 
		is the number of samples onto a single sequence (sequence_length), where implicitly x_i /from V for i .
		We want to introduce a linear transformation onto this so it makes a learnable rotation into the 
		embedding space 
		"""
		#embedding size must be even
		assert (d_model% 2 == 0), 'd_model must be div by 2'
		#Create all thetas (theta_i) for i in range(0,ndim/2) theta^(-(2i)/ndim)
		theta_j = torch.tensor([1 / theta**((2*i)/d_model) for i in range(d_model/2)])
		#creates absolute position based on seq_len
		m_i = torch.arange(seq_len, )
		#creates (m_i,theta_j) matrix 
		function_inputs = torch.outer(m_i, theta_j)
		#translated into polar
		self.rotary_transformation = torch.polar(torch.ones_like(function_inputs), function_inputs).unsqueeze(0).unsqueeze(2)

	def forward(
			self,
			x_n: torch.tensor,
	):
		#resampling input from embedding space into (batch_size, seq_len, embedding_size/2) 
		#(B, N, d_model) -> (B,N,d_model/2) polar transformation
		resampled_input = torch.view_as_complex(x_n.float().reshape(*x_n.shape[:-1], -1, 2))
		# F: ((1, N, 1, d_model/2), (B,N,H,d_model/2)) -> (B,N,H,d_model/2)
		rotated_batch = self.rotary_transformation * resampled_input
		# (B,N,H,d_model/2) -> (B,N,H, d_model/2, 2)
		rot_out = torch.view_as_real(rotated_batch)
		# (B,N,H,d_model/2, 2) -> (B,N,H,d_model)
		rot_out = rot_out.reshape(*x_n.shape)
		return rot_out.type_as()
	
"""
Dn Positional Encoding
Adds the first n-degree derivatives of the samples, creates lineal time dependence.
"""
class DnPositionalEncoding(nn.Module):
	def __init__(
			self,
			delta_t: timedelta,
			degree: int = 1,
			edge_order: int = 1
	):
		super().__init__()
		self.delta_t = delta_t.total_seconds()
		self.degree = degree
		self.edge_order = edge_order

	def forward(self, x_n):
		out = x_n.clone()
		for _ in range(1,self.degree+1):
			x_n = torch.gradient(x_n, spacing = (self.delta_t, ), dim = -1, edge_order=self.edge_order)
			out += x_n
		return out
	
"""
MLPs
PosWiseFFN
SeqWiseFFN
"""

class PosWiseFFN(nn.Module):
	def __init__(
			self,
			d_model: int,
			hidden_dim: int,
			activation: torch.nn = nn.SiLU,

	):
		super(PosWiseFFN, self).__init__()
		self.activation = activation
		self.w_1 = nn.Linear(d_model, hidden_dim)
		self.v = nn.Linear(d_model, hidden_dim)
		self.w_2 = nn.Linear(hidden_dim, d_model)
	def forward(self, x_n):
		#(B,N,d_model) -> (B,N,hidden_dim) -> (B,N,d_model)
		return self.w_2(self.activation(self.w_1(x_n)) + self.v(x_n))

class SeqWiseMLP(nn.Module):
	def __init__(
			self,
			seq_len: int,
			hidden_dim: int,
			activation = nn.SiLU,
	):
		super().__init__()
		self.activation = activation
		self.w_1 = nn.Linear(seq_len, hidden_dim)
		self.v = nn.Linear(seq_len, hidden_dim)
		self.w_2 = nn.Linear(hidden_dim, seq_len)
	def forward(
			self,
			x_n: torch.Tensor,
	):
		super().__init__()
		x_n = x_n.tranpose(-1,-2)
		return self.w_2(self.activation(self.w_1(x_n)) + self.v(x_n)).transpose(-1,-2)

# Moded Time series attention

class ModedTimeSeriesAttention(nn.Module):
	def __init__(
		self,
		d_model: int,
		seq_len: int,
		num_heads: int,
		kdim: int,
		vdim: int
	):
		super().__init__()
		"""
		attention = (concat({head_i}_{i=1}^{num_heads})W_fc)^T
		head_i = softmax(\frac{Q_{i}^{T}K}{\sqrt(d_{k})})V^{T}
		"""
		# Premises
		assert (vdim%num_heads == 0), 'k = \frac{hiden_dim}{num_heads} | k \in \mathbb(Z)^{+}'
		assert (seq_len%num_heads == 0), 'k = \frac{hiden_dim}{num_heads} | k \in \mathbb(Z)^{+}'
	
		self.num_heads = num_heads
		self.d_model = d_model
		self.vdim = vdim
		self.kdim = kdim
		self.head_dim = self.vdim//num_heads
		self.time_dim = self.vdim//seq_len
		
		# Projections into hidden
		self.W_q = nn.Linear(d_model, vdim)
		self.W_k = nn.Linear(kdim, vdim)
		self.W_v = nn.Linear(vdim, vdim)
		
		# fc projection
		self.W_fc = nn.Linear(self.head_dim*num_heads, d_model)
	
	def forward(self, queries, keys, values, mask = None):
		#Dimensions
		B,S,_ = queries.shape

		#Linear projection
		Q = self.W_q(queries)
		K = self.W_k(keys)
		V = self.W_v(values)

		# (B,S,vdim) ->(B,S,num_heads, head_dim)

		# Q = Q^T, K = K^T, V = V^T so that attention = soft(Q^T K / sqrt(d_k))V^T -> (B, num_heads, head_dim, seq_len)
		Q_time = Q.transpose(-1,-2).view(B,self.vdim, self.num_heads, self.time_dim).transpose(1,2)
		K_time = K.transpose(-1,-2).view(B,self.vdim, self.num_heads, self.time_dim).transpose(1,2)
		V_time = V.transpose(-1,-2).view(B,self.vdim, self.num_heads, self.time_dim).transpose(1,2)
		
		Q = Q.view(B,S,self.num_heads, self.head_dim).tranpose(1,2)
		K = K.view(B,S,self.num_heads, self.head_dim).tranpose(1,2)
		V = V.view(B,S,self.num_heads, self.head_dim).tranpose(1,2)

		# (B,time_heads, vdim, s/time_heads)-> (B,time_heads, vdim, s/time_heads) -> (B,S,num_heads, head_dim)
		out_time = Attention.flashattention(Q_time, K_time, V_time, scale = sqrt(self.seq_len)).transpose(1,2).view(B, self.vdim, self.seq_len).transpose(-1, -2)

		# (B,S,num_heads, head_dim) -> (B,S,hidden_dim)
		stan_out = Attention.flashattention(Q,K,V, mask, scale = sqrt(self.d_model)).view(B,S,self.num_heads*self.head_dim)

		out = out_time + stan_out

		return self.W_fc(out)