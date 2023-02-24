#include <vector>
#include <torch/extension.h>

#include <iostream>

std::vector<torch::Tensor> relative_attention(
    torch::Tensor input,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor queries,
    torch::Tensor rel_pos_enc,
    torch::Tensor relative_indices,
    int heads,
    float scale
    ) {


  auto shape = input.sizes();
  auto b = shape[0];
  auto h = shape[2];
  auto w = shape[3];

  auto att = torch::matmul(queries, keys.transpose(-2,1));
//  auto indices = torch::cat(torch::tensor({-1}),relative_indices.expand({heads,-1}));
  auto t = relative_indices.expand({heads,-1}).contiguous();
  at::IntArrayRef indices =  at::Tensor(t.data<int>(), t.data<int>() + t.numel())
  rel_pos_enc = rel_pos_enc.view(indices).unflatten(-1,h*w,h*w);

  att = att * scale * rel_pos_enc;


  auto out  = torch::matmul(values,torch::softmax(att,-2)).view({b,-1,h,w});

  return {out};
}
