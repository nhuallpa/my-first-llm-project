# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import tiktoken
from torch.utils.data import DataLoader
import os


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


def computing_attention_weight_for_x2():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)

    # Step 1: computed attention scores
    print(attn_scores_2)
    # normalization step
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())

    # Step 2: normalization step with softmax_naive
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())
    #  PyTorch implementation of softmax
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    # Step 3: calculating the context vector z(2)
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i
    print("Attention score for input 2:")
    print(context_vec_2)

    # calculating the context vector for ALL
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)

    print("- All attention score:")
    print(attn_scores)

    attn_scores = inputs @ inputs.T
    print("- All attention score with matrix multiplication:")
    print(attn_scores)

    print("- Normalize attention score")
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)

    # Step 4: verify all column all sum to 1:
    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    print("Row 2 sum:", row_2_sum)
    print("All row sums:", attn_weights.sum(dim=-1))

    print("- Compute all context vector with attention weights")
    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)

if __name__ == '__main__':
    computing_attention_weight_for_x2()
