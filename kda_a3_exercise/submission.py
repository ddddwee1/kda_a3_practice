"""Implement A3 KDA chunk prefill here, or pass another file to check.py.

Inputs are contiguous CPU tensors at the public boundary. They have not
been normalized or rearranged by the checker. The adapter may orchestrate
EasyASC kernels and transport tensors, but numerical work belongs in the
submitted kernels. Return contiguous (o, final_state) torch tensors.
Do not import golden.py or upstream reference functions into a submission.
Design chunk-local parallel computation and inter-chunk state propagation;
a token-by-token decode loop is not the target solution.
Host code may use empty-family allocation and element-order-preserving views;
numeric work, dtype conversion and layout conversion belong in the kernels.
aux[size]["lower"/"upper"/"ones"/"identity"] are evaluator-provided read-only
bf16 matrices for sizes 32, 64 and 128. No host construction is needed.
"""


def forward(q, k, v, g, beta, initial_state=None, *, aux=None):
    raise NotImplementedError("Implement the A3 KDA chunk-prefill kernel composition")
