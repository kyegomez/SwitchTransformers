import torch
from switch_transformers.model import SwitchTransformer

# Generate a random tensor of shape (1, 10) with values between 0 and 100
x = torch.randint(0, 100, (1, 10))

# Create an instance of the SwitchTransformer model
# num_tokens: the number of tokens in the input sequence
# dim: the dimensionality of the model
# heads: the number of attention heads
# dim_head: the dimensionality of each attention head
model = SwitchTransformer(
    num_tokens=100, dim=512, heads=8, dim_head=64
)

# Pass the input tensor through the model
out = model(x)

# Print the shape of the output tensor
print(out.shape)
