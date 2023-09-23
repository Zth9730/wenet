import torch




class AdaNorm(torch.nn.Module):
    """Construct an AdaNorm object."""
    def __init__(self, size, eps):
        super().__init__()
        self.epsilon = eps
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)
        input = input - mean
        mean = input.mean(-1, keepdim=True)
        graNorm = (1 / 10 * (input - mean) / (std + self.epsilon)).detach()
        input_norm = (input - input * graNorm) / (std + self.epsilon)
        return input_norm
        
    
# def adanorm(inputs, epsilon=1e-8):
#     mean = input.mean(-1, keepdim=True)
#     std = input.std(-1, keepdim=True)
#     input = input - mean
#     mean = input.mean(-1, keepdim=True)
#     graNorm = (1 / 10 * (input - mean) / (std + epsilon)).detach()
#     input_norm = (input - input * graNorm) / (std + epsilon)
#     return input_norm