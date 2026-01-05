import jittor as jt
from jittor import nn

class ResBlock(nn.Module):
    """
    A Residual Block module.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, 0.0)
        self.act = nn.SiLU()

    def execute(self, x):
        return x + self.act(self.linear(x))

class MedusaHead(nn.Module):
    """
    The Medusa Heads implemented in Jittor.
    Only contains the extra heads, not the base model.
    """
    def __init__(self, hidden_size, vocab_size, medusa_num_heads, medusa_num_layers):
        super().__init__()
        self.medusa_num_heads = medusa_num_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                *[ResBlock(hidden_size) for _ in range(medusa_num_layers)],
                nn.Linear(hidden_size, vocab_size, bias=False)
            )
            for _ in range(medusa_num_heads)
        ])

    def execute(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] (Jittor Var)
        Returns:
            List of [batch_size, seq_len, vocab_size] (Jittor Var)
        """
        results = []
        for head in self.heads:
            results.append(head(hidden_states))
        return results
