import torch_fl
import torch
dev = "flagos:0"
w = torch.randn(100, 16).to(dev)
idx = torch.randint(0, 100, (2, 8)).to(dev)
out = torch.nn.functional.embedding(idx, w)
a = torch.randn(4, 4).to(dev)
b = torch.randn(4, 4).to(dev)
c = (a + b).float().sum().item()
print("PROBE_OK embedding", tuple(out.shape), "add_sum", round(c, 3))
