"""GTrXL core: relative attention, detached segment memory and GRU gates.

Architecture adapted from Parisotto et al. (ICML 2020), section 3. This module
is a value predictor for trace patterning, not a reproduction of V-MPO/DMLab.
"""
import math

import torch
from torch import nn


class GRUGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.wr = nn.Linear(width, width, bias=False)
        self.ur = nn.Linear(width, width, bias=False)
        self.wz = nn.Linear(width, width, bias=False)
        self.uz = nn.Linear(width, width, bias=False)
        self.wg = nn.Linear(width, width, bias=False)
        self.ug = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.full((width,), 2.0))

    def forward(self, x, y):
        reset = torch.sigmoid(self.wr(y) + self.ur(x))
        update = torch.sigmoid(self.wz(y) + self.uz(x) - self.bias)
        candidate = torch.tanh(self.wg(y) + self.ug(reset * x))
        return (1 - update) * x + update * candidate


class RelativeAttention(nn.Module):
    def __init__(self, width, heads, memory):
        super().__init__()
        self.heads = heads
        self.memory = memory
        self.depth = width // heads
        self.width = width
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.relative = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.content_bias = nn.Parameter(torch.zeros(heads, self.depth))
        self.position_bias = nn.Parameter(torch.zeros(heads, self.depth))
        self.register_buffer('frequency', torch.exp(
            torch.arange(0, width, 2).float() * (-math.log(10000) / width)))

    def forward(self, x, memory, norm):
        batch, length, _ = x.shape
        previous = memory.shape[1]
        combined = norm(torch.cat((memory.detach(), x), dim=1))
        q = self.query(combined[:, previous:]).reshape(batch, length, self.heads, self.depth)
        k = self.key(combined).reshape(batch, previous + length, self.heads, self.depth)
        v = self.value(combined).reshape(batch, previous + length, self.heads, self.depth)
        # Explicit relative distances implement the Transformer-XL score without
        # the relative-shift optimization; no absolute segment position is used.
        distance = (previous + torch.arange(length, device=x.device))[:, None] - torch.arange(
            previous + length, device=x.device)[None, :]
        angle = distance.float()[..., None] * self.frequency
        position = torch.zeros(length, previous + length, self.width, device=x.device)
        position[..., 0::2] = angle.sin()
        position[..., 1::2] = angle[..., :position[..., 1::2].shape[-1]].cos()
        relative = self.relative(position).reshape(length, previous + length, self.heads, self.depth)
        content = torch.einsum('bqhd,bkhd->bhqk', q + self.content_bias, k)
        positional = torch.einsum('bqhd,qkhd->bhqk', q + self.position_bias, relative)
        score = (content + positional) / math.sqrt(self.depth)
        # Use the same sliding horizon for segment replay and one-step inference.
        excluded = (distance < 0) | (distance > self.memory)
        score = score.masked_fill(excluded[None, None], float('-inf'))
        attended = torch.einsum('bhqk,bkhd->bqhd', score.softmax(-1), v)
        return self.output(attended.reshape(batch, length, self.width))


class GTrXLBlock(nn.Module):
    def __init__(self, width, heads, memory):
        super().__init__()
        self.attention = RelativeAttention(width, heads, memory)
        self.attention_norm = nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        self.attention_gate = GRUGate(width)
        self.ff_gate = GRUGate(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(),
                                nn.Linear(4 * width, width))

    def forward(self, x, memory):
        y = self.attention_gate(x, self.attention(x, memory, self.attention_norm).relu())
        return self.ff_gate(y, self.ff(self.ff_norm(y)).relu())


class GTrXL(nn.Module):
    def __init__(self, width, heads, layers, memory, inputs=12):
        super().__init__()
        self.width = width
        self.memory = memory
        self.embed = nn.Linear(inputs, width)
        self.blocks = nn.ModuleList([GTrXLBlock(width, heads, memory) for _ in range(layers)])
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def initial_state(self, device):
        return [torch.zeros(1, 0, self.width, device=device) for _ in self.blocks]

    def forward(self, x, state=None):
        if state is None:
            state = self.initial_state(x.device)
        feature = self.embed(x)
        new_memory = []
        for block, memory in zip(self.blocks, state):
            new_memory.append(torch.cat((memory, feature), dim=1)[:, -self.memory:].detach())
            feature = block(feature, memory)
        return self.head(feature).squeeze(-1), new_memory
