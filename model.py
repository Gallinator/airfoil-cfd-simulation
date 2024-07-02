import torch.optim
from torch import nn


class Model(nn.Module):
    def __init__(self, grid_size: int, in_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = nn.Parameter(torch.ones(1, requires_grad=True))
        self.mu = nn.Parameter(torch.ones(1, requires_grad=True))
        self.seq = nn.Sequential(
            # Landmark + grids + alpha
            nn.Linear(in_size + grid_size * 2 + 1, 2048),
            nn.Tanh(),
            nn.Linear(2048, 4096),
            nn.Tanh(),
            nn.Linear(4096, 8192),
            nn.Tanh())
        self.v = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())
        self.u = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())
        self.p = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())
        self.uu = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())
        self.uv = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())
        self.vv = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())

    def forward(self, alpha, grid_x, grid_y, landmarks) -> tuple:
        in_data = torch.cat((alpha, landmarks, grid_x, grid_y), 1)
        x = self.seq(in_data)
        return self.v(x), self.u(x), self.p(x), self.uu(x), self.vv(x), self.uv(x)

    def get_rans_diffs(self, alpha, grid_x: torch.Tensor, grid_y: torch.Tensor, landmarks: torch.Tensor):
        u, v, p, uu, vv, uv = self.forward(alpha, grid_x, grid_y, landmarks)
        d_u_x = torch.autograd.grad(u, grid_x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_v_y = torch.autograd.grad(v, grid_y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_u_y = torch.autograd.grad(u, grid_y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_u_y2 = torch.autograd.grad(d_u_y, grid_y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_u_x2 = torch.autograd.grad(d_u_x, grid_x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_p_x = torch.autograd.grad(v, grid_x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_uu_x = torch.autograd.grad(uu, grid_x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_uv_y = torch.autograd.grad(uv, grid_y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_v_y2 = torch.autograd.grad(d_v_y, grid_y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_v_x = torch.autograd.grad(u, grid_x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_v_x2 = torch.autograd.grad(d_v_x, grid_x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_p_y = torch.autograd.grad(v, grid_y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_vv_y = torch.autograd.grad(vv, grid_y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        d_uv_x = torch.autograd.grad(uv, grid_x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        return (self.rans_continuity_diff(d_v_y, d_u_x),
                self.rans_momentum_x(u, v, d_u_x, d_u_y, d_p_x, d_u_x2, d_u_y2, d_uu_x, d_uv_y),
                self.rans_momentum_y(u, v, d_v_x, d_v_y, d_p_y, d_v_x2, d_v_y2, d_uv_x, d_vv_y))

    def rans_continuity_diff(self, d_v_y, d_u_x):
        return d_v_y + d_u_x

    def rans_momentum_x(self, u, v, d_u_x, d_u_y, d_p_x, d_u_x2, d_u_y2, d_uu_x, d_uv_y):
        return u * d_u_x + v * d_u_y + d_p_x * (1.0 / self.rho) - self.mu * (d_u_x2 + d_u_y2) + d_uu_x + d_uv_y

    def rans_momentum_y(self, u, v, d_v_x, d_v_y, d_p_y, d_v_x2, d_v_y2, d_uv_x, d_vv_y):
        return u * d_v_x + v * d_v_y + d_p_y * (1 / self.rho) - self.mu * (d_v_x2 + d_v_y2) + d_uv_x + d_vv_y
