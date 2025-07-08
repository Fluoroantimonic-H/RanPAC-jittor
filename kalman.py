# class KalmanFilter(nn.Module):
#     def __init__(self, state_dim, meas_dim, batch_size, F=None, H=None, Q=None, R=None):
#         super().__init__()
#         self.state_dim = state_dim
#         self.meas_dim = meas_dim
# 
#         # 初始化参数
#         self.F = F if F is not None else torch.randn(meas_dim, state_dim)
#         self.H = H if H is not None else torch.randn(state_dim, meas_dim)
# 
#         self.P = torch.eye(self.state_dim).repeat(1, 1)
# 
# 
#     def update_P(self):
#         self.P = self.F @ self.P @ self.F.transpose(-1, -2)
# 
# 
#     def execute(self, x_pred, z, is_test=False):
#         x_pred = x_pred.to('cpu')
#         z = z.to('cpu')
# 
#         # 观测残差: y = z - H x_pred
#         z_pred = torch.einsum('...ij,...j->...i', self.H, x_pred)       # H: 768 x 10000, x_pred: 10000
#         y = z - z_pred
# 
#         # 残差协方差: S = H P_pred H^T + R
#         S = self.H @ self.P @ self.H.transpose(-1, -2)
# 
#         # 卡尔曼增益: K = P_pred H^T S^{-1}
#         K = self.P @ self.H.transpose(-1, -2) @ torch.linalg.inv(S)
# 
#         h = 0.01
#         # 状态更新: x = x_pred + K y
#         x = (1 - h) * x_pred + h * torch.einsum('...ij,...j->...i', K, y)
# 
#         if not is_test:
#             # 协方差更新: P = (I - K H) P_pred
#             I = torch.eye(self.meas_dim, device=x.device)
#             self.P = (I - K @ self.H) @ self.P
# 
#             # 确保协方差矩阵对称
#             self.P = 0.5 * (self.P + self.P.transpose(-1, -2))
# 
#         # return x_pred
#         return x

import jittor as jt
from jittor import nn

class KalmanFilter(nn.Module):
    def __init__(self, state_dim, meas_dim, batch_size, F=None, H=None, Q=None, R=None):
        super().__init__()
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # 初始化参数
        self.F = F if F is not None else jt.randn(meas_dim, state_dim)
        self.H = H if H is not None else jt.randn(state_dim, meas_dim)

        self.P = jt.init.eye(self.state_dim)

        # ones = jt.ones(self.state_dim)
        # self.P = jt.diag(ones)

    def update_P(self):
        self.P = self.F @ self.P @ self.F.transpose(-1, -2)

    def execute(self, x_pred, z, is_test=False):
        # 观测残差: y = z - H x_pred
        # z_pred = jt.einsum('...ij,...j->...i', self.H, x_pred)
        # z_pred =  x_pred @ self.H.transpose()
        # y = z - z_pred
        #
        # # 残差协方差: S = H P H^T + R
        # S = self.H @ self.P @ self.H.transpose(-1, -2)
        #
        # # 卡尔曼增益: K = P H^T S^{-1}
        # K = self.P @ self.H.transpose(-1, -2) @ jt.linalg.inv(S)
        #
        # h = 0.01
        # # 状态更新: x = (1-h)x_pred + h * K y
        # # x = (1 - h) * x_pred + h * jt.einsum('...ij,...j->...i', K, y)
        # x = (1 - h) * x_pred + h * (y @ K.transpose(-1, -2))
        #
        # if not is_test:
        #     I = jt.init.eye(self.meas_dim)
        #     self.P = (I - K @ self.H) @ self.P
        #
        #     # 保证对称
        #     self.P = 0.5 * (self.P + self.P.transpose(-1, -2))

        return x_pred


  