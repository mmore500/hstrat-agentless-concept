import numpy as np
import torch

from ._field_func import field_func
from ._ker_c import ker_c
from ._roll_n import roll_n


# Lenia Step FFT version (faster)
class LeniaStepFFTC(torch.nn.Module):
    """Module pytorch that computes one Lenia Step with the fft version"""

    def __init__(
        self,
        C,
        R,
        T,
        c0,
        c1,
        r,
        rk,
        b,
        w,
        h,
        m,
        s,
        gn,
        is_soft_clip=False,
        SX=256,
        SY=256,
        speed_x=0,
        speed_y=0,
        device="cpu",
    ):
        torch.nn.Module.__init__(self)

        self.register_buffer("R", R)
        self.register_buffer("T", T)
        self.register_buffer("c0", c0)
        self.register_buffer("c1", c1)
        # self.register_buffer('r', r)
        self.register_parameter("r", torch.nn.Parameter(r))
        self.register_parameter("rk", torch.nn.Parameter(rk))
        self.register_parameter("b", torch.nn.Parameter(b))
        self.register_parameter("w", torch.nn.Parameter(w))
        self.register_parameter("h", torch.nn.Parameter(h))
        self.register_parameter("m", torch.nn.Parameter(m))
        self.register_parameter("s", torch.nn.Parameter(s))
        self.speed_x = speed_x
        self.speed_y = speed_y

        self.gn = 1
        self.nb_k = c0.shape[0]

        self.SX = SX
        self.SY = SY

        self.is_soft_clip = is_soft_clip
        self.C = C

        self.device = device
        self.to(device)
        self.kernels = torch.zeros((self.nb_k, self.SX, self.SY, 2)).to(
            self.device
        )

        self.compute_kernel()
        self.compute_kernel_env()

    def compute_kernel_env(self):
        """computes the kernel and the kernel FFT of the environnement from the parameters"""
        x = torch.arange(self.SX).to(self.device)
        y = torch.arange(self.SY).to(self.device)
        xx = x.view(-1, 1).repeat(1, self.SY)
        yy = y.repeat(self.SX, 1)
        X = (xx - int(self.SX / 2)).float()
        Y = (yy - int(self.SY / 2)).float()
        D = torch.sqrt(X**2 + Y**2) / (4)
        kernel = torch.sigmoid(-(D - 1) * 10) * ker_c(
            D,
            torch.tensor(np.array([0, 0, 0])).to(self.device),
            torch.tensor(np.array([0.5, 0.1, 0.1])).to(self.device),
            torch.tensor(np.array([1, 0, 0])).to(self.device),
        )
        kernel_sum = torch.sum(kernel)
        kernel_norm = kernel / kernel_sum
        # kernel_FFT = torch.rfft(kernel_norm, signal_ndim=2, onesided=False).to(self.device)
        kernel_FFT = torch.fft.rfftn(kernel_norm, dim=(0, 1)).to(self.device)

        self.kernel_wall = kernel_FFT

    def compute_kernel(self):
        """computes the kernel and the kernel FFT of the learnable channels from the parameters"""
        x = torch.arange(self.SX).to(self.device)
        y = torch.arange(self.SY).to(self.device)
        xx = x.view(-1, 1).repeat(1, self.SY)
        yy = y.repeat(self.SX, 1)
        X = (xx - int(self.SX / 2)).float()
        Y = (yy - int(self.SY / 2)).float()
        self.kernels = torch.zeros((self.nb_k, self.SX, self.SY // 2 + 1)).to(
            self.device
        )

        for i in range(self.nb_k):
            # distance to center in normalized space
            D = torch.sqrt(X**2 + Y**2) / ((self.R + 15) * self.r[i])

            kernel = torch.sigmoid(-(D - 1) * 10) * ker_c(
                D, self.rk[i], self.w[i], self.b[i]
            )
            kernel_sum = torch.sum(kernel)

            # normalization of the kernel
            kernel_norm = kernel / kernel_sum
            # plt.imshow(kernel_norm[0,0].detach().cpu()*100)
            # plt.show()

            # fft of the kernel
            # kernel_FFT = torch.rfft(kernel_norm, signal_ndim=2, onesided=False).to(self.device)
            kernel_FFT = torch.fft.rfftn(kernel_norm, dim=(0, 1)).to(
                self.device
            )

            self.kernels[i] = kernel_FFT

    def forward(self, input):
        input[:, :, :, 1] = torch.roll(
            input[:, :, :, 1], [self.speed_y, self.speed_x], [1, 2]
        )
        self.D = torch.zeros(input.shape).to(self.device)
        self.Dn = torch.zeros(self.C)

        # world_FFT = [torch.rfft(input[:,:,:,i], signal_ndim=2, onesided=False) for i in range(self.C)]
        world_FFT = [
            torch.fft.rfftn(input[:, :, :, i], dim=(1, 2))
            for i in range(self.C)
        ]

        ## speed up of the update for 1 channel creature by multiplying by all the kernel FFT

        # channel 0 is the learnable channel
        world_FFT_c = world_FFT[0]
        # multiply the FFT of the world and the kernels
        potential_FFT = self.kernels * world_FFT_c
        # ifft + realignself.SY//2+1
        potential = torch.fft.irfftn(potential_FFT, dim=(1, 2))
        potential = roll_n(potential, 2, potential.size(2) // 2)
        potential = roll_n(potential, 1, potential.size(1) // 2)
        # growth function
        gfunc = field_func[min(self.gn, 3)]
        field = gfunc(
            potential,
            self.m.unsqueeze(-1).unsqueeze(-1),
            self.s.unsqueeze(-1).unsqueeze(-1),
        )
        # add the growth multiplied by the weight of the rule to the total growth
        self.D[:, :, :, 0] = (self.h.unsqueeze(-1).unsqueeze(-1) * field).sum(
            0, keepdim=True
        )
        self.Dn[0] = self.h.sum()

        ###Base version for the case where we want the learnable creature to be  multi channel (which is not used in this notebook)

        # for i in range(self.nb_k):
        #   c0b=int((self.c0[i]))
        #   c1b=int((self.c1[i]))

        #   world_FFT_c = world_FFT[c0b]
        #   potential_FFT = complex_mult_torch(self.kernels[i].unsqueeze(0), world_FFT_c)

        #   potential = torch.irfft(potential_FFT, signal_ndim=2, onesided=False)
        #   potential = roll_n(potential, 2, potential.size(2) // 2)
        #   potential = roll_n(potential, 1, potential.size(1) // 2)

        #   gfunc = field_func[min(self.gn, 3)]
        #   field = gfunc(potential, self.m[i], self.s[i])

        #   self.D[:,:,:,c1b]=self.D[:,:,:,c1b]+self.h[i]*field
        #   self.Dn[c1b]=self.Dn[c1b]+self.h[i]

        # apply wall
        world_FFT_c = world_FFT[self.C - 1]
        potential_FFT = self.kernel_wall * world_FFT_c
        potential = torch.fft.irfftn(potential_FFT, dim=(1, 2))
        potential = roll_n(potential, 2, potential.size(2) // 2)
        potential = roll_n(potential, 1, potential.size(1) // 2)
        gfunc = field_func[3]
        field = gfunc(potential, 1e-8, 10)
        for i in range(self.C - 1):
            c1b = i
            self.D[:, :, :, c1b] = self.D[:, :, :, c1b] + 1 * field
            self.Dn[c1b] = self.Dn[c1b] + 1

        ## Add the total growth to the current state
        if not self.is_soft_clip:

            output_img = torch.clamp(
                input + (1.0 / self.T) * self.D, min=0.0, max=1.0
            )
            # output_img = input + (1.0 / self.T) * ((self.D/self.Dn+1)/2-input)

        else:
            output_img = torch.sigmoid(
                (input + (1.0 / self.T) * self.D - 0.5) * 10
            )
            # output_img = torch.tanh(input + (1.0 / self.T) * self.D)

        return output_img
