
import numpy as np
import torch
from Torch_utils import training_stats
from Torch_utils import misc
from Torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, gen_z, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------
class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)


    def run_G(self, z, nmap, sync):

        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z)
            if self.style_mixing_prob > 0:
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), skip_w_avg_update=True)[:, cutoff:]

        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(nmap, ws)

        return img, ws


    def run_D(self, img, sync):

        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
         
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img)

        return logits

    
    @staticmethod
    def MinPool(Is, Igr, kernel=(9, 9), thresh=0.034): # note: 系统误差
        # Is, Igr: (B, C, W, H)
        Is_mask = Is != 0

        rela_I = Igr - Is
        rela_I_sign = torch.sign(rela_I)
        rela_I_value = torch.abs(rela_I)

        padding = int((kernel[0] - 1) / 2)
        min_pool_I = -torch.max_pool2d(-rela_I_value, kernel, 1, padding)
        min_pool_I = min_pool_I * rela_I_sign

        high_pass_mask = rela_I_value > thresh
        mask = torch.logical_and(high_pass_mask, Is_mask)

        min_pool_I = min_pool_I[mask]

        return min_pool_I
    

    def accumulate_gradients(self, phasename, real_img, nmap, gen_z, gain):
        
        # Gmain: Maximize logits for generated images.     D(G(x)) -> max | G
        # Dmain: Minimize logits for generated images.     D(G(x)) -> min | D
        # Dmain: Maximize logits for real images.          D(real) -> max | D
        
        do_G_main = (phasename == 'Gmain')
        do_G_reg  = (phasename == 'Greg')
        do_D_main = (phasename == 'Dmain')
        do_D_reg  = (phasename == 'Dreg')

        loss_Gmain = 0
        loss_Gpl = 0

        loss_Dgen = 0
        loss_Dreal = 0
        loss_Dr1 = 0


        if do_G_main:
            gen_img, _ = self.run_G(gen_z, nmap, sync=(not do_G_reg))
            gen_logits = self.run_D(gen_img, sync=False)

            loss_Gmain = torch.nn.functional.softplus(-gen_logits)
            loss_Gmain.mean().mul(gain).backward()

            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/G/loss', loss_Gmain)



        if do_G_reg:
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            gen_img, gen_ws = self.run_G(gen_z[:batch_size], nmap[:batch_size], sync=True)

            pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            # with conv2d_gradfix.no_weight_gradients():
            pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()

            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)

            self.pl_mean.copy_(pl_mean.detach())

            pl_penalty = (pl_lengths - pl_mean).square()
            
            loss_Gpl = pl_penalty * self.pl_weight
            (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

            training_stats.report('Loss/pl_penalty', pl_penalty)
            training_stats.report('Loss/G/reg', loss_Gpl)
            



        if do_D_main:
            gen_img, _ = self.run_G(gen_z, nmap, sync=False)
            gen_logits = self.run_D(gen_img, sync=False)

            loss_Dgen = torch.nn.functional.softplus(gen_logits)
            loss_Dgen.mean().mul(gain).backward()
            
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())



        if do_D_main or do_D_reg:
            real_img_tmp = real_img.detach().requires_grad_(do_D_reg)
            nmap = nmap.detach().requires_grad_(do_D_reg)
            real_logits = self.run_D(real_img_tmp, sync=True)

            training_stats.report('Loss/scores/real', real_logits)
            training_stats.report('Loss/signs/real', real_logits.sign())

            if do_D_main:
                loss_Dreal = torch.nn.functional.softplus(-real_logits)
                loss_Dreal.mean().mul(gain).backward() 

                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            if do_D_reg:

                with conv2d_gradfix.no_weight_gradients():
                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                r1_penalty = r1_grads.square().sum([1,2,3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)

                (real_logits * 0 +  loss_Dr1).mean().mul(gain).backward()

                training_stats.report('Loss/r1_penalty', r1_penalty)
                training_stats.report('Loss/D/reg', loss_Dr1)













#----------------------------------------------------------------------------
class StyleGAN2Loss_v1(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)


    def run_G(self, z, nmap, sync):

        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z)
            if self.style_mixing_prob > 0:
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), skip_w_avg_update=True)[:, cutoff:]

        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(nmap, ws)

        return img, ws


    def run_D(self, img, sync):

        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
         
        with misc.ddp_sync(self.D, sync):
            main_logits, branch_logits = self.D(img)

        return main_logits, branch_logits

    
    def accumulate_gradients(self, phasename, real_img, nmap, gen_z, gain):
        
        # Gmain: Maximize logits for generated images.     D(G(x)) -> max | G
        # Dmain: Minimize logits for generated images.     D(G(x)) -> min | D
        # Dmain: Maximize logits for real images.          D(real) -> max | D
        
        do_G_main = (phasename == 'Gmain')
        do_G_reg  = (phasename == 'Greg')
        do_D_main = (phasename == 'Dmain')
        do_D_reg  = (phasename == 'Dreg')

        loss_Gmain_0 = 0
        loss_Gmain_1 = 0
        loss_Gpl = 0

        loss_Dgen = 0
        loss_Dreal = 0
        loss_Dr1 = 0


        if do_G_main:
            gen_img, _ = self.run_G(gen_z, nmap, sync=(not do_G_reg))
            main_logits, branch_logits = self.run_D(gen_img, sync=False)

            loss_Gmain_0 = torch.nn.functional.softplus(-main_logits)
            loss_Gmain_1 = torch.nn.functional.softplus(-branch_logits)
            (loss_Gmain_0 + loss_Gmain_1).mean().mul(gain).backward()

            training_stats.report('Loss/G/loss', (loss_Gmain_0 + loss_Gmain_1))



        if do_G_reg:
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            gen_img, gen_ws = self.run_G(gen_z[:batch_size], nmap[:batch_size], sync=True)

            pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            # with conv2d_gradfix.no_weight_gradients():
            pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]

            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())
            pl_penalty = (pl_lengths - pl_mean).square()
            
            loss_Gpl = pl_penalty * self.pl_weight
            (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

            training_stats.report('Loss/G/reg', loss_Gpl)



        if do_D_main:
            gen_img, _ = self.run_G(gen_z, nmap, sync=False)
            main_logits, branch_logits = self.run_D(gen_img, sync=False)

            loss_Dgen_0 = torch.nn.functional.softplus(main_logits)
            loss_Dgen_1 = torch.nn.functional.softplus(branch_logits)
            (loss_Dgen_0 + loss_Dgen_1).mean().mul(gain).backward()
            
            training_stats.report('Loss/signs/fake_main', main_logits.sign())
            training_stats.report('Loss/signs/fake_branch', branch_logits.sign())



        if do_D_main or do_D_reg:
            real_img_tmp = real_img.detach().requires_grad_(do_D_reg)
            nmap = nmap.detach().requires_grad_(do_D_reg)
            main_logits, branch_logits = self.run_D(real_img_tmp, sync=True)

            training_stats.report('Loss/signs/real_main', main_logits.sign())
            training_stats.report('Loss/signs/real_branch', main_logits.sign())

            if do_D_main:
                loss_Dreal_0 = torch.nn.functional.softplus(-main_logits)
                loss_Dreal_1 = torch.nn.functional.softplus(-branch_logits)
                (loss_Dreal_0 + loss_Dreal_1).mean().mul(gain).backward() 

                training_stats.report('Loss/D/loss', loss_Dgen_0 + loss_Dgen_1 + loss_Dreal_0 + loss_Dreal_1)

            if do_D_reg:

                with conv2d_gradfix.no_weight_gradients():
                    r1_grads = torch.autograd.grad(outputs=[main_logits.sum(), branch_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                r1_penalty = r1_grads.square().sum([1,2,3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)

                (main_logits * 0 + branch_logits * 0 + loss_Dr1).mean().mul(gain).backward()

                training_stats.report('Loss/r1_penalty', r1_penalty)
                training_stats.report('Loss/D/reg', loss_Dr1)
