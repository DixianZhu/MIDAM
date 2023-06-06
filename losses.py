import torch 
import torch.nn as nn 
import torch.nn.functional as F

class AUCMLoss(torch.nn.Module):
    """
    AUCM Loss with squared-hinge function: a novel loss function to directly optimize AUROC
    
    inputs:
        margin: margin term for AUCM loss, e.g., m in [0, 1]
        imratio: imbalance ratio, i.e., the ratio of number of postive samples to number of total samples
    outputs:
        loss value 
    
    Reference: 
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T., 
        Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification. 
        International Conference on Computer Vision (ICCV 2021)
    Link:
        https://arxiv.org/abs/2012.03173
    """
    def __init__(self, margin=1.0, imratio=None):
        super(AUCMLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.margin = margin
        self.p = imratio
        # https://discuss.pytorch.org/t/valueerror-cant-optimize-a-non-leaf-tensor/21751
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) #cuda()
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device) #.cuda()
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) #.cuda()
        
    def forward(self, y_pred, y_true):
        if self.p is None:
           self.p = (y_true==1).float().sum()/y_true.shape[0]   
        f_ps = y_pred[y_true==1].reshape(-1,1) 
        f_ns = y_pred[y_true==0].reshape(-1,1) 
        y_pred = y_pred.reshape(-1, 1) # be carefull about these shapes
        y_true = y_true.reshape(-1, 1) 
        tmp = (1-self.p)*torch.mean((y_pred - self.a)**2*(1==y_true).float()) + \
                    self.p*torch.mean((y_pred - self.b)**2*(0==y_true).float()) 
        loss = tmp + 2*self.alpha*(self.p*(1-self.p)*self.margin + \
                    torch.mean((self.p*y_pred*(0==y_true).float() - (1-self.p)*y_pred*(1==y_true).float())) )- \
                    self.p*(1-self.p)*self.alpha**2
        loss = loss/((1-self.p)*self.p)
        return loss

class CrossEntropyLoss(torch.nn.Module):
    """
    Cross Entropy Loss with Sigmoid Function
    Reference: 
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = F.binary_cross_entropy_with_logits  # with sigmoid
    def forward(self, y_pred, y_true):
        return self.criterion(y_pred, y_true.float().cuda())  


class MIDAM_softmax_pooling_loss(nn.Module):
    r"""
    Multiple Instance Deep AUC Maximization with stochastic Smoothed-MaX (MIDAM-smx) Pooling. This loss is used for optimizing the AUROC under Multiple Instance Learning (MIL) setting. 
    The Smoothed-MaX Pooling is defined as

    .. math::
        h(\mathbf w; \mathcal X) = \tau \log\left(\frac{1}{|\mathcal X|}\sum_{\mathbf x\in\mathcal X}\exp(\phi(\mathbf w; \mathbf x)/\tau)\right)

    where :math:`\phi(\mathbf w;\mathbf x)` is the prediction score for instance :math:`\mathbf x` and :math:`\tau>0` is a hyperparameter. 
    We optimize the following AUC loss with the Smoothed-MaX Pooling:

    .. math::
        \min_{\mathbf w\in\mathbb R^d,(a,b)\in\mathbb R^2}\max_{\alpha\in\Omega}F\left(\mathbf w,a,b,\alpha\right)&:= \underbrace{\hat{\mathbb E}_{i\in\mathcal D_+}\left[(h(\mathbf w; \mathcal X_i) - a)^2 \right]}_{F_1(\mathbf w, a)} \\
        &+ \underbrace{\hat{\mathbb E}_{i\in\mathcal D_-}\left[(h(\mathbf w; \mathcal X_i) - b)^2 \right]}_{F_2(\mathbf w, b)} \\
        &+ \underbrace{\alpha (c+ \hat{\mathbb E}_{i\in\mathcal D_-}h(\mathbf w; \mathcal X_i) - \hat{\mathbb E}_{i\in\mathcal D_+}h(\mathbf w; \mathcal X_i)) - \frac{\alpha^2}{2}}_{F_3(\mathbf w, \alpha)},

    The optimization algorithm for solving the above objective is implemented as :obj:`~libauc.optimizers.MIDAM`. The stochastic pooling loss only requires partial data from each bag when do optimization. For the more details about the formulations, please refer to the original paper [1]_.

    Args:
        data_len (int): number of training samples.
        margin (float, optional): margin parameter for AUC loss (default: 0.5).
        tau (float): temperature parameter for smoothed max pooling (default: 0.1).
        gamma (float, optional): moving average parameter for pooling operation (default: 0.9).
        eta (float, optional): step size for updating dual variable (default: 1.0).
        device (torch.device, optional): device for running the code. default: none (use GPU if available)

    Example:
        >>> Loss_fn = MIDAM_softmax_pooling_loss(data_len=data_len, margin=margin, tau=tau, gamma=gamma)
        >>> preds = torch.randn(32, 1, requires_grad=True)
        >>> target = torch.empty(32 dtype=torch.long).random_(1)
        >>> # in practice, index should be the indices of your data (bag-index for multiple instance learning).
        >>> loss_fn(y_pred=preds, y_true=target, index=torch.arange(32)) 
        >>> loss_fn.backward()

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning, pp. xxxxx-xxxxx. PMLR, 2023.
           https://prepare-arxiv?
    """
    def __init__(self, data_len, margin=0.5, tau=0.1, gamma=0.9, eta=1.0):
        '''
        :param margin: margin for squred hinge loss
        '''
        super(MIDAM_softmax_pooling_loss, self).__init__()
        self.gamma = gamma
        self.eta = eta
        self.tau = tau
        self.data_len = data_len
        self.s = torch.tensor([0.0]*data_len).view(-1, 1).cuda()  
        self.a = torch.zeros(1, dtype=torch.float32, requires_grad=True, device='cuda')
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=True, device='cuda')
        self.alpha = torch.zeros(1, dtype=torch.float32, requires_grad=False, device='cuda')
        self.margin = margin

    def update_smoothing(self, decay_factor):
        self.gamma = self.gamma/decay_factor
        self.eta = self.eta/decay_factor

    def forward(self, y_pred, y_true, ids): 
        ids = ids.cuda()
        self.s[ids] = (1-self.gamma) * self.s[ids] + self.gamma * y_pred.detach()
        vs = self.s[ids]
        ids_p = (y_true == 1)
        ids_n = (y_true == 0)
        s_p = vs[ids_p]   
        s_n = vs[ids_n]   
        logs_p = self.tau*torch.log(s_p)
        logs_n = self.tau*torch.log(s_n)
        gw_ins_p = y_pred[ids_p]/s_p 
        gw_ins_n = y_pred[ids_n]/s_n
        gw_p = torch.mean(2*self.tau*(logs_p-self.a.detach())*gw_ins_p)
        gw_n = torch.mean(2*self.tau*(logs_n-self.b.detach())*gw_ins_n)
        gw_s = self.alpha.detach()* self.tau * (torch.mean(gw_ins_n) - torch.mean(gw_ins_p))
        ga = torch.mean((logs_p - self.a)**2)
        gb = torch.mean((logs_n - self.b)**2)
        loss = gw_p + gw_n + gw_s + ga + gb
        g_alpha = (self.margin + logs_n.mean() - logs_p.mean()) - self.alpha.detach()
        self.alpha = torch.maximum(self.alpha + self.eta * g_alpha, torch.tensor(0.0))
        return loss

class MIDAM_attention_pooling_loss(nn.Module):
    r"""
    Multiple Instance Deep AUC Maximization with stochastic Attention (MIDAM-att) Pooling is used for optimizing the AUROC under Multiple Instance Learning (MIL) setting. 
    The Attention Pooling is defined as

    .. math::
        h(\mathbf w; \mathcal X) = \sigma(\mathbf w_c^{\top}E(\mathbf w; \mathcal X)) = \sigma\left(\sum_{\mathbf x\in\mathcal X}\frac{\exp(g(\mathbf w; \mathbf x))\delta(\mathbf w;\mathbf x)}{\sum_{\mathbf x'\in\mathcal X}\exp(g(\mathbf w; \mathbf x'))}\right),

    where :math:`g(\mathbf w;\mathbf x)` is a parametric function, e.g., :math:`g(\mathbf w; \mathbf x)=\mathbf w_a^{\top}\text{tanh}(V e(\mathbf w_e; \mathbf x))`, where :math:`V\in\mathbb R^{m\times d_o}` and :math:`\mathbf w_a\in\mathbb R^m`. 
    And :math:`\delta(\mathbf w;\mathbf x) = \mathbf w_c^{\top}e(\mathbf w_e; \mathbf x)` is the prediction score from each instance, which will be combined with attention weights.
    We optimize the following AUC loss with the Attention Pooling:

    .. math::
        \min_{\mathbf w\in\mathbb R^d,(a,b)\in\mathbb R^2}\max_{\alpha\in\Omega}F\left(\mathbf w,a,b,\alpha\right)&:= \underbrace{\hat{\mathbb E}_{i\in\mathcal D_+}\left[(h(\mathbf w; \mathcal X_i) - a)^2 \right]}_{F_1(\mathbf w, a)} \\
        &+ \underbrace{\hat{\mathbb E}_{i\in\mathcal D_-}\left[(h(\mathbf w; \mathcal X_i) - b)^2 \right]}_{F_2(\mathbf w, b)} \\
        &+ \underbrace{\alpha (c+ \hat{\mathbb E}_{i\in\mathcal D_-}h(\mathbf w; \mathcal X_i) - \hat{\mathbb E}_{i\in\mathcal D_+}h(\mathbf w; \mathcal X_i)) - \frac{\alpha^2}{2}}_{F_3(\mathbf w, \alpha)},

    The optimization algorithm for solving the above objective is implemented as :obj:`~libauc.optimizers.MIDAM`. The stochastic pooling loss only requires partial data from each bag when do optimization. For the more details about the formulations, please refer to the original paper [1]_.

    Args:
        data_len (int): number of training samples.
        margin (float, optional): margin parameter for AUC loss (default: 0.5).
        gamma (float, optional): moving average parameter for numerator and denominator on attention calculation (default: 0.9).
        eta (float, optional): step size for updating dual variable (default: 1.0).
        device (torch.device, optional): device for running the code (use GPU if available) (default: None).

    Example:
        >>> Loss_fn = MIDAM_attention_pooling_loss(data_len=data_len, margin=margin, tau=tau, gamma=gamma)
        >>> preds = torch.randn(32, 1, requires_grad=True)
        >>> denoms = torch.rand(32, 1, requires_grad=True) + 0.01
        >>> target = torch.empty(32 dtype=torch.long).random_(1)
        >>> # in practice, index should be the indices of your data (bag-index for multiple instance learning).
        >>> # denoms should be the stochastic denominator values output from your model.
        >>> loss_fn(sn=preds, sd=denoms, y_true=target, index=torch.arange(32)) 
        >>> loss_fn.backward()

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning, pp. xxxxx-xxxxx. PMLR, 2023.
           https://prepare-arxiv?
    """
    def __init__(self, data_len, margin=0.5, gamma_1=0.9, gamma_2=0.9, eta=1.0):
        '''
        :param margin: margin for squred hinge loss
        '''
        super(MIDAM_attention_pooling_loss, self).__init__()
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.eta = eta
        self.data_len = data_len
        self.sn = torch.tensor([1.0]*data_len).view(-1, 1).cuda()
        self.sd = torch.tensor([1.0]*data_len).view(-1, 1).cuda()        
        self.a = torch.zeros(1, dtype=torch.float32, requires_grad=True, device='cuda')
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=True, device='cuda')
        self.alpha = torch.zeros(1, dtype=torch.float32, requires_grad=False, device='cuda')
        self.margin = margin

    def update_smoothing(self, decay_factor):
        self.gamma_1 = self.gamma_1/decay_factor
        self.gamma_2 = self.gamma_2/decay_factor
        self.eta = self.eta/decay_factor

    def forward(self, y_pred, y_true, ids): 
        sn, sd = y_pred 
        ids = ids.cuda()
        self.sn[ids] = (1-self.gamma_1) * self.sn[ids] + self.gamma_1 * sn.detach()
        self.sd[ids] = (1-self.gamma_2) * self.sd[ids] + self.gamma_2 * sd.detach()
        vsn = self.sn[ids]
        vsd = torch.clamp(self.sd[ids], min=1e-8)
        snd = vsn / vsd
        snd = torch.sigmoid(snd)
        gsnd = snd * (1-snd)
        ids_p = (y_true == 1)
        ids_n = (y_true == 0)
        snd_p = snd[ids_p]
        snd_n = snd[ids_n]
        gsnd_p = gsnd[ids_p]
        gsnd_n = gsnd[ids_n]
        gw_att_p = gsnd_p*(1/vsd[ids_p]*sn[ids_p] - vsn[ids_p]/(vsd[ids_p]**2)*sd[ids_p]) 
        gw_att_n = gsnd_n*(1/vsd[ids_n]*sn[ids_n] - vsn[ids_n]/(vsd[ids_n]**2)*sd[ids_n])
        gw_p = torch.mean(2*(snd_p-self.a.detach())*gw_att_p)
        gw_n = torch.mean(2*(snd_n-self.b.detach())*gw_att_n)
        gw_s = self.alpha.detach() * (torch.mean(gw_att_n) - torch.mean(gw_att_p))
        ga = torch.mean((snd_p - self.a)**2)
        gb = torch.mean((snd_n - self.b)**2)
        loss = gw_p + gw_n + gw_s + ga + gb
        g_alpha = (self.margin + snd_n.mean() - snd_p.mean()) - self.alpha.detach()
        self.alpha = torch.maximum(self.alpha + self.eta * g_alpha, torch.tensor(0.0))
        return loss
