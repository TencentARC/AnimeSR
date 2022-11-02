from collections import OrderedDict

from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class DegradationModel(SRModel):
    """Degradation model for real-world, hard-to-synthesis degradation."""

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        self.l1_pix = build_loss(train_opt['l1_opt']).to(self.device)
        self.l2_pix = build_loss(train_opt['l2_opt']).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        # we reverse the order of lq and gt for convenient implementation
        self.lq = data['gt'].to(self.device)
        if 'lq' in data:
            self.gt = data['lq'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # l1 loss
        l_l1 = self.l1_pix(self.output, self.gt)
        l_total += l_l1
        loss_dict['l_l1'] = l_l1
        # l2 loss
        l_l2 = self.l2_pix(self.output, self.gt)
        l_total += l_l2
        loss_dict['l_l2'] = l_l2

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
