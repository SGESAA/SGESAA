from torch.optim.lr_scheduler import MultiStepLR


def adjust_learning_rate_resnet(optimizer, milestones, gamma=0.1):
    """
    Sets the learning rate to the initial LR decayed by 10 on every predefined epochs
    Ref: AutoAugment
    """

    # if epoch <= 90:
    #     return MultiStepLR(optimizer, [30, 60, 80])
    # elif epoch <= 270:  # autoaugment
    #     return MultiStepLR(optimizer, [90, 180, 240])
    # else:
    #     raise ValueError('invalid epoch=%d for resnet scheduler' % epoch)
    return MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)


class MultiStepLR_HotFix(MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        super(MultiStepLR_HotFix, self).__init__(optimizer, milestones, gamma,
                                                 last_epoch)
        self.milestones = list(milestones)
