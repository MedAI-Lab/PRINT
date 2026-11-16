"""Encoder unfreezing and learning-rate schedule."""
def depth(epoch):
    if not isinstance(epoch, int) or not 1 <= epoch <= 150:
        raise ValueError(epoch)
    if epoch <= 30:
        return 0
    if epoch <= 60:
        return 4
    return 6 if epoch <= 90 else 8


def encoder_lr(epoch):
    depth(epoch)
    return 1.875e-5 * min(1.0, max(0.0, (epoch - 30) / 20))
