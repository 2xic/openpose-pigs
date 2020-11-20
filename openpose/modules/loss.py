def l2_loss(input, target, mask, batch_size):
	overfit = True
	if input.shape != target.shape:
		print((input.shape, target.shape, mask.shape))
	loss = (input - target) * mask
	loss = (loss * loss) / 2 / batch_size
	return loss.sum()
