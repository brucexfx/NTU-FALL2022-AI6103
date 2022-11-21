# Notice: This file only contains the logic implementation of CutMix, excluding the model

# A bouding box function
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    # calculate the rw,rh of B
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # calculate the rx,rx of B
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # set a restriction for the bounding area to not exceed the whole sample
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    # return bounding box coordinates
    return bbx1, bby1, bbx2, bby2

# CutMix Method
for i, (input, target) in enumerate(train_loader):
   # measure data loading time
   data_time.update(time.time() - end)
   input = input.cuda()
   target = target.cuda()

   r = np.random.rand(1)
   if args.beta > 0 and r < args.cutmix_prob:
      # 1.set lambda，under beta distribution
      lam = np.random.beta(args.beta, args.beta)
      # 2.find two random samples
      rand_index = torch.randperm(input.size()[0]).cuda()
      target_a = target
      target_b = target[rand_index]
      # 3.create bounding box B
      bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
      # 4.replace the region B in sampleA with B in sample B
      input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
      # 5.adjust lamda
      lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
      # 6.put the new sample into training
      output = model(input)
      # 7.set weight according to lamda
      loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
