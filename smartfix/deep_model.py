"""Q4 - Method B (optional): ResNet18 patch features + nearest neighbour.

TODO (weeks 4-5):
  - frozen torchvision resnet18(weights="DEFAULT"), forward hooks on layer2 + layer3
  - upsample layer3 to the layer2 size, concatenate -> one vector per patch
  - memory bank = patches of all normal train images (subsample if RAM is tight)
  - test: min distance of each patch to the bank (torch.cdist) -> map,
    upsample to 256, Gaussian blur; image score = max of the map
"""
