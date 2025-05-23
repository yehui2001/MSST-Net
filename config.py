class Args:
    def __init__(self):
        # Model parameters
        self.hsi_channel = 31
        self.msi_channel = 3
        self.n_features = 64
        self.n_resblocks = 5
        self.n_heads = 4
        self.n_blocks = 4 #SpeT中Block的数量