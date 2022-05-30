
b0 = {"embed_dims":8,#32
      "num_stages":4,
      "num_layers":[2, 2, 2, 2],
      "num_heads":[1, 2, 5, 8],
      "patch_sizes":[7, 3, 3, 3],
      "strides":[4, 2, 2, 2],
      'sr_ratios':[8, 4, 2, 1],
      "mlp_ratio":4}

b2 = {"embed_dims":64,
      "num_stages":4,
      "num_layers":[3, 3, 6, 3],
      "num_heads":[1, 2, 5, 8],
      "patch_sizes":[7, 3, 3, 3],
      "strides":[4, 2, 2, 2],
      'sr_ratios':[8, 4, 2, 1],
      "mlp_ratio":4}

config = {'b0':b0,'b2':b2}