DataParallel(
  (module): ResNetPlus3(
    (expland_channel): Conv1d(1, 12, kernel_size=(15,), stride=(2,), padding=(7,), bias=False)
    (conv1): Conv1d(12, 64, kernel_size=(15,), stride=(2,), padding=(7,), bias=False)
    (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv1d(12, 64, kernel_size=(31,), stride=(2,), padding=(15,), bias=False)
    (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv3): Conv1d(12, 64, kernel_size=(45,), stride=(2,), padding=(22,), bias=False)
    (bn3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv4): Conv1d(12, 64, kernel_size=(9,), stride=(2,), padding=(4,), bias=False)
    (bn4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv11): Conv1d(64, 16, kernel_size=(1,), stride=(1,), bias=False)
    (se): SELayer(
      (avg_pool): AdaptiveAvgPool1d(output_size=1)
      (fc): Sequential(
        (0): Linear(in_features=64, out_features=4, bias=False)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=4, out_features=64, bias=False)
        (3): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): Sequential(
      (0): SEBasicBlock(
        (conv1): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=64, out_features=4, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=4, out_features=64, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (1): SEBasicBlock(
        (conv1): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=64, out_features=4, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=4, out_features=64, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (2): SEBasicBlock(
        (conv1): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=64, out_features=4, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=4, out_features=64, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
    (layer2): Sequential(
      (0): SEBasicBlock(
        (conv1): Conv1d(64, 128, kernel_size=(7,), stride=(2,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=128, out_features=8, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=8, out_features=128, bias=False)
            (3): Sigmoid()
          )
        )
        (downsample): Sequential(
          (0): Conv1d(64, 128, kernel_size=(1,), stride=(2,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (1): SEBasicBlock(
        (conv1): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=128, out_features=8, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=8, out_features=128, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (2): SEBasicBlock(
        (conv1): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=128, out_features=8, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=8, out_features=128, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (3): SEBasicBlock(
        (conv1): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=128, out_features=8, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=8, out_features=128, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
    (layer3): Sequential(
      (0): SEBasicBlock(
        (conv1): Conv1d(128, 256, kernel_size=(7,), stride=(2,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=256, out_features=16, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=256, bias=False)
            (3): Sigmoid()
          )
        )
        (downsample): Sequential(
          (0): Conv1d(128, 256, kernel_size=(1,), stride=(2,), bias=False)
          (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (1): SEBasicBlock(
        (conv1): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=256, out_features=16, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=256, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (2): SEBasicBlock(
        (conv1): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=256, out_features=16, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=256, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (3): SEBasicBlock(
        (conv1): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=256, out_features=16, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=256, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (4): SEBasicBlock(
        (conv1): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=256, out_features=16, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=256, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (5): SEBasicBlock(
        (conv1): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=256, out_features=16, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=256, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
    (layer4): Sequential(
      (0): SEBasicBlock(
        (conv1): Conv1d(256, 512, kernel_size=(7,), stride=(2,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(512, 512, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=512, out_features=32, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=512, bias=False)
            (3): Sigmoid()
          )
        )
        (downsample): Sequential(
          (0): Conv1d(256, 512, kernel_size=(1,), stride=(2,), bias=False)
          (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (1): SEBasicBlock(
        (conv1): Conv1d(512, 512, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(512, 512, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=512, out_features=32, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=512, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (2): SEBasicBlock(
        (conv1): Conv1d(512, 512, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv1d(512, 512, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (bn2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SELayer(
          (avg_pool): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=512, out_features=32, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=512, bias=False)
            (3): Sigmoid()
          )
        )
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
    (avgpool): AdaptiveAvgPool1d(output_size=1)
    (maxpool2): AdaptiveMaxPool1d(output_size=1)
    (head): Sequential(
      (0): Linear(in_features=1024, out_features=128, bias=True)
      (1): Hswish()
      (2): Linear(in_features=128, out_features=10, bias=True)
    )
    (dense2): Linear(in_features=10, out_features=1, bias=True)
  )
)