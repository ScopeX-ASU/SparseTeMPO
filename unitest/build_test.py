
import torch

import torch.fft

from pyutils.config import configs
from core import builder


def main() -> None:
    configs.load("./configs/metaconv_test/train/train.yml", recursive=True)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=int(configs.run.random_state) if int(configs.run.deterministic) else None,
        dpe=None,  # only pass forward function, not the entire Module
    )
    print(model)

if __name__ == "__main__":
    main()