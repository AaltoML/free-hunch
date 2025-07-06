import logging
import torch.distributed as dist

def setup_logger():
    rank = dist.get_rank() if dist.is_initialized() else 0
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)