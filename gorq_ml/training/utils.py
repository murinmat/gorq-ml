import torch
import matplotlib.pyplot as plt
from clearml.logger import Logger


def log_image(
        data: torch.Tensor,
        *,
        title: str,
        series: str,
        iteration: int,
        max_history: int,
        colormap: str | None = None,
) -> None:
    to_plot = data.detach().cpu().numpy()
    if colormap is not None:
        to_plot = plt.get_cmap(colormap)(to_plot)[..., :3]
    Logger.current_logger().report_image(
        title=title,
        series=series,
        iteration=iteration,
        max_image_history=max_history,
        image=to_plot
    )
