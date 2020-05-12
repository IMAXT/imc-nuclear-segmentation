import voluptuous as vo
from pathlib import Path

segmentation = vo.Schema(
    {
        vo.Optional('perform_full_analysis', default=True): bool,
        vo.Required('ref_channel'): vo.All(int, vo.Range(min=1)),
        vo.Optional('min_distance', default=3): vo.All(int, vo.Range(min=3, max=10)),
        vo.Optional('gb_ksize', default=0): vo.All(int, vo.Range(min=0, max=10)),
        vo.Optional('gb_sigma', default=2.0): vo.All(float, vo.Range(min=0, max=10)),
        vo.Optional('adapThresh_blockSize', default=15): vo.All(int, vo.Range(min=3, max=100)),
        vo.Optional('adapThresh_constant', default=-7.5): vo.All(float, vo.Range(min=-10, max=0)),
        vo.Optional('normalized_factor', default=30): vo.All(int, vo.Range(min=0, max=50)),
        vo.Optional('aic_apply_intensity_correction', default=False): bool,
        vo.Optional('aic_sigma', default=5): vo.All(int, vo.Range(min=1, max=20)),

    }
)

schema = vo.Schema(
    {
        vo.Required('input_path'): vo.All(vo.IsDir(), vo.Coerce(Path)),
        vo.Required('output_path'): vo.Coerce(Path),
        vo.Required('n_buff'): vo.All(int, vo.Range(min=1, max=4)),
        vo.Required('segmentation'): segmentation,
    }
)
