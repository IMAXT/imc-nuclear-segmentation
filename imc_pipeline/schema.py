import voluptuous as vo

segmentation = vo.Schema(
    {
        vo.Optional('perform_full_analysis', default=True): bool,
        vo.Required('ref_channel'): int,
        vo.Optional('min_distance', default=3): int,
        vo.Optional('gb_ksize', default=0): int,
        vo.Optional('gb_sigma', default=2): float,
        vo.Optional('adapThresh_blockSize', default=15): int,
        vo.Optional('adapThresh_constant', default=-7.5): float,
        vo.Optional('normalized_factor', default=30): int,
        vo.Optional('aic_apply_intensity_correction', default=False): bool,
        vo.Optional('aic_sigma', default=5): int,

    }
)

schema = vo.Schema(
    {
        vo.Required('input_path'): str,
        vo.Required('output_path'): str,
        vo.Required('n_buff'): int,
        vo.Required('segmentation'): segmentation,
    }
)
