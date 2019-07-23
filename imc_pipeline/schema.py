import voluptuous as vo

segmentation = vo.Schema(
    {
        vo.Required('ref_channel'): int,
        vo.Optional('min_distance', default=3): int,
        vo.Optional('gb_ksize', default=0): int,
        vo.Optional('gb_sigma', default=2): float,
        vo.Optional('adapThresh_blockSize', default=15): int,
        vo.Optional('adapThresh_constant', default=-7.5): float,
    }
)

schema = vo.Schema(
    {
        vo.Required('img_path'): str,
        vo.Required('output_path'): str,
        vo.Required('n_buff'): int,
        vo.Required('normalized_factor'): int,
        vo.Required('segmentation'): segmentation,
    }
)
