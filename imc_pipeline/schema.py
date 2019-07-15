import voluptuous as vo

schema = vo.Schema(
    {
        vo.Required('img_path'): str,
        vo.Required('output_path'): str,
        vo.Required('n_buff'): int,
        vo.Required('ref_channel'): int,
        vo.Required('normalized_factor'): int,
    }
)
