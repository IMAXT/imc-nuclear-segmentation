import voluptuous as vo

schema = vo.Schema(
    {
        vo.Required('img_format'): str,
        vo.Required('img_format_out'): str,
        vo.Required('cat_format'): str,
        vo.Required('img_path'): str,
        vo.Required('output_key'): str,
        vo.Required('output_key_ref'): str,
        vo.Required('output_key_mask'): str,
        vo.Required('output_key_cat'): str,
        vo.Required('ref_channel'): int,
        vo.Required('normalized_factor'): int,
    }
)
