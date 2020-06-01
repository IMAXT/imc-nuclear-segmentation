import voluptuous as vo
from pathlib import Path


def Odd():
    def f(val):
        if (val > 0) and (val % 2) == 0:
            raise vo.Invalid("must be odd number")
        return val

    return f


segmentation = vo.Schema(
    {
        vo.Optional("perform_full_analysis", default=True): bool,
        vo.Required("ref_channel"): vo.All(int, vo.Range(min=1)),
        vo.Optional("min_distance", default=3): vo.All(int, vo.Range(min=3, max=10)),
        vo.Optional("gb_ksize", default=0): vo.All(int, vo.Range(min=0, max=10), Odd()),
        vo.Optional("gb_sigma", default=2): vo.All(
            vo.Any(int, float), vo.Range(min=0, max=10)
        ),
        vo.Optional("adapThresh_blockSize", default=15): vo.All(
            int, vo.Range(min=3, max=100), Odd()
        ),
        vo.Optional("adapThresh_constant", default=-7.5): vo.All(
            vo.Any(int, float), vo.Range(min=-10, max=0)
        ),
        vo.Optional("normalized_factor", default=30): vo.All(
            int, vo.Range(min=0, max=50)
        ),
        vo.Optional("aic_apply_intensity_correction", default=False): bool,
        vo.Optional("aic_sigma", default=5): vo.All(
            vo.Any(int, float), vo.Range(min=1, max=20)
        ),
    }
)

schema = vo.Schema(
    {
        vo.Required("input_path"): vo.All(vo.IsDir(), vo.Coerce(Path)),
        vo.Required("output_path"): vo.Coerce(Path),
        vo.Required("n_buff"): vo.All(int, vo.Range(min=1, max=4)),
        vo.Required("segmentation"): segmentation,
    }
)
