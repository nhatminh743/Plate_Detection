from __future__ import absolute_import, division, print_function, unicode_literals
import copy

__all__ = ["build_post_process"]

from CTCDecode import CTCLabelDecode  # Only importing what you use

def build_post_process(config, global_config=None):
    support_dict = [
        "CTCLabelDecode",  # Only keeping what you use
    ]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, f"post process only supports {support_dict}"
    module_class = eval(module_name)(**config)
    return module_class