# SPDX-License-Identifier: GPL-3.0-or-later
from .nodes import LoadCompressedMageFlow, LoadCompressedRTIMageFlow

NODE_CLASS_MAPPINGS = {"LoadCompressedMageFlow": LoadCompressedMageFlow}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadCompressedMageFlow": "Load Compressed Mage-Flow"}
NODE_CLASS_MAPPINGS["LoadCompressedRTIMageFlow"] = LoadCompressedRTIMageFlow
NODE_DISPLAY_NAME_MAPPINGS["LoadCompressedRTIMageFlow"] = "Load Compressed RTI Mage-Flow"
