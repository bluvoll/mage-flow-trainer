# SPDX-License-Identifier: GPL-3.0-or-later
from .nodes import LoadCompressedMageFlow, LoadCompressedRTIMageFlow
from .weighted_text import MageFlowWeightedTextEncode

NODE_CLASS_MAPPINGS = {"LoadCompressedMageFlow": LoadCompressedMageFlow}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadCompressedMageFlow": "Load Compressed Mage-Flow"}
NODE_CLASS_MAPPINGS["LoadCompressedRTIMageFlow"] = LoadCompressedRTIMageFlow
NODE_DISPLAY_NAME_MAPPINGS["LoadCompressedRTIMageFlow"] = "Load Compressed RTI Mage-Flow"
NODE_CLASS_MAPPINGS["MageFlowWeightedTextEncode"] = MageFlowWeightedTextEncode
NODE_DISPLAY_NAME_MAPPINGS["MageFlowWeightedTextEncode"] = "Mage-Flow Weighted Text Encode"
