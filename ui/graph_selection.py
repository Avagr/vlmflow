# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, Optional

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class NodeType(Enum):
    AFTER_ATTN = "after_attn"
    AFTER_FFN = "after_ffn"
    FFN = "ffn"
    ORIGINAL = "original"  # The original tokens


def _format_block_hierachy_string(blocks: List[str]) -> str:
    return " ▸ ".join(blocks)


@dataclass
class GraphNode:
    layer: int
    token: int
    type: NodeType

    def is_in_residual_stream(self) -> bool:
        return self.type in [NodeType.AFTER_ATTN, NodeType.AFTER_FFN]

    def get_residual_predecessor(self) -> Optional["GraphNode"]:
        """
        Get another graph node which points to the state of the residual stream before
        this node.

        Retun None if current representation is the first one in the residual stream.
        """
        scheme = {
            NodeType.AFTER_ATTN: GraphNode(
                layer=max(self.layer - 1, 0),
                token=self.token,
                type=NodeType.AFTER_FFN if self.layer > 0 else NodeType.ORIGINAL,
            ),
            NodeType.AFTER_FFN: GraphNode(
                layer=self.layer,
                token=self.token,
                type=NodeType.AFTER_ATTN,
            ),
            NodeType.FFN: GraphNode(
                layer=self.layer,
                token=self.token,
                type=NodeType.AFTER_ATTN,
            ),
            NodeType.ORIGINAL: None,
        }
        node = scheme[self.type]
        if node.layer < 0:
            return None
        return node

    def get_name(self) -> str:
        return _format_block_hierachy_string(
            [f"L{self.layer}", f"T{self.token}", str(self.type.value)]
        )

    def get_predecessor_block_name(self) -> str:
        """
        Return the name of the block standing between current node and its predecessor
        in the residual stream.
        """
        scheme = {
            NodeType.AFTER_ATTN: [f"L{self.layer}", "attn"],
            NodeType.AFTER_FFN: [f"L{self.layer}", "ffn"],
            NodeType.FFN: [f"L{self.layer}", "ffn"],
            NodeType.ORIGINAL: ["Nothing"],
        }
        return _format_block_hierachy_string(scheme[self.type])

    def get_head_name(self, head: Optional[int]) -> str:
        path = [f"L{self.layer}", "attn"]
        if head is not None:
            path.append(f"H{head}")
        return _format_block_hierachy_string(path)

    def get_neuron_name(self, neuron: Optional[int]) -> str:
        path = [f"L{self.layer}", "ffn"]
        if neuron is not None:
            path.append(f"N{neuron}")
        return _format_block_hierachy_string(path)


class UiGraphNode(GraphNode):
    @staticmethod
    def from_json(json: Dict[str, Any]) -> Optional["UiGraphNode"]:
        try:
            layer = json["cell"]["layer"]
            token = json["cell"]["token"]
            type = NodeType(json["item"])
            return UiGraphNode(layer, token, type)
        except (TypeError, KeyError):
            return None


@dataclass
class UiGraphEdge:
    source: UiGraphNode
    target: UiGraphNode
    weight: float

    @staticmethod
    def from_json(json: Dict[str, Any]) -> Optional["UiGraphEdge"]:
        try:
            source = UiGraphNode.from_json(json["from"])
            target = UiGraphNode.from_json(json["to"])
            if source is None or target is None:
                return None
            weight = float(json["weight"])
            return UiGraphEdge(source, target, weight)
        except (TypeError, KeyError):
            return None


@dataclass
class GraphSelection:
    node: Optional[UiGraphNode]
    edge: Optional[UiGraphEdge]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> Optional["GraphSelection"]:
        try:
            node = UiGraphNode.from_json(json["node"])
            edge = UiGraphEdge.from_json(json["edge"])
            return GraphSelection(node, edge)
        except (TypeError, KeyError):
            return None