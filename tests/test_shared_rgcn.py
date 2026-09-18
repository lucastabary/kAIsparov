"""Tests for the `shared_rgcn` backend — the weight-tied twin of `rgcn`."""

from __future__ import annotations

import torch

from kaisparov.core.game import ChessGame
from kaisparov.models.factory import load_backend, load_backend_spec
from kaisparov.models.rgcn.processor import RGCNProcessor


def test_backend_spec_is_registered():
    spec = load_backend_spec("shared_rgcn")
    assert spec.name == "shared_rgcn"
    assert spec.processor_class is load_backend("shared_rgcn").PROCESSOR_CLASS


def test_message_passing_weights_are_shared_across_steps():
    spec = load_backend_spec("shared_rgcn")
    model = spec.model_class(hidden_dim=8)

    # One conv, not one per step: no `conv1`/`conv2`/... in the state dict.
    conv_params = [k for k in model.state_dict() if k.startswith("chess_rgcn.conv.")]
    assert conv_params, "expected a single shared conv named `conv`"
    assert not [k for k in model.state_dict() if k.startswith("chess_rgcn.conv1.")]

    # Depth is free: more message-passing steps, identical parameter count.
    counts = {
        steps: sum(p.numel() for p in spec.model_class(hidden_dim=8, num_steps=steps).parameters())
        for steps in (2, 4, 8)
    }
    assert len(set(counts.values())) == 1

    rgcn_params = sum(
        p.numel() for p in load_backend_spec("rgcn").model_class(hidden_dim=8).parameters()
    )
    assert counts[4] < rgcn_params  # tying is what makes it smaller


def test_forward_and_decode_produce_a_legal_move():
    spec = load_backend_spec("shared_rgcn")
    model = spec.model_class.create_agent(device=torch.device("cpu"), hidden_dim=8)
    processor = spec.processor_class()

    game = ChessGame()
    data = processor.graphify(game)
    action_scores, value = model(data)

    assert action_scores.shape == (data.edge_index.shape[1],)
    assert value.shape == (1,)

    action = processor.process_output((action_scores, value), game, deterministic=True)
    source, dest = action.move_coords[0], action.move_coords[1]
    assert dest in game.possible_moves(source)


def test_encoding_is_shared_with_rgcn():
    """Only the network differs, so the graph must be bit-identical to `rgcn`'s."""
    processor = load_backend_spec("shared_rgcn").processor_class()
    assert isinstance(processor, RGCNProcessor)

    game = ChessGame()
    shared = processor.graphify(game)
    reference = RGCNProcessor().graphify(game)
    assert torch.equal(shared.x, reference.x)
    assert torch.equal(shared.edge_index, reference.edge_index)
    assert torch.equal(shared.edge_type, reference.edge_type)
