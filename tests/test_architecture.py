"""A checkpoint is rebuilt as the network it was trained as — from its run, or its weights.

Runs recorded before the ``features`` key existed carry only ``model`` and
``hidden_dim``; their feature set has to be read off the input layer, or today's
default would be applied to weights trained on something else.
"""

from __future__ import annotations

import pytest
import torch

from kaisparov.core.game import ChessGame
from kaisparov.models.architecture import Architecture
from kaisparov.models.factory import build_agent, load_agent, resolve_architecture
from kaisparov.tracking.run import RunManager
from kaisparov.training.config import build_resume_config

CPU = torch.device("cpu")


def _tracked_run(tmp_path, arch: Architecture, *, record_features: bool = True):
    """A run trained as ``arch``, with one checkpoint, as RunManager writes it."""
    config = arch.to_dict()
    if not record_features:
        del config["features"]  # a run from before the key existed
    model, _ = build_agent(arch, CPU)
    run = RunManager(root=tmp_path, model=arch.model, config=config, seed=0, device="cpu")
    run.save_checkpoint(model, 1, {})
    return run.run_id, run.dir / "checkpoints" / "epoch1.pth", model


def _same_weights(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.parametrize("model", ["rgcn", "shared_rgcn"])
@pytest.mark.parametrize("features", ["pieces", "pieces_control"])
def test_a_tracked_checkpoint_comes_back_as_it_was_trained(tmp_path, model, features):
    arch = Architecture(model=model, hidden_dim=6, features=features)
    _, checkpoint, trained = _tracked_run(tmp_path, arch)

    loaded = load_agent(checkpoint, CPU)
    assert loaded.architecture == arch
    assert _same_weights(loaded.model, trained)
    assert loaded.processor.features.name == features
    loaded.model(loaded.processor.graphify(ChessGame()))  # model and processor fit


@pytest.mark.parametrize("features", ["pieces", "pieces_control"])
def test_a_run_that_predates_the_features_key_is_read_off_its_weights(tmp_path, features):
    arch = Architecture(model="shared_rgcn", hidden_dim=6, features=features)
    _, checkpoint, _ = _tracked_run(tmp_path, arch, record_features=False)
    assert resolve_architecture(checkpoint) == arch


def test_an_untracked_file_is_read_off_its_weights(tmp_path):
    arch = Architecture(model="rgcn", hidden_dim=5, features="pieces")
    model, _ = build_agent(arch, CPU)
    path = tmp_path / "loose.pth"
    torch.save(model.state_dict(), path)
    assert resolve_architecture(path) == arch


def test_an_explicit_override_wins_and_a_misfit_says_what_it_tried(tmp_path):
    arch = Architecture(model="rgcn", hidden_dim=6, features="pieces")
    _, checkpoint, _ = _tracked_run(tmp_path, arch)
    assert resolve_architecture(checkpoint, hidden_dim=9).hidden_dim == 9
    with pytest.raises(RuntimeError, match="features=pieces_control"):
        load_agent(checkpoint, CPU, features="pieces_control")


def test_resuming_an_old_run_keeps_the_features_it_was_trained_on(tmp_path):
    # The parent predates the key; the resumed run must not silently switch to the
    # current default, whatever that is.
    for features in ("pieces", "pieces_control"):
        arch = Architecture(model="rgcn", hidden_dim=6, features=features)
        run_id, _, _ = _tracked_run(tmp_path / features, arch, record_features=False)
        config = build_resume_config(run_id, {}, runs_dir=str(tmp_path / features))
        assert config.architecture == arch


def test_an_unknown_feature_set_is_refused_in_a_config():
    from kaisparov.training.config import TrainConfig

    with pytest.raises(ValueError, match="Unknown node feature set"):
        TrainConfig.from_dict({"features": "does_not_exist"})
