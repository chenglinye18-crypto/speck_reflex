"""No-hardware DynapCNN, Samna configuration and Specksim helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.specksim import SpecksimNetwork, from_sequential

from .model import INPUT_SHAPE


def build_dynapcnn(model, *, device_type: str, dvs_input: bool = False) -> tuple[DynapcnnNetwork, object, bool, str]:
    network = DynapcnnNetwork(model, input_shape=INPUT_SHAPE, discretize=True, dvs_input=dvs_input)
    # COMPATIBILITY_PATCH: current Sinabs uses layer2core_map, not the old
    # chip_layers_ordering argument. make_config is offline and opens no device.
    config = network.make_config(device=device_type, layer2core_map="auto")
    valid, message = samna.speck2f.validate_configuration(config)
    return network, config, bool(valid), str(message)


def specksim_events(raw_events: np.ndarray) -> np.ndarray:
    events = np.empty(len(raw_events), dtype=SpecksimNetwork.output_dtype)
    for field in ("x", "y", "t", "p"):
        events[field] = raw_events[field]
    if len(events):
        events["t"] -= events["t"][0]
    return events


def run_specksim(network, raw_events: np.ndarray) -> np.ndarray:
    return from_sequential(network, input_shape=INPUT_SHAPE)(specksim_events(raw_events))


def generate_flash_binary(config: object, destination: Path, io_sel: int | None) -> int:
    """Create a local binary only. This function has no hardware-write path."""
    if io_sel is None:
        raise ValueError("--io-sel is required with --generate-flash-binary; no universal safe default is documented")
    config.factory_config.io_sel = io_sel
    binary = samna.speck2f.configuration_to_flash_binary(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(binary))
    return len(binary)
