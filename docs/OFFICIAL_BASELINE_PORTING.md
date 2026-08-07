# Official baseline porting

The runner derives from the unmodified Sinabs v3.1.3 N-MNIST quick start, Specksim guide and
Save Samna Config As Binary guide. All divergences are marked `# COMPATIBILITY_PATCH:`:

1. Fix random seeds at 17.
2. Use inference batch size 1 instead of training batch size 4.
3. Download only the official Tonic N-MNIST test split and use sample 0.
4. Convert genuine Tonic event fields to Specksim uint32 fields; no events are synthesized.
5. Use current `layer2core_map="auto"` and local `make_config`; never call hardware `.to(...)`.
6. Work around Sinabs 3.1.3/Torch 2.10 state-reset incompatibility by replacing state views
   with detached zero buffers in the wrapper, without modifying the installed package.

The model topology and Tonic `ToFrame(..., n_time_bins=100)` preprocessing remain official.
No older Sinabs/Samna package was installed. Results are written to
`experiments/official_baselines/nmnist/artifacts/results.json`; generated flash binaries and
downloaded data are ignored by Git.

Weights use the deterministic Xavier initialization from the official notebook. This smoke test
checks the chain, not accuracy.

## Verified result

- Genuine N-MNIST test sample 0: label 0, 5,293 events, raster `[100, 2, 34, 34]`.
- Official model construction and Sinabs software forward: passed.
- `DynapcnnNetwork` construction and quantized software forward: passed.
- Speck 2f automatic mapping: passed on cores `{0: 0, 1: 1, 2: 2, 3: 3}`.
- Samna configuration generation and validation: passed (`SpeckConfiguration`, valid).
- Specksim inference: passed; it emitted no output spikes with the untrained deterministic weights.
  Sinabs 3.1.3 emits warnings for ignored wrapper/container modules while including mapped layers.
- Local flash binary generation: passed, 362,496 bytes. No flash write occurred.
- Real hardware: unavailable and never accessed.
