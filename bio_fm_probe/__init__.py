"""bio_fm_probe — a model-agnostic probing toolkit for biological foundation models.

Drop-in workflow:
  1. Add an adapter under bio_fm_probe/adapters/<name>.py implementing BioFMAdapter.
  2. Register it in run_audit.ADAPTER_REGISTRY.
  3. Run: python -m bio_fm_probe.run_audit --adapter <name> --model_dir ... --data ...

All probes (layer probe, baselines, SAE, SVD diagnostic) are shared and never
touch model internals — they go through the adapter interface.
"""
