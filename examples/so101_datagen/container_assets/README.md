# Imported containers

Adapted from NVIDIA [SimReady Assets](https://huggingface.co/datasets/nvidia/simready-assets), revision
`4763fbfaa4f8e97658e68b57d01b4c86659e08b5`, and [Palatial NV Asset Library](https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library), revision `69a87329036cec0440403351c12230970c4e67d8`.
Both sources are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: NVIDIA;
Palatial in collaboration with NVIDIA Robotics, respectively. Individual metadata records source,
revision, license, albedo paths/checksums and conversion details.

| Style | Source asset |
|---|---|
| kitchenware_tray | NVIDIA kitchenware_tray_a01 |
| crate | NVIDIA crate_blue_a01 |
| metal_tray_small | NVIDIA container_system_sterilising_b05 |
| metal_tray_medium | NVIDIA container_system_sterilising_b04 |
| metal_tray_long | NVIDIA container_system_sterilising_b02 |
| wooden_tray | PalatialSim D00984 |
| white_dish | PalatialSim D00935 |

Visual USD meshes are triangulated, centered in XY, aligned with the long axis along x, and normalized
by a conservative usable floor short half-width. Original face-varying UVs are retained; authored
repetition is baked into the coordinates (crate 3×, steel 4×). Original albedo PNGs are converted locally under
the source license; new steel/wood/white textures are resampled to at most 1024 px. MuJoCo's classic
renderer uses albedo and material shading, including stronger steel specularity; normal/ORM maps and
full PBR reflections are not imported.

Collision meshes use CoACD decomposition or the source's authored convex parts. For the new steel,
wood and white assets, overlapping decomposed floor pieces are removed from the known empty usable
interior; exterior wall hulls remain. A continuous support slab matches the measured source floor
height. Its contact response prevents tunneling or impulses at floor seams. Probe-grid regressions
verify that small dropped objects settle across the usable interior of every imported asset.

At runtime XY scales preserve aspect ratio; Z scales independently to the configured wall height.
The crate is an explicitly low-profile adaptation. Container floor height and usable extents follow
the scaled asset. Imported long axes are capped at a configurable 32 cm by default. Procedural bins
remain the default; choose one imported style with `generation.container.style`, or a weighted mix
with `generation.container.style_weights`.

Imported containers use their source appearance with probability 0.5, configured through
`generation.container.original_texture_prob`. Set 0 for sampled colors only, or 1 for original
appearances wherever labels remain unambiguous. Native albedos use white tint. Metadata supplies
material names and color annotations: blue plastic, gray steel, brown wood, or white dish. Repeated
native labels fall back to distinct sampled colors. Scene sidecars record the sampled material.

Generated `.obj` and `.png` files are ignored by Git. Only metadata, attribution and this setup
recipe are tracked. They are **not downloaded again for each episode**. A fresh checkout needs this
one-time preparation before imported styles can be used; procedural bins need no preparation.
Existing scanned-object assets continue using their separate download caches.

Use an isolated conversion environment so these dependencies do not change the generator environment
or `uv.lock`:

```bash
uv venv /tmp/so101-container-tools
uv pip install --python /tmp/so101-container-tools/bin/python \
  usd-core==26.11 coacd==1.0.14 scipy==1.18.1 numpy==2.2.6 pillow==12.3.0 huggingface-hub
/tmp/so101-container-tools/bin/python examples/so101_datagen/prepare_containers.py
```

The script fetches only the selected source asset packages at the pinned revisions, converts them,
and reuses complete local outputs on subsequent invocations. Select a subset with
`--styles wooden_tray white_dish`; `--force` rebuilds it. For offline conversion, use
`--source-root /path/to/packages`, containing the source asset directories listed above. Source and
converted texture checksums and collision-part counts are verified before replacing local files.
Normal episode generation does not import USD or CoACD.

The imported colliders currently have 29–72 convex pieces per container, compared with five boxes
for the procedural bin. These follow authored/decomposed rim and handle details; this is a faithful
conversion, not a minimal collision representation. Replacing them with fewer hulls requires checking
that the hollow interior, floor stability and grasp clearances still behave correctly.
