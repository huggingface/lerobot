# SO-101 synthetic data generator

Generate local annotated episodes with front and wrist cameras, then optionally merge and upload them.
The default task mix remains `place_in: 0.5`, `place_near: 0.3`, `pick_lift: 0.2`.

## Choosing tasks

Set `generation.tasks.weights` in the config or replace the dictionary from the CLI. Weights are
relative, need not add up to one, and must be finite and nonnegative with a positive total. Omitted
and zero-weight tasks are disabled. Unknown task names are rejected.

For example, generate a small local collection-only dataset:

```bash
MUJOCO_GL=egl PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m examples.so101_datagen.generate \
  --repo_id=local/so101_collect --output_dir=outputs/so101_collect \
  --num_shards=4 --episodes_per_shard=10 --num_workers=6 \
  --generation.tasks.weights='{"collect_objects":1.0}' \
  --push_to_hub=false
```

A mixture uses the same interface:

```bash
--generation.tasks.weights='{"place_in":0.5,"place_near":0.2,"collect_objects":0.3}'
```

Choose a fresh output directory when changing the recipe; completed shards are reused on resume.

## Longer tasks

`collect_objects` collects two named objects into one container. Either order is permitted; the
expert picks the initially nearer object first and reserves separate landing positions. It keeps
one physics simulation, reuses the wrist pose between transfers, and checks that earlier placements
remain inside the container. Rolling and temporary rim/gripper contact during later transfers are
allowed; final success requires both complete footprints inside and neither object touching the
gripper. Collection bins use 14–20 cm interiors, configured through
`generation.container.collection_inner_size`; single-object containers retain their 8–13 cm range.
Scenes without enough capacity and wall clearance are rejected without changing object sizes.

Each transfer has its own retry budget (`generation.recovery.max_attempts`, default 3). Deliberate
misses are sampled only for the episode's first grasp. Local recovery does not reset completed
transfers. Single-object tasks keep their 40-second limit; longer tasks use
`generation.max_long_episode_seconds` (default 120). Longer tasks require `prepare()` followed by
`simulate()`; static `prepare(script=True)` is intentionally unsupported because later transfers
need observed physics.

Plans use objective IDs so retries do not add future failures to the initial plan. Generation
sidecars record the objective IDs, transfer/object identities, grasp outcomes and completed-transfer
frame boundaries. Each declared transfer is an intentional action with its own identity.

`ordered_transfers` performs two explicitly ordered transfers. Half the proposed scenes put one object
in a container and the other next to a stationary third object, sampling both orders. The other half
say “put A in the container, then put B next to A”: A is the moved reference, so B also goes inside
the container beside A's observed current position. Final success checks both containment and adjacency.
Ordered tasks use collection-sized procedural containers to reserve enough capacity for this variant.
All variants verify the completed transfer sequence and preserve earlier goals during local retries.
Enable with `--generation.tasks.weights='{"ordered_transfers":1}'`.

`clear_container` starts with two named objects inside a shallow tray and moves them to separate free
table locations, in either order. It uses one simulation, accounts for the tray floor in spawning,
grasp positions and lift checks, and preserves completed table placements through local retries.
Initial contents must fit and be reachable; unrelated objects stay outside the tray. Table destinations
are reserved before execution and refreshed against observed geometry if an earlier placement rolls
into one. Final success requires both objects physically supported on the table, clear of the tray and
each other, with the tray empty. Clearance planning uses the actual rotated tray footprint. Trays use
`generation.container.clearing_wall_height` (1.2–2.2 cm), independently of other task families.
Enable with `--generation.tasks.weights='{"clear_container":1}'`; it remains opt-in.


`sort_objects` sorts three objects from two color groups into two distinctly named containers. Two
objects share one color and destination, with distinct shape names (for example, a red block and a
red cylinder); the third uses another color and container. This family uses colored primitives so
color-group instructions remain true, preserving their existing size ranges. Other families retain
scanned objects and source textures. The expert compares estimated travel distance for all six transfer orders,
reserves non-overlapping landing points and refreshes free floor space from observed placements.
Every completed object must stay in its assigned bin; a partial or incorrectly sorted result fails.
Procedural interiors use `generation.container.sorting_inner_size` (12–16 cm), allowing room for two
containers in the reachable workspace; imported sizes retain their configured bounds. Layouts without
capacity, reachable centers or unique labels are rejected. Defaults remain unchanged.
Enable with `--generation.tasks.weights='{"sort_objects":1}'`.


`put_all_in` transfers **every graspable object on the table** into one named container, starting
with two or three objects. It differs from `collect_objects`, which selects two explicitly named
objects and can leave distractors behind. Source sizes and materials are retained. Collision-free
landing positions reserve 8 mm wall margins and 12 mm footprint gaps; capacity and reachability
are checked before execution, and occupied reservations refresh against observed physics.

`arrange_row` places three named objects in a specified left-to-right order as viewed by the static
front camera. It reserves reachable table centers, requires final centers within 2 cm, and rejects
overlapping footprints or an incorrect projected order. Other table objects remain protected.

`swap_objects` exchanges two original positions through a reserved free buffer: A to buffer, B to
A's original position, then A to B's original position. Three distinct transfer identities preserve
both intentional picks of A in the plan. The buffer goal remains protected until A is intentionally
picked again; B's completed final goal remains protected throughout. Final centers must be within
2 cm of the original positions.

`unblock_then_place` moves a blocker aside before placing its target in a bin. Initial scenes keep
both objects visible and physically separate. The selected target approach is checked against a
counterfactual with the blocker restored to its settled initial pose: the original blocker must
obstruct the gripper, while the parked blocker must clear it. Each preflight and executed approach
is rechecked, including retries and carry-drop recovery. The proof is stored as `task_evidence` in
generation sidecars. This initial family uses two objects, with unchanged size ranges.

All four families reuse continuous physics, local retries and the 120-second long-task limit.
They are opt-in through the same dictionary, for example:

```bash
--generation.tasks.weights='{"put_all_in":1,"arrange_row":1,"swap_objects":1,"unblock_then_place":1}'
```

Imported containers are opt-in through `generation.container.style`:

| Style | Native appearance |
|---|---|
| `kitchenware_tray` | Blue plastic tray |
| `crate` | Blue crate, adapted to a low wall height |
| `metal_tray_small`, `metal_tray_medium`, `metal_tray_long` | Stainless-steel tray proportions |
| `wooden_tray` | Wooden serving tray with handles (PalatialSim D00984) |
| `white_dish` | White dish (PalatialSim D00935) |

Or mix them within one run using relative weights:

```bash
--generation.container.style_weights='{"metal_tray_small":1,"metal_tray_medium":1,"metal_tray_long":1,"wooden_tray":1,"white_dish":1}'
```

An empty style-weight dictionary uses `style`. Unknown names, negative or non-finite weights, and a
zero-sum nonempty mix are rejected. Procedural containers remain the default. Imported usable short
interiors use `generation.container.asset_inner_size` (12–15 cm), further bounded by
`generation.container.max_asset_length` (32 cm overall long axis). XY proportions are retained;
Z follows `generation.container.wall_height` (or `clearing_wall_height` for clearing tasks). Their visual meshes use original UVs and convex wall
collisions with one continuous supporting floor at the source floor height. See
[asset attribution and conversion notes](container_assets/README.md).

Imported containers use their original source albedo in 50% of eligible samples. Adjust
`generation.container.original_texture_prob` (0–1); use 1 to favor natural metal, wood and white
appearances. Names and color annotations match the actual material, and duplicate names fall back to
distinct recolored assets. Sampled scenes record style and appearance for replay.
Scanned GSO and RoboCasa objects already retain their native textures. Primitives retain their named
colors because they do not have a source texture.

New task families remain opt-in pending visual review; the default three-task weights are unchanged.
