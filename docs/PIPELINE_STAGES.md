# Pipeline Stages

`BackgroundRemovalPipeline.run` records these ten stages in
`PipelineResult.stages` (see `STAGE_KEYS`), so every decision of the
automatic removal is inspectable. Save them with
`bgremover viz.save_stage_grid` (the `stages` CLI subcommand) or
`scripts/run_demo.py`.

| # | Key | Content | Produced by |
|---|-----|---------|-------------|
| 1 | `input` | Original `uint8` RGB image, copied | `bgremover.io.load_image` / caller array |
| 2 | `resized` | Working-resolution `uint8` RGB (`work_max_side`) | `fundamentals.resize_max_side` (`cv2.resize`, `INTER_AREA`) |
| 3 | `denoised` | Smoothed working image (`denoise`: gaussian/median/none) | `filters.denoise` |
| 4 | `distance` | `float32` Lab distance to the background model in `[0, 1]` (Lab 100 = completely different) | `color.estimate_zoned_background_model` (16 perimeter zones, edge/variance/model-verified) + `color.background_distance_map` |
| 5 | `edges` | Boolean Canny edge map (`canny_low`/`canny_high` fractions + hysteresis) | `edges.canny` on `fundamentals.to_gray` output |
| 6 | `seed_mask` | Boolean raw foreground: Otsu on the **non-denoised** distance map, so blur never shifts the boundary | `segmentation.otsu_mask` on raw Lab distances |
| 7 | `grown_mask` | Boolean background grown from verified border seeds: Lab distance **and** sub-0.12 edge energy required, blocked by the radius-2 dilated edge barrier (`use_edge_barrier`) | `segmentation.grow_background` on Lab with `edges.combined_edge_response` + `edges.edge_barrier` as the gate |
| 8 | `cleaned_mask` | Boolean foreground: grown complement, plus connected sub-threshold background absorbed back (chromatic gate), minus barrier pixels the colour model reclaims, plus saliency seeds, then small-object removal, hole filling, `keep_largest` components, topological solidification and raw-colour boundary relocalisation | `morphology.remove_small_objects`, `morphology.fill_holes`, `color.saliency_foreground_seeds`, `morphology.solidify_foreground`, `pixels.label_components` |
| 9 | `alpha` | `float32` matte in `[0, 1]`, feathered by normalised spatial-distance weighting inside a 3-5 px trimap band, working resolution | `refine.guided_alpha_feathering` (`morphology.trimap_from_mask`) |
| 10 | `rgba` | Final `uint8` RGBA at the **original** resolution: halo-decontaminated colours composited with the upscaled matte | `refine.decontaminate_edges`, `refine.upscale_mask`, `refine.compose_rgba` |

Notes:

- The background model fits one or two Lab modes (`n_background_modes`)
  on `border_frac` frame zones verified in three passes (no edge
  crossings, no variance outliers, agreement with the fitted modes),
  with median + MAD trimming (`trim_k`), so a torso touching the bottom
  edge or shoulders crossing the sides cannot drag it.
- Stage 7 grows on the Lab field with an adaptive tolerance
  (`max(grow_tolerance, trim_k * spread)`) and an energy gate of 0.12
  on the fused Sobel+Canny response, so growth cannot jump facial
  boundaries or clothing edges even when tones look alike.
- Stage 8 resolves barrier pixels the growth could not reach by falling
  back to the colour model (kept only when the raw Otsu seed agrees),
  absorbs smooth shadow-like areas through a chromatic gate (shadows
  preserve background hue), and force-fills every region enclosed by
  the silhouette, so hollow faces or spectacle lenses stay solid.
- Reference scores: circle IoU 1.0000, rectangle 1.0000, studio mug
  0.9995, portrait with border-touching clothing 0.8449, two-tone
  background 0.9949 (all 388 tests passing).
