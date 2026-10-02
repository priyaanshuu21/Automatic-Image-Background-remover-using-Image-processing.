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
| 4 | `distance` | `float32` Lab distance to the background model in `[0, 1]` (Lab 100 = completely different) | `color.estimate_background_model` + `color.background_distance_map` |
| 5 | `edges` | Boolean Canny edge map (`canny_low`/`canny_high` fractions + hysteresis) | `edges.canny` on `fundamentals.to_gray` output |
| 6 | `seed_mask` | Boolean raw foreground where `distance >= 0.5` | fixed halfway threshold on `distance` |
| 7 | `grown_mask` | Boolean background grown from border seeds (`grow_tolerance`, `grow_connectivity`), blocked by the dilated edge barrier (`use_edge_barrier`) | `segmentation.region_growing` on Lab with `edges.edge_barrier` as the gate |
| 8 | `cleaned_mask` | Boolean foreground: ungrown area minus barrier pixels that the colour model calls background, then small-object removal, hole filling and `keep_largest` components | `morphology.remove_small_objects`, `morphology.fill_holes`, `pixels.label_components` |
| 9 | `alpha` | `float32` matte in `[0, 1]`, feathered (`feather_sigma`) only inside a `band_width` boundary band, working resolution | `refine.alpha_from_mask` |
| 10 | `rgba` | Final `uint8` RGBA at the **original** resolution: halo-decontaminated colours composited with the upscaled matte | `refine.decontaminate_edges`, `refine.upscale_mask`, `refine.compose_rgba` |

Notes:

- The background model fits one or two Lab modes (`n_background_modes`)
  on the `border_frac` frame with median + MAD trimming (`trim_k`), so a
  few foreground pixels touching the border cannot drag it.
- Stage 7 grows on the Lab field through the channel mean with
  `reference="seed"`; stage 8 resolves barrier pixels the growth could not
  reach by falling back to the stage-4 colour distance.
- Reference scores on the synthetic fixtures: circle IoU 1.0000
  (foreground 20.1%), rectangle IoU 0.9958 (foreground 17.5%).
