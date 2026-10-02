# Syllabus Map

How each classic image processing topic of the mini-project syllabus maps to
an implemented, tested module. Every entry names the module, its key
functions and the test file that locks the behaviour.

| Topic | Module | Key functions | Tests |
|-------|--------|---------------|-------|
| Image model, sampling, quantisation | `fundamentals.py` | `describe_image`, `to_gray`, `spatial_resample`, `requantize`, `resize_max_side` | `test_fundamentals.py` |
| Pixel relationships, connectivity | `pixels.py` | `neighbor_offsets`, `neighbors`, `distance`, `label_components`, `component_stats`, `border_touching_labels` | `test_pixels.py` |
| Point processing | `point_ops.py` | `apply_lut`, `negative`, `contrast_stretch`, `threshold_binary`, `intensity_slice`, `log_transform`, `power_law` | `test_point_ops.py` |
| Neighbourhood filtering | `filters.py` | `correlate2d`, `convolve2d`, `box_kernel`, `gaussian_kernel`, `mean_filter`, `gaussian_filter`, `median_filter`, `unsharp_mask`, `laplacian_sharpen`, `denoise` | `test_filters.py` |
| Histograms | `histogram.py` | `compute_histogram`, `cumulative_distribution`, `equalize_gray`, `equalize_color_value`, `histogram_stats` | `test_histogram.py` |
| Colour spaces and colour segmentation | `color.py` | `rgb_to_hsv`, `hsv_to_rgb`, `rgb_to_hsi`, `rgb_to_lab`, `lab_to_rgb`, `euclidean_color_distance`, `hsv_slice`, `estimate_background_model`, `background_distance_map` | `test_color.py` |
| Thresholding and region segmentation | `segmentation.py` | `otsu_threshold`, `otsu_mask`, `region_growing`, `hysteresis_threshold`, `region_adjacency`, `merge_regions`, `merge_similar_regions` | `test_segmentation.py` |
| Binary and grayscale morphology | `morphology.py` | `structuring_element`, `dilate`, `erode`, `opening`, `closing`, `hit_or_miss`, `thinning`, `thickening`, `fill_holes`, `remove_small_objects`, `remove_objects_touching_border` | `test_morphology.py` |
| Edge detection | `edges.py` | `gradients`, `gradient_magnitude`, `gradient_direction`, `sobel_edges`, `prewitt_edges`, `laplacian_edges`, `log_edges`, `canny`, `canny_stages`, `edge_barrier`, `overlay_edges` | `test_edges.py` |
| Frequency transforms (analysis only) | `transforms.py` | `dft2`, `idft2`, `magnitude_spectrum`, `dct2`, `idct2`, `hadamard_matrix`, `hadamard_transform` | `test_transforms.py` |
| Alpha refinement and export | `refine.py` | `alpha_from_mask`, `decontaminate_edges`, `compose_rgba`, `upscale_mask` | `test_refine.py` |
| Pipeline orchestration | `pipeline.py` | `BackgroundRemovalPipeline`, `PipelineResult`, `STAGE_KEYS` | `test_pipeline.py` |
| Visualisation | `viz.py` | `checkerboard`, `composite_over_checkerboard`, `stage_to_rgb`, `save_stage_grid` | `test_viz.py` |
| Command line | `cli.py` | `build_parser`, `main` (`remove`, `batch`, `stages`, `analyze`, `info`) | `test_cli.py` |
| Desktop app | `gui_controller.py`, `gui.py` | `GuiController`, `BackgroundRemoverApp`, `launch` | `test_gui.py` |
| Configuration and I/O | `config.py`, `io.py` | `PipelineConfig`, `load_image`, `save_image` | `test_config.py`, `test_io.py` |
| Architecture guard | `tests/test_architecture.py` | (parametrised over every module) | self-testing |

Deliberately decoupled: `transforms.py` never feeds the pipeline (frequency
analysis is inspection-only), and `viz.py`/`gui.py` never influence a mask.
