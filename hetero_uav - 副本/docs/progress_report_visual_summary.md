# Progress Report Visual Summary

Current progress in one sentence: the framework and fixed-capacity 3v2-to-5v4 transfer phenomenon are established, but strict algorithm superiority is not yet proven.

## Displayable figures
- `outputs\progress_report_figures\fig01_experiment_pipeline.png`: Experiment Pipeline (algorithm flow).
- `outputs\progress_report_figures\fig02_method_comparison_bar.png`: Method Comparison (metric results).
- `outputs\progress_report_figures\fig03_transfer_quality_3v2_vs_5v4.png`: 3v2 Seen vs 5v4 Zero-Shot (metric results).
- `outputs\progress_report_figures\fig04_transfer_retention.png`: Transfer Retention (metric results).
- `outputs\progress_report_figures\fig05_ablation_evidence.png`: Component Evidence (metric results).
- `outputs\progress_report_figures\fig06_training_curves.png`: Training Curves (metric results).
- `outputs\progress_report_figures\fig07_trajectory_3v2_normal_best.png`: 3v2 Normal Best Trajectory (ACMI trajectory).
- `outputs\progress_report_figures\fig08_trajectory_5v4_zero_shot.png`: 5v4 Zero-Shot Trajectory (ACMI trajectory).
- `outputs\progress_report_figures\fig07_trajectory_3v2_representative.png`: cropped 3v2 representative combat trajectory for slides.
- `outputs\progress_report_figures\fig08_trajectory_5v4_representative.png`: cropped 5v4 representative attack-transfer trajectory for slides.
- `outputs\progress_report_figures\fig09_paper_readiness_gap.png`: Paper-Readiness Gap (paper-readiness gap).
- `outputs\progress_report_figures\fig10_progress_summary_dashboard.png`: Progress Summary Dashboard (dashboard).

## What to say for each figure

- Fig. 1: explain the experimental pipeline and the fixed-capacity scope.
- Fig. 2: compare current variants without claiming statistical superiority.
- Fig. 3-4: explain that win rate transfers while elimination quality drops.
- Fig. 5: present component evidence for wrapped heading and geometry curriculum.
- Fig. 6: show training progress trends from existing logs.
- Fig. 7-8 original full-episode views: use as complete trajectory references.
- Fig. 7-8 representative views: use in the main report slides because they crop around launch/hit windows and show direction, launch, and hit markers.
- Fig. 9-10: close with safe claims and remaining gaps.

## Trajectory caveats

The 5v4 representative trajectory is selected to show attack-transfer behavior. It does not mean every 5v4 episode is an elimination win.

The current MAV behavior is better described as forward-survival / loose support. Do not claim that the trajectory fully matches a strict rear-support MAV behavior from the heterogeneous MAV/UAV paper.

## Safe wording

Use: fixed-capacity 3v2-to-5v4 zero-shot transfer phenomenon, proof-of-concept, single-run evidence, component evidence, paper-readiness gaps remain.

Avoid: solved zero-shot combat transfer, full TAM-HAPPO reproduction, full BRMA-MAPPO reproduction, statistically superior, arbitrary-scale generalization.
