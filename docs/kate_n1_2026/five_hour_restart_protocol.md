# Five-Hour Restart Protocol

Date: 2026-06-27

## Purpose

The user requested that the segmentation benchmark keep moving in restartable
work blocks. Codex cannot wake itself outside an active session, but each resumed
session should continue from this protocol instead of re-planning from scratch.

## Active Goal

Complete the reproducible SOTA brain MRI segmentation benchmark defined in:

```text
docs/kate_n1_2026/research_goal_and_completion_criteria.md
```

Current priority is Criterion 5 and Criterion 7:

- apply TTA label-ensemble scoring to real outputs from multiple methods;
- validate whether TTA uncertainty predicts test-retest or consensus error.

## Restart Checklist

At the start of each resumed block:

1. Check `git status --short`.
2. Check latest commit with `git log -1 --oneline`.
3. Read `data/kate_n1_2026/method_status_matrix.csv`.
4. Do not touch unrelated dirty brain-age files unless the user asks.
5. Pick the highest-priority incomplete criterion from
   `research_goal_and_completion_criteria.md`.
6. Make one concrete reproducible improvement, verify it, and commit only small
   code/manifests/CSV/reports.

## Current Completed Restartable Block

Two real label-ensemble branches are now populated:

- SynthSeg 2018 9-angle rotation sweep;
- FastSurfer Long v2 2018 +/-3 degree rotation pair.

## Next Block

Preferred next block:

1. Build the first test-retest dataset manifest for segmentation/TTA validation.
2. Start with SIMON if raw or suitable T1 derivatives are accessible.
3. If raw SIMON is unavailable, record the blocker and use available derivatives
   only as secondary evidence.
4. Connect TTA metrics to repeatability metrics, not only to single-scan volume
   summaries.

Fallback next block:

1. Add a BrainChop `tissue_fast` TTA/QC comparator if anatomical test-retest
   inputs remain blocked.
2. Keep it QC-only and do not promote it as anatomical segmentation.

## Stop Condition For A Block

End a block only after one of these is true:

- a focused commit is created;
- a real blocker is documented in a tracked report/status row;
- a long-running process has been started with a clear log path and status
  command.
