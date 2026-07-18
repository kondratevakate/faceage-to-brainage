# Repository branch model

The repository uses four long-lived branches with distinct responsibilities.

| Branch | Purpose | Content rule |
| --- | --- | --- |
| `main` | Integrated, reproducible project state | Small code, curated results, methods, and documentation from both workstreams |
| `faceage` | Face-age and photo/MRI avatar workstream | Face rendering, face-age, avatar geometry, and MRI face-target evaluation |
| `brainage` | Brain-age and MRI robustness workstream | Brain-age models, preprocessing, SIMON/travelling-subject analyses, and uncertainty/QC |
| `gh-pages` | Published website only | Deployment copy of `project_page/index.html` and `project_page/assets/` |

`faceage` and `brainage` start from the same integrated baseline. Workstream
changes are developed on the corresponding branch and merged into `main` only
when the code, small results, and scientific interpretation are reproducible.
The branches are organizational boundaries, not independent repositories or
claims that either modality is validated.

The editable website source lives under `project_page/` on `main`. The
`gh-pages` branch is a deployment artifact and should not contain research code,
raw data, model weights, caches, or unpublished subject-level outputs.

Short-lived topic branches should use `faceage/<topic>` or `brainage/<topic>`
when a change needs review before reaching the workstream branch. Obsolete
branch tips may be preserved as `archive/...` tags before their remote branches
are deleted.
