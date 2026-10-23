This is a modified, reduced preprocessing and feature-extraction distribution of
[CLAM](https://github.com/mahmoodlab/CLAM), based on upstream commit
`53e2409d4a8189c682c173382964a85f114f923c` and subsequent local SDPC/MPP adaptations.

Original CLAM authors: Mahmood Lab and CLAM contributors.
CLAM reference: Lu et al., *Data-efficient and weakly supervised computational
pathology on whole-slide images*, Nature Biomedical Engineering (2021).
DOI: 10.1038/s41551-020-00682-w.

The modified distribution retains the upstream GNU GPL version 3 license in
LICENSE.md. Modified on 2026-10-23: retained segmentation, contour checks,
coordinate stitching and slide readers; added a portable batch CLI, local encoder
loading, output validation, documentation and tests; removed MIL training,
evaluation, heatmap generation, obsolete patch-image storage and cohort scripts.

The Windows SDPC decoder is installed separately through `sdpc-win==3.0`.
Decoder binaries and model checkpoints are not redistributed here. Their licenses
and access requirements remain those of their respective providers. UNI and
H-optimus weights must be obtained through their official access procedures.
