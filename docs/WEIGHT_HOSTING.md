# Hosting final weights

The evidence collector intentionally did not upload binary weights. They remain
on the participant's Mac in Downloads/vlm3d_final/vlm3d_weeklong_deliverables/.
Use exactly these files after confirming redistribution permissions:

| Submission | Filename | SHA-256 |
|---|---|---|
| cd1082ec | task1_weights.zip | 5f252b1b2da96516303762fde1ec28af428849edb83afb7150435562b71d2d27 |
| a8799c78 | task2_weights.zip | ab8417b0e0b7395988ecdd748ad1fc543fc6f1d9866a420b95150d28bf793ebc |
| 888a2309 | task3_weights_full_47149.zip | b06dea5916646cda857f815d595e609523384566fcd602cbc34ad3decf13722a |

The Task2 ZIP is approximately 2.9 billion bytes. Choose authorized hosting with
appropriate single-file limits; do not assume a normal GitHub repository upload
can accept it. If hosting requires splitting an archive, publish all pieces,
reassembly instructions and the SHA-256 of the reconstructed original. Do not
silently repackage or change model contents without recording a new artifact.

Once hosted, add a versioned URL beside each filename in CHALLENGE_FORM.md and
provenance/ARTIFACTS.json, and change the form's availability selection to the
actual host. Upstream pretrained model links alone do not supply your trained
heads, LoRA state, pathology classifier, calibration or reranker.

Keep scans/reports, access tokens, private account records and Docker exports
out of normal Git. Repository visibility and licenses must be reviewed before
public upload; no automated publication is performed by this package.
