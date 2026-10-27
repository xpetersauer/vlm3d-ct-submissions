# Documentation validation — 2026-10-27

GitHub upload helpers were tested offline on synthetic files: lossless splitting,
exact-hash mismatch rejection, no silent preparation overwrite, public/draft
availability gates, and remote asset size/digest comparison. No actual GitHub
repository, release, or binary upload was created during these tests.

- All 86 collected text-file hashes match the uploaded evidence index.
- All 49 packaged historical source files match their collected snapshots byte
  for byte. No inference bug was silently repaired in the published snapshots.
- Methodology word counts: Task1 263; Task2 249; Task3 255.
- Index confirms missing only the old Task-3 baseline-fixed Docker archive;
  skipped and error lists are empty.
- Identical Task-1 weight hashes, repaired versus unreachable entrypoint order,
  and all-zero graph mixing coefficients were checked programmatically.
- Basic credential-signature scan found no matching hardcoded tokens/keys.
  This is not comprehensive privacy, copyright or license clearance.
- No binary checkpoints were loaded, and no Docker/GPU inference or official
  evaluation was run. Archive/weight hashes were computed by the Mac collector,
  not independently rehashed from binary files on this machine.

Remaining unresolved fields are explicit: Task1/3 official score mapping, full
platform IDs where needed, historical optimizer/training metadata, team details,
license decisions and authorized weight URLs. Documentation must not imply that
those checks have passed. Source snapshots have not been extracted from retained
Docker archives to establish binary source-to-submission equivalence.
