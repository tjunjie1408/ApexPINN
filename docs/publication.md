# Source, dataset and model publication

This is the intended release structure. No GitHub push or Hugging Face upload is performed by this
preparation. A published artifact is evidence of a release, not proof that unfinished research stages passed.

## GitHub: source and readable evidence

Keep reusable package code, CLI scripts, configurations, thin notebooks, tests, infrastructure recipes
and selected generated reports/figures. Keep the README's status and limitations aligned with the
released evidence. Existing tracked scientific records stay intact; do not rewrite their history to
make the tree look smaller.

Large data/checkpoints, credentials, Terraform state, local instructions, MCP receipts, downloaded tools
and one-off verification scripts are excluded from the source release. Local exclusions and shared
ignore rules serve different purposes; do not force-add private files.

Some historical cloud input archives contain local agent instructions. Do not upload entire workstation
folders or reuse those archives as public datasets. Build a separate publication package from explicit
data/model allowlists. Preserve historical archives locally for compatible preparation and recovery.

## Hugging Face: dataset release

Use a dataset repository for the selected, versioned data release. Include a dataset card describing
provenance, event/season selection, schema, units, exclusions, fold assignment, training-only normalization,
proxy limitations and the difference between development data and the untouched temporal holdout.
Use Parquet and machine-readable file hashes; keep a manifest tied to the source/config revision.

Dataset redistribution terms remain to be established before publication. A data-access library's
software license does not by itself establish the terms for redistributing all upstream data.
No dataset license has been selected by this preparation.
Code and model licenses are separate release decisions; this cleanup does not introduce a license grant.

See [dataset cards](https://huggingface.co/docs/hub/datasets-cards) and
[Hub license metadata](https://huggingface.co/docs/hub/repositories-licenses).

## Hugging Face: model release

Use a model repository for explicitly named research checkpoints and a model card. Label each model's
fold, phi, variant, ensemble member, training/calibration populations and completed evaluation scope.
Retain negative findings and calibration limitations. Do not describe a selected fold model as the
final frozen deployment predictor while T14 remains unimplemented.

An Equinox `.eqx` file requires the matching model template and static configuration to load. Include
the compatible package revision, locked environment, method/config, normalization metadata, learned
noise/calibration information and hashes for the selected weights. Preserve the associated declaration,
job record, evaluation and diagnostics as release evidence. Full optimizer checkpoints may be kept in
a separately labeled resume archive. Renaming an `.eqx` file does not create a Transformers model.

See [model cards](https://huggingface.co/docs/hub/model-cards). Public dataset/model repository names,
licenses, visibility and release scope remain user choices. No placeholder URL is a published artifact.

## Evidence needed for a showcase

Tie the GitHub revision, data revision, model files and reported metrics together with hashes and stable
release identifiers. Verify a fresh download and the documented model-load/prediction path using the
supported remote environment. Keep release verification separate from training and scientific gates.

Before describing the full pit-strategy project as complete, finish the remaining evaluation decisions,
predictor freeze, simulator/benchmarks and holdout stages documented in [project status](status.md).
Until then, the accurate portfolio scope is the completed forecasting pipeline and its recorded evidence.
