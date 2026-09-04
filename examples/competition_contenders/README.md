# GPU competition contender examples

These examples produce standalone Git-ready contender directories for both live
tracks. Each tree contains only `Dockerfile`, `run.sh`, `gpu_transform.cu`, and
one immutable `variant.env`. The CUDA helper processes every decoded RGB frame on
the assigned GPU; `nvidia-smi`, a device allocation, and a synchronized CUDA
kernel all fail closed before media output. FFmpeg only decodes and performs the
final H.264 encode. No scoring model is included: collected bytes go to VidAIO's
canonical CPU scoring/audit path.

The three earning profiles intentionally trade quality against byte rate, so testnet
can observe ranking and distinct output digests. A fourth `baseline` profile per track
is the non-earning reference executable. They are illustrative examples, not claims
about which profile wins a real committed corpus.

The upscaling profiles consume the complete task contract from the runner's read-only
hidden `.vidaio-next-upscale-task-<input_sha256>` sidecar. Each file is canonical JSON
plus one newline, e.g.
`{"target_height":1080,"target_width":1920,"upscale_factor":2}\n`. A batch may mix
2x and 4x tasks, and the contender must emit the exact committed dimensions rather than
guessing from rounded low-resolution geometry. The sidecar exposes no pristine
reference identity or bytes. Compression profiles use scale 1 and receive no sidecar.

Materialize a new directory for every independently enrolled solution. The
command refuses an existing destination:

```sh
python examples/competition_contenders/materialize.py \
  --track compression --variant quality \
  --destination /tmp/vidaio-next-compression-quality
python examples/competition_contenders/materialize.py \
  --track compression --variant compact \
  --destination /tmp/vidaio-next-compression-compact
python examples/competition_contenders/materialize.py \
  --track upscaling --variant quality \
  --destination /tmp/vidaio-next-upscaling-quality
python examples/competition_contenders/materialize.py \
  --track upscaling --variant compact \
  --destination /tmp/vidaio-next-upscaling-compact
python examples/competition_contenders/materialize.py \
  --track compression --variant baseline \
  --destination /tmp/vidaio-next-compression-baseline
python examples/competition_contenders/materialize.py \
  --track upscaling --variant baseline \
  --destination /tmp/vidaio-next-upscaling-baseline
```

Initialize each generated directory as its own fresh remote Git repository. Do
not put repository credentials in its URL or files. Record both enrolled pins:

```sh
git -C /tmp/vidaio-next-compression-quality init
git -C /tmp/vidaio-next-compression-quality add Dockerfile run.sh gpu_transform.cu variant.env
git -C /tmp/vidaio-next-compression-quality commit -m 'vidaio-next compression quality'
git -C /tmp/vidaio-next-compression-quality rev-parse HEAD
git -C /tmp/vidaio-next-compression-quality rev-parse 'HEAD^{tree}'
```

Push through the operator's ordinary credential helper, enroll the
credential-free HTTPS URL plus exact commit/tree pins, and let the production
`GitRepoProvider` make a new isolated checkout. Never initialize one generated
tree and then mutate its profile after enrollment.

Initialize and pin each baseline tree exactly the same way, but never enroll it under
a hotkey. Put its credential-free repository URL, exact commit/tree SHA, archived
artifact/provenance identities, version, and image digest in the manifest's `baseline`
block. Before anchoring, build that exact tree with the same fresh
Modal runner contract that will execute the competition and retain the returned
`image_digest`; the authenticated anchor request must use that exact value. The
orchestrator opens the commitment again before building and halts before any contender
runs if the manifest tree, active tokenomics digest, or built baseline image differs.

The required `reward_param_digest` is not an operator-selected label. It is SHA-256
over canonical JSON of the full active `TokenomicsConfig`; the control service rejects
any supplied digest other than its own `reward_parameter_digest(config)` result. This
binds the pre-enrollment commitment to the policy that will actually pay the epoch.

The base CUDA images are pinned to their linux/amd64 registry digests. Image
builds require network access to the Ubuntu package repository; contender runs
do not. The Modal runner passes no secret, identity token, volume, or network to
the Sandbox and forces a new Sandbox per batch.
