# Architecture overview — services, trust boundaries, and data flow

This is the visual entry point to VidAIO. It explains what runs where, which data is
trusted, how a miner output becomes a weight, and how that result can be reproduced on
an ordinary CPU. [VALIDATING.md](VALIDATING.md) covers the operational side of running
a validator against this topology.

## Release snapshot

| Area | State |
|---|---|
| Two-track inference, competition scoring, deterministic weights | Implemented and CI-covered |
| CPU-only own-audit and beacon-audit; findings are report-only | Implemented and CI-covered |
| Control/service topology and GPU-backed miner ingress | Central CPU plane on the authority hosts; one independent CPU/TLS miner-edge host per miner identity |
| Public testnet registrations, live S3/IAM, archive RPC measurements, alert delivery, soak | Proved in the live testnet run and soak |
| Mainnet release | Released — live on netuid 85 after the successful testnet soak |

This distinction matters: the repository can prove deterministic behavior locally, but
only a real deployment can prove credentials, network paths, chain cadence, GPU runtime,
and operational alerting together.

## 0. What this subnet does

VidAIO (Bittensor subnet 85) rewards two video tasks:

- **Compression** — reduce the encoded size while retaining useful visual quality.
- **Upscaling** — reconstruct a higher-resolution result from a degraded miner input.

Those tasks run in two economic arenas:

- **Inference** continuously challenges registered miners and folds their round results
  into an EWMA. Its pool is 80% in IDLE, 60% in a PODIUM window, and 10% in a
  CROWN window. Inside that pool the compression/upscaling split stays 0.8/0.2.
- **Competition** executes submitted solutions in isolated GPU sandboxes. The latest
  successfully applied global result opens one seven-day half-open reward window:
  40% goes to its podium without a breakthrough (PODIUM), or 90% goes to the podium
  when rank one beats the archived executable baseline by at least 5% (CROWN). Outside
  a window, competition receives zero and the protocol sends IDLE's fixed 20% to the
  canonical non-earner sink.

The invariant underneath both arenas is:

> Every number that can change emissions must be independently recomputable from
> durable evidence on CPU.

GPU miners may produce outputs, and competition sandboxes may execute solutions on GPU.
They do not get to produce an opaque, trusted score. The authority computes the score
with the release-locked CPU scoring stack, and auditors independently replay that same
calculation on CPU.

## 1. Logical system map

Measurement and finalization plane:

```mermaid
flowchart LR
    subgraph PRIVATE["Private challenge preparation"]
        OP["Operator ingest"] --> CS["Challenge Service\nmaster, segment, degradation, commit"]
    end

    subgraph GPU["Untrusted GPU execution"]
        IM["Inference miner"]
        CM["Competition image"]
    end

    subgraph AUTH["Owner-operated scoring authority"]
        IV["Inference Validator\ndispatch, receipts, EWMA, archive"]
        CO["Competition Orchestrator\nbuild, run, archive"]
        BR["Baseline Registry\nversioned executable + provenance"]
        SW["CPU Scoring Worker\nmetrics, gates, packet digest"]
        AF["Authority Finalizer\nsnapshot, reward window, vector, epoch log"]
        SA["Authority API\nfinalized pointer"]
    end

    OBJ["S3-compatible evidence store\nsealed live, public after retirement"]
    CHAIN["Bittensor chain\ncommitments and epoch anchor"]

    CS -->|"input plus held-out reference"| IV
    IV <-->|"input / output plus receipt"| IM
    CO <-->|"sandbox run / output"| CM
    BR -->|"active per-track baseline"| CO
    IV <-->|"score request / packet"| SW
    CO <-->|"score request / packet"| SW
    IV -->|"folded inference state"| AF
    CO -->|"committed competition inputs"| AF
    IV -->|"inference evidence"| OBJ
    CO -->|"competition evidence"| OBJ
    BR -->|"baseline archives and provenance"| OBJ
    CS -->|"challenge commitment"| CHAIN
    CO -->|"competition commitment"| CHAIN
    AF -->|"epoch log, _FINALIZED last"| OBJ
    AF -->|"epoch digest"| CHAIN
    AF -->|"verified CROWN source"| BR
    AF --> SA
```

Validator and reporting plane:

```mermaid
flowchart TB
    SA["Authority API pointer"] --> EPOCH["Authenticated epoch inputs"]
    OBJ["S3 epoch log and evidence"] --> EPOCH
    CHAIN["Chain anchor and close-block metagraph"] --> EPOCH

    subgraph VALIDATOR["Each validator identity — separate processes and state"]
        WS["Thin Weight-setter"]
        OWN["Own-auditor\nfull CPU replay"]
        BEACON["Beacon-auditor\nfull CPU replay"]
    end

    EPOCH --> WS
    EPOCH --> OWN
    EPOCH --> BEACON
    CHAIN -->|"future block hash"| BEACON
    WS -->|"set weights and anchor publication"| CHAIN
    WS -->|"publish confirmed landed vector"| OBJ

    OWN -->|"signed report"| AR["Audit Results API"]
    BEACON -->|"signed report"| AR
    CHAIN -->|"verify reporter registration"| AR
    OWN -->|"dispute/inconclusive log plus metric"| HUMAN["Operator alerting\nand manual remediation"]
    BEACON -->|"dispute/inconclusive log plus metric"| HUMAN
    AR --> DASH["Dashboard"] --> HUMAN
```

Three separations in this diagram are intentional:

1. The pristine reference never goes to a miner. The validator receives it through a
   trusted challenge handoff and supplies it only to the scorer and evidence archive.
2. The scoring worker is stateless with respect to durable evidence. The inference
   validator archives inference artifacts; the competition orchestrator archives
   competition artifacts.
3. Auditors report to the Audit Results API and operator alerting. They have no control
   edge into the weight-setter.

The optional customer-serving plane is omitted here to keep the consensus path legible;
its separate registry and gateway relationship appears in the competition section.

## 2. Compute and trust boundary

```mermaid
flowchart LR
    subgraph UNTRUSTED["Untrusted, acceleration allowed"]
        GPUM["GPU inference miner"]
        GC["GPU competition solution"]
    end

    subgraph TRUSTED["Release-locked authority path"]
        CAN["Canonicalize output"]
        CPU["CPU metrics and gates"]
        PKT["Digest-addressed score packet"]
        LOG["Finalized epoch log"]
        EVID["S3 media and bundles\nnamed by the audit manifest"]
    end

    subgraph REPLAY["Independent validator path"]
        ACAN["Canonicalize from evidence"]
        ACPU["Same CPU metrics and gates"]
        CMP["Tolerance-bounded metric and score replay\nexact artifact and weight-digest checks"]
        REPORT["Signed report-only finding"]
    end

    GPUM -->|"video bytes"| CAN
    GC -->|"video bytes"| CAN
    CAN --> CPU --> PKT --> EVID
    EVID -->|"audit-manifest digests"| LOG
    EVID --> ACAN --> ACPU --> CMP --> REPORT
    LOG -->|"expected earnings and weights"| CMP
```

At launch, PieAPP and every other payout-affecting metric are pinned to CPU. A future
GPU scorer is acceptable only if its output is evidence, not authority: a CPU auditor
must still reproduce the earned score within the release-locked tolerance.

## 3. The two scoring lanes

Both tracks enter one canonical scoring contract, but their transformations and gates
are different. The anti-gaming quality delta is measured against the **miner input**,
not against an unrelated pristine-reference baseline.

```mermaid
flowchart TB
    subgraph COMMON["Scoring request snapshots"]
        REF["Pristine reference\nheld out from miner"]
        INPUT["Degraded miner input"]
        OUTPUT["Miner output"]
        CANON["Verify digests, private-copy, canonicalize,\nand probe reference, input, and output"]
    end

    subgraph COMP["Compression lane"]
        CVMAF["Primary VMAF\nreference versus output"]
        CDELTA["Primary plus NEG VMAF\nminer input versus output"]
        CPER["CPU tone, grayscale, and chroma checks\nminer input versus output"]
        CRATE["Byte ratio\noutput versus miner input"]
        CGATE{"All validity and\nanti-gaming gates pass?"}
        CFORM["Compression formula\n0.7 gain plus 0.3 VMAF"]
        CSCORE["Compression score"]
    end

    subgraph UPSCALE["Upscaling lane"]
        UBIND["Exact launch 2x scale and target-geometry binding"]
        UVMAF["Primary VMAF\nreference versus output"]
        UDELTA["Primary plus NEG VMAF\nminer input versus output"]
        UPER["CPU tone, grayscale, and chroma checks\nminer input versus output"]
        UPIE["CPU PieAPP\nreference versus output,\ndigest-bound frame window"]
        UGATE{"All validity and\nanti-gaming gates pass?"}
        UFORM["Upscaling formula\nPieAPP quality plus content length"]
        USCORE["Upscaling score"]
    end

    REF --> CANON
    INPUT --> CANON
    OUTPUT --> CANON

    CANON --> CVMAF
    CANON --> CDELTA
    CANON --> CPER
    CANON --> CRATE
    CVMAF --> CGATE
    CDELTA --> CGATE
    CPER --> CGATE
    CRATE --> CGATE
    CVMAF --> CFORM
    CRATE --> CFORM
    CGATE -->|"yes"| CFORM
    CFORM --> CSCORE
    CGATE -->|"no"| CZERO["Zero plus stable reason code"]

    CANON --> UBIND
    CANON --> UVMAF
    CANON --> UDELTA
    CANON --> UPER
    CANON --> UPIE
    UBIND --> UGATE
    UVMAF --> UGATE
    UDELTA --> UGATE
    UPER --> UGATE
    UPIE --> UFORM
    UGATE -->|"yes"| UFORM
    UFORM --> USCORE
    UGATE -->|"no"| UZERO["Zero plus stable reason code"]
```

The audit comparison uses tolerance hysteresis at a gate boundary: tiny permitted
metric drift may not flip an authority pass into an audit failure. Away from a boundary,
scores and derived earnings remain strict and deterministic.

## 4. Deployed topology

The deployed stack deliberately runs trust boundaries as separate containers,
with separate state directories, health checks, and metrics endpoints.

```mermaid
flowchart LR
    subgraph PRE["One-shot gates — never runtime services"]
        SPF["production-static-preflight\nrelease/config validation only"]
        LPF["production-live-preflight\narchive + wallet + miner + S3 + capacity"]
    end

    subgraph BACKEND["Backend services"]
        SW["scoring-worker\nHTTP 8201 · metrics 9103"]
        CS["challenge-service\nHTTP 8210 · metrics 9105"]
        AR["audit-results-api\nHTTP 8710 · metrics 9112"]
        CO["orchestrator\nHTTP 8500 · metrics 9104"]
        BR["baseline-registry\nHTTP 8720 · metrics 9123"]
    end

    subgraph PRIMARY["Primary authority host — separate processes/cgroups"]
        AUTH["authority-node\ninference validator + pointer API + finalizer\npointer 8700 · metrics 9101/9111/9120"]
        THIN1["thin-validator-node 1\nmetrics 9102"]
        BA1["beacon-auditor 1\nmetrics 9121"]
        OA1["own-auditor 1\nmetrics 9122"]
    end

    subgraph V2["Independent validator host/project/state"]
        THIN2["thin-validator-node 2\nmetrics 9102"]
        BA2["beacon-auditor 2\nmetrics 9121"]
        OA2["own-auditor 2\nmetrics 9122"]
    end

    subgraph INGRESS["Six fresh independent CPU miner-edge hosts/projects/IPs"]
        MC["3 compression hosts\nquality · balanced · compact\npublic TLS 8300 → private miner"]
        MU["3 upscaling hosts\nquality · balanced · compact\npublic TLS 8300 → private miner"]
    end

    subgraph MODAL["Fresh Modal environments"]
        MGPU["Inference GPU workers\nquality, balanced, compact"]
        CGPU["Competition GPU sandboxes\npinned contender images"]
    end

    UI["dashboard\nHTTP 8600 · metrics 9109"]
    RPC["Public archive RPC\nprimary plus optional future fallback"]
    S3["S3-compatible evidence store"]

    CS <-->|"challenge handout"| AUTH
    AUTH <-->|"score request and response"| SW
    AUTH <--> MC
    AUTH <--> MU
    AUTH -->|"authenticated epoch pointers"| THIN1
    AUTH -->|"authenticated epoch pointers"| BA1
    AUTH -->|"authenticated epoch pointers"| OA1
    AUTH -->|"private HTTPS epoch pointers"| THIN2
    AUTH -->|"private HTTPS epoch pointers"| BA2
    AUTH -->|"private HTTPS epoch pointers"| OA2
    MC <-->|"authenticated HTTPS"| MGPU
    MU <-->|"authenticated HTTPS"| MGPU
    CO <--> CGPU
    BR -->|"active baseline"| CO
    CO <-->|"score request and response"| SW
    AUTH --> S3
    CO --> S3
    BR --> S3
    CS <--> RPC
    CO <--> RPC
    AR <--> RPC
    AUTH <--> RPC
    THIN1 <--> RPC
    BA1 <--> RPC
    OA1 <--> RPC
    THIN2 <--> RPC
    BA2 <--> RPC
    OA2 <--> RPC
    S3 --> THIN1
    S3 --> THIN2
    THIN1 -->|"landed-vector publication"| S3
    THIN2 -->|"landed-vector publication"| S3
    S3 --> BA1
    S3 --> OA1
    S3 --> BA2
    S3 --> OA2
    BA1 --> AR
    OA1 --> AR
    BA2 --> AR
    OA2 --> AR
    AR --> UI
    AUTH --> UI
    BR --> UI
```

The primary authority host runs no earning miner and has no local-miner startup
dependency. Each reference miner identity runs the same parameterized miner-edge
composition on a fresh host/public IPv4 address: one hardened CPU
miner ingress plus its containerized IP-SAN TLS edge. The ingress authenticates and bounds
artifact-v2 transport, delegates media transformation to a fresh Modal GPU worker, and
never scores. Scoring and every audit recomputation remain in ordinary CPU services. This
host/IP isolation is economically load-bearing because reward dedup is global by
advertised IP before track ranking.

The product gateway, optional product-serving selection worker, champion backend, and
autoupdater are product/operations components, not required containers in the
consensus stack. The executable-baseline registry is different: current schema-v15 earning
competitions require it. Absence of the optional product components must not be mistaken
for a missing scoring or emissions path.
`production-static-preflight` and `production-live-preflight` are distinct one-shot
profiles. Only the latter has the exact `--live`, 7,200-block archive-depth, and real
miner identity/URL arguments. Neither starts, owns, or depends on the runtime services
drawn beside it.

## 5. Service ownership

| Service or role | Primary code | Owns durable state | Responsibility |
|---|---|---|---|
| Challenge Service | owner-operated, not in this repo | Challenge catalog, immutable segment manifests, and sealed-source metadata | Ingest, duplicate screening, lossless master, per-track duration eligibility, constrained winnable DAG draw from the selected segment, commit before dispatch |
| Inference Validator | `vidaio/validator/` | Round state and inference evidence | Census-aware dispatch, signed artifact-v2 transport, deadlines, receipts, duplicate resolution, miner-attributable zeroes |
| Scoring Worker | `vidaio/scoring_worker/`, `vidaio/scoring/` | None beyond process telemetry | Stateless CPU canonicalization, metrics, gates, score packet generation |
| Competition Orchestrator | `vidaio/competition/` | Competition state and competition evidence | Enrollment, reproducible build, isolated GPU run, per-subject scoring, evidence matrix, and lifecycle |
| Authority Finalizer | `vidaio/authority/`, `vidaio/epoch/` | Epoch working state, then immutable S3 log | Snapshot already-folded EWMA, independently check the fold, derive economic order and the global reward window, dedup/payouts, predecessor link, finalization and chain anchor |
| Scoring Authority API | `vidaio/authority/service.py` | No alternate copy of the log | Serve the latest finalized pointer and expected digest |
| Thin Weight-setter | `vidaio/weightsetter/` | Submission cursor, accepted intent, and landed-vector publication state | Authorize exact epoch-log bytes, submit safely, publish the confirmed landed vector, recover from chain/RPC uncertainty |
| Own-auditor | `vidaio/auditor/`, `vidaio/audit/` with `audit_mode=own_audit` | Its own cursor, signed-report outbox, logs | Full CPU replay on its own cadence; report only. The legacy `vidaio/weightsetter/own_audit.py` helper is not production wiring. |
| Beacon-auditor | `vidaio/auditor/`, `vidaio/audit/` with `audit_mode=beacon` | Its own cursor, signed-report outbox, logs | Full CPU replay bound to a post-finalization beacon; report only |
| Audit Results API | `vidaio/audit_api/` | Signed findings keyed by auditor, epoch, and mode | Central append-only investigation record |
| Dashboard | owner-operated, not in this repo | Presentation/cache state only | Surface health, metrics, findings, chain-confirmed published weight history, and the effective reward window; keep the executable-baseline registry visibly distinct |
| Baseline registry | `vidaio/registry/` | Per-track immutable executable versions and promotion latch | Seed each track with archived baseline v0; activate only an independently verified CROWN winner; serve the active baseline beyond its reward window |
| Serving registry | `vidaio/registry/` | Optional product-serving selection | Select what the optional product gateway serves; separate from reward-window economics and the executable-baseline ratchet |
| Gateway | owner-operated, not in this repo | Product-plane job state | Route organic customer jobs; outside the consensus loop |
| Reference miner ingress | `vidaio/miner/` | Miner-local model and task state | Authenticated artifact-v2 endpoint around a replaceable private backend |
| Autoupdater | `vidaio/autoupdater/` | Version observation state | Report release/version state; it does not silently redefine consensus behavior |

## 6. Life of one finalized epoch

```mermaid
sequenceDiagram
    participant C as Chain
    participant CS as Challenge Service
    participant V as Inference Validator
    participant M as GPU Miner
    participant SW as CPU Scoring Worker
    participant CO as Competition Orchestrator
    participant CG as Modal Competition GPU
    participant S3 as S3-compatible Store
    participant F as Authority Finalizer
    participant API as Authority API
    participant WS as Thin Weight-setter
    participant OA as Own-auditor
    participant BA as Beacon-auditor
    participant AR as Audit Results API

    CS->>C: anchor challenge commitment
    CS->>V: private miner input plus selected eligible segment reference
    V->>M: miner input only, signed task and deadline
    M->>V: output bytes plus signed receipt
    V->>SW: input, output, reference and binding metadata
    SW-->>V: CPU score packet plus content digest
    V->>S3: archive inference evidence
    V->>V: atomically fold this round into EWMA
    CO->>C: anchor competition commitment before enrollment
    CO->>CG: run pinned contender image on hidden input
    CG-->>CO: candidate output
    CO->>SW: competition input, output, reference and binding
    SW-->>CO: CPU score packet plus content digest
    CO->>S3: archive competition evidence matrix
    CO->>F: provide committed competition inputs
    F->>F: snapshot and verify fold, then derive competition, reward window, dedup and payouts
    F->>S3: audit manifest, epoch log, then _FINALIZED
    F->>C: anchor epoch digest
    F->>API: publish finalized pointer

    par Scheduled weight submission
        WS->>API: fetch finalized pointer
        WS->>S3: fetch exact epoch-log bytes
        WS->>C: verify anchored digest and chain context
        WS->>C: reconcile uid/hotkey targets and submit the safe vector
        C-->>WS: confirm exact landed vector
        WS->>S3: publish landed vector and publication record
        WS->>C: anchor publication digest
    and Independent own-audit
        OA->>API: fetch finalized pointer
        OA->>S3: fetch all committed evidence
        OA->>C: independently verify anchor and close-block metagraph
        OA->>OA: CPU replay every scored item and earning
        OA->>OA: persist signed finding in durable outbox
        OA->>AR: upload signed own-audit finding
    and Independent beacon-audit
        BA->>API: fetch finalized pointer
        C-->>BA: post-finalization block hash
        BA->>S3: fetch all committed evidence
        BA->>C: independently verify anchor and close-block metagraph
        BA->>BA: CPU replay every scored item and earning
        BA->>BA: persist signed finding in durable outbox
        BA->>AR: upload signed beacon-audit finding
    end
```

For DISPUTED or INCONCLUSIVE results, each auditor emits a structured CRITICAL/WARNING
log and metric suitable for external alerting before central delivery. Every signed
report is persisted in a durable outbox and retried, so an unavailable Audit Results API
cannot erase a finding. Audit status cannot interrupt scheduled weight submission:
investigation and remediation are manual operator actions.

## 7. Evidence and anchoring graph

```mermaid
flowchart TB
    subgraph BEFORE["Before dispatch"]
        SOURCE["Lossless source"] --> MANIFEST["Immutable clip name/hash/duration manifest"]
        MANIFEST --> SEALED["Selected eligible segment reference"]
        PLAN["Challenge plan"] --> PREIMAGE["Hidden seeded DAG preimage"]
        PREIMAGE --> INPUT["Miner input"]
        PREIMAGE --> COMMIT["Pre-dispatch commitment"]
        COMMIT --> CA["Challenge anchor on chain"]
    end

    subgraph MEASURE["Response and CPU measurement"]
        INPUT --> OUTPUT["Miner output"]
        INPUT --> RECEIPT["Miner-signed artifact receipt"]
        OUTPUT --> RECEIPT
        CA --> RECEIPT
        SEALED --> PACKET["Digest-addressed CPU score packet"]
        INPUT --> PACKET
        OUTPUT --> PACKET
    end

    subgraph AUDITEVID["Recomputable item evidence"]
        PREIMAGE -->|"publish after retirement"| REVEAL["Public DAG reveal"]
        SEALED -->|"release after retirement/completion"| PUBLICREF["Public pristine reference"]
        CA --> BUNDLE["Per-item audit bundle"]
        RECEIPT --> BUNDLE
        INPUT --> BUNDLE
        OUTPUT --> BUNDLE
        PACKET --> BUNDLE
        REVEAL --> BUNDLE
        PUBLICREF --> BUNDLE
        PACKET --> PDIG["Score-packet digest"]
        PDIG --> MERKLE["Merkle root and inclusion proof"]
        BUNDLE --> MANIFEST["Audit manifest\nlists bundle and packet refs"]
        PDIG --> MANIFEST
        MERKLE --> MANIFEST
    end

    subgraph EPOCHSET["Finalized epoch set"]
        PREV["Previous epoch-log digest"] --> EPOCH["Epoch log bytes"]
        MANIFEST --> EPOCH
        CENSUS["Close-block census"] --> EPOCH
        ECON["Verified EWMA, competition result, reward window,\ndedup and authority u16 vector"] --> EPOCH
        EPOCH --> FINAL["_FINALIZED marker written last"]
        FINAL --> EA["Epoch digest anchored on chain"]
        FINAL --> PTR["Authority API pointer"]
    end

    EPOCH --> AVAILABLE{"Log bytes, pointer, and\nchain anchor available?"}
    PTR --> AVAILABLE
    EA --> AVAILABLE
    AVAILABLE -->|"no"| HOLD["HOLD this cycle and retry"]
    AVAILABLE -->|"yes"| AUTHZ{"sha256(log bytes) equals\npointer digest equals\nchain anchor?"}
    AUTHZ -->|"no"| REFUSE["REFUSE this candidate"]
    AUTHZ -->|"yes"| WEIGHTS["Reconcile bindings and submit\npinned-SDK max-grid vector"]
    WEIGHTS --> LANDED["Confirmed landed vector"]
    LANDED --> PUB["WEIGHT_VECTOR artifact plus\nPublicationRecord"]
    PUB --> PA["Publication digest anchored on chain"]

    PUBLICREF --> REPLAY["Third-party CPU replay"]
    BUNDLE --> REPLAY
    MANIFEST --> REPLAY
    EA --> REPLAY
    REPLAY --> FINDING["Signed audit finding"]
```

The weight-setter authenticates the log with the equality
`sha256(epoch_log_bytes) == pointer_digest == on_chain_anchor`. It can also defer for
ordinary safety conditions such as an unavailable archive RPC, an unfinalized or empty
log, an unparseable payload, wrong tempo, or unresolved chain intent. Those are chain
submission safeguards, not audit enforcement.

The left side depicts an inference item. A competition item replaces the degradation-DAG
reveal with its manifest-committed `(input digest, reference digest, 2x factor,
target width, target height)`
binding and becomes publicly recomputable when the completion gate releases its pristine
reference.

The authority log commits `weight_u16` on VIDAIO's deterministic **sum-grid**. It is not
the runtime wire representation. The real chain adapter first verifies every positive
`(uid, hotkey)` target against a fresh metagraph. A deregistered or recycled UID rejects
the complete attempt before any write: it is neither paid to a new occupant nor removed
and renormalized into survivors. With the exact target set verified, pinned Bittensor
10.5 emits the deterministic **max-grid** vector, exactly
`max_normalize_u16(epoch_log.weight_u16)`. The confirmed max-grid vector—not the
authority sum-grid—is what gets durably published and anchored. Consequently the epoch
log's weight-vector digest and the post-submit publication digest normally differ, but
their mapping is deterministic and independently testable.

## 8. Mutable state and system of record

| State | System of record | Recovery property |
|---|---|---|
| Challenge ingest, plans, and sealed-reference metadata | Challenge Service database plus configured secret material | A restart cannot silently redraw an already committed challenge |
| In-flight inference rounds | Inference Validator state | Deadlines and receipts survive process restart |
| Competition lifecycle | Orchestrator database | State transitions are explicit and resumable |
| Inputs, outputs, references, packets, manifests, epoch logs | S3-compatible bucket, versioned/locked by deployment policy | Content digests, retention policy, and `_FINALIZED` written last |
| Epoch authenticity | Bittensor chain anchor | Independent validators bind downloaded bytes to an immutable digest |
| Weight submission | Chain plus weight-setter cursor/accepted intent and S3 publication | Retry safely; publish only the vector confirmed to have landed |
| Own-audit and beacon-audit progress | Separate auditor cursors and outboxes | One mode cannot advance, overwrite, or suppress the other |
| Audit findings | Signed report in a durable outbox, structured log/metric, then Audit Results API | Central outage delays delivery but does not lose the signed report |
| Release identity | Git tree plus `data/ci-pass` | Deployment preflight proves CI ran on the exact shipping source tree |

## 9. Weight setting and audit findings are independent

```mermaid
flowchart LR
    TICK["Scheduled weight tick"] --> GET{"Pointer and finalized\nepoch log available?"}
    GET -->|"no"| WAIT["HOLD this cycle\nretry safely"]
    GET -->|"yes"| VERIFY{"Bytes, digest, anchor, schema,\nand submission context valid?"}
    VERIFY -->|"no"| REFUSE["REFUSE this candidate\ndo not submit altered weights"]
    VERIFY -->|"yes"| VECTOR["Load authority sum-grid\nweight_u16"]
    VECTOR --> SAFE{"Chain submission\nsafe now?"}
    SAFE -->|"no"| WAIT
    SAFE -->|"yes"| RECON{"Exact uid/hotkey target set\nstill valid?"}
    RECON -->|"no"| WAIT
    RECON -->|"yes"| SET["SDK max-normalize exact sum-grid,\nset_weights, confirm runtime vector"]
    SET --> PUB["Publish and anchor confirmed\nmax-grid runtime vector"]

    LOG["Anchored evidence"] --> OWN["Own-audit CPU replay"]
    LOG --> BEACON["Beacon-audit CPU replay"]
    OWN --> REPORT["Signed finding in durable outbox\ndispute/inconclusive log plus metric"]
    BEACON --> REPORT
    REPORT --> API["Audit Results API"]
    API --> HUMAN["Dashboard, investigation,\nmanual remediation"]
```

There is deliberately no arrow from `REPORT` back into `SET`. Validators always attempt
to set the authority-derived vector on schedule once the epoch itself is authentic and
the chain operation is safe, subject only to the mandatory current `(uid, hotkey)`
exact-target check described above. Audit results inform people; they do not automatically
freeze the fleet or rewrite emissions.

## 10. Competition, reward-window, and executable-baseline lifecycles

```mermaid
stateDiagram-v2
    [*] --> SCHEDULED
    SCHEDULED --> ENROLLING
    ENROLLING --> FINALIZING_SUBMISSIONS
    FINALIZING_SUBMISSIONS --> VALIDATING
    VALIDATING --> BUILDING
    BUILDING --> EVALUATING
    EVALUATING --> SCORING
    SCORING --> AWAITING_END_TIME
    AWAITING_END_TIME --> COMPLETED
    SCHEDULED --> FAILED
    VALIDATING --> FAILED
    BUILDING --> FAILED
    ENROLLING --> CANCELLED
    COMPLETED --> [*]
    FAILED --> [*]
    CANCELLED --> [*]
```

```mermaid
flowchart TB
    REPO["Submitted repository"] --> SNAP["Pinned source snapshot"]
    SNAP --> IMAGE["Reproducible sandbox image"]
    IMAGE --> GPU["Fresh isolated GPU run"]
    GPU --> OUT["Candidate outputs"]
    OUT --> CPU["CPU scoring and evidence"]
    CPU --> RESULT["Global machine-ranked CompetitionResult\napplied at epoch close"]
    RESULT --> WINDOW{"Winner beats archived baseline\nby at least 5%?"}
    WINDOW -->|"no"| PODIUM["PODIUM reward window\n60% inference · 40% competition"]
    WINDOW -->|"yes"| CROWN["CROWN reward window\n10% inference · 90% competition"]
    PODIUM --> EXPIRE["At ends_at: effective IDLE\n80% inference · 20% sink"]
    CROWN --> EXPIRE

    BASE["Active per-track archived baseline\nversioned executable + provenance"] --> CPU
    CROWN --> RELEASE["Release exact winner source\nbefore earning epoch commits"]
    RELEASE --> LATCH["Verified CROWN promotion latch"]
    LATCH --> RERUN["Rebuild exact sealed winner archive\nand rerun committed item matrix"]
    RERUN -->|"verified"| BASE

    CPU --> CDB["Completed competition DB evidence"]
    REVIEW["Append-only human review state"] --> CDB
    CDB -.->|"separate promotion policy"| REG["Serving registry champion"]
    CUSTOMER["Enterprise customer"] --> GW["Gateway"]
    GW -->|"read current track"| REG
    REG -->|"selected champion identity"| GW
    GW -->|"dispatch to configured endpoint"| BACKEND["Champion backend adapter\nnot an emissions authority"]
```

The economic podium and reward state are derived entirely from committed machine
evidence; human review cannot change the economic rank. There is one global reward
window: every successfully applied newer cycle replaces any older window and starts a
new half-open interval `[applied_at, ends_at)`. Production fixes that duration at 168
hours; a testnet-only override may shorten it so both activation and expiry can be
observed. PODIUM and CROWN both pay the result's top three at 70/20/10 within the fixed
competition pool. Missing or deregistered ranks are not redistributed; their shares go
to the canonical sink.

The executable ratchet is a distinct per-track state machine. Each track starts with an
archived public reference implementation at baseline v0. Only a fully verified CROWN
winner can replace that track's active baseline, after the exact sealed submission is
released before the earning epoch commits, then rebuilt and rerun against the committed
matrix. The active executable remains the
baseline after its seven-day payout window expires. A separate product promotion policy
may decide what an optional customer gateway serves; neither serving choice nor human
review can alter emissions.

Test deployments enroll known example contenders and create only fresh GPU resources
whose names start with `vidaio-next-`; teardown removes those resources when the run is
finished.

## 11. Where emissions go

The latest global reward state selects fixed pools. The inference pool always preserves
the 0.8/0.2 compression/upscaling split; the competition pool always uses the result's
70/20/10 podium.

```mermaid
flowchart TB
    E["Total subnet emissions E"] --> STATE{"Effective RewardWindowState\nat epoch close time"}
    STATE -->|"IDLE"| IDLE["80% inference · 0% competition · 20% sink"]
    STATE -->|"PODIUM"| POD["60% inference · 40% competition"]
    STATE -->|"CROWN"| CR["10% inference · 90% competition"]

    IDLE --> INF["Inference pool"]
    IDLE --> SINK["Canonical non-earner sink UID"]
    POD --> INF
    POD --> CP["Competition podium 70 / 20 / 10"]
    CR --> INF
    CR --> CP

    INF --> COMPRESS["Compression = inference × 0.8\n64% / 48% / 8% of E"]
    INF --> UPSCALE["Upscaling = inference × 0.2\n16% / 12% / 2% of E"]
    COMPRESS --> ELIG["Score floor, dedup, top-five rank curve"]
    UPSCALE --> ELIG
    CP --> EARNERS["Payable podium hotkeys"]
    ELIG --> EARNERS

    COMPRESS -.->|"unallocated"| SINK
    UPSCALE -.->|"unallocated"| SINK
    CP -.->|"missing rank"| SINK
```

Inference eligibility is not “whoever showed up gets a full share”:

```mermaid
flowchart LR
    ROUNDS["Round scores"] --> EWMA["EWMA with inactivity decay"]
    EWMA --> FLOOR{"Absolute score at least 0.10?"}
    FLOOR -->|"no"| ZERO["No payout"]
    FLOOR -->|"yes"| DEDUP["Coldkey and concrete-IP dedup"]
    DEDUP --> RANK["Deterministic rank"]
    RANK --> TOP{"Top five?"}
    TOP -->|"no"| ZERO
    TOP -->|"yes"| CURVE["Rank shares 5 : 4 : 3 : 2 : 1"]
    CURVE --> TRACK["Apply track and current inference-pool share"]
```

Unspecified advertisement addresses (`0.0.0.0` and `::`) are exempt from IP dedup so
miners without a serving axon do not collapse into one identity. A concrete shared IP
is deduped, as is a shared coldkey. Coldkey collusion and reliance on one CPU PieAPP
implementation are explicitly accepted launch risks.

## 12. Failure isolation

| Failure | What stops | What continues |
|---|---|---|
| Miner timeout or invalid output | That miner item receives its deterministic result/zero | Other miners and tracks finish the round |
| Scoring worker unavailable | Affected scoring items are non-punitively skipped/not accumulated | Other services and existing immutable evidence remain available |
| Per-item S3 evidence write fails | That item cannot enter the economic fold | Other evidence and rounds remain intact |
| Epoch-log S3 write or finalization fails | The epoch is not advertised or anchored as complete | Previous on-chain weights remain in effect |
| Authority API unavailable | Weight-setter and both auditor discovery loops retry because they cannot authenticate a new pointer | Previous weights and already stored evidence remain intact |
| Archive RPC unavailable or rate-limited | Chain-dependent preflight, anchoring, or submission retries safely | No fabricated census, anchor, or weight vector is accepted |
| Own-audit or beacon-audit FAIL/INCONCLUSIVE | A signed finding, structured log, and metric are emitted; provisioned monitoring may alert | Scheduled authenticated weight submission is not interrupted |
| Audit Results API unavailable | Central delivery waits in the signed-report outbox | Structured logs/metrics persist; weight setting continues |
| One auditor mode crashes | That mode restarts from its own cursor | The other auditor and weight-setter remain independent |
| GPU sandbox or contender build fails | That contender/competition follows its explicit failure transition | Inference scoring and finalized prior epochs continue |
| CROWN baseline promotion build/rerun fails | The per-track promotion latch remains pending and the prior executable stays active | The already anchored reward window and weight-setting continue; the next competition for that track waits for remediation |

## 13. Known seams and accepted launch risks

- The live public-testnet run validated S3 policy/IAM, cursor-floor genesis,
  wallet registration, archive-RPC rate limits versus anchor cadence, miner
  advertisements, real Modal GPU behavior, alert delivery, and a multi-epoch soak
  before release.
- The serving `ChampionBackend` and self-service product routing are not consensus
  dependencies and are not treated as complete merely because competition emissions
  are live.
- Coldkey-level collusion and a single-model CPU PieAPP metric are accepted launch
  risks rather than hidden claims of solved problems.
- Competition is earning live. The epoch log commits the exact pre-enrollment raw
  commitment receipt (subnet, payload/digest, inclusion/hash, and finalized height), and
  authority/auditors independently archive-read it, alongside the full item matrix and sealed contender
  archives, the archived baseline executable and provenance, every packet/bundle, the
  chain-time `applied_at`, global cycle, `CompetitionResult`, and `RewardWindowState`.
- Mainnet deployment was gated on the real testnet evidence and soak, not only on
  local CI.

## 14. Deeper reading

- the project design record — why validators land byte-identical weights
- [VALIDATING.md](VALIDATING.md) — validator deployment and operating behavior
- The per-package `README.md` files under `vidaio/` — service-level contracts
  (weight-setter, auditor, authority, audit API, epoch schema)
