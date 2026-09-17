# OSPool Cluster 14996562 Hold Review

Snapshot taken at `2026-09-11T15:58:50Z`.

## Campaign

- Batch: `batch_20260909_012610`
- Submitted jobs: 10,000
- Completed jobs: 9,874 (98.74%)
- Running jobs: 81
- Held jobs: 45
- Idle jobs: 0
- Returned archives among held jobs: 0

The campaign output directory is:

```text
/ospool/ap40/data/fang.han/cardiac_spect_srm/batch_20260909_012610
```

## Hold Classification

| Jobs | Process IDs               | Hold code |    Age at review | Starts | Reason                                                          |
| ---: | ------------------------- | --------: | ---------------: | -----: | --------------------------------------------------------------- |
|   42 | See scheduler query below |      47/0 | 0.55-24.93 hours |    1-4 | Exceeded the execution site's 20-hour duration limit            |
|    2 | 9903, 9905                |    13/256 |      23.91 hours |      0 | Transient OSDF director connection failure (`no route to host`) |
|    1 | 9318                      |      12/2 |       0.08 hours |      3 | Expected output archive was missing during transfer             |

Held jobs do not consume execute slots, but they also do not recover automatically merely by remaining held.

## Runtime Context

The 9,874 completed jobs had cumulative remote-wall-clock percentiles:

| Percentile |     Runtime |
| ---------: | ----------: |
|        p50 | 11.60 hours |
|        p90 | 16.34 hours |
|        p95 | 21.12 hours |
|        p99 | 28.65 hours |
|    maximum | 48.10 hours |

`RemoteWallClockTime` can accumulate across starts, so values above 20 hours do not imply one uninterrupted execution exceeded the site limit.

## Recommendation

1. Release the two code-13 OSDF transfer holds once. Their failure occurred before execution and was caused by a transient network outage.
2. Do not expect the 42 code-47 jobs to change while held. Release them only if recovering the final fraction of campaign statistics is worth another full attempt; some have already started up to four times.
3. Treat the code-12 missing-output job as a failed worker attempt. It can be released once for diagnosis, but remove it if the archive is missing again.
4. If 98.74% plus the remaining running jobs is sufficient, wait for the 81 running jobs to settle and then remove all residual held jobs as queue cleanup.

No jobs were released or removed during this review.

## Commands

Review current holds:

```bash
ssh ospool 'condor_q 14996562 -hold -af ProcId HoldReasonCode HoldReasonSubCode NumJobStarts HoldReason'
```

Release only the two transient OSDF failures:

```bash
ssh ospool 'condor_release 14996562.9903 14996562.9905'
```

Remove all jobs still held after the running jobs settle:

```bash
ssh ospool 'condor_rm -constraint "ClusterId == 14996562 && JobStatus == 5"'
```
