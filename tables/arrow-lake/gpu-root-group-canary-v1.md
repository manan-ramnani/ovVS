# Arrow Lake GPU root-group capacity gate

Status: **owner/helper route parked on this software stack**. The standalone
gate changes no production dispatch or CAGRA kernel.

## Question and scope

The promoted CAGRA walk still under-occupies the iGPU at batch one because it
assigns one workgroup per query. A proposed exact owner/helper route would use
four cooperative workgroups for one query and device-scope root barriers. The
necessary admission condition is a named-kernel
`max_num_work_groups >= 4` result before any cooperative launch.

The default-off canary uses the current CAGRA workgroup size of 128 and the
conservative `128/4` local-memory admission footprint of 2,640 bytes per
workgroup. It selects exactly one integrated Intel Level Zero GPU, gates both
required SYCL extension macros, queries the executable kernel, and launches
nothing when capacity is insufficient. CTest adds a 20-second external timeout
for any future admitted launch.

## Result

| Required groups | Reported maximum | Local size | Local memory/group | Producer exit | Cooperative launches |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 128 | 2,640 B | 77 | 0 |

The selected device was `Intel(R) Graphics` through oneAPI Unified Runtime over
Level Zero driver `1.15.39183+1`. The tool reported `fallback=false`,
`canary_admitted=false`, and `integration_status=standalone_not_routed`.

This does not show that a root barrier executed incorrectly: the safe contract
refused to launch because the prerequisite capacity was absent. It also makes
no latency, throughput, energy, or acceleration claim. A production
owner/helper kernel would include additional traversal state and work and would
still require its own exact named-kernel query; implementing it after this
necessary gate failed is not justified on the measured Arrow Lake stack.

## Disposition and provenance

The four-workgroup batch-one route is parked on Arrow Lake. Reopen it only for
a materially changed runtime/driver, a new SKU, or a design with a different
measured cooperative requirement. No BIOS or firmware change is requested.
Continue B2 only through software paths that do not require unsupported
cross-workgroup synchronization, while retaining the promoted one-workgroup
kernel.

The clean producer revision is `6deb173`; the executable SHA-256 is
`1BB0C51DE5F2527EBB81ED9E23C5520BB3A20DD6FAA44C7FAD2212AB249604AD`.
The immutable JSON and its digest are in
[`evidence/gpu-root-group-canary-v1/`](evidence/gpu-root-group-canary-v1/README.md).
