# Backend interfaces / 后端接口

New HybrIK image inference and GVHMR video inference are experimental and unavailable. Compatibility packages are planned for a future update; this source package does not include them. The interface details below describe the included clients for integration development, not an installation procedure that enables these features.

HybrIK 图片新推理和 GVHMR 视频新推理尚未开放，本版不包含兼容包。下文是已有客户端的开发接口说明，不是启用这两项功能的安装教程。

## Capabilities and status / 能力与状态

`GET /api/capabilities` returns `kairyth-capabilities/v1`, including `release_profile`, `profile_valid` and the two feature entries `hybrik_image_inference` / `gvhmr_video_inference`. Each entry includes `enabled`, `state`, `reason`, `experimental` and `message_key`.

The current release reports `disabled_experimental`; backend status is `release-unavailable`, and new inference POSTs to `/api/image-poses` and `/api/motions` return HTTP 503 before upload parsing. External paths or a successful resource check do not enable them. Existing-pose import, motion-to-pose extraction, stored-motion editing and other workbench operations remain separate.

`GET /api/health` reports core availability through `service_ready`. Its `ok` / `rigging` fields describe rigging setup, not whether every workbench page is offline. A missing or invalid release marker disables the two inference entries with `disabled_configuration` without closing the core service. Keep the checked-in `release-profile.json` with the installation.

`ready` / `installed` describe preflight only. `runtime_verified:false` and `rights_verified:false` do not establish model execution or usage permission. Public-disabled backends are not probed or loaded by status requests.

## Pose worker contract / 姿势工作进程契约

Client implementation: [`pose_backend_client.py`](../motionbuilder_app/pose_backend_client.py).

```text
<configured-python> -B <configured-worker.py> --request <job>/backend_request.json
```

The request uses `schema: "kairyth-hybrik/v1"` and fields `output_dir`, `input_image`, `runtime_root`, `name` and `source_image_name`. The request and input image belong to the current job, and the job directory is the worker's working directory.

stdout may carry version-matching JSONL `stage`, `error` and `completed` events; progress is 0–100. Events are not a substitute for result validation. Fixed result files are `hybrik_raw.npz`, `pose_soma30.npz`, `constraints.json` and `backend_result.json`.

The client validates path containment, fixed filenames, byte/ZIP expansion bounds, NPY headers, dtypes, shapes, finite values and SOMA-30 conventions. NPZ loading disallows object arrays and pickle. Failure, cancellation or invalid output must not become a successful pose-library entry.

## Video / detector / extractor contract

The video client sequence is prepare → independent detector JSON → inference → extraction. A compatible implementation must support `--prepare_only`, `--tracking_file`, matching cache identities and output semantics. Detector output uses `person-tracks/v1`; extraction produces the workbench's source-motion bundle. An arbitrary upstream checkout is not guaranteed to implement this contract.

The [configuration module](../motionbuilder_app/backend_config.py) names root/interpreter/worker settings such as `KAIRYTH_HYBRIK_ROOT`, `KAIRYTH_HYBRIK_PYTHON`, `KAIRYTH_HYBRIK_WORKER`, `KAIRYTH_GVHMR_ROOT`, `KAIRYTH_GVHMR_EXTRACTOR` and `KAIRYTH_DETECTOR_ROOT`. Windows-relative paths resolve against the workbench project root; WSL interpreter settings are Linux absolute paths. These client settings do not supply an adapter or enable unavailable release features.

Only execute trusted workers. Output validation is not a sandbox; an external interpreter runs code with the account's permissions. Backend code, detector dependencies, model weights and body-model resources retain their own licenses. See [license details](LICENSE_DECISIONS.md) and [resource usage](RESEARCH_BACKEND_USAGE.md).
