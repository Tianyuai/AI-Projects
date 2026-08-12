# Final branch review fixes report

日期：2026-08-12
实现提交：`7898d8e` (`fix: harden recall experiment evidence boundaries`)

## 范围与约束

- 完整阅读 `docs/superpowers/specs/2026-08-11-modular-candidate-recall-experiment-harness-design.md` 与 `docs/superpowers/plans/2026-08-11-modular-candidate-recall-experiment-harness.md`。
- 按 systematic debugging 先追踪根因，再按 TDD 为每项 finding 增加最小回归覆盖并观察预期失败，最后做最小修复。
- 未联网、未执行 live 调用、未读取 `.env`、未使用或修改预算账本。
- 用户自有未跟踪路径 `data/budget_ledger.sqlite3`、`deliverables/`、`docs/evidence/identifier-map-semantic-audit-2026-08-10.json` 保持存在且未暂存。

## Finding 1 — regenerated `±1` 比较字段错误（Critical）

- 核验：成立。`compare_regenerated()` 比较的是固定 Gold 分母 `gold_association_count`，因此实际 Gold 命中数偏差不会触发失败。
- RED：新增命中数偏差 `+2`、分母不变的回归，实际错误结论为 `passed`，期望 `failed`。
- GREEN：改为比较 `gold_hit_count`；保留 macro recall 与 historical retention 门槛。
- Focused：`tests/recall_experiments/test_evaluator.py`，15 passed。

## Finding 2 — DeepSeek snapshot/live 成本与预算身份（Important）

- 核验：成立。预算控制器拒绝 `llm_calls > 0` 且 `cost_cny is None` 的 reservation；原 replay 估算正是未知成本，因此会在读取 snapshot 前失败。
- RED：回归要求仅从 manifest 中的 LLM sealed usage 推导 input/output token、cost 与 elapsed 上界；原 helper 不存在。另加 live runtime 缺少 budget/pricing 身份必须拒绝的回归，原实现错误放行。
- GREEN：replay 在 manifest 已由 `DependencySnapshotReader` 验证后，从相同 manifest bytes 的封存 LLM usage 生成 initial/repair reservation 上界；不使用 OpenAlex usage，不猜 live model 价格。live runtime 必须显式绑定 `backend_identity`、`budget_policy_sha256`、`pricing_policy_sha256`，否则 `config_mismatch`。
- 风险边界：缺少 sealed LLM usage 的 replay 只按封存的零成本事实运行；live 价格仍必须由外部已绑定 runtime 提供，本实现未硬编码任何 DeepSeek live price。

## Finding 3 — generation failure attempt/report 丢失且错误归类（Important）

- 核验：成立。`RecallGenerationFailure` 原先直接越过 runner，既不写 generation artifact，也不保留 attempt/report，更不会继续剩余 attempts；CLI 又把所有失败统一改为 `snapshot_unavailable`。
- RED：fail-once generator 在 attempt-01 抛出 `generation_failure`、attempt-02 成功；原实现直接抛异常。
- GREEN：runner 捕获结构化 generation failure，写失败 generation artifact、保留 failure code/attempt、继续调度；报告记录 failure 与 generation provenance。retrieval 仅当全部错误确为 `snapshot_unavailable` 才使用该 code，否则为 `retrieval_infrastructure_failure`。不足有效重复单独为 `insufficient_valid_repeats`。
- Focused：`tests/recall_experiments/test_runner.py`，10 passed；CLI 12 passed。

## Finding 4 — DeepSeek citation seed 与限额（Important）

- 核验：成立。LLM payload 剥离了动作必须回传的 `seed_canonical_id`；citation handler 直接采用模型 action limit，未受 recipe execution limit 约束。
- RED：合法冻结 seed payload 缺少 `seed_canonical_id`；action limit 99 在 recipe limit 9 下仍向 backend 传 99。
- GREEN：只在 `seed_candidates[].seed_canonical_id` 暴露可执行 seed reference；Gold/其他 identifier 继续 fail-closed 扫描。handler 使用 `min(action.limit, context.max_results_per_action)`。
- Focused：generation + citation tests，48 passed。

## Finding 5 — report/compare 执行身份未绑定（Important）

- 核验：成立。原 report 仅有 `run_id` 与 attempts；compare 只抽取 recall result，无法证明 recipe/sample/prompt/backend/snapshot/action/runtime/budget/pricing/generation 身份一致。
- RED：两个数值完全相同但 recipe SHA 不同的 v1 report 被错误比较为 `passed`；live runtime 只有 backend label 而无 budget/pricing identity 也被放行。
- GREEN：在既有 recall result schema 外新增 `candidate-recall-report-v1` execution identity envelope，绑定：
  - method、recipe SHA、sample SHA；
  - prompt SHA、generator type/model、generation backend call/snapshot provenance；
  - retrieval backend、snapshot manifest SHA、actions SHA；
  - action/result limits、candidate-pool policy；
  - repeat/max-attempt、runtime live authorization；
  - runtime backend、budget policy、pricing policy identity。
- Compare fail-closed：legacy+legacy 仍可读取；v1+v1 必须 envelope 完全一致；legacy/v1 混用或任何 v1 mismatch 都返回 `config_mismatch`。既有 result model/version 未重写。
- Focused：CLI + runner + artifact/generation tests，61 passed（最终 CLI 12 passed）。

## Finding 6 — hash-check 后按路径重开（TOCTOU）（Important）

- 核验：成立。`formal_run.py` 先读 bytes/hash，再通过 `read_jsonl(path, ...)` 重开；historical adapter 也在 verification 后多次重新 `read_text/read_bytes`。
- RED：回归将 path-based `read_jsonl` 设为 trap，原 `load_queries()` 命中 trap。
- GREEN：formal-run Gold、business result、execution 都直接解析 `_read_bound_bytes()` 返回的同一 bytes。historical adapter 引入 immutable `_VerifiedSource(path, content, sha256)`，后续 JSON/JSONL、source hash 与 snapshot manifest inspection 均消费 verified bytes，不再重开已验证源。
- Focused：input 15 passed；historical replay + inventory 26 passed。

## Finding 7 — inventory 非文本 family 判定（Minor）

- 核验：成立。整体兼容性原逻辑硬编码要求 `text_search + title_search`，遗漏设计允许的 citation 非文本 family。
- RED：`text_search + citation_expand` 无法通过兼容性 helper（原 helper/能力不存在）。
- GREEN：明确要求至少一个 text family，以及 `title_search | citation_expand` 中至少一个 non-text family。
- Focused：inventory 13 passed。

## Finding 8 — recipe/prompt model 不一致（Minor）

- 核验：成立。recipe loader 绑定 prompt bytes/SHA，但未比较 prompt artifact `model` 与 generator recipe `model`。
- RED：recipe `different-model` + prompt `deepseek-v4-flash` 被错误接受。
- GREEN：从已绑定 prompt bytes 解析 model，并在 load 阶段要求精确一致。
- Focused：recipes 21 passed。

## 最终验证

- `python -m pytest tests/recall_experiments -q`：`227 passed in 6.59s`。
- tracked Python Ruff：`All checks passed!`。
- `python -m mypy src scripts`：`Success: no issues found in 135 source files`。
- `git diff --check`：通过（仅 Git 的 LF→CRLF 工作区提示，无 whitespace error）。
- 未将中途启动后按主代理要求终止的全仓 offline pytest 计为验证证据。

## 剩余风险

- v1 compare 有意采用严格 identity 全等；需要比较“不同 recipe 但被人工证明等价”的未来场景时，必须先设计独立、版本化的 compatibility identity，不能放宽当前 fail-closed 规则。
- live pricing 仍由授权组合根提供；缺少 hash-bound pricing/budget identity 时现在会拒绝运行。
- generation failure artifact 记录结构化安全错误与 provenance，不保存 provider 原始敏感响应。

## Independent-review follow-up

The post-implementation review identified five additional audit-boundary gaps. Each was
reproduced with a minimal failing regression before implementation:

1. Live policy identity: RED accepted caller-declared invalid, zero, or controller-mismatched
   hashes. GREEN derives the budget hash from the injected controller's canonical budget,
   parses and hashes the supplied verified pricing-policy bytes, and requires the injected LLM
   backend to expose the same pricing-policy identity. Missing or inconsistent evidence fails
   closed; no live price is guessed.
2. Action/snapshot TOCTOU: RED showed execution consumed bytes before identity reopened the
   path. GREEN has fixed/manual generators retain the SHA-256 of the exact action bytes they
   parsed, while replay constructs the snapshot reader and report identity from one manifest
   byte buffer. Regression tests mutate the paths afterward and prove the consumed identity and
   behavior remain bound to the original bytes.
3. Legacy compare bypass: RED showed deleting `execution_identity` from two v1 reports still
   allowed comparison. GREEN requires every `candidate-recall-report-v1` report to contain an
   equal v1 identity envelope. Only two reports explicitly marked
   `candidate-recall-report-legacy-v0` may take the legacy compatibility path; mixed, unknown,
   or identity-less v1 inputs return `config_mismatch`.
4. Generation call audit: RED retained only final-call provenance after repair. GREEN persists a
   typed receipt for each initial/repair call, including kind, usage, provenance, errors, and
   terminal state, plus `repair_count`, through success/failure generation artifacts and the
   report provenance envelope.
5. Citation prompt contract: RED found the system instruction prohibited the same seed ID the
   validator requires. GREEN explicitly permits only the supplied `seed_canonical_id` to be
   returned verbatim for `citation_expand`; other identifiers remain prohibited.

Fresh verification after the follow-up:

- focused tests: `104 passed in 2.77s`
- recall suite: `231 passed in 7.59s`
- Ruff (changed recall/storage/tests scope): `All checks passed!`
- mypy (recall package plus dependency snapshot): `Success: no issues found in 29 source files`
- `git diff --check`: no whitespace errors (only the repository's expected LF/CRLF notices)

No network or live provider call was made, `.env` was not read, and the three user-owned
untracked paths listed above remain untouched and unstaged.

## Final fail-closed review follow-up

Four remaining review findings were reproduced independently and fixed with RED/GREEN tests:

1. Missing historical evidence: RED showed `historical_run=None` was incorrectly sent through
   pairwise identity comparison and became `config_mismatch`. GREEN validates the current
   report's declared schema/identity by itself, then emits the evaluator's truthful
   `insufficient_historical_evidence` / `not_provable` result. Invalid current reports still fail
   closed.
2. Live runtime introspection: RED proved a caller-built fake backend plus caller-declared hashes
   could execute. The underlying live adapters do not yet expose a shared public, immutable,
   versioned identity contract covering provider/dependency, adapter/model/version,
   endpoint/operation, actual pricer, and controller. The safe minimal GREEN therefore removes
   the caller `backend_identity` and pricing-bytes inputs from `build_live_runtime` and disables
   recall live-runtime construction/acceptance before any provider call. Constructing through
   the official boundary returns `live_runtime_unavailable`; injected declarations return
   `config_mismatch`. Live support must remain unavailable until those actual adapters expose
   the complete introspectable contract; private-field inspection is deliberately not used as
   evidence.
3. Formal-live enforcement identity: RED showed controllers with different `formal_live` and
   reservation TTL values had the same budget hash. GREEN adds a read-only controller policy
   fingerprint over a versioned controller identity, the complete `SearchBudget`, `formal_live`,
   and `reservation_ttl_seconds`. Because live recall is now disabled, no controller lacking
   `formal_live=True` can reach execution.
4. Attempt status semantics: RED showed generation failures used `generation_failure` as the
   status. GREEN records every failed attempt as `attempt_status="failed"` and keeps the reason
   only in `failure_code`. The attempt model now rejects any other status and requires failed
   status/failure code to appear together.

Fresh final verification:

- focused RED/GREEN checks: `5 passed`, followed by runner scope `11 passed`
- recall suite: `235 passed in 6.95s`
- Ruff (changed control/recall/tests scope): `All checks passed!`
- mypy (budget controller plus recall package): `Success: no issues found in 29 source files`

No network or live provider call was made, `.env` was not read, and the user-owned untracked
paths remain untouched and unstaged.

## Strict execution-identity validation follow-up

The final review found that a v1 execution identity containing only
`identity_schema_version` still passed current-only validation. RED tests demonstrated that the
version shell, deletion of every required top-level field, malformed/zero SHA-256 values,
boolean-as-integer values, invalid repeat bounds, generator binding contradictions, and replay
marked as live were accepted.

GREEN introduces a frozen, extra-forbid `ExecutionIdentity` model that exactly covers the
current `_execution_identity` payload: method, recipe/sample/prompt hashes, generator type/model,
retrieval backend, snapshot/actions hashes, action/result limits, candidate-pool policy,
repeat/max-attempt counts, live authorization, and the complete versioned runtime identity.
Comparison now canonicalizes both reports through that model before equality testing, and the
producer validates/canonicalizes the identity before writing it.

Conditional validation requires:

- nonzero, correctly formatted SHA-256 identities and strict integer/boolean types;
- manual/fixed generators to bind actions and omit prompt/model;
- DeepSeek generators to bind prompt/model and omit fixed action bytes;
- sealed replay to be non-live and bind the same manifest hash in the outer and runtime identity;
- snapshot-unavailable offline execution to use an explicit versioned unavailable runtime and no
  manifest hash;
- live execution to be authorized, snapshot-free, and to carry a complete versioned live runtime
  whose search/citation/LLM dependencies share the controller and pricing fingerprints.

The first full verification exposed one useful regression: the pre-existing empty runtime for an
omitted snapshot became `config_mismatch` before producing the truthful `snapshot_unavailable`
attempt. That state now has an explicit `candidate-recall-unavailable-runtime-v1` identity rather
than weakening the successful replay rules.

Fresh verification:

- strict-identity focused tests: `27 passed in 1.26s`
- recall suite: `260 passed in 7.23s`
- Ruff: `All checks passed!`
- mypy: `Success: no issues found in 29 source files`
- `git diff --check`: no whitespace errors (only expected LF/CRLF notices)

No network/live execution or `.env` read occurred. User-owned untracked paths remain untouched
and unstaged.
