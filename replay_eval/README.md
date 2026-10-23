# replay_eval: Replay Evaluation

`replay_eval` dùng một lần full generation để đánh giá nhiều early-exit method trên cùng một trajectory.

## Luồng chạy

1. Runner tạo agent `replay_eval`.
2. Agent gọi model một lần để lấy full reasoning trace.
3. Agent replay từng prefix `r_1:t`.
4. Mỗi method gồm 3 phần:
   - `checkpoint` (`S`): quyết định có kiểm tra tại vị trí này không.
   - `verifier` (`pi`): chấm prefix hiện tại đã đủ để trả lời chưa.
   - `stop_rule` (`G`): biến điểm verifier thành quyết định dừng.
5. Khi method dừng, agent append `</think>` rồi gọi model thật để sinh final answer.
6. Runner ghi kết quả từng method vào `additional_logs.method_results`.

Full reasoning trace chỉ được dùng làm trajectory để replay. Nếu verifier cần temporary answer hoặc trial answer, nó vẫn gọi model qua `generate_temp_answer()`. Hàm này sinh answer mới cho prefix ở replay position hiện tại; cache của nó chỉ tránh gọi lại model khi cùng một position đã có temporary answer trước đó.

Nếu full trajectory chưa sinh `</think>`, kể cả khi đã dùng hết
`max_completion_token`, replay agent không còn đánh dấu cả example là generation
failure. Nó giữ reasoning đã có, append force-quit `</think>`, rồi gọi model một
lần nữa với budget riêng `max_final_answer_completion_token`. Với config mặc
định, token logic của answer là:

```text
prompt_tokens + thinking_tokens (tối đa 32000) + answer_tokens (tối đa 4096)
```

Runner ghi `prompt_tokens`, `thinking_tokens`, `answer_tokens`, và
`completion_tokens` ở top level. Chi tiết force-quit nằm trong
`additional_logs`: `forced_final_answer`, `thinking_limit_reached`,
`final_answer_truncated`, `final_answer_prompt_tokens`, và
`logical_total_tokens`.

## Continue after confirmed exits (trap-analysis samples)

The shared flag applies to every enabled (uncommented) method in replay v1 and v2:

```yaml
agent_kwargs:
  replay_eval:
    exit: false  # default: true
```

With `exit: true`, each method stops at its first confirmed exit and keeps its
existing result name. With `exit: false`, every confirmed exit produces a final
answer, and replay continues along the same original trajectory with the same
method configuration and evolving state. Rejected checkpoints produce no answer
sample. Temporary/trial answers still use fresh model calls through
`generate_temp_answer()` and its position cache.

Names count **confirmed exits**, independently for each method and example:
`deer:1`, `deer:2`, ... . Rejected checkpoints do not consume a number. The actual
replay position remains in `position`; the suffix is not a token position or an
index over all checkpoints. Every method also saves `deer:full`, containing the
full-trajectory answer (or the existing forced-finalization fallback), even if it
never confirms an exit. One-shot baseline checkpoint schedules remain one-shot.

Continued results include `base_method: "deer"`, `exit: false`, and
`exit_index: 1` (or `null` for `:full`). Confirmed samples keep
`stopped_early: true` to mark the exit decision, although replay continues;
`:full` uses `false`. State/logs are frozen at each sample and verifier token
usage is cumulative through that checkpoint. Existing evaluators score each
answer independently. Final answers may be generated after replay and shared
between methods at the same prefix, as in ordinary replay; they never become
part of the reasoning trajectory.

Use these samples to compare an earlier accepted exit with later accepted exits
under identical method settings, including whether a later answer recovers from
an earlier incorrect exit. This mode does not generate answers for rejected
checkpoints or classify traps itself.

Override the YAML setting with `--exit false` (or `--exit=true`) through
`scripts/runner.py`, `python -m replay_eval.v2`, or the evaluation bash wrappers.
For standalone v2 configuration, use `runtime.exit`.

V2 incremental replay (`rerun_all: false`) recognizes a continued method as
complete only when its matching `base_method:full` row exists. Numbered rows
alone cause that method to replay from the beginning and replace its previous
continued-result group. Ordinary first-exit results do not satisfy a continued
run; both modes can coexist in an incrementally merged output. Use a new output
file when switching modes with record-level `--resume`, which skips completed
records. Final-answer-only reruns preserve the sample names and metadata.

## Token usage theo method

Mỗi phần tử trong `additional_logs.method_results` là một `MethodResult` và có các field usage:

```json
{
  "method": "dtsr",
  "prompt_tokens": 1234,
  "completion_tokens": 56,
  "verifier_prompt_tokens": 789,
  "verifier_completion_tokens": 12
}
```

- `prompt_tokens` / `completion_tokens`: token của LLM call sinh final answer cho method đó sau khi early-exit. Method không dừng sớm và dùng `y_full` từ trajectory sẽ không có thêm final-answer call nên hai field này có thể là `null`.
- `verifier_prompt_tokens` / `verifier_completion_tokens`: tổng token của các temporary answer mà verifier của method đó sử dụng trong quá trình replay. Khi nhiều method dùng chung một cached temporary answer, mỗi method đều báo usage của answer đó; vì vậy cộng các field này giữa các method sẽ tính lặp shared API calls. Verifier không dùng temporary answer thì hai field này là `0`.

Verifier có LLM call phải cộng usage vào `ctx.state` để replay agent attribute đúng theo từng method. Các built-in verifier đang dùng convention này:

```python
state = ctx.state
state["_sufficiency_prompt_tokens"] = state.get("_sufficiency_prompt_tokens", 0) + (
    score_response.prompt_tokens or 0
)
state["_sufficiency_completion_tokens"] = state.get(
    "_sufficiency_completion_tokens", 0
) + (score_response.completion_tokens or 0)
```

Với probe verifier:

```python
state["_probe_prompt_tokens"] = state.get("_probe_prompt_tokens", 0) + (
    response.prompt_tokens or 0
)
state["_probe_completion_tokens"] = state.get("_probe_completion_tokens", 0) + (
    response.completion_tokens or 0
)
```

`ReplayEvaluatorAgent` cộng các key `_probe_*`, `_sufficiency_*`, `_dynasor_*`, `_answer_coverage_*`, `_codestop_*`, và `_puma_probe_*` thành `verifier_prompt_tokens` và `verifier_completion_tokens` khi serialize `MethodResult`. Nếu thêm verifier LLM mới, hoặc reuse một trong các nhóm key này, hoặc thêm helper aggregation tương ứng trong `replay_eval/replay.py`.

## LLM của verifier

Verifier có thể dùng solver LLM hiện tại (`same`) hoặc tự load LLM riêng (`custom`). Cả hai đều dùng text completion.

```yaml
agent_kwargs:
  replay_eval:
    methods:
      - factory: replay_eval.methods.build_dtsr_method
        kwargs:
          name: dtsr
          llm: same
          probe_tokens: 20
          completion_kwargs:
            temperature: 0
            max_completion_token: 64
          reflection_signals:
            - "wait"
            - "Wait"
            - "but"
            - "But"
      - factory: replay_eval.methods.build_dtsr_method
        kwargs:
          name: flash_think
          llm: custom
          llm_name: deepinfra:Qwen/Qwen2.5-7B-Instruct
          chat_template:
            type: hf
            model_id: Qwen/Qwen2.5-7B-Instruct
            add_generation_prompt: true
            think_start_token: "<think>"
            think_end_token: "</think>"
          completion_kwargs:
            temperature: 0
            max_completion_token: 64
```

Prompt được render qua chat template nếu có, sau đó handler tự thêm đoạn đóng thinking như `<think>\n\n</think>\n` để tránh model tiếp tục sinh reasoning khi template mặc định mở thinking.

## Cấu hình

`replay_eval` nằm trong cùng section `method` với các method khác. Chọn nó bằng:

```bash
python scripts/runner.py \
  --agent replay_eval \
  --dataloader gsm8k \
  --max-total-thinking-token 32000 \
  --max-final-answer-completion-token 4096 \
  --limit 1
```

PUMA trong `replay_eval` dùng `SemanticStepCheckpoint`, nên checkpoint chạy ở boundary của reasoning step thay vì scan từng word/token. Có thể đặt `embedding_device: auto`, `cuda`, `mps`, hoặc `cpu` cho embedding model; similarity score được đưa về Python `float` sau khi compute.

PUMA trial answer không lấy từ full generation. Khi một step được flag để trial, PUMA gọi `generate_temp_answer()` để sinh trial answer mới từ prefix tương ứng. Nếu `verify_batch_token` làm nhiều checkpoint map về cùng prompt position, cache của `generate_temp_answer()` sẽ reuse đúng trial answer đã sinh cho position đó để tránh duplicate model call.

PUMA tính confidence bằng cùng token-probability geometric mean như DEER. Phần
được chấm dừng tại answer terminator đầu tiên (ví dụ `}` hoặc `</think>`) và vẫn
bị giới hạn bởi `probe_tokens`; full trial answer vẫn được giữ để extract và so
sánh answer. Log `puma_trials[*].confidence_tokens` ghi số token thực sự tham gia
confidence score.

## Thêm method mới

Tạo factory trả về `EarlyExitMethod` trong `replay_eval/methods.py`, rồi thêm vào:

```yaml
agent_kwargs:
  replay_eval:
    methods:
      - factory: replay_eval.methods.build_deer_method
        kwargs:
          name: deer
          threshold: 0.95
```
