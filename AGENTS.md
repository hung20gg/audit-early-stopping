## Notices

- Do not edit `llm/` if not specifically instructed. It is a submodule and should be updated via `git submodule update --remote llm`.
- Must not edit `archive`
- Early-exit generation must preserve the original rendered prompt across generation steps. `Response.input` stores that prompt, and follow-up completion calls should use `Response.get_generation_content()` so the prompt remains `input + content`.
- `generate_temp_answer()` is the fresh temporary/trial-answer model-call path. Its cache only avoids repeating the same temporary answer for the same replay position; it must not be replaced with the full-generation answer.
- Update the runner input params should also update the bash scripts
