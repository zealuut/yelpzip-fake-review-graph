# routeL_abnormal_aux_head

Route L evaluates whether the abnormal/text branch can be explicitly supervised
before its vectors are passed into the D1 graph pipeline. For this route,
"strict D1 base" means fresh training under the D1 `ce6a9d6` protocol in the
same environment. It does not mean freezing or copying D1 review vectors,
graph edges, or checkpoints.

This route uses the existing D1 data protocol with `mask_source=full_text`.
Despite the historical `llm_masked_logic` encoder name, these runs do not use
LLM span masks unless a future config explicitly changes `mask_source`.

The runner executes `D1_NO_AUX_fresh_control` first and stops if the resulting
`Base_LogicAE_CB` graph AUC does not clear the configured D1 floor. Aux-head
results are not interpretable until this no-aux fresh control passes.

The phase-1 queue runs:

- `D1_AUX_full`
- `A1_no_cross_attention`
- `A5_no_logic_bilstm`
- `A8_no_gate`
- `A16_aux_on_final_review_vector`
- `A17_aux_text_only`

Future internal ablations can be added by appending variants in
`configs/abnormal_aux_phase1.json`; the runner forwards each variant's
architecture flags to `graph.run_final_experiment`.
