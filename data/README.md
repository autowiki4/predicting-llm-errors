# Data

## `raw/gsm8k_test.jsonl` (bundled)

The 1,319-question test split of **GSM8K** (Grade School Math 8K), downloaded unchanged from
<https://github.com/openai/grade-school-math>.

Each line is a JSON object:

```json
{"question": "Janet’s ducks lay 16 eggs per day. ...",
 "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n... \n#### 18"}
```

The text after `####` is the final answer. `<<...>>` are calculator annotations, which
`llm_reliability.data` strips.

To fetch the 7,473-question train split as well:

```bash
llmrel download-data --split train      # -> data/raw/gsm8k_train.jsonl
```

### Licence and citation

GSM8K is released under the MIT License, Copyright (c) 2021 OpenAI.

> Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). *Training Verifiers to Solve Math Word
> Problems.* arXiv:2110.14168.
