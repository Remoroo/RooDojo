# Speech · RooDojo

Two open frontiers. Harnesses are locked, baselines are pending — the
engine has not yet started iterating on either.

| Workflow | Domain | Headline metric | Status |
|---|---|---|---|
| [`tts-neural-voice`](./tts-neural-voice) | text-to-speech | `mel_recon_loss` on locked eval | open frontier |
| [`asr-speech-recognition`](./asr-speech-recognition) | speech-to-text | `wer` on locked eval, target ≤ 5% | open frontier |

## What "open frontier" means

The contract (`program.md`) is locked, the eval-set placeholders are reserved,
and the universal log shape (`results.tsv`) is committed empty. When the
agent picks up these workflows, the first commit will be the baseline; every
subsequent commit appends one row to `results.tsv` regardless of outcome.

These two workflows ride the same universal contract as the RL and robotics
workflows, so any improvement loop the engine learns elsewhere transfers here
with no special-casing.
