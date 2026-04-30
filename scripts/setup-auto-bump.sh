#!/usr/bin/env bash
# One-time setup for the RooDojo auto-bump pipeline.
#
# After running this, every push to one of the 7 workflow repos will
# automatically bump the corresponding pin in this RooDojo meta-repo.
# No manual `git submodule update && git push` ever again.
#
# Prereqs:
#   - The `gh` CLI (https://cli.github.com), authenticated with an
#     account that has write access to all 7 workflow repos.
#   - A fine-grained PAT with Contents: read+write on Remoroo/RooDojo.
#     Create one at https://github.com/settings/personal-access-tokens
#     ("Beta" GitHub PATs). Repository access = Remoroo/RooDojo.
#
# Usage:
#   ./scripts/setup-auto-bump.sh <pat-token>
#
# The token is shipped as a Secret named ROODOJO_TOKEN to each of the
# 7 workflow repos via `gh secret set`. After setup, you can rotate
# the token by running this script again with a new value.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <fine-grained-PAT>"
  echo ""
  echo "Create a PAT here:"
  echo "  https://github.com/settings/personal-access-tokens"
  echo "  - Repository access: Remoroo/RooDojo"
  echo "  - Permissions: Contents (read+write)"
  exit 2
fi

TOKEN="$1"
REPOS=(
  Remoroo/ppo-bipedal-hardcore
  Remoroo/dog-run-locomotion
  Remoroo/eye-in-hand-calibration
  Remoroo/tts-neural-voice
  Remoroo/asr-speech-recognition
  Remoroo/cifar10-speedrun
  Remoroo/higgs-boost
)

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: gh CLI not installed. brew install gh"
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "Error: gh CLI not authenticated. Run: gh auth login"
  exit 1
fi

echo "Installing ROODOJO_TOKEN secret in 7 workflow repos..."
for repo in "${REPOS[@]}"; do
  printf "  %-50s ... " "$repo"
  if printf "%s" "$TOKEN" | gh secret set ROODOJO_TOKEN --repo "$repo" --body - >/dev/null 2>&1; then
    echo "ok"
  else
    echo "FAILED"
    exit 1
  fi
done

echo ""
echo "Done. From now on:"
echo "  - Push from inside any workflow → bump-roodojo action runs → RooDojo pin updates."
echo "  - Watch action runs at https://github.com/<repo>/actions"
echo ""
echo "First test:"
echo "  cd vision/cifar10-speedrun"
echo "  git commit --allow-empty -m 'ci: smoke-test auto-bump'"
echo "  git push"
echo "  # Then check https://github.com/Remoroo/cifar10-speedrun/actions"
echo "  # And: cd .. && git fetch && git log --oneline origin/main -3"
