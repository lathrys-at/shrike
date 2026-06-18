#!/usr/bin/env bash
# Swap a staged skill variant into the live skill dir, so the QA eval (which
# reads shrike-skills/create-cards/{SKILL.md,references/examples.md} from a hardcoded
# path) runs against it. Variants live in tests/manual/skill_quality/variants/<name>/.
#
#   use-variant.sh v0      # restore the control (current/main skill)
#   use-variant.sh a|b|c   # apply a tightening variant
#
# shrike-cli.md is constant across variants and is left untouched.
set -euo pipefail

name="${1:?usage: use-variant.sh <v0|a|b|c>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
src="$root/tests/manual/skill_quality/variants/$name"
dst="$root/shrike-skills/create-cards"

[ -f "$src/SKILL.md" ] || { echo "no SKILL.md in $src" >&2; exit 1; }
[ -f "$src/examples.md" ] || { echo "no examples.md in $src" >&2; exit 1; }

cp "$src/SKILL.md" "$dst/SKILL.md"
cp "$src/examples.md" "$dst/references/examples.md"

echo "applied variant '$name':"
printf '  SKILL.md    %5s words\n' "$(wc -w < "$dst/SKILL.md")"
printf '  examples.md %5s words\n' "$(wc -w < "$dst/references/examples.md")"
