#!/usr/bin/env bash
# Script to generate a pdf that contains all chapters and sections of the book
# Instructions (linux) - by doing the following in the terminal:
#   1. install pandoc and texlive (with xelatex) by running:
#      sudo apt update && sudo apt install -y pandoc texlive texlive-xetex texlive-fonts-recommended texlive-latex-extra fonts-dejavu
#   2. navigate to the repository root
#   3. make the script executable by running: chmod +x build_book_pdf.sh
#   4. generate the book pdf by running: ./build_book_pdf.sh
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$ROOT_DIR/book"
OUTPUT_PDF="$OUTPUT_DIR/book.pdf"

BOOK_TITLE="Practical Generative AI – From Zero to Hero"
AUTHOR="Dennis V. Hansen"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
REPO_URL="https://github.com/dennisdeh/genai-zero-to-hero"
GIT_COMMIT="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

mkdir -p "$OUTPUT_DIR"

TMP_FRONT_MATTER="$(mktemp)"
TMP_HEADER_TEX="$(mktemp)"
TMP_LUA_FILTER="$(mktemp --suffix=.lua)"

# -------------------------------------------------------------------
# Pandoc filter: allow line breaks in long inline code (file paths, module names)
# after "/", "." and "_", so that they do not run off the page
# -------------------------------------------------------------------
cat > "$TMP_LUA_FILTER" <<'EOF'
local map = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["#"] = "\\#",
  ["$"] = "\\$", ["%"] = "\\%", ["&"] = "\\&", ["~"] = "\\textasciitilde{}",
  ["^"] = "\\textasciicircum{}",
  ["_"] = "\\_\\allowbreak{}", ["/"] = "/\\allowbreak{}", ["."] = ".\\allowbreak{}",
}
function Code(el)
  if not FORMAT:match("latex") then return nil end
  local text = el.text:gsub(".", function(c) return map[c] or c end)
  return pandoc.RawInline("latex", "\\texttt{" .. text .. "}")
end
EOF

# -------------------------------------------------------------------
# LaTeX page/header/footer configuration
# -------------------------------------------------------------------
cat > "$TMP_HEADER_TEX" <<EOF
% Page geometry
\\usepackage[a4paper,margin=3cm]{geometry}

% Header / footer
\\usepackage{fancyhdr}
\\pagestyle{fancy}
\\fancyhf{}

% Header
\\fancyhead[L]{${AUTHOR}}
\\fancyhead[R]{\\href{${REPO_URL}}{${REPO_URL}}}

% Footer
\\fancyfoot[C]{\\thepage}

\\renewcommand{\\headrulewidth}{0.4pt}
\\renewcommand{\\footrulewidth}{0pt}

% Avoid lines running into the margin: more flexible spacing, and wrap long lines in code blocks
\\setlength{\\emergencystretch}{3em}
\\usepackage{fvextra}
\\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\\\\{\\}}
\\RecustomVerbatimEnvironment{verbatim}{Verbatim}{breaklines,breakanywhere}
EOF

# -------------------------------------------------------------------
# Front page (Pandoc Markdown + YAML)
# -------------------------------------------------------------------
cat > "$TMP_FRONT_MATTER" <<EOF
---
title: "$BOOK_TITLE"
author: "$AUTHOR"
date: "$TIMESTAMP"
---

\

**Repository:** [$REPO_URL]($REPO_URL)

\

**Git commit:** \`$GIT_COMMIT\`

\clearpage
\tableofcontents
\clearpage
EOF

# -------------------------------------------------------------------
# Discover book chapters (lexicographical order)
# -------------------------------------------------------------------
mapfile -t BOOK_FILES < <(
  find "$ROOT_DIR" \
    -type f \
    -name 'book_*.md' \
    | sort
)

if [[ "${#BOOK_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: No files matching 'book_*.md' found." >&2
  exit 1
fi

echo "Files included:"
printf '  %s\n' "${BOOK_FILES[@]}"
echo

# -------------------------------------------------------------------
# Build PDF
# -------------------------------------------------------------------
pandoc \
  "$TMP_FRONT_MATTER" \
  "${BOOK_FILES[@]}" \
  --number-sections \
  --pdf-engine=xelatex \
  --include-in-header="$TMP_HEADER_TEX" \
  --lua-filter="$TMP_LUA_FILTER" \
  -V monofont="DejaVu Sans Mono" \
  -V monofontoptions="Scale=0.85" \
  -o "$OUTPUT_PDF"

rm "$TMP_FRONT_MATTER" "$TMP_HEADER_TEX" "$TMP_LUA_FILTER"

echo "PDF generated: $OUTPUT_PDF"