#!/usr/bin/env bash
# Convert an existing editable PPTX with the project-local LibreOffice build.
# The output keeps the input stem: a DRAFT PPTX produces a DRAFT PDF.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tool_dir="$project_dir/artifacts/pdf-tools"
soffice="$tool_dir/extracted/opt/libreoffice7.6/program/soffice"
font_config="$tool_dir/fonts.conf"
font_file="$tool_dir/fonts/NotoSansCJKsc-Regular.otf"
input="${1:-$project_dir/slides/ODEBrain_DRAFT_pending_experiments.pptx}"
output_dir="${2:-$project_dir/slides}"

[[ -f "$input" ]] || { echo "PPTX missing: $input" >&2; exit 2; }
[[ -x "$soffice" ]] || { echo "Local LibreOffice missing: $soffice" >&2; exit 2; }
[[ -f "$font_file" ]] || { echo "Local CJK font missing: $font_file" >&2; exit 2; }
command -v gs >/dev/null || { echo "Ghostscript (gs) is required for page validation" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 is required for PPTX page validation" >&2; exit 2; }

expected_pages="$(python3 - "$input" <<'PY'
import re, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    print(sum(bool(re.fullmatch(r'ppt/slides/slide[0-9]+\.xml', n)) for n in z.namelist()))
PY
)"
[[ "$expected_pages" -gt 0 ]] || { echo "No slides found in PPTX" >&2; exit 2; }

mkdir -p "$output_dir" "$tool_dir/work" "$tool_dir/render"
cat > "$font_config" <<EOF
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <include ignore_missing="yes">/etc/fonts/fonts.conf</include>
  <dir>$tool_dir/fonts</dir>
  <alias>
    <family>Microsoft YaHei</family>
    <prefer><family>Noto Sans CJK SC</family></prefer>
  </alias>
</fontconfig>
EOF
work_dir="$(mktemp -d "$tool_dir/work/export.XXXXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT
stem="$(basename "${input%.pptx}")"
temp_pdf="$work_dir/$stem.pdf"
output_pdf="$output_dir/$stem.pdf"

export FONTCONFIG_FILE="$font_config"
export SAL_USE_VCLPLUGIN=svp
"$soffice" "-env:UserInstallation=file://$work_dir/profile" \
  --headless --convert-to pdf:impress_pdf_Export --outdir "$work_dir" "$input"
[[ -s "$temp_pdf" ]] || { echo "LibreOffice produced no PDF" >&2; exit 1; }

actual_pages="$(gs -q -dNODISPLAY -sPDFNAME="$temp_pdf" \
  -c 'PDFNAME (r) file runpdfbegin pdfpagecount = quit')"
[[ "$actual_pages" == "$expected_pages" ]] || {
  echo "Page mismatch: PPTX=$expected_pages PDF=$actual_pages" >&2; exit 1;
}

render_dir="$(mktemp -d "$tool_dir/render/${stem}.XXXXXXXX")"
gs -q -dBATCH -dNOPAUSE -sDEVICE=png16m -r60 \
  -sOutputFile="$render_dir/page-%02d.png" "$temp_pdf"
rendered_pages="$(find "$render_dir" -maxdepth 1 -name 'page-*.png' | wc -l)"
[[ "$rendered_pages" -eq "$expected_pages" ]] || {
  echo "Rendered page mismatch: expected=$expected_pages actual=$rendered_pages" >&2; exit 1;
}

if command -v montage >/dev/null; then
  montage "$render_dir"/page-*.png -tile 4x \
    -geometry 400x225+8+8 -background '#d7dfe4' "$render_dir/contact.png"
fi

cp -f -- "$temp_pdf" "$output_pdf"
echo "PDF: $output_pdf"
echo "Slides/pages/rendered PNGs: $expected_pages/$actual_pages/$rendered_pages"
echo "Visual proof: $render_dir/contact.png"
echo "Inspect the contact sheet and full-size pages for font substitution or cropping."
