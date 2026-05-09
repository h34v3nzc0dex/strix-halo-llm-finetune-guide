#!/usr/bin/env python3
"""Fix Ubuntu mainline kernel .deb run-parts double-dir bug.

Why:
  Mainline kernel .debs invoke `run-parts` with two directories at once,
  which run-parts doesn't accept. This breaks postinst/postrm/preinst/prerm
  on Ubuntu 24.04+. Each call must be split into two, each guarded by a
  directory-existence test using `if/fi` (NOT `&&`) so a missing
  /usr/share/kernel/X.d doesn't propagate exit-1 out of the trigger script.

Usage (per kernel install):
  cd /path/to/workspace/kernels/<ver>
  for f in linux-image*.deb linux-modules*.deb linux-headers-*-generic_*amd64.deb; do
      d=$(basename "$f" .deb)
      mkdir -p extracted/"$d"
      dpkg-deb -R "$f" extracted/"$d"
  done
  python3 /path/to/scripts/fix-kernel-run-parts.py \
      extracted/linux-image*/DEBIAN/{preinst,postinst,prerm,postrm} \
      extracted/linux-modules*/DEBIAN/postinst \
      extracted/linux-headers-*-generic_*/DEBIAN/postinst
  for d in extracted/*; do dpkg-deb --build "$d" "$(basename $d)-fixed.deb"; done
  sudo dpkg -i linux-headers-*-all.deb *-fixed.deb

Two run-parts flavours appear:
  A) DEB_MAINT_PARAMS=... run-parts ... --arg=$version \
         --arg=$image_path /etc/kernel/X.d /usr/share/kernel/X.d
  B) DEB_MAINT_PARAMS=... run-parts ... --arg=$version \
         /etc/kernel/X.d /usr/share/kernel/X.d
"""
import pathlib
import re
import sys

if len(sys.argv) < 2:
    raise SystemExit("usage: fix-kernel-run-parts.py <script>...")

PAT_WITH_IMG = re.compile(
    r"""(?m)
    ^(?P<indent>[ \t]*)
    DEB_MAINT_PARAMS="\$\*"\s+run-parts\s+--report\s+--exit-on-error\s+
    --arg=\$version\s*\\\n
    [ \t]*--arg=(?P<imgpath>"?\$image_path"?)\s+
    (?P<dir1>/etc/kernel/[A-Za-z._]+)\s+(?P<dir2>/usr/share/kernel/[A-Za-z._]+)
    """,
    re.VERBOSE,
)

PAT_NO_IMG = re.compile(
    r"""(?m)
    ^(?P<indent>[ \t]*)
    DEB_MAINT_PARAMS="\$\*"\s+run-parts\s+--report\s+--exit-on-error\s+
    --arg=\$version\s*\\\n
    [ \t]*(?P<dir1>/etc/kernel/[A-Za-z._]+)\s+(?P<dir2>/usr/share/kernel/[A-Za-z._]+)
    """,
    re.VERBOSE,
)


def repl_with_img(m: re.Match) -> str:
    indent = m.group("indent")
    img = m.group("imgpath")
    dir1 = m.group("dir1")
    dir2 = m.group("dir2")
    rp = f'DEB_MAINT_PARAMS="$*" run-parts --report --exit-on-error --arg=$version --arg={img}'
    return (
        f"{indent}if [ -d {dir1} ]; then {rp} {dir1}; fi\n"
        f"{indent}if [ -d {dir2} ]; then {rp} {dir2}; fi"
    )


def repl_no_img(m: re.Match) -> str:
    indent = m.group("indent")
    dir1 = m.group("dir1")
    dir2 = m.group("dir2")
    rp = 'DEB_MAINT_PARAMS="$*" run-parts --report --exit-on-error --arg=$version'
    return (
        f"{indent}if [ -d {dir1} ]; then {rp} {dir1}; fi\n"
        f"{indent}if [ -d {dir2} ]; then {rp} {dir2}; fi"
    )


for path in sys.argv[1:]:
    p = pathlib.Path(path)
    text = p.read_text()
    new, n1 = PAT_WITH_IMG.subn(repl_with_img, text)
    new, n2 = PAT_NO_IMG.subn(repl_no_img, new)
    n = n1 + n2
    if n == 0:
        print(f"{path}: no buggy run-parts found")
    else:
        p.write_text(new)
        print(f"{path}: split {n} run-parts call(s) ({n1} with-img, {n2} no-img)")
