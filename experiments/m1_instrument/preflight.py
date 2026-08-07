"""Mile 1 preflight: confirm remote artifacts exist BEFORE downloading 8 GB."""
from __future__ import annotations

from huggingface_hub import HfApi

from probe.pins import LENS_REPO, LENS_REVISION, MODEL_ID

api = HfApi()
ok = True

print("--- model ---")
try:
    info = api.model_info(MODEL_ID)
    print(f"[OK]   {MODEL_ID}  gated={info.gated}")
except Exception as exc:
    ok = False
    print(f"[FAIL] {MODEL_ID}  {type(exc).__name__}")
    print("       candidates on the hub:")
    try:
        for m in api.list_models(search="Qwen3.5-4B", limit=12):
            print(f"         {m.id}")
    except Exception as e2:
        print(f"         search failed: {e2}")

print()
print("--- lens repo ---")
try:
    files = api.list_repo_files(LENS_REPO)
    print(f"[OK]   {LENS_REPO}  {len(files)} files on main")
    hits = sorted(f for f in files if "qwen" in f.lower())
    print(f"       qwen-related entries ({len(hits)}):")
    for f in hits[:25]:
        print(f"         {f}")
except Exception as exc:
    ok = False
    print(f"[FAIL] {LENS_REPO}  {type(exc).__name__}: {exc}")

print()
print("--- lens revision ---")
try:
    refs = api.list_repo_refs(LENS_REPO)
    names = [b.name for b in refs.branches] + [t.name for t in refs.tags]
    print(f"       refs: {names}")
    if LENS_REVISION in names:
        print(f"[OK]   revision {LENS_REVISION} exists")
        rf = api.list_repo_files(LENS_REPO, revision=LENS_REVISION)
        print(f"       {len(rf)} files at that revision")
        for f in sorted(rf)[:25]:
            print(f"         {f}")
    else:
        ok = False
        print(f"[FAIL] revision {LENS_REVISION} NOT among refs")
except Exception as exc:
    ok = False
    print(f"[FAIL] refs  {type(exc).__name__}: {exc}")

print()
print("PREFLIGHT PASS" if ok else "PREFLIGHT FAIL -- do not download")
raise SystemExit(0 if ok else 1)
