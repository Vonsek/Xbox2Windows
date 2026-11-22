"""Synthetic PSO references and archived shader aliases."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pixel_aot_check import assembly
from pso_readiness import report


def with_rounding(instructions):
    """Insert the rounding forms and a multiply that pixel_aot lowers before the pack."""
    index = next(i for i, line in enumerate(instructions) if line.startswith("v_cvt_pkrtz"))
    extra = ["v_floor_f32_e32 v12, v12", "v_fract_f32_e32 v13, v13", "v_trunc_f32_e32 v15, v15",
             "v_add_f32_e32 v12, v12, v13", "v_mul_f32_e32 v15, v15, v12"]
    return instructions[:index] + extra + instructions[index:]


def check_scanner_tracks_pixel_aot():
    import pixel_aot
    from scan_unsupported_shaders import is_instruction_supported, is_opcode_recognized
    for name in pixel_aot.ALU:
        opcode = f"v_{name}_f32"
        assert is_opcode_recognized(opcode) and is_opcode_recognized(opcode + "_e32"), opcode
        assert is_instruction_supported(f"{opcode}_e32 v1, v2, v3"), opcode
    for name, (arity, _) in pixel_aot.INTEGER.items():
        opcode = f"v_{name}"
        assert is_opcode_recognized(opcode), opcode
        operands = ", ".join(f"v{k}" for k in range(1, arity + 2))
        assert is_instruction_supported(f"{opcode} {operands}"), opcode
        if arity <= 2:
            assert is_opcode_recognized(opcode + "_e32"), opcode
            assert is_instruction_supported(f"{opcode}_e32 {operands}"), opcode
    assert not is_instruction_supported("v_bfi_b32_e32 v0, v1, v2, v3")
    assert not is_instruction_supported("v_xor_b32 v0, v1")
    assert not is_instruction_supported("v_not_b32 v0, -v1")
    # The SOPK compares and s_cmovk_i32 take an SGPR and the 16-bit immediate as LLVM prints it.
    for opcode in (*(f"s_cmpk_{r}_{t}32" for r in ("eq", "lg", "gt", "ge", "lt", "le") for t in "iu"), "s_cmovk_i32"):
        assert is_opcode_recognized(opcode), opcode
        assert is_instruction_supported(f"{opcode} s44, 0x8000"), opcode
        assert not is_instruction_supported(f"{opcode} s44, 0x10000"), opcode
        assert not is_instruction_supported(f"{opcode} v1, 0x1"), opcode


def check():
    check_scanner_tracks_pixel_aot()
    code = b"synthetic native bytes"
    digest = hashlib.sha256(code).hexdigest()
    pixel = {"stage": "pixel", "sha256": digest}
    mesh = {"stage": "mesh", "sha256": digest}
    manifest = {"shaders": [
        dict(pixel, native_file="pixel.bin"), dict(pixel, native_file="alias.bin"),
        mesh, {"stage": "vertex"},
    ]}
    pipelines = {"pipelines": [
        {"id": "missing-root", "effective_root_signature": "missing", "shaders": [pixel]},
        {"id": "mesh-evidence", "effective_root_signature": "present", "shaders": [mesh]},
        {"id": "absent-record", "effective_root_signature": "unknown",
         "shaders": [dict(pixel, sha256="0" * 64)]},
        {"id": "pixel", "effective_root_signature": "present", "shaders": [pixel]},
    ]}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pixel.bin").write_bytes(code)
        for path, data in (("manifest.json", manifest), ("pipelines.json", pipelines)):
            (root / path).write_text(json.dumps(data), encoding="utf-8")
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parents[1] /
                                   "tools" / "pso_readiness.py"), "manifest.json", "pipelines.json"],
                                   cwd=root, capture_output=True, text=True, check=True)
        actual = json.loads(completed.stdout)
        assert (actual["archive_records"], actual["records_with_native_sha256"],
                actual["unique_native_stage_hashes"]) == (4, 3, 2)
        missing, mesh_row, absent, untested = actual["pipelines"]
        assert missing["first_blocker"] == {"reason": "missing_root_signature"}
        assert mesh_row["shaders"]["mesh"]["status"] == "stage_not_assessed"
        assert absent["shaders"]["pixel"]["status"] == "shader_not_in_manifest"
        assert untested["status"] == "needs_evidence"
        assert untested["shaders"]["pixel"]["archive_records"] == 2
        # Temporary paths can contain a directory alias on Windows CI.
        unnormalized_root = root / ".." / root.name
        for instructions, status, blocker, detail in (
                (assembly(False), "needs_pso_validation", None, None),
                (with_rounding(assembly(False)), "needs_pso_validation", None, None),
                ([], "blocked", "disassembly_failed", "empty instruction stream"),
                (ValueError("Unknown instruction at byte 0"), "blocked", "disassembly_failed",
                 "Unknown instruction at byte 0"),
                (["s_endpgm"], "blocked", "pixel_lowering_rejected", None),
                (["unknown_instruction"], "blocked", "unsupported_opcodes", "unknown_instruction")):
            patch_args = {"side_effect": instructions} if isinstance(instructions, Exception) else {"return_value": instructions}
            with patch("pixel_aot.disassemble", **patch_args) as scan:
                rows = report(manifest, pipelines, unnormalized_root, "test-llvm")["pipelines"]
                assert rows[0]["first_blocker"] == {"reason": "missing_root_signature"}
                assert rows[2]["status"] == "needs_evidence" and rows[2]["first_blocker"] is None
                assert rows[-1]["status"] == status, rows[-1]
                if blocker:
                    assert rows[-1]["first_blocker"]["reason"] == blocker
                    assert rows[-1]["first_blocker"]["stage"] == "pixel"
                    if detail is not None:
                        assert rows[-1]["first_blocker"]["detail"] == detail
                else:
                    assert rows[-1]["first_blocker"] is None
                scan.assert_called_once_with(code, "test-llvm")
        (root / "pixel.bin").write_bytes(b"mismatched bytes")
        try:
            report(manifest, pipelines, root, "test-llvm")
        except ValueError as exc:
            assert "does not match SHA-256" in str(exc)
        else:
            raise AssertionError("accepted native bytes under the wrong hash")

        outside = {"shaders": [dict(pixel, native_file="../pixel.bin")]}
        try:
            report(outside, pipelines, unnormalized_root, "test-llvm")
        except ValueError as exc:
            assert "outside the manifest directory" in str(exc)
        else:
            raise AssertionError("accepted a native path outside the manifest directory")
    print("PSO census, evidence states, and native hash control passed")


if __name__ == "__main__":
    check()
