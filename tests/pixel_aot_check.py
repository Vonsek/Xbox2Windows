"""Exercise offline pixel translation with independently assembled test programs."""
from pathlib import Path
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pixel_aot
from pixel_aot import ALU, disassemble, native_program

# First word of each line of every program lower() accepted in this run. tools/form_coverage.py counts
# these when this test passes; a program a check expects to be rejected raises before it is recorded.
ACCEPTED = set()
SOPC_OPS = (
    "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32",
    "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32",
    "s_cmp_eq_u64", "s_cmp_lg_u64",
    "s_bitcmp0_b32", "s_bitcmp1_b32",
    "s_bitcmp0_b64", "s_bitcmp1_b64",
)
VOP2_INT_OPS = (
    "v_lshrrev_b32_e32", "v_ashrrev_i32_e32", "v_lshlrev_b32_e32",
    "v_and_b32_e32", "v_or_b32_e32", "v_xor_b32_e32",
    "v_add_nc_u32_e32", "v_sub_nc_u32_e32",
)


def lower(instructions, *args, **kwargs):
    source = pixel_aot.lower(instructions, *args, **kwargs)
    ACCEPTED.update(line.split()[0] for line in instructions if line.strip())
    return source


def write_record(path):
    """Save ACCEPTED where CTest's FORM_COVERAGE_RECORD points; nothing is written without it."""
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(f"{word}\n" for word in sorted(ACCEPTED)), encoding="utf-8")


RAMP = (0, 16, 37, 64, 101, 153, 219, 254)


def assembly(swap):
    red, blue = ("v14", "v12") if swap else ("v12", "v14")
    return [
        "s_mov_b64 s[50:51], exec",
        "s_wqm_b64 exec, exec",
        "s_mov_b32 m0, s24",
        "v_interp_p1_f32_e32 v8, v0, attr0.x",
        "v_interp_p1_f32_e32 v9, v0, attr0.y",
        "v_interp_p2_f32_e32 v9, v1, attr0.y",
        "v_interp_p2_f32_e32 v8, v1, attr0.x",
        "s_and_b64 exec, exec, s[50:51]",
        "s_mov_b32 s21, 0",
        "s_load_dwordx8 s[28:35], s[20:21], null",
        "s_mov_b32 s40, 1",
        "s_mov_b32 s41, 2",
        "s_mov_b32 s42, 3",
        "s_mov_b32 s43, 4",
        "s_waitcnt lgkmcnt(0)",
        "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
        "s_waitcnt vmcnt(0)",
        f"v_cvt_pkrtz_f16_f32_e32 v16, {red}, v13",
        f"v_cvt_pkrtz_f16_f32_e32 v17, {blue}, v15",
        "exp mrt0 v16, v16, v17, v17 done compr vm",
        "s_endpgm",
    ]


def rgb_sample() -> list[str]:
    program = assembly(False)
    sample = next(i for i, line in enumerate(program) if line.startswith("image_sample"))
    program[sample] = program[sample].replace("v[12:15]", "v[12:14]").replace("dmask:0xf", "dmask:0x7")
    program.insert(sample, "v_mov_b32_e32 v15, 0.5")
    return program


DUAL_CASES = ("forward", "reverse", "reuse", "partial")


def dual_program(case="forward", *, swap_weights=False):
    program = assembly(False)
    first = next(i for i, line in enumerate(program) if line.startswith("image_sample"))
    sample_a = program[first]
    sample_b = sample_a.replace("v[12:15]", "v[20:23]").replace("s[28:35]", "s[56:63]")
    load_b = "s_load_dwordx8 s[56:63], s[20:21], 0x20"
    if case == "reuse":
        # Finish A before reloading the same descriptor registers for B.
        program[first:first+1] = [sample_a, "s_waitcnt vmcnt(0)",
                                  load_b.replace("56:63", "28:35"), "s_waitcnt lgkmcnt(0)",
                                  sample_b.replace("56:63", "28:35")]
    elif case in ("forward", "reverse", "partial"):
        program[first:first+1] = [sample_a, sample_b] if case != "reverse" else [sample_b, sample_a]
        program.insert(program.index("s_mov_b32 s40, 1"), load_b)
    else:
        raise ValueError(f"Unknown dual-texture case: {case}")
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    a, b = ("0x3e800000", "0x3f400000") if swap_weights else ("0x3f400000", "0x3e800000")
    ops = []
    for i in range(4):
        ops.extend([f"v_mul_legacy_f32_e32 v{12+i}, {a}, v{12+i}",
                    f"v_fma_f32 v{12+i}, {b}, v{20+i}, v{12+i}"])
    if case == "partial":
        # Scale A's red channel while B is still outstanding.
        wait = program.index("s_waitcnt vmcnt(0)")
        first_mul = ops.pop(0).replace(", v12", ", v24")
        program[wait:wait] = ["s_waitcnt vmcnt(1)", "v_mov_b32_e32 v24, v12", first_mul]
        pos += 3
    program[pos:pos] = ops
    return program


def translate_dual(instructions):
    return lower(instructions, 20, 24, [1, 2, 3, 4], second_texture_offset=32)


def shader_record(blob: bytes, name: bytes) -> bytes:
    encoder = zlib.compressobj(wbits=-15)
    payload = encoder.compress(blob) + encoder.flush()
    return (bytes([len(name) + 1]) + name + b"\0\0"
            + struct.pack("<4I", len(payload), 0, len(payload), len(blob)) + payload)


def envelope(code: bytes, name: bytes = b"fixture", *, private_data: bytes | None = None, directory_end=None,
             compute_width=None, mesh_limits=None) -> bytes:
    program_start = 288 if directory_end is not None else 256
    image = bytearray(program_start) + code
    image[:4] = b"FSSM"
    struct.pack_into("<I", image, 192, program_start - 40)
    if directory_end is not None:
        extent, end_field = directory_end
        struct.pack_into("<I", image, 4, 0x101)
        struct.pack_into("<Q", image, 200, extent)
        struct.pack_into("<Q", image, 232, end_field)
    native = b"XBCH" + bytes(12) + struct.pack("<II", len(image), 0) + zlib.compress(image)
    program = (bytes(24) + b"XDAT" + struct.pack("<II", 1, 0)
               + struct.pack("<III", 13, 48, len(native)) + native)
    if compute_width is not None:
        program = program[:2] + struct.pack("<H", 13 if mesh_limits else 5) + program[4:]
    sections = [b"DXIL" + struct.pack("<I", len(program)) + program]
    if compute_width is not None:
        psv = bytearray(52)
        struct.pack_into("<I", psv, 0, 48)
        psv[28] = 13 if mesh_limits else 5
        if mesh_limits:
            psv[31] = 2
            struct.pack_into("<HH", psv, 16, *mesh_limits)
        struct.pack_into("<3I", psv, 40, compute_width, 1, 1)
        sections.append(b"PSV0" + struct.pack("<I", len(psv)) + psv)
    if private_data is not None:
        sections.append(b"PRIV" + struct.pack("<I", len(private_data)) + private_data)
    offset = 32 + 4 * len(sections)
    offsets = bytearray()
    for section in sections:
        offsets.extend(struct.pack("<I", offset))
        offset += len(section)
    blob = b"DXBC" + bytes(16) + struct.pack("<3I", 1, offset, len(sections)) + offsets + b"".join(sections)
    return shader_record(blob, name)


def must_reject(function, *args):
    try:
        function(*args)
    except (ValueError, KeyError):
        return
    raise AssertionError(f"{function.__name__} accepted invalid input")


def check_container_selector(directory: Path, code: bytes, library: str | None = None) -> None:
    archive = directory / "same-name.rpf"
    records = [envelope(code, private_data=metadata) for metadata in (b"first", b"second")]
    containers = []
    for record in records:
        archive.write_bytes(record)
        container, native = native_program(archive, "fixture")
        assert native == code
        containers.append(container)
    digests = [hashlib.sha256(blob).hexdigest() for blob in containers]
    first, second = records
    broken = b"\x08fixture\0\0" + struct.pack("<4I", 2, 0, 2, 8) + b"\x07\xff"
    archive.write_bytes(first + second)
    must_reject(native_program, archive, "fixture")
    for digest, blob in zip(digests, containers, strict=True):
        assert native_program(archive, "fixture", container_sha256=digest.upper()) == (blob, code)

    for order in (broken + first, first + broken):
        archive.write_bytes(order)
        assert native_program(archive, "fixture", container_sha256=digests[0]) == (containers[0], code)
    # A bad frame can claim bytes covering a subsequent record of the same name.
    archive.write_bytes(b"\x08fixture\0\0" + struct.pack("<4I", len(first) + 2, 0, len(first) + 2, 8)
                        + b"\x07\xff" + first)
    assert native_program(archive, "fixture", container_sha256=digests[0]) == (containers[0], code)
    archive.write_bytes(first * 257 + second)
    assert native_program(archive, "fixture", container_sha256=digests[1]) == (containers[1], code)

    archive.write_bytes(envelope(code))
    intact, _ = native_program(archive, "fixture")
    corrupt = intact[:-1] + bytes([intact[-1] ^ 1])
    archive.write_bytes(first + shader_record(corrupt, b"fixture"))
    try:
        native_program(archive, "fixture", container_sha256=hashlib.sha256(corrupt).hexdigest())
    except zlib.error as error:
        assert "incorrect data check" in str(error), error
    else:
        raise AssertionError("Container selection hid a corrupt native stream")

    for contents, digest, reason in (
        (first + second, hashlib.sha256(code).hexdigest(), "No record matches the name and container SHA-256"),
        (first + second, "bad", "Invalid decoded container SHA-256"),
        (first + broken + first + second, digests[0], "Multiple records match the name and container SHA-256"),
    ):
        archive.write_bytes(contents)
        try:
            native_program(archive, "fixture", container_sha256=digest)
        except ValueError as error:
            assert str(error) == reason, error
        else:
            raise AssertionError("Container selection accepted an invalid digest or ambiguous record")

    if library is None:
        return
    archive.write_bytes(first + second)
    output = directory / "selected.hlsl"
    command = [sys.executable, "-B", Path(pixel_aot.__file__), archive, "fixture", "--llvm", library,
               "--texture-sgpr", "20", "--interp-sgpr", "24", "--sampler-words", "1", "2", "3", "4",
               "--output", output]
    ambiguous = subprocess.run(command, capture_output=True, text=True)
    assert ambiguous.returncode == 1 and "Multiple records match the name" in ambiguous.stderr
    for digest in digests:
        result = subprocess.run([*command, "--container-sha256", digest], capture_output=True, text=True, check=True)
        summary = json.loads(result.stdout)
        assert summary["container_sha256"] == digest
        assert summary["native_sha256"] == hashlib.sha256(code).hexdigest()
        assert output.read_text(encoding="utf-8") == lower(assembly(False), 20, 24, [1, 2, 3, 4])
    wrong = subprocess.run([*command, "--container-sha256", "0" * 64], capture_output=True, text=True)
    assert wrong.returncode == 1 and "No record matches the name and container SHA-256" in wrong.stderr


def translate(instructions):
    return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor=64)


def with_constants(body):
    """Load constant words 3 to 6 into s44 to s47, then run body on the sample in v12 to v15."""
    program = assembly(False)
    pos = program.index("s_mov_b32 s40, 1")
    program.insert(pos, "s_load_dwordx4 s[44:47], s[20:21], 0x40")
    pos = program.index("s_waitcnt vmcnt(0)")
    program[pos:pos] = [
        "s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc",
        "s_waitcnt lgkmcnt(0)",
    ]
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = body
    return program


# s_movk_i32 sign-extends its 16-bit immediate. s_bfe_u32 takes bytes 1 and 3 of each result, and each byte is
# widened as a binary16 subnormal and scaled by 2^24 to the exact integer: 0x8000 gives 0xffff8000 (bytes 0x80 and
# 0xff) and 0x7fff gives 0x00007fff (bytes 0x7f and 0x00). Zero extension would export 0 for byte 3 of 0x8000.
MOVK_BODY = [
    "s_movk_i32 s48, 0x8000",
    "s_movk_i32 s51, 0x7fff",
    "s_bfe_u32 s49, s48, 0x80018",
    "s_bfe_u32 s50, s48, 0x80008",
    "s_bfe_u32 s52, s51, 0x80008",
    "s_bfe_u32 s53, s51, 0x80018",
    *[line for channel, source in enumerate(("s49", "s50", "s52", "s53"))
      for line in (f"v_cvt_f32_f16_e32 v{12 + channel}, {source}", f"v_mul_f32_e32 v{12 + channel}, 0x4b800000, v{12 + channel}")],
]
MOVK_EXPECTED = [255.0, 128.0, 127.0, 0.0]


def check_movk_draw(directory, compiler, renderer, backend):
    source = directory / "movk.hlsl"
    shader = directory / "movk.cso"
    case = directory / "movk.case"
    words = [(imm - 0x10000 if imm & 0x8000 else imm) & 0xffffffff for imm in (0x8000, 0x7fff)]
    assert [float((words[0] >> 24) & 0xff), float((words[0] >> 8) & 0xff),
            float((words[1] >> 8) & 0xff), float((words[1] >> 24) & 0xff)] == MOVK_EXPECTED
    source.write_text(translate(with_constants(MOVK_BODY)), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    constants = bytes(256)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    case.write_bytes(constants + struct.pack("<4f", *MOVK_EXPECTED) * 64)
    subprocess.run(draw, check=True)
    # Zero extension clears byte 3 of 0x8000; sign-extending 0x7fff as negative sets byte 3.
    for channel, value in ((0, 0.0), (3, 255.0)):
        wrong = list(MOVK_EXPECTED)
        wrong[channel] = value
        case.write_bytes(constants + struct.pack("<4f", *wrong) * 64)
        result = subprocess.run(draw, capture_output=True, text=True)
        if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"A wrong s_movk_i32 extension escaped detection in channel {channel}")


def with_scalar_constant_reads() -> list[str]:
    program = with_constants(["v_mov_b32_e32 v12, s61", "v_mov_b32_e32 v13, s62", "v_mov_b32_e32 v14, s63"])
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc")
    program[load:load + 1] = [
        "s_buffer_load_dword s61, s[44:47], null",
        "s_buffer_load_dword s62, s[44:47], 0x8",
        "s_buffer_load_dword s63, s[44:47], 0xfc",
    ]
    return program


def affine_coordinates() -> list[str]:
    program = assembly(False)
    high_address = "s_mov_b32 s21, 0"
    program.remove(high_address)
    restore = program.index("s_and_b64 exec, exec, s[50:51]")
    program[restore:restore] = [
        high_address,
        "s_load_dwordx4 s[44:47], s[20:21], 0x40",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dwordx4 s[60:63], s[44:47], null",
        "s_waitcnt lgkmcnt(0)",
        "v_add_f32_e32 v8, 0.5, v8",
        "v_mul_legacy_f32_e32 v8, s62, v8",
        "v_fma_f32 v8, 0.5, v8, 0",
    ]
    return program


PAIR_LOADS = ("s_buffer_load_dwordx2 s[60:61], s[44:47], 0xc",
              "s_buffer_load_dwordx2 s[62:63], s[44:47], 0x14",
              "s_buffer_load_dwordx2 s[64:65], s[44:47], 0xf8")


def with_scalar_pair_reads() -> list[str]:
    """Read constant words 3 and 4, 5 and 6, 62 and 63 as pairs; the first pair spans two float4 rows."""
    program = with_constants([
        "v_mul_f32_e32 v12, s60, v12",
        "v_max_f32_e32 v12, s64, v12",
        "v_add_f32_e32 v13, s61, v13",
        "v_sub_f32_e32 v14, s62, v14",
        "v_mov_b32_e32 v15, s63",
        "v_mul_f32_e32 v15, s65, v15",
    ])
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc")
    program[load:load + 1] = PAIR_LOADS
    return program


def scalar_pair_case(directory: Path, scale: float, bias: float, minuend: float, alpha: float,
                     floor: float, gain: float) -> Path:
    """Expected colors for with_scalar_pair_reads(). Every step is exact in binary32."""
    def exact(value):
        assert single(value) == value, value
        return value
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, *map(exact, (scale, bias, minuend, alpha)))
    struct.pack_into("<2f", case, 248, exact(floor), exact(gain))
    for y in RAMP:
        for x in RAMP:
            r, g, b = (single(c / 255) for c in (x, y, 128))
            color = (max(floor, exact(scale * r)), exact(bias + g), exact(minuend - b), exact(gain * alpha))
            case.extend(struct.pack("<4f", *map(half_rtz, color)))
    path = directory / "scalar-pairs.case"
    path.write_bytes(case)
    return path


def curve():
    return with_constants([
        "v_max_f32_e32 v18, s46, v12",
        "v_log_f32_e32 v18, v18",
        "v_rcp_f32_e32 v19, s44",
        "v_mul_legacy_f32_e32 v18, v19, v18",
        "v_exp_f32_e32 v18, v18",
        "v_sub_f32_e32 v19, 1.0, v13",
        "v_fma_f32 v12, s45, v18, -v19",
        "v_fmac_f32_e32 v12, s47, v13",
        "v_min_f32_e32 v12, 1.0, v12",
        "v_mul_legacy_f32_e64 v12, v12, 1.0 clamp",
    ])


def shaping():
    return with_constants([
        "v_max_f32_e32 v18, s46, v12",
        "v_sqrt_f32_e32 v18, v18",
        "v_mul_f32_e32 v18, s45, v18",
        "v_add_f32_e32 v19, s44, v13",
        "v_rsq_f32_e32 v19, v19",
        "v_add_f32_e32 v18, v19, v18",
        "v_mul_f32_e32 v20, s44, v13",
        "v_floor_f32_e32 v21, v20",
        "v_fract_f32_e32 v20, v20",
        "v_fma_f32 v12, s47, v20, v18",
        "v_sub_f32_e32 v12, v12, v21",
        "v_mul_f32_e64 v12, v12, 1.0 clamp",
    ])


def truncate_values() -> list[str]:
    return with_constants([
        "v_add_f32_e32 v12, s44, v12",
        "v_trunc_f32_e32 v12, v12",
        "v_trunc_f32_e32 v13, s45",
        "v_trunc_f32_e64 v14, -|s46|",
        "v_trunc_f32_e64 v15, s47 clamp",
    ])


def sopk_scalar() -> list[str]:
    return with_constants([
        "s_mov_b32 s50, s44",
        "s_mulk_i32 s50, 0x3",
        "s_mov_b32 s51, s45",
        "s_addk_i32 s51, 0xfffe",
        "v_mov_b32_e32 v12, s50",
        "v_mov_b32_e32 v13, s51",
    ])


def rounding():
    """Round scaled samples up and to nearest even, and a constant tie through the VOP3 encoding."""
    return with_constants([
        "v_mul_f32_e32 v18, s44, v12",
        "v_ceil_f32_e32 v12, v18",
        "v_mul_f32_e32 v19, s44, v13",
        "v_rndne_f32_e32 v13, v19",
        "v_rndne_f32_e64 v14, s45",
        "v_ceil_f32_e64 v15, -s46",
    ])


def select(wave64=False):
    """Compare into lane masks and select; LLVM names the masks by wavefront width."""
    vcc, first, second = ("vcc", "s[52:53]", "s[54:55]") if wave64 else ("vcc_lo", "s52", "s53")
    return with_constants([
        f"v_cmp_nlt_f32_e64 {second}, v12, v13",
        f"v_cndmask_b32_e64 v14, 0, 1.0, {second}",
        f"v_cmp_eq_f32_e32 {vcc}, v12, v13",
        f"v_cndmask_b32_e32 v15, s47, v15, {vcc}",
        f"v_cmp_lt_f32_e32 {vcc}, s44, v12",
        "v_sub_f32_e32 v18, 1.0, v12",
        f"v_cmp_ge_f32_e64 {first}, v13, s45",
        f"v_cndmask_b32_e64 v13, v13, s46, {first}",
        f"v_cndmask_b32_e32 v12, v12, v18, {vcc}",
    ])


def three_way():
    """Median, minimum and maximum of three operands, each result one of its inputs."""
    return with_constants([
        "v_med3_f32 v15, s47, v13, v12",
        "v_max3_f32 v14, v12, v14, v13",
        "v_med3_f32 v12, v12, s44, s45",
        "v_min3_f32 v13, |s46|, v13, v12",
    ])


def absolute_value():
    program = assembly(False)
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = [
        "v_sub_f32_e32 v12, 0.5, v12",
        "v_sub_f32_e64 v12, |v12|, 0",
    ]
    return program


def unpack_bytes():
    # Each output takes a different byte of a constant word, in e32 and e64 forms.
    return with_constants([
        "v_cvt_f32_ubyte0_e32 v12, s44",
        "v_cvt_f32_ubyte1_e64 v13, s45",
        "v_cvt_f32_ubyte2_e32 v14, s46",
        "v_cvt_f32_ubyte3_e64 v15, s47 clamp",
    ])


def offset_nibbles():
    # The low four bits of each constant word as a signed offset in sixteenths.
    return with_constants([
        "v_cvt_off_f32_i4_e32 v12, s44",
        "v_cvt_off_f32_i4_e64 v13, s45",
        "v_cvt_off_f32_i4_e32 v14, s46",
        "v_cvt_off_f32_i4_e64 v15, s47 clamp",
    ])


def ceiling():
    return with_constants([
        "v_ceil_f32_e32 v12, s44",
        "v_ceil_f32_e64 v13, -|s45|",
        "v_ceil_f32_e64 v14, |s46|",
        "v_ceil_f32_e64 v15, |s47| clamp",
    ])


def int_minmax_pair(encoding):
    return with_constants([
        "v_mov_b32_e32 v18, s45",
        f"v_min_i32_{encoding} v12, s44, v18",
        f"v_max_i32_{encoding} v13, s44, v18",
        f"v_min_u32_{encoding} v14, s44, v18",
        f"v_max_u32_{encoding} v15, s44, v18",
    ])


def int_minmax_triple(kind):
    return with_constants([
        "v_mov_b32_e32 v18, s45",
        "v_mov_b32_e32 v19, s46",
        f"v_min3_{kind} v12, s44, v18, v19",
        f"v_max3_{kind} v13, s44, v18, v19",
        f"v_med3_{kind} v14, s44, v18, v19",
        f"v_med3_{kind} v15, s46, v18, v18",
    ])


def frexp_mant_values():
    return with_constants([
        "v_frexp_mant_f32_e32 v12, s44",
        "v_frexp_mant_f32_e64 v13, -|s45|",
        "v_frexp_mant_f32_e64 v14, |s46|",
        "v_frexp_mant_f32_e64 v15, s47 clamp",
    ])


def ldexp_values():
    return with_constants([
        "v_mov_b32_e32 v18, s44",
        "v_ldexp_f32 v12, v18, s45",
        "v_ldexp_f32 v13, -|s44|, 2",
        "v_ldexp_f32 v14, s46, -2",
        "v_ldexp_f32 v15, s47, 3 clamp",
    ])


def reciprocal_iflag():
    return with_constants([
        "v_rcp_iflag_f32_e32 v12, s44",
        "v_rcp_iflag_f32_e64 v13, -|s45|",
        "v_rcp_iflag_f32_e64 v14, |s46|",
        "v_rcp_iflag_f32_e64 v15, |s47| clamp",
    ])


# The gfx10 predicate number is a four-bit truth table. Bits 0..3 select
# less, equal, greater and unordered respectively, independent of the HLSL.
PREDICATES = ("f", "lt", "eq", "le", "gt", "lg", "ge", "o",
              "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru")


def predicate_group(first, sdwa=False):
    # Copy runtime constants into vector registers using supported operations.
    body = ["v_sub_f32_e64 v18, s45, 0", "v_sub_f32_e64 v19, s47, 0"]
    for channel, op in enumerate(PREDICATES[first:first + 4]):
        compare = (f"v_cmp_{op}_f32_sdwa vcc_lo, s44, v18 src0_sel:DWORD src1_sel:DWORD" if sdwa
                   else f"v_cmp_{op}_f32_e32 vcc_lo, s44, v18")
        body.extend([compare,
                     f"v_cndmask_b32_e32 v{12 + channel}, 0, v19, vcc_lo"])
    return with_constants(body)


# Three-bit truth table for signed and unsigned 32-bit integer comparisons (bits 0..2: <, ==, >).
INT_PREDICATES = ("f", "lt", "eq", "le", "gt", "ne", "ge", "t")


def int_predicate_group(kind, encoding, first, *, wave64=False):
    # Copy runtime constants with v_mov_b32 so all 32 bits survive unchanged.
    dst = ("vcc" if wave64 else "vcc_lo") if encoding == "e32" else ("s[52:53]" if wave64 else "s52")
    body = ["v_mov_b32_e32 v18, s45", "v_mov_b32_e32 v19, s47"]
    for channel, op in enumerate(INT_PREDICATES[first:first + 4]):
        body.extend([f"v_cmp_{op}_{kind}_{encoding} {dst}, s44, v18",
                     f"v_cndmask_b32_{encoding} v{12 + channel}, 0, v19, {dst}"])
    return with_constants(body)


# v_cmp_class_f32 is true when the class of S0 is in the mask S1 (LLVM SIInstrFlags::ClassFlags).
# Each channel tests one property: +inf against its own bit, a quiet NaN against both NaN bits,
# -0 against the +0 bit and a positive denormal against its bit, so neither the sign of zero
# nor a denormal flushed to zero can pass.
CLASS_BITS = (0x7f800000, 0x7fc00000, 0x80000000, 0x00000001)
CLASS_BODY = [
    "v_mov_b32_e32 v19, 1.0",
    "v_cmp_class_f32_e64 vcc_lo, s44, 0x200",
    "v_cndmask_b32_e32 v12, 0, v19, vcc_lo",
    "v_cmp_class_f32_e64 vcc_lo, s45, 3",
    "v_cndmask_b32_e32 v13, 0, v19, vcc_lo",
    "v_mov_b32_e32 v18, 64",
    "v_cmp_class_f32_e32 vcc_lo, s46, v18",
    "v_cndmask_b32_e32 v14, 0, v19, vcc_lo",
    "v_cmp_class_f32_e64 s0, s47, 0x80",
    "v_cndmask_b32_e64 v15, 0, v19, s0",
]


def check_class_draw(directory, compiler, renderer, backend):
    source = directory / "class.hlsl"
    shader = directory / "class.cso"
    case = directory / "class.case"
    source.write_text(translate(with_constants(CLASS_BODY)), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *CLASS_BITS)
    expected = [1.0, 1.0, 0.0, 1.0]
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
    subprocess.run(draw, check=True)
    # A wrong class in any channel flips its result; the renderer must report each flip.
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = 1.0 - wrong_values[k]
        case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"A wrong class result escaped detection in channel {k}")


# v_fract_f32 clamps x - floor(x) to 0x3f7fffff. For -2**-30 the unclamped difference rounds to 1.0;
# the other inputs are exact and stay below the clamp.
FRACT_INPUTS = (-2.0 ** -30, -0.25, 2.75, -2.0 ** -23)
FRACT_CLAMP = struct.unpack("<f", struct.pack("<I", 0x3f7fffff))[0]


def fract_values():
    return with_constants([
        "v_fract_f32_e32 v12, s44",
        "v_fract_f32_e64 v13, |s45|",
        "v_fract_f32_e64 v14, -|s46|",
        "v_fract_f32_e32 v15, s47",
    ])


RECIPROCAL_BODY = [
    "v_rcp_f32_e64 v12, |s44|",
    "v_rcp_f32_e64 v13, -|s45|",
    "v_rsq_f32_e64 v14, |s46|",
    "v_rsq_f32_e64 v15, |s47| clamp",
]


def check_reciprocal_draw(directory, compiler, renderer, backend):
    source = directory / "reciprocal.hlsl"
    shader = directory / "reciprocal.cso"
    case = directory / "reciprocal.case"
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, -4.0, 8.0, -16.0, -0.25)
    case.write_bytes(constants + struct.pack("<4f", 0.25, -0.125, 0.25, 1.0) * 64)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for mutation in (False, True):
        body = RECIPROCAL_BODY.copy()
        if mutation:
            # 1 / abs(-16) is 1/16, while its reciprocal square root is 1/4.
            body[2] = body[2].replace("v_rsq", "v_rcp")
        source.write_text(translate(with_constants(body)), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        if mutation:
            wrong = subprocess.run(draw, capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError("Substituting reciprocal for reciprocal square root escaped detection")
        else:
            subprocess.run(draw, check=True)
# VOP3 (e64) encodings that the translator already accepts but no other fixture draws: floor, fract, max, min and
# fmac with source modifiers, and the 16 comparisons writing an SGPR mask from negated operands. Negating both
# operands swaps the less and greater relations and keeps equality and unordered.
E64_ARITH = (
    (["v_floor_f32_e64 v12, -|s44|", "v_fract_f32_e64 v13, -s45",
      "v_max_f32_e64 v14, -s46, |s47|", "v_min_f32_e64 v15, s46, -|s47|"],
     [-3.0, 0.75, 4.0, -4.0], ((0, 2.0), (1, 0.25), (2, -3.0), (3, 3.0))),
    (["v_mov_b32_e32 v12, s46", "v_fmac_f32_e64 v12, -s44, |s45|",
      "v_mov_b32_e32 v13, 0", "v_mov_b32_e32 v14, 0", "v_mov_b32_e32 v15, 0"],
     [-0.125, 0.0, 0.0, 0.0], ((0, 6.125),)),
)
E64_ARITH_INPUTS = (2.5, 1.25, 3.0, -4.0)


def predicate_group_e64(first):
    body = ["v_sub_f32_e64 v18, s45, 0", "v_sub_f32_e64 v19, s47, 0"]
    for channel, op in enumerate(PREDICATES[first:first + 4]):
        body.extend([f"v_cmp_{op}_f32_e64 s0, -s44, -v18", f"v_cndmask_b32_e64 v{12 + channel}, 0, v19, s0"])
    return with_constants(body)


def check_e64_draws(directory, compiler, renderer, backend):
    source = directory / "e64.hlsl"
    shader = directory / "e64.cso"
    def add_case(cases, constants, expected, status):
        path = directory / f"e64-{len(cases)}.case"
        path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((path, status))

    def run_cases(cases):
        command = [renderer, backend, "--check-pixel-batch", shader, *(path for path, _ in cases)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 77:
            result.check_returncode()
        expected = [f"case={i} status={status}" for i, (_, status) in enumerate(cases)]
        actual = [line for line in result.stdout.splitlines() if line.startswith("case=")]
        if actual != expected or result.returncode != int(any(status != "pass" for _, status in cases)):
            raise AssertionError(f"Expected {expected}\n{result.stdout}\n{result.stderr}")

    x, y, z, w = E64_ARITH_INPUTS
    assert [math.floor(-abs(x)), -y - math.floor(-y), max(-z, abs(w)), min(z, -abs(w))] == E64_ARITH[0][1]
    assert [-x * abs(y) + z, 0.0, 0.0, 0.0] == E64_ARITH[1][1]
    for body, expected, wrongs in E64_ARITH:
        source.write_text(translate(with_constants(body)), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, *E64_ARITH_INPUTS)
        cases = []
        add_case(cases, constants, expected, "pass")
        for channel, value in wrongs:
            wrong_values = list(expected)
            wrong_values[channel] = value
            add_case(cases, constants, wrong_values, "mismatch")
        add_case(cases, constants, expected, "pass")
        run_cases(cases)
    inputs = ((-1, 1, 0), (-0.0, 0.0, 1), (1, -1, 2), (math.nan, 1, 3), (1, math.nan, 3))
    negated = {0: 2, 1: 1, 2: 0, 3: 3}
    for first in range(0, 16, 4):
        source.write_text(translate(predicate_group_e64(first)), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        cases = []
        for left, right, relation in inputs:
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, left, right, 0, 1)
            expected = [float((op >> negated[relation]) & 1) for op in range(first, first + 4)]
            add_case(cases, constants, expected, "pass")
            # A lowering that dropped the negations would follow the original relation.
            unnegated = [float((op >> relation) & 1) for op in range(first, first + 4)]
            if unnegated != expected:
                add_case(cases, constants, unnegated, "mismatch")
        run_cases(cases)


def bitwise_program():
    # Emit exact finite binary32 colors by placing result bits directly in their mantissas.
    return with_constants([
        "v_mov_b32_e32 v18, s46",
        "v_mov_b32_e32 v19, s47",
        "v_lshlrev_b32_e32 v20, s44, v18",
        "v_lshrrev_b32_e32 v20, s45, v20",
        "v_ashrrev_i32_e32 v21, 20, v19",
        "v_not_b32_e32 v22, v18",
        "v_bfi_b32 v23, 0x00700000, v18, v19",
        "v_xor_b32_e32 v22, v19, v22",
        *[instruction for index in range(4) for instruction in (
            f"v_and_b32_e32 v{12+index}, 0x007f0000, v{20+index}",
            f"v_or_b32_e32 v{12+index}, 0x3f800000, v{12+index}",
        )],
    ])


def check_bitwise_draw(directory, compiler, renderer, backend):
    shader = directory / "bits.cso"
    source = directory / "bits.hlsl"
    case = directory / "bits.case"
    for logical in (False, True):
        program = bitwise_program()
        if logical:
            program = [line.replace("v_ashrrev_i32", "v_lshrrev_b32") for line in program]
        source.write_text(translate(program), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        for left, right, a, b in ((36, 33, 0x40d10000, 0xc0250000), (35, 37, 0x3e270000, 0xbf620000)):
            words = (((a << (left % 32)) & 0xffffffff) >> (right % 32),
                     (b - 2**32) >> 20, ~a ^ b, (a & 0x00700000) | (b & ~0x00700000))
            colors = struct.pack("<4I", *[0x3f800000 | (value & 0x007f0000) for value in words])
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, left, right, a, b)
            case.write_bytes(constants + colors * 64)
            draw = [renderer, backend, "--check-pixel-case", shader, case]
            if logical:
                wrong = subprocess.run(draw, capture_output=True, text=True)
                if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                    raise AssertionError("Logical shift substituted for arithmetic shift escaped detection")
            else:
                subprocess.run(draw, check=True)


def check_fract_draw(directory, compiler, renderer, backend):
    source = directory / "fract.hlsl"
    shader = directory / "fract.cso"
    case = directory / "fract.case"
    source.write_text(translate(fract_values()), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *FRACT_INPUTS)
    a, b, c, d = FRACT_INPUTS
    expected = [min(single(x - math.floor(x)), FRACT_CLAMP) for x in (a, abs(b), -abs(c), d)]
    assert expected[0] == FRACT_CLAMP and single(FRACT_INPUTS[0] - math.floor(FRACT_INPUTS[0])) == 1.0
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    case.write_bytes(constants + struct.pack("<4f", *map(half_rtz, expected)) * 64)
    subprocess.run(draw, check=True)
    # An unclamped fract exports 1.0 for the first input; the renderer must tell it from the clamp.
    case.write_bytes(constants + struct.pack("<4f", 1.0, *map(half_rtz, expected[1:])) * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("An unclamped fract result escaped detection")


# v_mad_f32 and v_mac_f32 round the product before the add and flush binary32 denormals.
# 4097 * 4097 = 2**24 + 2**13 + 1 rounds to 2**24 + 2**13, so the unfused sum is 0 where a fused one is 1.0.
# 2**-127 is a denormal input: flushed, the second sum is 0.5; preserved, it is 1.0.
MAD_INPUTS = (4097.0, -16785408.0, 2.0 ** -127, 2.0 ** 126)
MAD_BODY = [
    "v_mad_f32 v12, s44, s44, s45",
    "v_mov_b32_e32 v13, s47",
    "v_mad_f32 v13, s46, v13, 0.5",
    "v_mov_b32_e32 v15, s44",
    "v_mov_b32_e32 v14, s45",
    "v_mac_f32_e32 v14, s44, v15",
    "v_mad_f32 v15, 0.5, 4.0, 0.5",
]


def check_mad_draw(directory, compiler, renderer, backend):
    source = directory / "mad.hlsl"
    shader = directory / "mad.cso"
    case = directory / "mad.case"
    source.write_text(translate(with_constants(MAD_BODY)), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *MAD_INPUTS)
    a, c, tiny, large = MAD_INPUTS
    unfused = single(single(a * a) + c)
    assert unfused == 0.0 and single(a * a + c) == 1.0
    assert single(tiny) == tiny and 0 < tiny < 2.0 ** -126 and single(tiny * large) == 0.5
    expected = [unfused, 0.5, unfused, 2.5]
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    case.write_bytes(constants + struct.pack("<4f", *map(half_rtz, expected)) * 64)
    subprocess.run(draw, check=True)
    # A fused multiply-add or a denormal-preserving one exports 1.0; the renderer must tell each case apart.
    for wrong_values in ([1.0, 0.5, unfused, 2.5], [unfused, 1.0, unfused, 2.5], [unfused, 0.5, 1.0, 2.5]):
        case.write_bytes(constants + struct.pack("<4f", *map(half_rtz, wrong_values)) * 64)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"A fused or denormal-preserving result escaped detection: {wrong_values}")


def check_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "predicates.hlsl"
    shader = directory / "predicates.cso"
    case = directory / "predicates.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    # Isolate each negated relation so another predicate cannot supply the unordered test. An
    # unordered relational fcmp alone is not enough: the Windows 10 in-box WARP 10.0.19041
    # evaluates fcmp ult, ule, ugt and uge as their ordered forms.
    for op in ("nge", "nlg", "ngt", "nle", "nlt"):
        isolated = with_constants(["v_sub_f32_e64 v18, s45, 0",
                                   f"v_cmp_{op}_f32_e32 vcc_lo, s44, v18",
                                   "v_cndmask_b32_e32 v12, 0, 1.0, vcc_lo"])
        source.write_text(translate(isolated), encoding="ascii")
        listing = directory / f"{op}.ll"
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main",
                        "-Fc", listing, "-Fo", shader, source], check=True)
        dxil = listing.read_text(encoding="utf-8")
        if not any(text in dxil for text in ("@dx.op.isSpecialFloat.f32(i32 8,", "fcmp uno ", "fcmp ueq ")):
            raise AssertionError(f"{op.upper()} lacks an explicit unordered comparison in DXIL")
    inputs = ((-1, 1, 0), (-0.0, 0.0, 1), (1, -1, 2),
              (math.nan, 1, 3), (1, math.nan, 3))
    for first in range(0, 16, 4):
        source.write_text(translate(predicate_group(first)), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        for left, right, relation in inputs:
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, left, right, 0, 1)
            pixel = struct.pack("<4f", *((op >> relation) & 1 for op in range(first, first + 4)))
            case.write_bytes(constants + pixel * 64)
            try:
                subprocess.run(draw, check=True)
            except subprocess.CalledProcessError as error:
                channels = ", ".join(
                    f"{channel}={op}" for channel, op in enumerate(PREDICATES[first:first + 4])
                )
                error.add_note(
                    f"Predicate draw on {backend}, first={first}, left={left!r}, right={right!r}, channels {channels}"
                )
                raise
    # Keep the unordered oracle and change only the NaN operand to equal.
    struct.pack_into("<f", constants, 16, 1.0)
    case.write_bytes(constants + pixel * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing NaN with a finite equal operand escaped detection")


def check_int_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "int_predicates.hlsl"
    shader = directory / "int_predicates.cso"
    case = directory / "int_predicates.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    # Normal and signed-zero IEEE-754 bit patterns avoid denormal/NaN canonicalization while covering
    # signed/unsigned order flips (-0.0 vs +0.0, +1.0 vs -1.0) and integer-vs-float order (-2.0 vs -1.0).
    pairs = (
        (0x80000000, 0x00000000),
        (0xbf800000, 0xbf800000),
        (0x3f800000, 0xbf800000),
        (0xc0000000, 0xbf800000),
    )

    def relation(left_u, right_u, kind):
        if kind == "i32":
            left_v = left_u - 0x100000000 if left_u & 0x80000000 else left_u
            right_v = right_u - 0x100000000 if right_u & 0x80000000 else right_u
        else:
            left_v, right_v = left_u, right_u
        return 0 if left_v < right_v else (1 if left_v == right_v else 2)

    for kind in ("i32", "u32"):
        for encoding in ("e32", "e64"):
            for first in (0, 4):
                source.write_text(translate(int_predicate_group(kind, encoding, first)), encoding="ascii")
                subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
                for left_u, right_u in pairs:
                    rel = relation(left_u, right_u, kind)
                    constants = bytearray(256)
                    struct.pack_into("<IIff", constants, 12, left_u, right_u, 0.0, 1.0)
                    pixel = struct.pack("<4f", *((op >> rel) & 1 for op in range(first, first + 4)))
                    case.write_bytes(constants + pixel * 64)
                    subprocess.run(draw, check=True)

    # Negative control: signed shader evaluated against the unsigned oracle on (0x80000000, 0x00000000).
    source.write_text(translate(int_predicate_group("i32", "e32", 0)), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    constants = bytearray(256)
    struct.pack_into("<IIff", constants, 12, 0x80000000, 0x00000000, 0.0, 1.0)
    unsigned_rel = relation(0x80000000, 0x00000000, "u32")
    wrong_pixel = struct.pack("<4f", *((op >> unsigned_rel) & 1 for op in range(4)))
    case.write_bytes(constants + wrong_pixel * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Signed/unsigned integer comparison mismatch escaped detection")
# The SDWA comparisons with whole-dword selections repeat the 16-predicate draws, including NaN on either side.
# For each group a reference that inverts the unordered (NaN) result must report a mismatch.
def check_sdwa_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "sdwa_predicates.hlsl"
    shader = directory / "sdwa_predicates.cso"
    case = directory / "sdwa_predicates.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    inputs = ((-1, 1, 0), (-0.0, 0.0, 1), (1, -1, 2), (math.nan, 1, 3), (1, math.nan, 3))
    for first in range(0, 16, 4):
        source.write_text(translate(predicate_group(first, sdwa=True)), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        for left, right, relation in inputs:
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, left, right, 0, 1)
            expected = [float((op >> relation) & 1) for op in range(first, first + 4)]
            case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
            subprocess.run(draw, check=True)
        wrong = [1.0 - value for value in expected]
        case.write_bytes(constants + struct.pack("<4f", *wrong) * 64)
        result = subprocess.run(draw, capture_output=True, text=True)
        if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"An inverted unordered SDWA comparison escaped detection, first={first}")


def half_rtz(x):
    bits, = struct.unpack("<H", struct.pack("<e", x))
    rounded, = struct.unpack("<e", struct.pack("<H", bits))
    return struct.unpack("<e", struct.pack("<H", bits - (abs(rounded) > abs(x))))[0]


# v_cvt_f32_f16 widens the binary16 in the low half of S0; the high half holds junk that must be ignored.
# The second channel applies -|x| to -5.0, the third is the largest finite half and the fourth is the
# smallest subnormal half, 2**-24, scaled by 2**24 so a flushed subnormal exports 0 instead of 1.0.
HALF_WORDS = (0xdead3c00, 0x0000c500, 0x00007bff, 0xbeef0001)
HALF_BODY = [
    "v_cvt_f32_f16_e32 v12, s44",
    "v_cvt_f32_f16_e64 v13, -|s45|",
    "v_cvt_f32_f16_e32 v14, s46",
    "v_cvt_f32_f16_e32 v15, s47",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]


def half_value(word):
    return struct.unpack("<e", struct.pack("<H", word & 0xffff))[0]


def check_half_draw(directory, compiler, renderer, backend):
    source = directory / "half.hlsl"
    shader = directory / "half.cso"
    case = directory / "half.case"
    source.write_text(translate(with_constants(HALF_BODY)), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *HALF_WORDS)
    first, second, third, fourth = map(half_value, HALF_WORDS)
    expected = [first, -abs(second), third, fourth * 2.0 ** 24]
    assert expected == [1.0, -5.0, 65504.0, 1.0] and fourth == 2.0 ** -24
    # Reading the high half, dropping the negation, a wrong exponent and a flushed subnormal.
    wrong = [half_value(HALF_WORDS[0] >> 16), 5.0, 32752.0, 0.0]
    for x in (*expected, *wrong):
        assert half_rtz(x) == x, x
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
    subprocess.run(draw, check=True)
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = wrong[k]
        case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
        result = subprocess.run(draw, capture_output=True, text=True)
        if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"A wrong half conversion escaped detection in channel {k}")

    # A separate export observes the literal and two scalar results as half encodings.
    body = [
        "v_cvt_f32_f16_e32 v12, 0x3c00",
        "s_ashr_i64 s[72:73], 0xf000, 2",
        "v_cvt_f32_f16_e32 v13, s72",
        "s_bfm_b64 s[72:73], 16, 0",
        "s_and_b64 s[72:73], s[72:73], 0x3c00",
        "v_cvt_f32_f16_e32 v14, s72",
        *HALF_BODY[3:],
    ]
    source.write_text(translate(with_constants(body)), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    scalars = [half_value(0x3c00), half_value(0xf000 // 4), half_value((2**16 - 1) & 0x3c00), fourth * 2.0 ** 24]
    assert scalars == [1.0, 1.0, 1.0, 1.0]
    case.write_bytes(constants + struct.pack("<4f", *scalars) * 64)
    subprocess.run(draw, check=True)
    # A shift by zero leaves 0xf000 (-8192.0) in green.
    wrong_values = [scalars[0], half_value(0xf000), *scalars[2:]]
    case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
    result = subprocess.run(draw, capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("A wrong s_ashr_i64 count escaped detection in channel 1")


def pack_f16_pair(lo, hi):
    lo_bits = lo if isinstance(lo, int) else struct.unpack("<H", struct.pack("<e", lo))[0]
    hi_bits = hi if isinstance(hi, int) else struct.unpack("<H", struct.pack("<e", hi))[0]
    return lo_bits | (hi_bits << 16)


def rne_f16(value):
    if math.isnan(value):
        return math.nan
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


# Packed binary16 VOP3P arithmetic: exercises all five v_pk_*_f16 instructions, op_sel/op_sel_hi,
# neg_lo/neg_hi, clamp, NaN minNum semantics, RNE tie-to-even rounding, and gfx1013 zero-extension
# of inline f16 constants (0.5 -> lo=0.5, hi=0.0).
VOP3P_BODY = [
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[0,1]",
    "v_pk_mul_f16 v23, v22, v20 neg_hi:[1,0]",
    "v_pk_fma_f16 v24, v23, v21, s47 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
    "v_pk_min_f16 v25, v24, 0x40007e00",
    "v_pk_max_f16 v26, v24, 0.5 clamp",
    "v_cvt_f32_f16_e32 v12, v25",
    "v_pk_add_f16 v27, v25, 0 op_sel:[1,0]",
    "v_cvt_f32_f16_e32 v13, v27",
    "v_cvt_f32_f16_e32 v14, v26",
    "v_pk_add_f16 v28, v26, 0 op_sel:[1,0]",
    "v_cvt_f32_f16_e32 v15, v28",
]
VOP3P_SETS = (
    # Set 1: v24=(0.875, 0.25); channel 0 tests minNum(0.875, qNaN)=0.875, channel 3=0.25 proves inline 0.5 hi=0.0.
    (pack_f16_pair(1.5, 2.0), pack_f16_pair(0.5, -1.0), pack_f16_pair(0.5, 1.0), pack_f16_pair(0.5, 0.5)),
    # Set 2: v24=(2.5, -1.0); channel 2 clamps 2.5 -> 1.0 and channel 3 clamps -1.0 -> 0.0.
    (pack_f16_pair(2.0, 3.0), pack_f16_pair(1.0, 0.0), pack_f16_pair(1.0, 2.0), pack_f16_pair(0.5, 1.0)),
    # Set 3: exact halfway ties at 1.0 + 2^-11 (-> even 1.0) and -(1.0 + 2^-10) - 2^-11 (-> even -(1.0 + 2^-9),
    # distinguishing RNE from RTZ f32tof16 in channel 1).
    (pack_f16_pair(0xbc01, 0x3c00), pack_f16_pair(0x9000, 0x9000), pack_f16_pair(4.0, 4.0), pack_f16_pair(0.25, 0.25)),
)


def vop3p_reference(words):
    def split16(w):
        return half_value(w), half_value(w >> 16)

    def min_num(a, b):
        return b if math.isnan(a) else (a if math.isnan(b) else min(a, b))

    def max_num(a, b):
        return b if math.isnan(a) else (a if math.isnan(b) else max(a, b))

    def sat(x):
        return 0.0 if math.isnan(x) or x <= 0.0 else (1.0 if x >= 1.0 else x)

    s44, s45, s46, s47 = (split16(w) for w in words)
    v22 = (rne_f16(s44[1] + (-s45[0])), rne_f16(s44[0] + s45[1]))
    v23 = (rne_f16(v22[0] * s46[0]), rne_f16((-v22[1]) * s46[1]))
    v24 = (rne_f16(v23[0] * s47[1] + s47[0]), rne_f16(v23[1] * s47[0] + s47[1]))
    v25 = (rne_f16(min_num(v24[0], math.nan)), rne_f16(min_num(v24[1], 2.0)))
    v26 = (rne_f16(sat(max_num(v24[0], 0.5))), rne_f16(sat(max_num(v24[1], 0.0))))
    return [v25[0], v25[1], v26[0], v26[1]]


def check_vop3p_draw(directory, compiler, renderer, backend):
    source = directory / "vop3p.hlsl"
    shader = directory / "vop3p.cso"
    case = directory / "vop3p.case"
    program = with_constants(VOP3P_BODY)
    source.write_text(translate(program), encoding="ascii")
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]
    subprocess.run(compile_shader, check=True)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for words in VOP3P_SETS:
        expected = vop3p_reference(words)
        assert all(not math.isnan(x) and half_rtz(x) == x for x in expected), expected
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)

    # Negative mutations against Set 1 and Set 2 must produce pixel_mismatch.
    for words, old, new, label in (
        (VOP3P_SETS[0],
         "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[0,1]",
         "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1]", "neg_lo"),
        (VOP3P_SETS[0],
         "v_pk_mul_f16 v23, v22, v20 neg_hi:[1,0]",
         "v_pk_mul_f16 v23, v22, v20", "neg_hi"),
        (VOP3P_SETS[0],
         "v_pk_min_f16 v25, v24, 0x40007e00",
         "v_pk_max_f16 v25, v24, 0x40007e00", "min vs max"),
        (VOP3P_SETS[0],
         "v_pk_max_f16 v26, v24, 0.5 clamp",
         "v_pk_max_f16 v26, v24, 0.5 op_sel_hi:[1,0] clamp", "inline hi zero-extension"),
        (VOP3P_SETS[1],
         "v_pk_max_f16 v26, v24, 0.5 clamp",
         "v_pk_max_f16 v26, v24, 0.5", "clamp"),
    ):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        case.write_bytes(constants + struct.pack("<4f", *vop3p_reference(words)) * 64)
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        subprocess.run(compile_shader, check=True)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


# v_cvt_f16_f32 rounds to nearest even and, on GFX10, writes only the low half of the destination. Channels 0
# and 1 read the whole word: v12 starts as 1.0, so after the conversion it is 1 + h * 2^-23 when the high half
# is kept, and (v12 - 1) * 2^23 splits into the low and high bytes of the binary16 encoding h. Channel 2
# widens -|S0| back to binary32 and channel 3 widens a subnormal result scaled by 2^24. Each set lists wrong
# references (channel, value) for the property it isolates.
NARROW_BODY = [
    "v_mov_b32_e32 v12, 1.0",
    "v_cvt_f16_f32_e32 v12, s44",
    "v_subrev_f32_e32 v12, 1.0, v12",
    "v_mul_f32_e32 v12, 0x4b000000, v12",
    "v_mul_f32_e32 v13, 0x3b800000, v12",
    "v_floor_f32_e32 v13, v13",
    "v_fmac_f32_e32 v12, 0xc3800000, v13",
    "v_cvt_f16_f32_e64 v14, -|s45|",
    "v_cvt_f32_f16_e32 v14, v14",
    "v_cvt_f16_f32_e32 v15, s46",
    "v_cvt_f32_f16_e32 v15, v15",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]
NARROW_SETS = (
    # Ties to even in the normal range and at the smallest subnormal.
    ((1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11, 1.5 * 2.0 ** -24), ((0, 1.0), (2, -(1 + 2.0 ** -10)), (3, 1.0))),
    # The largest value below the overflow tie, -0, and the tie between zero and 2^-24.
    ((65520 - 2.0 ** -8, -0.0, 2.0 ** -25), ((1, 124.0), (2, 0.0), (3, 1.0))),
    # The overflow tie gives infinity; rounding carries a subnormal into the smallest normal; a negative subnormal.
    ((65520.0, 2.0 ** -14 - 2.0 ** -25, -3 * 2.0 ** -24), ((1, 123.0), (2, -1023 * 2.0 ** -24), (3, 3.0))),
    # A negative subnormal encoding, the negation modifier and a binary32 subnormal input.
    ((-3 * 2.0 ** -24, 3.0, 2.0 ** -149), ((1, 0.0), (2, 3.0))),
)


def narrow_reference(x):
    if abs(x) >= 65520:
        return 0xfc00 if x < 0 else 0x7c00
    return struct.unpack("<H", struct.pack("<e", x))[0]


def check_narrow_draw(directory, compiler, renderer, backend):
    source = directory / "narrow.hlsl"
    shader = directory / "narrow.cso"
    case = directory / "narrow.case"
    source.write_text(translate(with_constants(NARROW_BODY)), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for inputs, wrongs in NARROW_SETS:
        words = [struct.unpack("<I", struct.pack("<f", x))[0] for x in inputs]
        assert [struct.unpack("<f", struct.pack("<I", w))[0] for w in words] == list(inputs)
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *words)
        h = narrow_reference(inputs[0])
        expected = [float(h & 0xff), float(h >> 8), -abs(half_value(narrow_reference(-abs(inputs[1])))),
                    half_value(narrow_reference(inputs[2])) * 2.0 ** 24]
        for x in (*expected, *(value for _, value in wrongs)):
            assert half_rtz(x) == x, x
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
        for channel, value in wrongs:
            assert struct.pack("<f", value) != struct.pack("<f", expected[channel])
            wrong_values = list(expected)
            wrong_values[channel] = value
            case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
            result = subprocess.run(draw, capture_output=True, text=True)
            if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
                raise AssertionError(f"A wrong narrowing escaped detection for {inputs} in channel {channel}")


def arithmetic():
    """Add, multiply, reversed subtraction and moves on the sample and constant words 8 to 11."""
    program = assembly(False)
    pos = program.index("s_mov_b32 s40, 1")
    program.insert(pos, "s_load_dwordx4 s[44:47], s[20:21], 0x40")
    pos = program.index("s_waitcnt vmcnt(0)")
    program[pos:pos] = [
        "s_buffer_load_dwordx4 s[44:47], s[44:47], 0x20",
        "s_waitcnt lgkmcnt(0)",
    ]
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = [
        "v_mov_b32_e32 v18, v12",
        "v_mul_f32_e32 v18, s44, v18",
        "v_add_f32_e64 v12, v18, s45 clamp",
        "v_mov_b32_e64 v19, s46",
        "v_subrev_f32_e32 v19, v13, v19",
        "v_mul_f32_e64 v19, v19, -2.0",
        "v_subrev_f32_e64 v13, s47, v19 clamp",
        "v_add_f32_e32 v14, -0.5, v14",
        "v_mov_b32_e32 v20, 0x3e800000",
        "v_mul_f32_e32 v20, v20, v18",
        "v_mov_b32_e32 v15, v20",
    ]
    return program


def arithmetic_case(directory, *constants):
    """Expected colors for arithmetic(). Each step must be exact in binary32, so neither the
    rounding of add, subtract and multiply nor fusion can change the result."""
    def exact(value):
        rounded, = struct.unpack("<f", struct.pack("<f", value))
        assert rounded == value, value
        return rounded
    scale, bias, pivot, offset = map(exact, constants)
    case = bytearray(256)
    struct.pack_into("<4f", case, 32, scale, bias, pivot, offset)
    ramp = (0, 16, 37, 64, 101, 153, 219, 254)
    for y in ramp:
        for x in ramp:
            r, g, b = (struct.unpack("<f", struct.pack("<f", c / 255))[0] for c in (x, y, 128))
            scaled = exact(scale * r)
            red = min(1, max(0, exact(scaled + bias)))
            green = min(1, max(0, exact(exact(exact(pivot - g) * -2.0) - offset)))
            color = (red, green, exact(-0.5 + b), exact(0.25 * scaled))
            case.extend(struct.pack("<4f", *map(half_rtz, color)))
    path = directory / "arithmetic.case"
    path.write_bytes(case)
    return path


def curve_case(directory, gain):
    gamma, floor, mix = 2.0, .125, .25
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, gamma, gain, floor, mix)
    for y in RAMP:
        for x in RAMP:
            g = y / 255
            r = gain * math.sqrt(max(floor, x / 255)) - (1 - g) + mix * g
            case.extend(struct.pack("<4f", half_rtz(min(1, max(0, r))), half_rtz(g), half_rtz(128/255), 1))
    path = directory / "curve.case"
    path.write_bytes(case)
    return path


def shaping_case(directory, gain):
    scale, floor, mix = 2.0, .125, .25
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, scale, gain, floor, mix)
    for y in RAMP:
        for x in RAMP:
            g = y / 255
            scaled = scale * g
            r = (gain * math.sqrt(max(floor, x / 255))
                 + 1.0 / math.sqrt(scale + g)
                 + mix * (scaled - math.floor(scaled))
                 - math.floor(scaled))
            case.extend(struct.pack("<4f", half_rtz(min(1, max(0, r))), half_rtz(g), half_rtz(128/255), 1))
    path = directory / "shaping.case"
    path.write_bytes(case)
    return path


def truncate_case(directory: Path, bias: float, green: float, blue: float, alpha: float) -> Path:
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, bias, green, blue, alpha)
    # modf returns the integer component as a float and preserves the sign of zero.
    green = math.modf(green)[1]
    blue = math.modf(-abs(blue))[1]
    alpha = max(0.0, min(1.0, math.modf(alpha)[1]))
    for _ in RAMP:
        for x in RAMP:
            red = math.modf(bias + x / 255)[1]
            case.extend(struct.pack("<4f", red, green, blue, alpha))
    path = directory / "truncate.case"
    path.write_bytes(case)
    return path


def sopk_case(directory, red: float, green: float) -> Path:
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, red, green, 0, 1)

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    results = (bits(red) * 3, bits(green) - 2)
    expected = [struct.unpack("<f", struct.pack("<I", value & 0xffffffff))[0]
                for value in results]
    for _ in RAMP:
        for x in RAMP:
            case.extend(struct.pack("<4f", half_rtz(expected[0]), half_rtz(expected[1]), half_rtz(128 / 255), 1))
    path = directory / "sopk.case"
    path.write_bytes(case)
    return path


def rndne(x):
    """Round a binary32 value to the nearest integer, ties to even (Python's round on a float)."""
    return float(round(x))


def rounding_case(directory, scale, tie, bias):
    """Expected colors for rounding(). The scale is a power of two, so each product is exact and
    only the rounding mode decides the result. The constants keep every result away from -0."""
    assert math.frexp(scale)[0] == 0.5, scale
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, scale, tie, bias, 0)
    for y in RAMP:
        for x in RAMP:
            r, g = single(x / 255), single(y / 255)
            color = (math.ceil(scale * r), rndne(scale * g), rndne(single(tie)), math.ceil(-single(bias)))
            case.extend(struct.pack("<4f", *map(half_rtz, map(float, color))))
    path = directory / "rounding.case"
    path.write_bytes(case)
    return path


def absolute_case(directory):
    case = bytearray(256)
    for y in RAMP:
        for x in RAMP:
            red = abs(0.5 - x / 255)
            case.extend(struct.pack("<4f", half_rtz(red), half_rtz(y / 255), half_rtz(128 / 255), 1))
    path = directory / "absolute.case"
    path.write_bytes(case)
    return path


def single(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def select_case(directory, fold, limit, capped, unequal):
    """Expected colors for select(): every step is exact in binary32, so no device tolerance applies."""
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, fold, limit, capped, unequal)
    ramp = (0, 16, 37, 64, 101, 153, 219, 254)
    for y in ramp:
        for x in ramp:
            r, g = single(x / 255), single(y / 255)
            color = (single(1 - r) if fold < r else r, capped if g >= limit else g,
                     0.0 if r < g else 1.0, 1.0 if r == g else unequal)
            case.extend(struct.pack("<4f", *map(half_rtz, color)))
    path = directory / "select.case"
    path.write_bytes(case)
    return path


# v_cube{id,sc,tc,ma}_f32 follow LLVM's ConstantFoldAMDGCNCubeIntrinsic. The direction sets cover all
# six faces, the y/x and z/x ties (the later axis in z, y, x order wins) and -0 on the major axis,
# which selects the positive face. Each set has one wrong reference for the property it isolates.
CUBE_BODY = [
    "v_mov_b32_e32 v18, s45",
    "v_mov_b32_e32 v19, s46",
    "v_cubeid_f32 v12, s44, v18, v19",
    "v_cubesc_f32 v13, s44, v18, v19",
    "v_cubetc_f32 v14, s44, v18, v19",
    "v_cubema_f32 v15, s44, v18, v19",
]
CUBE_SETS = (
    ((2.0, -2.0, 1.0), (0, 1.0)),
    ((-1.5, 0.25, -1.5), (0, 1.0)),
    ((-3.0, 1.0, 2.0), (3, 6.0)),
    ((0.0, 0.0, -0.0), (0, 5.0)),
    ((1.0, 0.5, -0.25), (1, -0.25)),
    ((0.25, 1.0, 0.5), (2, -0.5)),
)


def cube_reference(x, y, z):
    if abs(z) >= abs(x) and abs(z) >= abs(y):
        face, sc, tc, ma = (5.0, -x, -y, z) if z < 0 else (4.0, x, -y, z)
    elif abs(y) >= abs(x):
        face, sc, tc, ma = (3.0, x, -z, y) if y < 0 else (2.0, x, z, y)
    else:
        face, sc, tc, ma = (1.0, z, -y, x) if x < 0 else (0.0, -z, -y, x)
    return [face, sc, tc, ma + ma]


def check_cube_draw(directory, compiler, renderer, backend):
    source = directory / "cube.hlsl"
    shader = directory / "cube.cso"
    case = directory / "cube.case"
    source.write_text(translate(with_constants(CUBE_BODY)), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    faces = set()
    for direction, (channel, wrong_value) in CUBE_SETS:
        constants = bytearray(256)
        struct.pack_into("<3f", constants, 12, *direction)
        expected = cube_reference(*direction)
        faces.add(expected[0])
        assert all(half_rtz(x) == x for x in (*expected, wrong_value)) and wrong_value != expected[channel]
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
        wrong_values = list(expected)
        wrong_values[channel] = wrong_value
        case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
        result = subprocess.run(draw, capture_output=True, text=True)
        if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"A wrong cube result escaped detection for {direction}")
    assert faces == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
    assert math.copysign(1.0, cube_reference(0.0, 0.0, -0.0)[3]) < 0


SCALAR64_SETS = (
    # (a_lo, a_hi, b_lo, b_hi, sh_lo, sh_hi, bfm_w, bfm_off, bfe_src, bfe_ctrl_a, bfe_ctrl_b, alt)
    (0x89abcdef, 0x91234567, 0x55aa55aa, 0x7fffffff, 5, 37, 12, 28, 0x9a345f50, 0x00080004, 0x00100018, 0x55aa55aa),
    (0x0000a5a5, 0x0f0f0000, 0xffff5a5a, 0xf0f0ffff, 0, 100, 40, 16, 0x9a345f50, 0x00000000, 0x00000000, 0x00000000),
)


def scalar64():
    """Exercise 64-bit SOP1/SOP2 scalar ALU and 32-bit BFE, packing 40 output bits into exact binary16 floats."""
    program = assembly(False)
    pos = program.index("s_mov_b32 s40, 1")
    program.insert(pos, "s_load_dwordx4 s[44:47], s[20:21], 0x40")
    pos = program.index("s_waitcnt vmcnt(0)")
    program[pos:pos] = [
        "s_buffer_load_dwordx4 s[52:55], s[44:47], 0x10",
        "s_buffer_load_dwordx4 s[56:59], s[44:47], 0x20",
        "s_buffer_load_dwordx4 s[60:63], s[44:47], 0x30",
        "s_waitcnt lgkmcnt(0)",
    ]

    body = [
        "s_mov_b32 s64, 0",
        "s_mov_b32 s65, 0",
        "s_mov_b32 s66, 0",
        "s_mov_b32 s67, 0",
        "s_mov_b32 s78, 0",
        "s_mov_b32 s79, 0",
        "s_mov_b32 s82, s63",
        "s_mov_b32 s83, 0",
    ]

    def capture_scc(bit_index):
        body.extend([
            "s_mov_b32 s70, 0",
            "s_mov_b32 s71, 0",
            f"s_bfm_b64 s[74:75], 1, {bit_index}",
            "s_cmov_b64 s[70:71], s[74:75]",
            "s_or_b64 s[66:67], s[66:67], s[70:71]",
        ])

    def mix_acc(pair):
        body.extend([
            "s_lshl_b64 s[68:69], s[64:65], 3",
            "s_lshr_b64 s[64:65], s[64:65], 61",
            "s_or_b64 s[64:65], s[64:65], s[68:69]",
            f"s_xor_b64 s[64:65], s[64:65], {pair}",
        ])

    # Bit 0: s_and_b64 + two consecutive s_cmov_b64 (inline -1 sign-extended and SCC preserved).
    body.extend([
        "s_and_b64 s[72:73], s[52:53], s[54:55]",
        "s_cmov_b64 s[78:79], -1",
    ])
    capture_scc(0)
    mix_acc("s[72:73]")
    mix_acc("s[78:79]")

    # Bit 1: s_or_b64 (full 64-bit union saved in s[84:85] + SCC probe on s[82:83]).
    body.append("s_or_b64 s[84:85], s[52:53], s[54:55]")
    mix_acc("s[84:85]")
    body.append("s_or_b64 s[72:73], s[82:83], 0")
    capture_scc(1)

    # Bit 2: s_xor_b64 (register pair and zero-extended 32-bit hex literal).
    body.append("s_xor_b64 s[72:73], s[52:53], s[54:55]")
    mix_acc("s[72:73]")
    body.append("s_xor_b64 s[72:73], s[82:83], 0x55aa55aa")
    capture_scc(2)
    mix_acc("s[72:73]")

    # Bit 3: s_andn2_b64 (register pair + SCC probe).
    body.append("s_andn2_b64 s[72:73], s[52:53], s[54:55]")
    mix_acc("s[72:73]")
    body.append("s_andn2_b64 s[72:73], s[82:83], 0xf0f0f0f")
    capture_scc(3)

    # Bit 4: s_orn2_b64 (register pair + inline -1 proving 64-bit all-ones inversion).
    body.append("s_orn2_b64 s[72:73], s[52:53], s[54:55]")
    mix_acc("s[72:73]")
    body.append("s_orn2_b64 s[72:73], s[82:83], -1")
    capture_scc(4)

    # Bit 5: s_not_b64 + s_brev_b64 (s_brev_b64 placed before capture_scc to verify SCC preservation).
    body.append("s_not_b64 s[72:73], s[52:53]")
    mix_acc("s[72:73]")
    body.extend([
        "s_not_b64 s[72:73], s[84:85]",
        "s_brev_b64 s[86:87], s[52:53]",
    ])
    capture_scc(5)
    mix_acc("s[86:87]")

    # Bit 6: s_lshl_b64 across small/zero and >=32/wrapped shift counts.
    body.append("s_lshl_b64 s[72:73], s[52:53], s56")
    mix_acc("s[72:73]")
    body.append("s_lshl_b64 s[72:73], s[52:53], s57")
    mix_acc("s[72:73]")
    body.append("s_lshl_b64 s[72:73], s[82:83], s57")
    capture_scc(6)

    # Bit 7: s_lshr_b64 across small/zero and >=32/wrapped shift counts.
    body.append("s_lshr_b64 s[72:73], s[52:53], s56")
    mix_acc("s[72:73]")
    body.append("s_lshr_b64 s[72:73], s[52:53], s57")
    mix_acc("s[72:73]")
    body.append("s_lshr_b64 s[72:73], s[82:83], s56")
    capture_scc(7)

    # Bit 8: s_ashr_i64 (signed) + s_bfm_b64 (preserving SCC, plus w=0 and w=32 boundaries).
    body.append("s_ashr_i64 s[72:73], s[52:53], s56")
    mix_acc("s[72:73]")
    body.append("s_ashr_i64 s[72:73], s[52:53], s57")
    mix_acc("s[72:73]")
    body.extend([
        "s_ashr_i64 s[72:73], s[82:83], s56",
        "s_bfm_b64 s[86:87], s58, s59",
    ])
    capture_scc(8)
    mix_acc("s[86:87]")
    body.append("s_bfm_b64 s[88:89], 0, 5")
    mix_acc("s[88:89]")
    body.append("s_bfm_b64 s[88:89], 32, 16")
    mix_acc("s[88:89]")

    # Bit 9: s_bfe_u32 (SCC observed via s_cmov_b64 into acc) and s_bfe_i32 (SCC observed in bit 9).
    body.extend([
        "s_mov_b32 s78, 0",
        "s_mov_b32 s79, 0",
        "s_bfe_u32 s72, s60, s61",
        "s_cmov_b64 s[78:79], 0x13572468",
        "s_bfe_u32 s73, s60, s62",
    ])
    mix_acc("s[72:73]")
    mix_acc("s[78:79]")
    body.extend([
        "s_bfe_i32 s72, s60, s61",
        "s_bfe_i32 s73, s60, s62",
    ])
    capture_scc(9)
    mix_acc("s[72:73]")

    # s_nand_b64, s_nor_b64 and s_xnor_b64: the register pair, then an SCC probe. Channel 3 exports
    # only SCC bits 0-9, so each probe's SCC reaches acc through s_cmov_b64 of a distinct literal.
    # nand(-1, -1) and nor(alt, -1) are zero, and xnor(alt, -1) = alt is zero only in Set 2.
    for op, probe, marker in (("s_nand_b64", "-1, -1", "0x2468ace0"),
                              ("s_nor_b64", "s[82:83], -1", "0x13579bdf"),
                              ("s_xnor_b64", "s[82:83], -1", "0x0fedcba9")):
        body.append(f"{op} s[72:73], s[52:53], s[54:55]")
        mix_acc("s[72:73]")
        body.extend([
            "s_mov_b32 s78, 0",
            "s_mov_b32 s79, 0",
            f"{op} s[72:73], {probe}",
            f"s_cmov_b64 s[78:79], {marker}",
        ])
        mix_acc("s[72:73]")
        mix_acc("s[78:79]")

    # Fold all 64 bits of acc into bits [29:0] of s64: acc ^ (acc >> 30) ^ (acc >> 60), both shifts of the
    # original acc, so every bit reaches the export. Repeating x ^= x >> 30 instead gives x ^ (x >> 60)
    # and drops bits 30-59. Then export 4 x 10-bit fields as 1 + k/1024.
    body.extend([
        "s_lshr_b64 s[68:69], s[64:65], 30",
        "s_lshr_b64 s[70:71], s[64:65], 60",
        "s_xor_b64 s[64:65], s[64:65], s[68:69]",
        "s_xor_b64 s[64:65], s[64:65], s[70:71]",
    ])
    for channel, (src, ctrl) in enumerate((("s64", "0xa0000"), ("s64", "0xa000a"),
                                           ("s64", "0xa0014"), ("s66", "0xa0000"))):
        body.extend([
            f"s_bfe_u32 s80, {src}, {ctrl}",
            "s_mov_b32 s81, 0",
            "s_lshl_b64 s[80:81], s[80:81], 13",
            "s_or_b64 s[80:81], s[80:81], 0x3f800000",
            f"v_mov_b32_e32 v{12 + channel}, s80",
        ])

    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = body
    return program


def scalar64_reference(program, words):
    """Independent Python ISA reference for scalar64() using wide integers."""
    mask32, mask64 = 0xffffffff, 0xffffffffffffffff
    regs = {}
    scc = None
    channels = {}

    def read32(tok):
        if tok.startswith("s"):
            return regs[tok] & mask32
        return int(tok, 0) & mask32

    def read64(tok, signed=False):
        if tok.startswith("s["):
            lo, hi = map(int, tok[2:-1].split(":"))
            return (regs[f"s{lo}"] & mask32) | ((regs[f"s{hi}"] & mask32) << 32)
        if tok.startswith("0x"):
            v = int(tok, 16)
            return v | (0xffffffff00000000 if signed and (v & 0x80000000) else 0)
        return int(tok, 0) & mask64

    def write64(tok, val):
        lo, hi = map(int, tok[2:-1].split(":"))
        val &= mask64
        regs[f"s{lo}"] = val & mask32
        regs[f"s{hi}"] = (val >> 32) & mask32

    for line in program:
        op, _, rest = line.partition(" ")
        ops = [t.strip() for t in rest.split(",")] if rest else []
        if op == "s_buffer_load_dwordx4":
            lo = int(ops[0][2:-1].split(":")[0])
            start = (0 if ops[2] == "null" else int(ops[2], 16)) // 4
            for j in range(4):
                regs[f"s{lo + j}"] = words[start + j]
        elif op == "s_mov_b32" and ops[0].startswith("s") and not ops[1].startswith("s2"):
            regs[ops[0]] = read32(ops[1])
        elif op == "s_and_b64" and ops[0].startswith("s["):
            v = read64(ops[1]) & read64(ops[2])
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_or_b64":
            v = read64(ops[1]) | read64(ops[2])
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_xor_b64":
            v = read64(ops[1]) ^ read64(ops[2])
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_andn2_b64":
            v = read64(ops[1]) & (~read64(ops[2]) & mask64)
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_orn2_b64":
            v = read64(ops[1]) | (~read64(ops[2]) & mask64)
            write64(ops[0], v)
            scc = v != 0
        elif op in ("s_nand_b64", "s_nor_b64", "s_xnor_b64"):
            a, b = read64(ops[1]), read64(ops[2])
            v = ~{"s_nand_b64": a & b, "s_nor_b64": a | b, "s_xnor_b64": a ^ b}[op] & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_not_b64":
            v = ~read64(ops[1]) & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_brev_b64":
            x = read64(ops[1])
            v = int(f"{x:064b}"[::-1], 2)
            write64(ops[0], v)
        elif op == "s_cmov_b64":
            assert scc is not None
            if scc:
                write64(ops[0], read64(ops[1]))
        elif op == "s_lshl_b64":
            v = (read64(ops[1]) << (read32(ops[2]) & 63)) & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_lshr_b64":
            v = read64(ops[1]) >> (read32(ops[2]) & 63)
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_ashr_i64":
            x = read64(ops[1], signed=True)
            if x >= (1 << 63):
                x -= 1 << 64
            v = (x >> (read32(ops[2]) & 63)) & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_bfm_b64":
            w = read32(ops[1]) & 63
            off = read32(ops[2]) & 63
            v = (((1 << w) - 1) << off) & mask64
            write64(ops[0], v)
        elif op in ("s_bfe_u32", "s_bfe_i32"):
            src = read32(ops[1])
            ctrl = read32(ops[2])
            off = ctrl & 31
            w = (ctrl >> 16) & 127
            if w == 0:
                v = 0
            elif op == "s_bfe_u32":
                v = (src >> off) & ((1 << min(w, 32)) - 1)
            else:
                if off + w < 32:
                    raw = (src >> off) & ((1 << w) - 1)
                    if raw & (1 << (w - 1)):
                        raw -= 1 << w
                else:
                    raw = (src - (1 << 32) if src & 0x80000000 else src) >> off
                v = raw & mask32
            regs[ops[0]] = v
            scc = v != 0
        elif op == "v_mov_b32_e32" and ops[0] in ("v12", "v13", "v14", "v15"):
            bits = regs[ops[1]] & mask32
            val, = struct.unpack("<f", struct.pack("<I", bits))
            assert half_rtz(val) == val, hex(bits)
            channels[ops[0]] = val
    return tuple(channels[f"v{12 + c}"] for c in range(4)), regs["s66"] & 0x3ff


def scalar64_case(directory, values):
    constants = bytearray(256)
    struct.pack_into("<12I", constants, 16, *values)
    words = struct.unpack("<64I", constants)
    color, _ = scalar64_reference(scalar64(), words)
    path = directory / "scalar64.case"
    path.write_bytes(constants + struct.pack("<4f", *color) * 64)
    return path
# Scalar ALU producers in the scalar() fixture. Each SCC result sets its own bit of s60 through
# s_cselect_b32; each result value is mixed into the hash in s61. a to d are constant words 3 to 6.
SCALAR_STEPS = (
    ("s_add_u32 s62, s44, s45", True), ("s_sub_u32 s62, s46, s47", True),
    ("s_add_i32 s62, s46, s45", True), ("s_sub_i32 s62, s46, s45", True),
    ("s_min_u32 s62, s45, s44", True), ("s_min_i32 s62, s44, s45", True),
    ("s_max_u32 s62, s44, s45", True), ("s_max_i32 s62, s45, s44", True),
    ("s_and_b32 s62, s44, s47", True), ("s_xor_b32 s62, s44, s45", True),
    ("s_lshr_b32 s62, s46, 31", True),
    ("s_cmp_eq_u32 s44, s45", False), ("s_cmp_lg_u32 s44, s45", False),
    ("s_cmp_gt_i32 s45, s44", False), ("s_cmp_ge_i32 s44, s45", False),
    ("s_cmp_lt_i32 s44, s45", False), ("s_cmp_le_i32 s45, s44", False),
    ("s_cmp_gt_u32 s44, s45", False), ("s_cmp_ge_u32 s45, s44", False),
    ("s_cmp_lt_u32 s45, s44", False), ("s_cmp_le_u32 s44, s45", False),
    ("s_cmp_eq_i32 s47, -1", False), ("s_cmp_lg_i32 s46, 0x80000000", False),
)
# Results that leave SCC unused or unchanged; s_cselect_b32 here selects between two registers.
SCALAR_VALUES = ("s_mul_i32 s62, s44, s46", "s_or_b32 s62, s45, s46", "s_andn2_b32 s62, s44, s45",
                 "s_orn2_b32 s62, s47, s44", "s_not_b32 s62, s46", "s_lshl_b32 s62, s44, s47",
                 "s_ashr_i32 s62, s46, s44", "s_cmp_lt_u32 s44, s45", "s_cselect_b32 s62, s46, s47")
# Two constant sets; every SCC bit differs between them, so both arms of each select are drawn.
SCALAR_WORDS = ((0xfffffff0, 0x20, 0x80000000, 1), (5, 5, 0x7fffffff, 0xffffffff))


def scalar_body():
    body = ["s_mov_b32 s60, 0", "s_movk_i32 s61, 0xff80"]
    for bit, (step, value) in enumerate(SCALAR_STEPS):
        flag = 1 << bit
        body += [step, f"s_cselect_b32 s63, {flag if flag <= 64 else hex(flag)}, 0", "s_or_b32 s60, s60, s63"]
        if value:
            body += ["s_mul_i32 s61, s61, 31", "s_add_u32 s61, s61, s62"]
    for step in SCALAR_VALUES:
        body.append(step)
        if step.startswith("s_cselect") or not step.startswith("s_cmp"):
            body += ["s_mul_i32 s61, s61, 31", "s_add_u32 s61, s61, s62"]
    # Fold the 32-bit hash to 17 bits and spread 23 flag bits and 17 hash bits over four channels,
    # ten bits each, exported as 1 + k / 1024, which binary16 holds exactly.
    body += ["s_lshr_b32 s63, s61, 17", "s_xor_b32 s61, s61, s63", "s_and_b32 s61, s61, 0x1ffff",
             "s_and_b32 s56, s60, 0x3ff",
             "s_lshr_b32 s57, s60, 10", "s_and_b32 s57, s57, 0x3ff",
             "s_lshr_b32 s58, s60, 20", "s_lshl_b32 s63, s61, 3", "s_or_b32 s58, s58, s63",
             "s_and_b32 s58, s58, 0x3ff",
             "s_lshr_b32 s59, s61, 7"]
    for channel in range(4):
        body += [f"s_lshl_b32 s{56 + channel}, s{56 + channel}, 13",
                 f"s_or_b32 s{56 + channel}, s{56 + channel}, 1.0",
                 f"v_mov_b32_e32 v{12 + channel}, s{56 + channel}"]
    return body


def scalar():
    return with_constants(scalar_body())


def scalar_reference(body, words):
    """Run body on Python integers; returns the four exported channels and every select's arm."""
    mask = 0xffffffff
    regs = dict(zip(("s44", "s45", "s46", "s47"), words))
    scc, arms, out = None, [], {}

    def signed(x):
        return x - (1 << 32) if x >> 31 else x

    def read(token):
        if token in regs:
            return regs[token]
        if token == "1.0":
            return 0x3f800000
        return int(token, 0) & mask

    for line in body:
        op, operands = line.split(" ", 1)
        args = operands.split(", ")
        if op.startswith("v_mov_b32"):
            out[args[0]] = regs[args[1]]
            continue
        x, *rest = map(read, args[1:] if not op.startswith("s_cmp") else args)
        y = rest[0] if rest else None
        if op.startswith("s_cmp_"):
            relation, kind = op[6:8], op[9]
            left, right = (signed(x), signed(y)) if kind == "i" else (x, y)
            scc = {"eq": left == right, "lg": left != right, "gt": left > right, "ge": left >= right,
                   "lt": left < right, "le": left <= right}[relation]
            continue
        name = op[2:]
        if name == "mov_b32":
            result = x
        elif name == "movk_i32":
            result = (x - 0x10000 if x & 0x8000 else x) & mask
        elif name == "cselect_b32":
            assert scc is not None
            arms.append(scc)
            result = x if scc else y
        elif name in ("add_u32", "sub_u32"):
            wide = x + y if name == "add_u32" else x - y
            result, scc = wide & mask, not 0 <= wide <= mask
        elif name in ("add_i32", "sub_i32"):
            wide = signed(x) + signed(y) if name == "add_i32" else signed(x) - signed(y)
            result, scc = wide & mask, not -2**31 <= wide < 2**31
        elif name == "mul_i32":
            result = (x * y) & mask
        elif name in ("min_u32", "max_u32", "min_i32", "max_i32"):
            key = signed if name.endswith("i32") else (lambda v: v)
            scc = key(x) < key(y) if name.startswith("min") else key(x) > key(y)
            result = x if scc else y
        elif name == "not_b32":
            result = ~x & mask
            scc = result != 0
        else:
            result = {"and_b32": lambda: x & y, "or_b32": lambda: x | y, "xor_b32": lambda: x ^ y,
                      "andn2_b32": lambda: x & ~y, "orn2_b32": lambda: x | ~y,
                      "lshl_b32": lambda: x << (y & 31), "lshr_b32": lambda: x >> (y & 31),
                      "ashr_i32": lambda: signed(x) >> (y & 31)}[name]() & mask
            scc = result != 0
        regs[args[0]] = result
    channels = []
    for n in range(12, 16):
        value = struct.unpack("<f", struct.pack("<I", out[f"v{n}"]))[0]
        assert 1 <= value < 2 and half_rtz(value) == value, value
        channels.append(value)
    return channels, arms


def scalar_case(directory, words):
    """Constant words 3 to 6 as raw 32-bit patterns, then the uniform color expected at every texel."""
    case = bytearray(256)
    struct.pack_into("<4I", case, 12, *words)
    channels, _ = scalar_reference(scalar_body(), words)
    case.extend(struct.pack("<4f", *channels) * 64)
    path = directory / "scalar.case"
    path.write_bytes(case)
    return path


# Mutations a draw must detect: signedness of a shift and of a comparison, and the sources of a subtraction.
SCALAR_MUTATIONS = (("s_ashr_i32", "s_lshr_b32"), ("s_cmp_lt_i32", "s_cmp_lt_u32"),
                    ("s_sub_u32 s62, s46, s47", "s_sub_u32 s62, s47, s46"))


def scalar_detecting_words(old, new):
    """The first constant set whose reference changes under the mutation."""
    body = scalar_body()
    changed = [line.replace(old, new) for line in body]
    assert changed != body, old
    return next(words for words in SCALAR_WORDS
                if scalar_reference(changed, words)[0] != scalar_reference(body, words)[0])


def check_scalar_controls():
    body = scalar_body()
    program = scalar()
    source = translate(program)
    for expected in ("= asuint(params[0].w) + asuint(params[1].x);", " < asuint(params[0].w)) ? 1u : 0u;",
                     "= (asuint(params[1].y) < asuint(params[1].z)) ? 2u : 0u;",
                     "= ((~(asuint(params[1].y) ^ asuint(params[1].x)) & (asuint(params[1].y) ^ salu",
                     "= asuint(min(asint(asuint(params[0].w)), asint(asuint(params[1].x))));",
                     "= (asint(asuint(params[1].x)) > asint(asuint(params[0].w)));", "? 8192u : 0u;",
                     "= (asint(asuint(params[1].z)) == asint(4294967295u));", "? 2097152u : 0u;",
                     "= asuint(params[0].w) & ~asuint(params[1].x);", "= ~asuint(params[1].y);",
                     "= asuint(asint(asuint(params[1].y)) >> (asuint(params[0].w) & 31u));",
                     "= 4294967168u * 31u;", "? asuint(params[1].y) : asuint(params[1].z);",
                     " | 1065353216u;", "= asfloat(salu"):
        assert expected in source, expected
    # Both arms of every flag select are drawn across the two constant sets.
    flags = [scalar_reference(body, words)[1] for words in SCALAR_WORDS]
    assert len(flags[0]) == len(SCALAR_STEPS) + 1
    assert all(a != b for a, b in zip(flags[0][:-1], flags[1][:-1])), flags
    for old, new in SCALAR_MUTATIONS:
        scalar_detecting_words(old, new)
    # SCC exists only after a tracked write; the exec-mask operations overwrite it.
    base = assembly(False)
    wqm = base.index("s_wqm_b64 exec, exec")
    andexec = base.index("s_and_b64 exec, exec, s[50:51]")
    translate(base[:andexec] + ["s_cmp_eq_u32 1, 1", "s_cselect_b32 s62, 1, 0"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_cselect_b32 s62, 1, 0"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_cmp_eq_u32 1, 1"] + base[andexec:andexec + 1]
                + ["s_cselect_b32 s62, 1, 0"] + base[andexec + 1:])
    must_reject(translate, base[:wqm] + ["s_cmp_eq_u32 1, 1"] + base[wqm:wqm + 1]
                + ["s_cselect_b32 s62, 1, 0"] + base[wqm + 1:])
    # A scalar result is not a known constant, so it cannot supply a sampler word or address.
    for old, new in (("s_mov_b32 s40, 1", "s_add_u32 s40, 0, 1"), ("s_mov_b32 s21, 0", "s_and_b32 s21, 0, 0")):
        must_reject(translate, [line.replace(old, new) for line in base])
    # Neighbouring forms that the lowerer does not implement, operand kinds it cannot read, and
    # destinations other than plain scalar registers.
    first = "s_add_u32 s62, s44, s45"
    for new in ("s_addc_u32 s62, s44, s45", "s_subb_u32 s62, s44, s45", "s_mul_hi_u32 s62, s44, s45",
                "s_absdiff_i32 s62, s44, s45", "s_nand_b32 s62, s44, s45", "s_xnor_b32 s62, s44, s45",
                "s_bfe_u32 s62, v12, s45", "s_lshl1_add_u32 s62, s44, s45", "s_abs_i32 s62, s44",
                "s_brev_b32 s62, s44", "s_and_b64 s[62:63], s[44:46], s[46:47]",
                "s_add_u32 s62, s[44:45], s45", "s_add_u32 s62, s44", "s_add_u32 s62, s44, s45, s46",
                "s_add_u32 s62, -s44, s45", "s_add_u32 s62, s28, s45", "s_add_u32 s62, s20, s45",
                "s_add_u32 s62, s90, s45", "s_add_u32 s62, vcc_lo, s45", "s_add_u32 s62, exec_lo, s45",
                "s_add_u32 s62, v12, s45", "s_add_u32 m0, s44, s45", "s_add_u32 vcc_lo, s44, s45",
                "s_add_u32 s62, s44, |s45|",
                "s_cmp_eq_u64 s[44:46], s[46:47]", "s_bitcmp1_b32 v12, s45", "s_cmov_b32 s62, s44",
                "s_cmp_lt_i32 s44", "s_cmp_lt_f32 s44, s45"):
        changed = [line.replace(first, new) for line in program]
        assert changed != program, new
        must_reject(translate, changed)
    flag = program.index(first) + 1
    for new in ("s_cselect_b64 s[62:63], s44, s45", "s_cmov_b32 s63, 1", "s_cselect_b32 s63, s28, 0",
                "s_cselect_b32 vcc_lo, 1, 0"):
        must_reject(translate, program[:flag] + [new] + program[flag + 1:])
    # Before lgkmcnt(0) the constant words are still loading: neither read nor overwrite them.
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc") + 1
    for line in ("s_add_u32 s62, s44, 1", "s_add_u32 s45, 1, 1", "s_cmp_eq_u32 s46, 0",
                 "s_cselect_b32 s47, 1, 0"):
        must_reject(translate, program[:load] + ["s_cmp_eq_u32 1, 1", line] + program[load:])
    translate(program[:load] + ["s_cmp_eq_u32 1, 1", "s_cselect_b32 s62, 1, 0"] + program[load:])


# SOPK compares and the conditional move immediate. Each s_cmpk_* outcome sets its own bit of s60 through
# s_cselect_b32; s61 and s59 each take an s_cmovk_i32 whose SCC comes from an s_cmpk. sopk_program(k) exports the
# four bytes of SOPK_RESULTS[k] through v_cvt_f32_ubyte0-3, so each channel is an exact integer 0..255.
# The immediate is printed as LLVM prints it, 0x0 to 0xffff; _i32 sign-extends it and _u32 zero-extends it.
SOPK_FORMS = tuple(f"s_cmpk_{r}_{t}32" for r in ("eq", "lg", "gt", "ge", "lt", "le") for t in "iu") + ("s_cmovk_i32",)
SOPK_COMPARES = (
    "s_cmpk_eq_i32 s44, 0x8000", "s_cmpk_eq_i32 s45, 0x7fff",
    "s_cmpk_eq_u32 s45, 0x8000", "s_cmpk_eq_u32 s46, 0x7fff",
    "s_cmpk_lg_i32 s44, 0x8000", "s_cmpk_lg_i32 s45, 0x7fff",
    "s_cmpk_lg_u32 s45, 0x8000", "s_cmpk_lg_u32 s46, 0x7fff",
    "s_cmpk_gt_i32 s47, 0x0", "s_cmpk_gt_i32 s45, 0x7fff",
    "s_cmpk_gt_u32 s46, 0x8000", "s_cmpk_gt_u32 s45, 0x7fff",
    "s_cmpk_ge_i32 s44, 0x8000", "s_cmpk_ge_i32 s47, 0x0",
    "s_cmpk_ge_u32 s45, 0x8000", "s_cmpk_ge_u32 s46, 0x8000",
    "s_cmpk_lt_i32 s44, 0x8000", "s_cmpk_lt_i32 s47, 0x0",
    "s_cmpk_lt_u32 s45, 0x8000", "s_cmpk_lt_u32 s46, 0x8000",
    "s_cmpk_le_i32 s47, 0x0", "s_cmpk_le_i32 s45, 0x7fff",
    "s_cmpk_le_u32 s46, 0x8000", "s_cmpk_le_u32 s45, 0x7fff",
)
SOPK_MOVES = ("s_mov_b32 s61, s47", "s_cmpk_lt_i32 s47, 0x1", "s_cmovk_i32 s61, 0x8001",
              "s_mov_b32 s59, s46", "s_cmpk_ge_u32 s46, 0x8001", "s_cmovk_i32 s59, 0x7ffe")
SOPK_RESULTS = ("s60", "s61", "s59")
# Constant words 3 to 6 (s44 to s47). Every compare and both s_cmovk_i32 conditions differ between the sets, and
# a = -32768 = sext(0x8000) in the first set; b straddles 0x8000; d straddles the signed range limit.
SOPK_WORDS = ((0xffff8000, 0x8000, 0x7fff, 0x80000000), (0xffff7fff, 0x7fff, 0x10000, 0x7fffffff))


def sopk_body():
    body = ["s_mov_b32 s60, 0"]
    for bit, step in enumerate(SOPK_COMPARES):
        flag = 1 << bit
        body += [step, f"s_cselect_b32 s63, {flag if flag <= 64 else hex(flag)}, 0", "s_or_b32 s60, s60, s63"]
    return body + list(SOPK_MOVES)


def sopk_program(k, body=None):
    return with_constants([*(sopk_body() if body is None else body), f"v_mov_b32_e32 v18, {SOPK_RESULTS[k]}",
                           *(f"v_cvt_f32_ubyte{n}_e32 v{12 + n}, v18" for n in range(4))])


def sopk_reference(body, words):
    """Independent model of the RDNA 2 ISA definitions on Python integers: the words SOPK_RESULTS name, and the
    SCC of every compare in program order."""
    regs = dict(zip(("s44", "s45", "s46", "s47"), words))
    scc, outcomes = None, []

    def sext16(k):
        return k - 0x10000 if k & 0x8000 else k

    def signed(x):
        return x - (1 << 32) if x >> 31 else x

    for line in body:
        op, operands = line.split(" ", 1)
        args = operands.split(", ")
        value = (lambda t: regs[t] if t in regs else int(t, 0))
        if op.startswith("s_cmpk_"):
            relation, kind = op[7:9], op[10]
            k = int(args[1], 16)
            left, right = (signed(regs[args[0]]), sext16(k)) if kind == "i" else (regs[args[0]], k)
            scc = {"eq": left == right, "lg": left != right, "gt": left > right, "ge": left >= right,
                   "lt": left < right, "le": left <= right}[relation]
            outcomes.append(scc)
        elif op == "s_cmovk_i32":
            if scc:
                regs[args[0]] = sext16(int(args[1], 16)) & 0xffffffff
        elif op == "s_movk_i32":
            regs[args[0]] = sext16(int(args[1], 16)) & 0xffffffff
        elif op == "s_mov_b32":
            regs[args[0]] = value(args[1])
        elif op == "s_cselect_b32":
            regs[args[0]] = value(args[1]) if scc else value(args[2])
        elif op == "s_or_b32":
            regs[args[0]] = value(args[1]) | value(args[2])
            scc = regs[args[0]] != 0
        else:
            raise ValueError(line)
    return [regs[r] for r in SOPK_RESULTS], outcomes


# (exported result, old text, new text): signedness of a compare, sign- against zero-extension of the immediate,
# the equality boundary, and an s_cmovk_i32 that ignores SCC.
SOPK_MUTATIONS = (
    (0, "s_cmpk_lt_i32 s44, 0x8000", "s_cmpk_lt_u32 s44, 0x8000"),
    (0, "s_cmpk_gt_u32 s46, 0x8000", "s_cmpk_gt_i32 s46, 0x8000"),
    (0, "s_cmpk_eq_i32 s44, 0x8000", "s_cmpk_eq_u32 s44, 0x8000"),
    (0, "s_cmpk_ge_i32 s44, 0x8000", "s_cmpk_gt_i32 s44, 0x8000"),
    (1, "s_cmpk_lt_i32 s47, 0x1", "s_cmpk_lt_u32 s47, 0x1"),
    (1, "s_cmovk_i32 s61, 0x8001", "s_movk_i32 s61, 0x8001"),
    (2, "s_cmpk_ge_u32 s46, 0x8001", "s_cmpk_ge_i32 s46, 0x8001"),
)


def sopk_detecting_words(k, old, new):
    """The first constant set on which the mutation changes exported result k."""
    body = sopk_body()
    changed = [line.replace(old, new) for line in body]
    assert changed != body, old
    return next(words for words in SOPK_WORDS
                if sopk_reference(changed, words)[0][k] != sopk_reference(body, words)[0][k])


def check_sopk_controls():
    source = translate(sopk_program(0))
    assert set(SOPK_FORMS) <= ACCEPTED
    s44, s45, s46, s47 = (f"asuint(params[{n // 4}].{'xyzw'[n % 4]})" for n in range(3, 7))
    for expected in (
        f"= (asint({s44}) == -32768);", f"= (asint({s45}) == 32767);",
        f"= ({s45} == 32768u);", f"= ({s46} != 32767u);",
        f"= (asint({s47}) > 0);", f"= ({s46} > 32768u);",
        f"= (asint({s44}) >= -32768);", f"= ({s45} >= 32768u);",
        f"= (asint({s44}) < -32768);", f"= ({s46} < 32768u);",
        f"= (asint({s45}) <= 32767);", f"= ({s45} <= 32767u);",
        f"= (asint({s47}) < 1);", f"= ({s46} >= 32769u);",
        f" ? 4294934529u : {s47};", f" ? 32766u : {s46};",
    ):
        assert expected in source, expected
    # Every compare and both conditional moves take each outcome across the two constant sets.
    flags = [sopk_reference(sopk_body(), words)[1] for words in SOPK_WORDS]
    assert len(flags[0]) == len(SOPK_COMPARES) + 2
    assert all(a != b for a, b in zip(*flags)), flags
    for k, old, new in SOPK_MUTATIONS:
        sopk_detecting_words(k, old, new)
    # A register written by s_movk_i32 is a known constant; the compare reads it as a uint literal.
    known = translate(with_constants(["s_movk_i32 s50, 0x8000", "s_cmpk_eq_i32 s50, 0x8000",
                                      "s_cmovk_i32 s50, 0x1"]))
    assert "(asint(4294934528u) == -32768);" in known and " ? 1u : 4294934528u;" in known
    program = sopk_program(1)
    first, move = "s_cmpk_eq_i32 s44, 0x8000", "s_cmovk_i32 s61, 0x8001"
    for bad in (
        "s_cmpk_eq_i32 s44, 0x10000", "s_cmpk_eq_i32 s44, -1", "s_cmpk_eq_i32 s44, 1.0",
        "s_cmpk_eq_i32 s44", "s_cmpk_eq_i32 s44, 0x8000, 0x1", "s_cmpk_eq_i32 v12, 0x8000",
        "s_cmpk_eq_i32 s106, 0x8000", "s_cmpk_eq_i32 vcc_lo, 0x8000", "s_cmpk_eq_i32 m0, 0x8000",
        "s_cmpk_eq_i32 s[44:45], 0x8000", "s_cmpk_eq_i32 0x10, 0x8000", "s_cmpk_eq_i32 s90, 0x8000",
        "s_cmpk_eq_i32 s20, 0x8000", "s_cmpk_eq_i32 s28, 0x8000",
        # Neighbours that stay unlowered: the register conditional move and the hardware-register moves.
        "s_cmov_b32 s63, s44", "s_getreg_b32 s63, hwreg(HW_REG_MODE)",
        "s_setreg_b32 hwreg(HW_REG_MODE), s44",
    ):
        changed = [line.replace(first, bad) for line in program]
        assert changed != program, bad
        must_reject(translate, changed)
    for bad in ("s_cmovk_i32 s61, 0x10000", "s_cmovk_i32 s61, -1", "s_cmovk_i32 s61", "s_cmovk_i32 v12, 0x8001",
                "s_cmovk_i32 vcc_lo, 0x8001", "s_cmovk_i32 m0, 0x8001", "s_cmovk_i32 s70, 0x8001",
                "s_cmovk_i32 s61, s44", "s_cmovk_i32 s[60:61], 0x8001", "s_cmovk_i32 s106, 0x8001"):
        changed = [line.replace(move, bad) for line in program]
        assert changed != program, bad
        must_reject(translate, changed)
    # s_cmovk_i32 needs a tracked SCC; the exec-mask operations clear it.
    base = assembly(False)
    wqm = base.index("s_wqm_b64 exec, exec")
    andexec = base.index("s_and_b64 exec, exec, s[50:51]")
    translate(base[:andexec] + ["s_mov_b32 s62, 7", "s_cmpk_lg_u32 s62, 0x7", "s_cmovk_i32 s62, 0x1"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_mov_b32 s62, 7", "s_cmovk_i32 s62, 0x1"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_mov_b32 s62, 7", "s_cmpk_lg_u32 s62, 0x7"] + base[andexec:andexec + 1]
                + ["s_cmovk_i32 s62, 0x1"] + base[andexec + 1:])
    must_reject(translate, base[:wqm] + ["s_mov_b32 s62, 7", "s_cmpk_lg_u32 s62, 0x7"] + base[wqm:wqm + 1]
                + ["s_cmovk_i32 s62, 0x1"] + base[wqm + 1:])
    # Before lgkmcnt(0) the constant words are still loading: neither compare nor conditionally overwrite them.
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc") + 1
    must_reject(translate, program[:load] + ["s_cmpk_eq_u32 s44, 0x1"] + program[load:])
    must_reject(translate, program[:load] + ["s_mov_b32 s62, 7", "s_cmpk_lg_u32 s62, 0x7", "s_cmovk_i32 s45, 0x1"]
                + program[load:])
    translate(program[:load] + ["s_mov_b32 s62, 7", "s_cmpk_lg_u32 s62, 0x7", "s_cmovk_i32 s62, 0x1"] + program[load:])


def sopk_case(path, words, word):
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *words)
    path.write_bytes(constants + struct.pack("<4f", *[float((word >> 8 * n) & 0xff) for n in range(4)]) * 64)


def check_sopk_draws(directory, compiler, renderer, backend):
    source = directory / "sopk_compare.hlsl"
    shader = directory / "sopk_compare.cso"
    case = directory / "sopk_compare.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]
    for k in range(len(SOPK_RESULTS)):
        source.write_text(translate(sopk_program(k)), encoding="ascii")
        subprocess.run(compile_shader, check=True)
        for words in SOPK_WORDS:
            sopk_case(case, words, sopk_reference(sopk_body(), words)[0][k])
            subprocess.run(draw, check=True)
    # Each mutation changes the exported word on its detecting set (checked in the controls), so the draw against
    # the correct reference must report a mismatch.
    for k, old, new in SOPK_MUTATIONS:
        words = sopk_detecting_words(k, old, new)
        mutated = [line.replace(old, new) for line in sopk_body()]
        source.write_text(translate(sopk_program(k, mutated)), encoding="ascii")
        subprocess.run(compile_shader, check=True)
        sopk_case(case, words, sopk_reference(sopk_body(), words)[0][k])
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Replacing {old} with {new} escaped detection")


def check_sopk_render(directory, compiler, renderer, backend):
    """The SOPK compare draws as their own CTest entry per backend; exit 77 from the renderer is a skip."""
    try:
        check_sopk_draws(directory, compiler, renderer, backend)
    except subprocess.CalledProcessError as error:
        if error.returncode == 77:
            return 77
        raise
    return 0


# SDWA moves and half widening. In SDWA_PRESERVE_BODY each move writes a 16-bit field into WORD_0 of a register
# holding 1.0 and keeps its high half, so the word is 1 + f * 2^-23; (v - 1) * 2^23 is split into the low and high
# bytes of f. Channels 0 and 1 read WORD_1 of S0; channels 2 and 3 read BYTE_2 of S1 sign-extended to 16 bits.
# SDWA_PAD_BODY writes BYTE_0 of S0 to BYTE_3 and WORD_0 of S2 to WORD_1 with zero padding, and widens each half
# of S1, the low one through -|x|. Each constant set has wrong references (channel, value) for one selection.
def sdwa_readback(value, high):
    return [line.format(r=value, h=high) for line in SDWA_READBACK]


SDWA_READBACK = ["v_subrev_f32_e32 {r}, 1.0, {r}", "v_mul_f32_e32 {r}, 0x4b000000, {r}", "v_mul_f32_e32 {h}, 0x3b800000, {r}", "v_floor_f32_e32 {h}, {h}", "v_fmac_f32_e32 {r}, 0xc3800000, {h}"]
SDWA_PRESERVE_BODY = [
    "v_mov_b32_e32 v12, 1.0",
    "v_mov_b32_sdwa v12, s44 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
    *sdwa_readback("v12", "v13"),
    "v_mov_b32_e32 v14, 1.0",
    "v_mov_b32_sdwa v14, sext(s45) dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_2",
    *sdwa_readback("v14", "v15"),
]
SDWA_PAD_BODY = [
    "v_mov_b32_sdwa v12, s44 dst_sel:BYTE_3 dst_unused:UNUSED_PAD src0_sel:BYTE_0",
    "v_cvt_f32_f16_sdwa v13, s45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1",
    "v_cvt_f32_f16_sdwa v14, -|s45| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
    "v_mov_b32_sdwa v15, s46 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_0",
]
SDWA_SETS = (
    # A wrong WORD_0 selection; a zero-extended byte.
    (SDWA_PRESERVE_BODY, (0x12345678, 0x00c00000, 0), [52.0, 18.0, 192.0, 255.0], ((0, 120.0), (1, 86.0), (3, 0.0))),
    (SDWA_PRESERVE_BODY, (0xfedc0000, 0x007f0000, 0), [220.0, 254.0, 127.0, 0.0], ((0, 0.0), (3, 255.0))),
    # BYTE_1 instead of BYTE_0; the other half of S1 in channels 1 and 2; WORD_1 instead of WORD_0 of S2.
    (SDWA_PAD_BODY, (0x0000c340, 0xc5003c00, 0x3f80c0a0), [2.0, -5.0, -1.0, -5.0], ((0, -128.0), (1, 1.0), (2, -5.0), (3, 1.0))),
    (SDWA_PAD_BODY, (0x000047c0, 0x38004900, 0x00004120), [-2.0, 0.5, -10.0, 10.0], ((0, 32768.0), (1, 10.0), (2, -0.5), (3, 0.0))),
)


def sdwa_reference(body, words):
    """Selections and widening on the integer words, independent of the lowering."""
    x, y, z = words
    if body is SDWA_PRESERVE_BODY:
        first = x >> 16
        byte = (y >> 16) & 0xff
        second = (byte - 0x100 if byte & 0x80 else byte) & 0xffff
        return [float(first & 0xff), float(first >> 8), float(second & 0xff), float(second >> 8)]
    as_float = lambda w: struct.unpack("<f", struct.pack("<I", w))[0]
    return [as_float((x & 0xff) << 24), half_value(y >> 16), -abs(half_value(y)), as_float((z & 0xffff) << 16)]


def check_sdwa_draw(directory, compiler, renderer, backend):
    for name, body in (("sdwa_preserve", SDWA_PRESERVE_BODY), ("sdwa_pad", SDWA_PAD_BODY)):
        source = directory / f"{name}.hlsl"
        shader = directory / f"{name}.cso"
        source.write_text(translate(with_constants(body)), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    case = directory / "sdwa.case"
    for body, words, expected, wrongs in SDWA_SETS:
        assert sdwa_reference(body, words) == expected, (words, expected)
        shader = directory / ("sdwa_preserve.cso" if body is SDWA_PRESERVE_BODY else "sdwa_pad.cso")
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *words)
        for x in (*expected, *(value for _, value in wrongs)):
            assert half_rtz(x) == x, x
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
        for channel, value in wrongs:
            assert value != expected[channel]
            wrong_values = list(expected)
            wrong_values[channel] = value
            case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
            result = subprocess.run(draw, capture_output=True, text=True)
            if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
                raise AssertionError(f"A wrong SDWA selection escaped detection for {words} in channel {channel}")


def check_constant_descriptor_bounds() -> None:
    for offset in (0, 28, 32):
        program = [line.replace("s[20:21], 0x40", f"s[20:21], {offset:#x}")
                   for line in curve()]
        if offset == 32:
            assert lower(program, 20, 24, [1, 2, 3, 4], offset) == translate(curve())
        else:
            try:
                lower(program, 20, 24, [1, 2, 3, 4], offset)
            except ValueError as error:
                assert "descriptors overlap" in str(error), error
            else:
                raise AssertionError(f"Constant descriptor at {offset} overlaps texture 0")


def check_scalar_constant_reads() -> None:
    program = with_scalar_constant_reads()
    source = translate(program)
    for field in ("params[0].x", "params[0].z", "params[15].w"):
        assert f"= {field};" in source

    load = "s_buffer_load_dword s62, s[44:47], 0x8"
    for bad in (load.replace("0x8", "0x9"), load.replace("0x8", "0x100"), load.replace("0x8", "s60"),
                load.replace("s[44:47]", "s[28:31]"), load.replace("s[44:47]", "s[44:46]")):
        must_reject(translate, [bad if line == load else line for line in program])

    first = program.index("s_buffer_load_dword s61, s[44:47], null")
    wait = program.index("s_waitcnt lgkmcnt(0)", first)
    descriptor = program.index("s_load_dwordx4 s[44:47], s[20:21], 0x40")
    must_reject(translate, program[:descriptor + 1] + [load] + program[descriptor + 1:])
    must_reject(translate, program[:wait] + program[wait + 1:])  # Result read before the wait.
    must_reject(translate, program[:wait] + [load] + program[wait:])  # Outstanding write to the same SGPR.
    must_reject(translate, program[:first] + ["s_mov_b32 s46, 0"] + program[first:])
    must_reject(translate, [line.replace("s[44:47]", "s[46:49]") for line in program])

    alias = [line.replace("s63", "s46") for line in program]
    assert translate(alias) == source
    must_reject(translate, alias[:wait + 1] + [load] + alias[wait + 1:])


def check_affine_coordinates() -> None:
    program = affine_coordinates()
    final = "v_fma_f32 v8, 0.5, v8, 0"
    final_index = program.index(final)
    source = translate(program)
    assert f".Sample(samp, float2(alu{final_index}, uv.y))" in source

    exact_input = ["v_mov_b32_e32 v22, 0.5", *program]
    exact_input[final_index + 1] = final.replace("0.5", "v22")
    must_reject(translate, exact_input)
    mask_input = ["v_cmp_eq_f32_e64 s52, 0.5, 0.5", *program]
    mask_input[final_index + 1] = final.replace("0.5", "s52")
    must_reject(translate, mask_input)

    restore = program.index("s_and_b64 exec, exec, s[50:51]")
    exact_overwrite = "v_add_f32_e32 v8, -0.5, v8"
    must_reject(translate, program[:restore + 1] + [exact_overwrite] + program[restore + 1:])
    duplicate = [*program[:final_index], final.replace("v8, 0.5", "v22, 0.5"), *program[final_index:restore],
                 "v_add_f32_e32 v23, 0.0, v9", program[restore], exact_overwrite, *program[restore + 1:]]
    separate_coords = [line.replace("v[8:9]", "v[22:23]") for line in duplicate]
    assert f".Sample(samp, float2(alu{final_index}, alu{restore + 1}))" in translate(separate_coords)
    partial = program.index("v_interp_p2_f32_e32 v8, v1, attr0.x")
    must_reject(translate, program[:partial] + ["v_add_f32_e32 v8, 0.0, v8"] + program[partial:])
    wait = program.index("s_buffer_load_dwordx4 s[60:63], s[44:47], null") + 1
    must_reject(translate, program[:wait] + program[wait + 1:])

    for opcode in ("v_mul_legacy_f32_e32 v8, v9, v8", "v_fma_f32 v8, v9, v8, 0",
                   "v_fma_f32 v8, 0.5, |v8|, 0", "v_fma_f32 v8, 0.5, v8, 0 clamp"):
        changed = program.copy()
        changed[final_index] = opcode
        must_reject(translate, changed)


def check_scalar_helpers_with_vector_arithmetic() -> None:
    entry = assembly(False)
    entry_add = "v_add_f32_e32 v12, 0.0, v12"
    entry.insert(entry.index("s_waitcnt vmcnt(0)") + 1, entry_add)
    quad_add = "v_add_f32_e32 v8, 0.5, v8"
    scalar = "s_bfe_u32 s70, 1, 0x10000"
    for program, vector in ((entry, entry_add), (affine_coordinates(), quad_add)):
        position = program.index(vector)
        for index in (position, position + 1):
            source = translate(program[:index] + [scalar] + program[index:])
            assert "uint s_bfe_u32(" in source and " = s_bfe_u32(" in source


# v_fmaak_f32 computes S0 * S1 + K and v_fmamk_f32 computes S0 * K + S1, with K a 32-bit literal.
# Every product and sum is exact in binary32, so the reference does not depend on fusion. Swapping
# the literal with S1 changes each result.
LITERAL_FMA_INPUTS = (1.5, 2.5, -3.0, 0.75)
LITERAL_FMA_BODY = [
    "v_mov_b32_e32 v13, s45",
    "v_fmaak_f32 v12, s44, v13, 0x3e800000",
    "v_fmamk_f32 v13, s44, 0x40400000, v13",
    "v_mov_b32_e32 v15, s47",
    "v_fmaak_f32 v14, s46, v15, 0xbf000000",
    "v_fmamk_f32 v15, s46, 0x3f000000, v15",
]


def check_literal_fma_draw(directory, compiler, renderer, backend):
    source = directory / "literal_fma.hlsl"
    shader = directory / "literal_fma.cso"
    case = directory / "literal_fma.case"
    source.write_text(translate(with_constants(LITERAL_FMA_BODY)), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *LITERAL_FMA_INPUTS)
    a, b, c, d = LITERAL_FMA_INPUTS
    expected = [a * b + 0.25, a * 3.0 + b, c * d - 0.5, c * 0.5 + d]
    swapped = [a * 0.25 + b, a * b + 3.0, c * -0.5 + d, c * d + 0.5]
    for x in (*expected, *swapped):
        assert single(x) == x == half_rtz(x), x
    assert all(x != y for x, y in zip(expected, swapped))
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
    subprocess.run(draw, check=True)
    # Treating the literal as S1, or S1 as the literal, must change the exported color.
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = swapped[k]
        case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Swapped literal multiply-add operands escaped detection in channel {k}")
def check_scalar_pair_reads() -> None:
    program = with_scalar_pair_reads()
    source = translate(program)
    for field in ("(params[0].w * sample", "max(params[15].z, ", "(params[1].x + sample", "(params[1].y - sample",
                  "= params[1].z;", "(params[15].w * alu"):
        assert field in source, field

    load = PAIR_LOADS[1]
    for old, new in (("0x14", "0x15"), ("0x14", "0x16"),  # not dword-aligned
                     ("0x14", "0xfc"), ("0x14", "0x100"),  # the second word or both pass the 256 bound bytes
                     ("0x14", "s60"), ("0x14", "s60, 0x14"),  # register offset
                     ("s[62:63]", "s62"), ("s[62:63]", "s[62:62]"), ("s[62:63]", "s[62:64]"),
                     ("s[62:63]", "s[60:63]"), ("s[62:63]", "s[61:62]"), ("s[62:63]", "s[106:107]"),
                     ("s[44:47]", "s[28:31]"), ("s[44:47]", "s[44:45]"), ("s[44:47]", "s[44:46]"),
                     ("dwordx2", "dword"), ("dwordx2", "dwordx4"), ("s_buffer_load", "s_load"),
                     ("dwordx2 s[62:63]", "dwordx8 s[56:63]"), ("dwordx2 s[62:63]", "dwordx16 s[48:63]")):
        bad = load.replace(old, new)
        assert bad != load
        must_reject(translate, [bad if line == load else line for line in program])

    first = program.index(PAIR_LOADS[0])
    wait = program.index("s_waitcnt lgkmcnt(0)", first)
    descriptor = program.index("s_load_dwordx4 s[44:47], s[20:21], 0x40")
    must_reject(translate, program[:descriptor + 1] + [load] + program[descriptor + 1:])  # Descriptor still loading.
    must_reject(translate, program[:wait] + program[wait + 1:])  # Results read before the wait.
    for word in ("s62", "s63"):  # Neither word of the pair can be read or written until the wait.
        must_reject(translate, program[:wait] + [f"v_mov_b32_e32 v20, {word}"] + program[wait:])
        translate(program[:wait + 1] + [f"v_mov_b32_e32 v20, {word}"] + program[wait + 1:])
        must_reject(translate, program[:wait] + [f"s_mov_b32 {word}, 0"] + program[wait:])
        must_reject(translate, program[:wait] + [f"s_buffer_load_dword {word}, s[44:47], 0x8"] + program[wait:])
    must_reject(translate, program[:wait] + [load] + program[wait:])
    must_reject(translate, program[:wait] + ["s_mov_b64 s[62:63], exec"] + program[wait:])
    translate(program[:wait + 1] + [load, "s_waitcnt lgkmcnt(0)"] + program[wait + 1:])  # Reloading after the wait.
    must_reject(translate, program[:first] + ["s_mov_b32 s46, 0"] + program[first:])

    # A pair may land in the descriptor's own registers, which ends the descriptor.
    alias = [line.replace("s[64:65]", "s[46:47]").replace("s64", "s46").replace("s65", "s47") for line in program]
    assert alias != program and translate(alias) == source
    must_reject(translate, alias[:wait + 1] + [load] + alias[wait + 1:])


HINT_NAMES = ("s_clause", "s_setprio", "s_sleep")


def scheduling_hints():
    # Hints the compiler places between instructions. native_program ends the code at s_endpgm,
    # so s_code_end padding never reaches the translator and is not part of the fixture.
    program = assembly(False)
    program.insert(program.index("s_waitcnt vmcnt(0)"), "s_clause 0x1")
    program.insert(1, "s_setprio 2")
    program.insert(program.index("s_waitcnt vmcnt(0)"), "s_sleep 1")
    return program


def check_scheduling_hints():
    """Scheduling hints change timing only. Generated names carry the instruction index, so
    each hint must translate exactly like s_nop 0 at the same position."""
    plain = assembly(False)
    hinted = scheduling_hints()
    as_nops = [("s_nop 0" if line.split(" ")[0] in HINT_NAMES else line)
               for line in hinted]
    assert translate(hinted) == translate(as_nops)
    nop = translate(["s_nop 0"] + plain)
    for hint, limit in (("s_clause", 0x3f), ("s_setprio", 3), ("s_sleep", 0x7f)):
        assert translate([f"{hint} {limit}"] + plain) == nop
        assert translate([f"{hint} {hex(limit)}"] + plain) == nop
        must_reject(translate, [f"{hint} {limit + 1}"] + plain)
    # s_code_end stays unknown, and nothing may follow the program end or sit between the color
    # export and s_endpgm.
    must_reject(translate, ["s_code_end"] + plain)
    must_reject(translate, plain + ["s_code_end"])
    must_reject(translate, plain + ["s_nop 0"])
    must_reject(translate, plain[:-1] + ["s_setprio 0", plain[-1]])


def check_adapter_skip(directory: Path) -> None:
    compiler, renderer = "stub-dxc", "stub-offscreen"
    work = directory / "no-adapter"
    work.mkdir()

    def run_without_adapter(args: list[str | Path], *, check: bool = False) -> subprocess.CompletedProcess[None]:
        assert args[0] in (compiler, renderer)
        result = subprocess.CompletedProcess(args, 77 if args[0] == renderer else 0)
        if check:
            result.check_returncode()
        return result

    with patch.object(subprocess, "run", side_effect=run_without_adapter):
        assert check_render(work, compiler, renderer, "hardware") == 77


def attribute_color(rg):
    code = ["s_mov_b32 s30, s14", "s_mov_b32 m0, s30"]
    for phase in (1, 2):
        for i, channel in enumerate("wzxy"):
            code.append(f"v_interp_p{phase}_f32_e32 v{8+i}, v{phase-1}, attr0.{channel}")
    code.extend(["v_cvt_pkrtz_f16_f32_e32 v12, v8, v9", "v_cvt_pkrtz_f16_f32_e32 v13, v10, v11"])
    tail = "off, off" if rg else "v13, v13"
    return code + [f"exp mrt0 v12, v12, {tail} done compr vm", "s_endpgm"]


def translate_attribute(code):
    return lower(code, 8, 14, None)


def buffer_color(rg):
    # Use a copied low table address and an explicit zero high word.
    tail = "off, off" if rg else "v26, v26"
    return [
        "s_mov_b32 s30, s9", "s_mov_b32 s31, 0",
        "s_load_dwordx4 s[12:15], s[30:31], null", "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dwordx4 s[4:7], s[12:15], 0x10", "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v20, s7", "v_mov_b32_e32 v21, s4",
        "v_cvt_pkrtz_f16_f32_e32 v25, v20, v21",
        "v_cvt_pkrtz_f16_f32_e32 v26, v21, v20",
        f"exp mrt0 v25, v25, {tail} done compr vm", "s_endpgm",
    ]


def translate_buffer_color(instructions):
    return lower(instructions, 9, 24, None, 0)


def three_way_case(directory, low, high, cap, middle):
    """Expected colors for three_way(); no rounding occurs before the half export."""
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, low, high, cap, middle)
    for y in RAMP:
        for x in RAMP:
            r, g, b = single(x / 255), single(y / 255), single(128 / 255)
            clamped = sorted((r, low, high))[1]
            color = (clamped, min(abs(cap), g, clamped), max(r, b, g), sorted((middle, g, r))[1])
            case.extend(struct.pack("<4f", *map(half_rtz, color)))
    path = directory / "three_way.case"
    path.write_bytes(case)
    return path


def check_three_way_templates():
    """The HLSL forms must pick the expected operand for every operand order."""
    values = (-1.5, 0.0, 0.25, 0.25, 3.0)
    for a in values:
        for b in values:
            for c in values:
                pick = {name: eval(ALU[name][1].format("a", "b", "c"), {"min": min, "max": max},
                                   {"a": a, "b": b, "c": c}) for name in ("min3", "max3", "med3")}
                assert pick == {"min3": min(a, b, c), "max3": max(a, b, c),
                                "med3": sorted((a, b, c))[1]}, (a, b, c)
def pack_f16_pair(lo, hi):
    lo_b = struct.unpack("<H", struct.pack("<e", lo))[0]
    hi_b = struct.unpack("<H", struct.pack("<e", hi))[0]
    return lo_b | (hi_b << 16)


def rne_f16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


# Mixed-precision VOP3P FMA fixture covering v_fma_mix_f32, v_fma_mixlo_f16, v_fma_mixhi_f16,
# op_sel / op_sel_hi, |.| / -, clamp, inline f16 constants, and untouched-half preservation.
FMA_MIX_BODY = [
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_mov_b32_e32 v22, 0x40003c00",
    "v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0]",
    "v_fma_mixlo_f16 v22, s44, 0.5, v21 op_sel:[1,1,0]",
    "v_fma_mix_f32 v23, v22, 1.0, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]",
    "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp",
    "v_cvt_f32_f16_e32 v13, v22",
    "v_fma_mix_f32 v14, v22, 1.0, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]",
    "v_fma_mix_f32 v15, v12, 1.0, 0.5 clamp",
]

FMA_MIX_SETS = (
    # Set 1: mixed f16/f32 with |.|, -, op_sel_hi, and untouched-half preservation on both mixlo & mixhi.
    (pack_f16_pair(1.5, -2.0), pack_f16_pair(0.5, -0.25), 0.75, 0.25),
    # Set 2: saturates both v_fma_mixhi_f16 clamp and v_fma_mix_f32 clamp to [0, 1].
    (pack_f16_pair(3.0, -4.0), pack_f16_pair(-0.5, 0.5), 2.5, -1.0),
    # Set 3: exact binary16 halfway ties (odd mantissa -> round up under RNE vs down under RTZ).
    (pack_f16_pair(2.0 + 2 ** -9, 1.0), pack_f16_pair(0.5, 0.25), 1.0 + 2 ** -11 + 2 ** -12, 2 ** -11),
)


def fma_mix_reference(a_bits, b_bits, c_val, d_val):
    a_lo = struct.unpack("<e", struct.pack("<H", a_bits & 0xffff))[0]
    a_hi = struct.unpack("<e", struct.pack("<H", (a_bits >> 16) & 0xffff))[0]
    b_lo = struct.unpack("<e", struct.pack("<H", b_bits & 0xffff))[0]
    b_hi = struct.unpack("<e", struct.pack("<H", (b_bits >> 16) & 0xffff))[0]
    v12 = single(abs(a_hi) * (-b_lo) + c_val)
    lo = rne_f16(single(a_lo * 0.5 + d_val))
    hi = rne_f16(min(max(single(-abs(b_hi) * 2.0 + c_val), 0.0), 1.0))
    v15 = min(max(single(v12 + 0.5), 0.0), 1.0)
    return (half_rtz(v12), half_rtz(lo), half_rtz(hi), half_rtz(v15))


def check_fma_mix_draw(directory, compiler, renderer, backend):
    program = with_constants(FMA_MIX_BODY)
    source = directory / "fma_mix.hlsl"
    shader = directory / "fma_mix.cso"
    case = directory / "fma_mix.case"
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    for a_bits, b_bits, c_val, d_val in FMA_MIX_SETS:
        constants = bytearray(256)
        struct.pack_into("<2I2f", constants, 12, a_bits, b_bits, c_val, d_val)
        expected = fma_mix_reference(a_bits, b_bits, c_val, d_val)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
    # Mutations verified against Set 1 (modifiers, half selection, preservation) and Set 2 (clamp).
    for set_idx, old, new, label in (
        (0, "v_fma_mix_f32 v12, |s44|, -v19, v20", "v_fma_mix_f32 v12, s44, -v19, v20", "abs modifier"),
        (0, "v_fma_mix_f32 v12, |s44|, -v19, v20", "v_fma_mix_f32 v12, |s44|, v19, v20", "neg modifier"),
        (0, "op_sel:[1,1,0] op_sel_hi:[1,0,0]", "op_sel:[1,1,0]", "op_sel_hi half select"),
        (0, "v_fma_mixhi_f16 v22, -|v19|", "v_fma_mixlo_f16 v22, -|v19|", "mixhi half preservation"),
        (1, "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp",
         "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0]", "mixhi clamp"),
    ):
        a_bits, b_bits, c_val, d_val = FMA_MIX_SETS[set_idx]
        constants = bytearray(256)
        struct.pack_into("<2I2f", constants, 12, a_bits, b_bits, c_val, d_val)
        expected = fma_mix_reference(a_bits, b_bits, c_val, d_val)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        subprocess.run(compile_shader, check=True)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


def sopc_group() -> list[str]:
    body = [
        "s_mov_b32 s70, 0",
        "s_mov_b32 s71, 0",
        "s_mov_b32 s72, 1",
        "s_mov_b32 s73, 0",
    ]
    for op in (
        "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32",
        "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32",
    ):
        body.extend([f"{op} s44, s45", "s_cmov_b64 s[70:71], s[72:73]"])
    body.extend([
        "s_cmp_eq_u64 s[44:45], s[46:47]", "s_cmov_b64 s[70:71], s[72:73]",
        "s_cmp_lg_u64 s[44:45], s[46:47]", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp0_b32 s44, 5", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp1_b32 s44, 5", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp0_b64 s[44:45], 37", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp1_b64 s[44:45], 37", "s_cmov_b64 s[70:71], s[72:73]",
    ])
    return with_constants(body)


def vop2_integer_group() -> list[str]:
    return with_constants([
        "v_mov_b32_e32 v18, s45",
        "v_lshrrev_b32_e32 v12, s44, v18",
        "v_ashrrev_i32_e32 v13, s44, v18",
        "v_lshlrev_b32_e32 v14, s44, v18",
        "v_and_b32_e32 v15, s44, v18",
        "v_or_b32_e32 v12, s44, v18",
        "v_xor_b32_e32 v13, s44, v18",
        "v_add_nc_u32_e32 v14, s44, v18",
        "v_sub_nc_u32_e32 v15, s44, v18",
    ])
def pack_f16_pair(lo, hi):
    lo_b = struct.unpack("<H", struct.pack("<e", lo))[0]
    hi_b = struct.unpack("<H", struct.pack("<e", hi))[0]
    return lo_b | (hi_b << 16)


# VOP3P dot-product fixture covering all 7 RDNA 2 (gfx1030) dot-product forms:
# v_dot2_f32_f16 (op_sel, op_sel_hi, neg_lo, neg_hi, clamp), v_dot2_{i,u}32_{i,u}16,
# v_dot4_{i,u}32_{i,u}8, and v_dot8_{i,u}32_{i,u}4 (wrapping and saturating clamp).
DOT_BODY = [
    "v_mov_b32_e32 v18, s44",
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_dot2_f32_f16 v22, v18, 0x40003800, 0.5 op_sel:[1,0,0] op_sel_hi:[0,1,1] neg_lo:[1,0,1] neg_hi:[0,1,0]",
    "v_dot2_f32_f16 v12, v18, 0x3c003800, v22 clamp",
    "v_dot2_i32_i16 v23, v19, v20, v21 op_sel:[0,1,0] op_sel_hi:[1,0,1] clamp",
    "v_dot2_u32_u16 v24, v19, v20, v21 clamp",
    "v_dot4_i32_i8 v25, v19, v20, v21 clamp",
    "v_dot4_u32_u8 v26, v19, v20, v21",
    "v_dot8_i32_i4 v27, v19, v20, v21",
    "v_dot8_u32_u4 v28, v19, v20, v21 clamp",
    "v_dot8_u32_u4 v29, v23, 0x12341234, 0",
    "v_dot8_u32_u4 v29, v24, 0x43214321, v29",
    "v_cvt_f32_f16_e32 v13, v29",
    "v_mul_f32_e32 v13, 0x4b800000, v13",
    "v_dot8_u32_u4 v30, v25, 0x12341234, 0",
    "v_dot8_u32_u4 v30, v26, 0x43214321, v30",
    "v_cvt_f32_f16_e32 v14, v30",
    "v_mul_f32_e32 v14, 0x4b800000, v14",
    "v_dot8_u32_u4 v31, v27, 0x12341234, 0",
    "v_dot8_u32_u4 v31, v28, 0x43214321, v31",
    "v_cvt_f32_f16_e32 v15, v31",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]

DOT_SETS = (
    # Set 1: mixed signs, interior v_dot2_f32_f16 value, non-overflowing signed & unsigned integer dots.
    (pack_f16_pair(-0.5, 1.0), 0xfffc0005, 0x0003fffe, 0x00000010),
    # Set 2: positive signed saturation (INT32_MAX) and unsigned saturation (UINT32_MAX).
    (pack_f16_pair(2.0, 2.0), 0x7fff7fff, 0x7fff7fff, 0x7ffffff0),
    # Set 3: negative signed saturation (INT32_MIN) and high-end clamp on v_dot2_f32_f16.
    (pack_f16_pair(-2.0, 2.0), 0x8000ffff, 0x7fff7fff, 0x80000010),
    # Set 4: v_dot8_u32_u4 unsigned saturation (UINT32_MAX) vs unwrapped v_dot4_u32_u8 wraparound.
    (pack_f16_pair(0.5, -1.0), 0xffffffff, 0xffffffff, 0xfffffe00),
)


def dot_reference(f16_bits, x, y, acc):
    def sext(val, bits):
        mask = (1 << bits) - 1
        v = val & mask
        return v - (1 << bits) if v & (1 << (bits - 1)) else v

    def sat_i32(v):
        return max(-0x80000000, min(0x7fffffff, v)) & 0xffffffff

    def sat_u32(v):
        return min(0xffffffff, v) & 0xffffffff

    def nibble_dot(a, b):
        return sum(((a >> (4 * k)) & 0xf) * ((b >> (4 * k)) & 0xf) for k in range(8))

    a_lo = struct.unpack("<e", struct.pack("<H", f16_bits & 0xffff))[0]
    a_hi = struct.unpack("<e", struct.pack("<H", (f16_bits >> 16) & 0xffff))[0]
    v22 = single((-a_hi) * 0.5 + a_lo * (-2.0) + (-0.5))
    v12 = min(max(single(a_lo * 0.5 + a_hi * 1.0 + v22), 0.0), 1.0)

    acc_s = sext(acc, 32)
    v23 = sat_i32(sext(x, 16) * sext(y >> 16, 16) + sext(x >> 16, 16) * sext(y, 16) + acc_s)
    v24 = sat_u32((x & 0xffff) * (y & 0xffff) + ((x >> 16) & 0xffff) * ((y >> 16) & 0xffff) + acc)
    v25 = sat_i32(sum(sext(x >> (8 * k), 8) * sext(y >> (8 * k), 8) for k in range(4)) + acc_s)
    v26 = (sum(((x >> (8 * k)) & 0xff) * ((y >> (8 * k)) & 0xff) for k in range(4)) + acc) & 0xffffffff
    v27 = (sum(sext(x >> (4 * k), 4) * sext(y >> (4 * k), 4) for k in range(8)) + acc_s) & 0xffffffff
    v28 = sat_u32(nibble_dot(x, y) + acc)

    c1 = nibble_dot(v23, 0x12341234) + nibble_dot(v24, 0x43214321)
    c2 = nibble_dot(v25, 0x12341234) + nibble_dot(v26, 0x43214321)
    c3 = nibble_dot(v27, 0x12341234) + nibble_dot(v28, 0x43214321)
    assert c1 < 1024 and c2 < 1024 and c3 < 1024, (c1, c2, c3)
    return (half_rtz(v12), float(c1), float(c2), float(c3))


def check_dot_draw(directory, compiler, renderer, backend):
    program = with_constants(DOT_BODY)
    source = directory / "dot.hlsl"
    shader = directory / "dot.cso"
    case = directory / "dot.case"
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    for values in DOT_SETS:
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = dot_reference(*values)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
    for set_idx, old, new, label in (
        (0, "neg_lo:[1,0,1]", "neg_lo:[0,0,1]", "dot2_f32_f16 neg_lo"),
        (0, "v_dot2_i32_i16 v23", "v_dot2_u32_u16 v23", "dot2 signed vs unsigned"),
        (1, "v_dot4_i32_i8 v25, v19, v20, v21 clamp", "v_dot4_i32_i8 v25, v19, v20, v21", "dot4_i32_i8 clamp"),
        (0, "v_dot8_i32_i4 v27", "v_dot8_u32_u4 v27", "dot8 signed vs unsigned"),
        (3, "v_dot8_u32_u4 v28, v19, v20, v21 clamp", "v_dot8_u32_u4 v28, v19, v20, v21", "dot8_u32_u4 clamp"),
    ):
        values = DOT_SETS[set_idx]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = dot_reference(*values)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        subprocess.run(compile_shader, check=True)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")
# Packed 16-bit integer VOP3P fixture covering all 14 forms: add/sub/min/max (i16/u16),
# mul_lo_u16, mad (i16/u16), and lshlrev/lshrrev/ashrrev shifts, plus op_sel/op_sel_hi, clamp,
# and inline -1. Exports all four bytes of the packed accumulator as exact [0, 255] integers.
PK_INT16_BODY = [
    "v_mov_b32_e32 v18, s44",
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_pk_add_i16 v22, v18, v19 op_sel:[1,0] op_sel_hi:[0,1] clamp",
    "v_pk_add_u16 v23, v18, v19 clamp",
    "v_pk_sub_i16 v24, v18, v19 clamp",
    "v_pk_sub_u16 v25, v18, v19 clamp",
    "v_pk_mad_i16 v26, v18, v19, v20 clamp",
    "v_pk_mad_u16 v27, v18, v19, v20 clamp",
    "v_pk_min_i16 v28, v22, v24",
    "v_pk_max_i16 v29, v28, v26",
    "v_pk_min_u16 v28, v23, v25",
    "v_pk_max_u16 v28, v28, v27",
    "v_pk_mul_lo_u16 v29, v29, v28",
    "v_pk_add_i16 v29, v29, -1",
    "v_pk_lshlrev_b16 v22, v21, v29 op_sel:[0,1] op_sel_hi:[1,0]",
    "v_pk_lshrrev_b16 v23, v21, v28",
    "v_pk_ashrrev_i16 v24, v21, v18",
    "v_pk_add_u16 v30, v22, v23",
    "v_pk_sub_u16 v30, v30, v24",
    "v_pk_add_u16 v30, v30, v25",
    "v_pk_add_u16 v30, v30, v26",
    "v_pk_add_u16 v30, v30, v29",
    "v_pk_lshlrev_b16 v25, 8, v30 op_sel_hi:[0,1]",
    "v_pk_lshrrev_b16 v25, 8, v25 op_sel_hi:[0,1]",
    "v_pk_lshrrev_b16 v26, 8, v30 op_sel_hi:[0,1]",
    "v_pk_add_u16 v27, v25, 0 op_sel:[1,0] op_sel_hi:[0,0]",
    "v_pk_add_u16 v28, v26, 0 op_sel:[1,0] op_sel_hi:[0,0]",
    "v_cvt_f32_f16_e32 v12, v25",
    "v_mul_f32_e32 v12, 0x4b800000, v12",
    "v_cvt_f32_f16_e32 v13, v26",
    "v_mul_f32_e32 v13, 0x4b800000, v13",
    "v_cvt_f32_f16_e32 v14, v27",
    "v_mul_f32_e32 v14, 0x4b800000, v14",
    "v_cvt_f32_f16_e32 v15, v28",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]

PK_INT16_SETS = (
    # Set 1: mixed signed/unsigned lanes + shift counts >= 16 (testing & 15 masking and ashrrev sign fill).
    (0xf0080015, 0x0010fffe, 0x0007fff9, 0x00130012),
    # Set 2: saturates all 6 clamping ops (signed +/-32767/-32768 and unsigned 0/65535 boundaries).
    (0x70009000, 0x7000a000, 0x4000c000, 0x00020003),
    # Set 3: signed/unsigned extremes (INT16_MIN/MAX, UINT16_MAX) and max 4-bit shift count 15.
    (0x80007fff, 0xffff0001, 0x0002fffd, 0x000f0001),
)


def pk_int16_reference(w18, w19, w20, w21):
    def u16(w, hi):
        return (w >> 16) & 0xffff if hi else w & 0xffff

    def i16(w, hi):
        v = u16(w, hi)
        return v - 0x10000 if v >= 0x8000 else v

    def pack(lo, hi):
        return (lo & 0xffff) | ((hi & 0xffff) << 16)

    def sat_i(v):
        return max(-32768, min(32767, v)) & 0xffff

    def sat_u(v):
        return max(0, min(65535, v)) & 0xffff

    v22 = pack(sat_i(i16(w18, 1) + i16(w19, 0)), sat_i(i16(w18, 0) + i16(w19, 1)))
    v23 = pack(sat_u(u16(w18, 0) + u16(w19, 0)), sat_u(u16(w18, 1) + u16(w19, 1)))
    v24 = pack(sat_i(i16(w18, 0) - i16(w19, 0)), sat_i(i16(w18, 1) - i16(w19, 1)))
    v25 = pack(sat_u(u16(w18, 0) - u16(w19, 0)), sat_u(u16(w18, 1) - u16(w19, 1)))
    v26 = pack(sat_i(i16(w18, 0) * i16(w19, 0) + i16(w20, 0)), sat_i(i16(w18, 1) * i16(w19, 1) + i16(w20, 1)))
    v27 = pack(sat_u(u16(w18, 0) * u16(w19, 0) + u16(w20, 0)), sat_u(u16(w18, 1) * u16(w19, 1) + u16(w20, 1)))
    v28 = pack(min(i16(v22, 0), i16(v24, 0)), min(i16(v22, 1), i16(v24, 1)))
    v29 = pack(max(i16(v28, 0), i16(v26, 0)), max(i16(v28, 1), i16(v26, 1)))
    v28 = pack(min(u16(v23, 0), u16(v25, 0)), min(u16(v23, 1), u16(v25, 1)))
    v28 = pack(max(u16(v28, 0), u16(v27, 0)), max(u16(v28, 1), u16(v27, 1)))
    v29 = pack((u16(v29, 0) * u16(v28, 0)) & 0xffff, (u16(v29, 1) * u16(v28, 1)) & 0xffff)
    v29 = pack((i16(v29, 0) - 1) & 0xffff, (i16(v29, 1) - 1) & 0xffff)
    v22 = pack((u16(v29, 1) << (u16(w21, 0) & 15)) & 0xffff, (u16(v29, 0) << (u16(w21, 1) & 15)) & 0xffff)
    v23 = pack(u16(v28, 0) >> (u16(w21, 0) & 15), u16(v28, 1) >> (u16(w21, 1) & 15))
    v24 = pack((i16(w18, 0) >> (u16(w21, 0) & 15)) & 0xffff, (i16(w18, 1) >> (u16(w21, 1) & 15)) & 0xffff)
    v30 = pack((u16(v22, 0) + u16(v23, 0)) & 0xffff, (u16(v22, 1) + u16(v23, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) - u16(v24, 0)) & 0xffff, (u16(v30, 1) - u16(v24, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) + u16(v25, 0)) & 0xffff, (u16(v30, 1) + u16(v25, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) + u16(v26, 0)) & 0xffff, (u16(v30, 1) + u16(v26, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) + u16(v29, 0)) & 0xffff, (u16(v30, 1) + u16(v29, 1)) & 0xffff)
    return tuple(float((v30 >> (8 * k)) & 0xff) for k in range(4))


def check_pk_int16_draw(directory, compiler, renderer, backend):
    program = with_constants(PK_INT16_BODY)
    source = directory / "pk_int16.hlsl"
    shader = directory / "pk_int16.cso"
    case = directory / "pk_int16.case"
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    for values in PK_INT16_SETS:
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = pk_int16_reference(*values)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
    for set_idx, old, new, label in (
        (0, "v_pk_ashrrev_i16 v24, v21, v18", "v_pk_lshrrev_b16 v24, v21, v18", "ashrrev sign extension"),
        (0, "v_pk_min_i16 v28, v22, v24", "v_pk_min_u16 v28, v22, v24", "signed vs unsigned min"),
        (1, "v_pk_sub_u16 v25, v18, v19 clamp", "v_pk_sub_u16 v25, v18, v19", "sub_u16 clamp"),
        (0, "op_sel:[1,0] op_sel_hi:[0,1] clamp", "op_sel_hi:[1,1] clamp", "op_sel cross-lane swap"),
        (1, "v_pk_mad_i16 v26, v18, v19, v20 clamp", "v_pk_mad_i16 v26, v18, v19, v20", "mad_i16 clamp"),
    ):
        values = PK_INT16_SETS[set_idx]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = pk_int16_reference(*values)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        subprocess.run(compile_shader, check=True)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


def check_controls(directory):
    program = bitwise_program()
    for before, after in (("s44, v18", "s44, |v18|"), ("20, v19", "20, v19 clamp"),
                          ("0x00700000, v18, v19", "0x00700000, v18")):
        altered = [line.replace(before, after) for line in program]
        assert altered != program
        must_reject(translate, altered)
    program = buffer_color(True)
    assert "float2 main(" in translate_buffer_color(program)
    assert "SamplerState" not in translate_buffer_color(program)
    must_reject(lambda p: lower(p, 8, 24, None, 0), program)
    must_reject(lambda p: lower(p, 9, 24, None, 32), program)
    must_reject(lambda p: lower(p, 9, 24, None, 0, second_texture_offset=32), program)
    must_reject(lambda p: lower(p, 9, 24, [0] * 4, 0), program)
    must_reject(lambda p: lower(p, 20, 24, None), assembly(False))
    must_reject(translate_buffer_color, [s for s in program if s != "s_waitcnt lgkmcnt(0)"])
    must_reject(translate_buffer_color, [s.replace("off, off", "off, v26") for s in program])
    check_scheduling_hints()
    check_constant_descriptor_bounds()
    check_scalar_constant_reads()
    check_affine_coordinates()
    check_scalar_helpers_with_vector_arithmetic()
    check_adapter_skip(directory)
    check_scalar_pair_reads()
    # Whole-dword SDWA comparisons lower exactly as the e32 forms; byte and word selections are refused.
    for first in range(0, 16, 4):
        assert translate(predicate_group(first, sdwa=True)) == translate(predicate_group(first)), first
    must_reject(translate, with_constants(["v_sub_f32_e64 v18, s45, 0",
        "v_cmp_lt_f32_sdwa vcc_lo, s44, v18 src0_sel:WORD_1 src1_sel:DWORD"]))
    sopc_source = translate(sopc_group())
    assert set(SOPC_OPS) <= ACCEPTED
    for expected in (
        "asint(asuint(params[0].w)) == asint(asuint(params[1].x))",
        "asint(asuint(params[0].w)) < asint(asuint(params[1].x))",
        "asuint(params[0].w) == asuint(params[1].x)",
        "asuint(params[0].w) < asuint(params[1].x)",
        "asuint(params[0].w) == asuint(params[1].y) && asuint(params[1].x) == asuint(params[1].z)",
        "asuint(params[0].w) != asuint(params[1].y) || asuint(params[1].x) != asuint(params[1].z)",
        "((asuint(params[0].w) & (1u << (5u & 31u))) == 0u)",
        "((asuint(params[0].w) & (1u << (5u & 31u))) != 0u)",
        "(((37u & 32u) != 0u ? (asuint(params[1].x) & (1u << (37u & 31u))) : (asuint(params[0].w) & (1u << (37u & 31u)))) == 0u)",
        "(((37u & 32u) != 0u ? (asuint(params[1].x) & (1u << (37u & 31u))) : (asuint(params[0].w) & (1u << (37u & 31u)))) != 0u)",
    ):
        assert expected in sopc_source, expected
    for bad_sopc in (
        "s_cmp_gt_u64 s[44:45], s[46:47]",
        "s_cmp_eq_i64 s[44:45], s[46:47]",
        "s_cmp_eq_u64 s[65:66], s[46:47]",
        "s_cmp_eq_u32 s106, s44",
        "s_cmp_eq_u64 s[105:106], s[46:47]",
    ):
        must_reject(translate, with_constants([bad_sopc]))
    vop2_source = translate(vop2_integer_group())
    assert set(VOP2_INT_OPS) <= ACCEPTED
    for expected in (
        "(asuint(alu20) >> (asuint(params[0].w) & 31u))",
        "asuint(asint(asuint(alu20)) >> (asuint(params[0].w) & 31u))",
        "(asuint(alu20) << (asuint(params[0].w) & 31u))",
        "(asuint(params[0].w) & asuint(alu20))",
        "(asuint(params[0].w) | asuint(alu20))",
        "(asuint(params[0].w) ^ asuint(alu20))",
        "(asuint(params[0].w) + asuint(alu20))",
        "(asuint(params[0].w) - asuint(alu20))",
    ):
        assert expected in vop2_source, expected
    for bad_vop2 in (
        "v_add_nc_u32_e64 v12, s44, v18",
        "v_sub_nc_u32_e64 v12, s44, v18",
        "v_sub_nc_u32_e32 v12, abs(v18), v18",
        "v_and_b32_e32 v12, -v18, v18",
        "v_and_b32_e32 v12, abs(v18), v18",
        "v_and_b32_e32 v12, v106, v18",
        "v_and_b32_e32 v12, s44, v106",
    ):
        must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", bad_vop2]))
    rgb = rgb_sample()
    translate(rgb)
    for old, new in (("v[12:14]", "v[12:15]"), ("dmask:0x7", "dmask:0xb")):
        must_reject(translate, [line.replace(old, new) for line in rgb])
    wait = rgb.index("s_waitcnt vmcnt(0)")
    must_reject(translate, rgb[:wait] + rgb[wait+1:])
    must_reject(translate, rgb[:wait] + ["v_mov_b32_e32 v14, 1.0"] + rgb[wait:])
    translate(rgb[:wait] + ["v_mov_b32_e32 v15, 1.0"] + rgb[wait:])
    # v_cvt_f32_f16 widens the low half through widen16 and applies modifiers after widening.
    body = translate(with_constants(HALF_BODY)).split("float4 main", 1)[1]
    assert body.count("widen16(asuint(") == 4 and "(-abs(widen16(" in body, body
    must_reject(translate, with_constants(["v_cvt_f32_f16_e32 v12, 1.0"]))
    must_reject(translate, with_constants(["v_cvt_f32_f16_sdwa v12, s44 src0_sel:WORD_1"]))
    check_scalar_controls()
    check_sopk_controls()
    # v_cvt_f16_f32 keeps the destination's high half and narrows the modified source through narrow16.
    body = translate(with_constants(NARROW_BODY)).split("float4 main", 1)[1]
    assert body.count("narrow16(") == 3 and body.count("& 0xffff0000u) | narrow16(") == 3, body
    assert "narrow16((-abs(" in body, body
    must_reject(translate, with_constants(["v_cvt_f16_f32_e64 v12, s44 clamp"]))
    must_reject(translate, with_constants(["v_cvt_f16_f32_sdwa v12, s44 dst_sel:WORD_1"]))
    # An inline integer keeps its bit pattern, and a destination holding interpolation state is refused.
    body = translate(with_constants(["v_cvt_f16_f32_e32 v12, -1", "v_cvt_f16_f32_e64 v13, -|0.5|"])).split("float4 main", 1)[1]
    assert "narrow16(asfloat(4294967295u))" in body and "narrow16((-abs(0.5f)))" in body, body
    for register in ("v0", "v1", "v8", "v9"):
        must_reject(translate, with_constants([f"v_cvt_f16_f32_e32 {register}, s44"]))
    archive = directory / "input.bin"
    code = bytes.fromhex("000080bf 000081bf")  # s_nop 0 followed by s_endpgm
    check_container_selector(directory, code)
    record = envelope(code)
    archive.write_bytes(b"another_fixture\0" + record)
    assert native_program(archive, "fixture")[1] == code
    long_name = b"synthetic_pixel_with_a_printable_length"
    archive.write_bytes(envelope(code, long_name))
    assert native_program(archive, long_name.decode())[1] == code
    for invalid in (record[:-1], record + b"\0" + record, envelope(code + code)):
        archive.write_bytes(invalid)
        must_reject(native_program, archive, "fixture")

    partial = dual_program("partial")
    wait = partial.index("s_waitcnt vmcnt(1)")
    # Keep the early read of A but issue B first. Now A is still pending.
    reversed_samples = partial.copy()
    reversed_samples[wait - 2:wait] = reversed(reversed_samples[wait - 2:wait])
    must_reject(translate_dual, reversed_samples)
    for count in (2, 63):
        changed = partial.copy()
        changed[wait] = f"s_waitcnt vmcnt({count})"
        must_reject(translate_dual, changed)
    for early_use in ("v_mov_b32_e32 v24, v20", "v_mov_b32_e32 v20, v12",
                      "v_mul_legacy_f32_e32 v12, 1.0, v20",
                      "v_cvt_pkrtz_f16_f32_e32 v24, v12, v20"):
        changed = partial.copy()
        changed.insert(wait + 1, early_use)
        must_reject(translate_dual, changed)
    # Each of B's four destinations remains protected from a constant write.
    for reg in range(20, 24):
        changed = partial.copy()
        changed.insert(wait + 1, f"v_mov_b32_e32 v{reg}, 0")
        must_reject(translate_dual, changed)
    # Packing A is legal here even though B is outstanding.
    translate_dual(partial[:wait + 1] + ["v_cvt_pkrtz_f16_f32_e32 v24, v12, v13"] + partial[wait + 1:])
    # A larger legal wait after draining does not invalidate completed samples.
    end_wait = partial.index("s_waitcnt vmcnt(0)")
    drained = partial[:end_wait + 1] + ["s_waitcnt vmcnt(63)"] + partial[end_wait + 1:]
    translate_dual(drained)
    must_reject(translate_dual, [op.replace("vmcnt(63)", "vmcnt(64)") for op in drained])

    # LLVM 15 prints "exp mrt0 v16, ..."; newer LLVM prints "exp mrt0, v16, ...".
    for program in (assembly(False), assembly(True), curve(), shaping(), arithmetic(), three_way(), rounding()):
        modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
        assert modern != program
        assert translate(modern) == translate(program)
        must_reject(translate, [line.replace("exp mrt0 ", "exp mrt0,, ") for line in program])

    # The literal of v_fmaak_f32 is its last operand and that of v_fmamk_f32 its second; a register there is refused.
    body = translate(with_constants(LITERAL_FMA_BODY)).split("float4 main", 1)[1]
    assert body.count("mad(") == 4 and "asfloat(1048576000u)" in body, body
    must_reject(translate, with_constants(["v_fmaak_f32 v12, s44, v13, v14"]))
    must_reject(translate, with_constants(["v_fmamk_f32 v12, s44, v13, 0x3e800000"]))
    must_reject(translate, with_constants(["v_fmaak_f32 v12, s44, 0x3e800000"]))
    for old, new in (("v_sqrt_f32_e32 v18, v18", "v_sqrt_f32_e32 v18, v18, v18"),
                     ("v_add_f32_e32 v19, s44, v13", "v_add_f32_e32 v19, v13")):
        must_reject(translate, [line.replace(old, new) for line in shaping()])
    check_three_way_templates()
    source = translate(three_way())
    for expected in ("max(min(params[1].z, sample16.y), min(max(params[1].z, sample16.y), sample16.x))",
                     "max(max(sample16.x, sample16.z), sample16.y)",
                     "max(min(sample16.x, params[0].w), min(max(sample16.x, params[0].w), params[1].x))",
                     "min(min(abs(params[1].y), sample16.y), alu"):
        assert expected in source, expected
    for old, new in (("v_med3_f32 v12, v12, s44, s45", "v_med3_f32 v12, v12, s44"),
                     ("v_med3_f32 v15, s47", "v_med3_i32 v15, |s47|"),
                     ("v_max3_f32 v14, v12", "v_max3_u32 v14, |v12|"),
                     ("v_min3_f32 v13, |s46|", "v_min3_f16 v13, |s46|")):
        changed = [line.replace(old, new) for line in three_way()]
        assert changed != three_way()
        must_reject(translate, changed)

    program = truncate_values()
    source = translate(program)
    assert "trunc(params[1].x)" in source
    must_reject(translate, [line.replace("v13, s45", "v13, s45, v12") for line in program])
    wait = program.index("s_waitcnt vmcnt(0)")
    program.remove("v_trunc_f32_e32 v13, s45")
    program.insert(wait, "v_trunc_f32_e32 v13, s45")
    must_reject(translate, program)

    # v_cmp_class_f32 classifies the raw encoding; a mask with modifiers or a non-numeric operand is refused.
    body = translate(with_constants(CLASS_BODY)).split("float4 main", 1)[1]
    assert body.count("class32(") == 4 and "& 512u" in body and "asuint(alu" in body, body
    must_reject(translate, with_constants(["v_cmp_class_f32_e64 vcc_lo, s44, -s45"]))
    must_reject(translate, with_constants(["v_cmp_class_f32_e64 vcc_lo, s44"]))
    program = sopk_scalar()
    source = translate(program)
    assert "(asuint(params[0].w) * 3u)" in source and "(asuint(params[1].x) + 4294967294u)" in source

    scc_program = with_constants([
        "s_mov_b32 s52, 0", "s_mov_b32 s53, 0", "s_mov_b32 s54, 0", "s_mov_b32 s55, 0",
        "s_mov_b32 s50, s44", "s_addk_i32 s50, 0xfffe",
        "s_mov_b32 s70, 0", "s_mov_b32 s71, 0", "s_mov_b32 s74, 1", "s_mov_b32 s75, 0",
        "s_cmov_b64 s[70:71], s[74:75]",
    ])

    assert "0x80000000u" in translate(scc_program)
    cmov = scc_program.index("s_cmov_b64 s[70:71], s[74:75]")
    after_mul = scc_program[:cmov] + ["s_mulk_i32 s50, 0x3"] + scc_program[cmov:]
    assert "0x80000000u" in translate(after_mul)

    for old, new in (("s_mulk_i32 s50, 0x3", "s_mulk_i32 s50, 0x10000"),
                     ("s_addk_i32 s51, 0xfffe", "s_addk_i32 s51, 0x10001"),
                     ("s_mov_b32 s50, s44", "s_mov_b32 s50, s28"),
                     ("s_mulk_i32 s50, 0x3", "s_mulk_i32 v50, 0x3"),
                     ("s_addk_i32 s51, 0xfffe", "s_addk_i32 s51, v45")):
        changed = [line.replace(old, new) for line in program]
        assert changed != program
        must_reject(translate, changed)

    wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    must_reject(translate, program[:wait] + ["s_mulk_i32 s44, 0x3"] + program[wait:])
    source = translate(rounding())
    for expected in ("= ceil(alu", "= round(alu", "= round(params[1].x);", "= ceil((-params[1].y));"):
        assert expected in source, expected
    # Neighbouring forms: other widths, output modifiers, a second source and the IEEE-flag reciprocal.
    for old, new in (("v_ceil_f32_e32 v12, v18", "v_ceil_f16_e32 v12, v18"),
                     ("v_ceil_f32_e32 v12, v18", "v_ceil_f64_e32 v[12:13], v[18:19]"),
                     ("v_rndne_f32_e32 v13, v19", "v_rndne_f16_e32 v13, v19"),
                     ("v_rndne_f32_e32 v13, v19", "v_rndne_f32_e32 v13, v19, v18"),
                     ("v_rndne_f32_e64 v14, s45", "v_rndne_f32_e64 v14, s45 mul:2"),
                     ("v_ceil_f32_e64 v15, -s46", "v_ceil_f32_e64 v15, -s46 div:2"),
                     ("v_ceil_f32_e64 v15, -s46", "v_rcp_iflag_f32_e64 v15, -s46, v12"),
                     ("v_ceil_f32_e64 v15, -s46", "v_ceil_f32_e64 v15, -s28")):
        changed = [line.replace(old, new) for line in rounding()]
        assert changed != rounding()
        must_reject(translate, changed)

    program = absolute_value()
    alternate = [line.replace("|v12|", "abs(v12)") for line in program]
    assert translate(program) == translate(alternate)
    # Each cube operation reads the three coordinates once and needs all three operands.
    body = translate(with_constants(CUBE_BODY)).split("float4 main", 1)[1]
    assert body.count("precise float3 cube") == 4 and body.count("* 2.0f)") == 1, body
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", "v_cubeid_f32 v12, s44, v18"]))
    # SDWA moves select and place fields; UNUSED_SEXT, byte selections for half widening and
    # non-canonical SDWA text are refused.
    body = translate(with_constants(SDWA_PRESERVE_BODY + SDWA_PAD_BODY)).split("float4 main", 1)[1]
    assert body.count("& 0xffff0000u) | ") == 2 and body.count("asint(") == 1 and body.count("widen16(") == 2, body
    must_reject(translate, with_constants(["v_mov_b32_sdwa v12, s44 dst_sel:BYTE_0 dst_unused:UNUSED_SEXT src0_sel:WORD_1"]))
    must_reject(translate, with_constants(["v_cvt_f32_f16_sdwa v12, s44 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1"]))
    must_reject(translate, with_constants(["v_mov_b32_sdwa v12, s44 src0_sel:WORD_1"]))
    # UNUSED_PRESERVE keeps bits only from a tracked numeric value; DWORD and padded writes may replace anything.
    for register in ("v0", "v1", "v8", "v9", "v20"):
        must_reject(translate, with_constants([f"v_mov_b32_sdwa {register}, s44 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1"]))
    body = translate(with_constants(["v_mov_b32_sdwa v12, -0.5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD",
                                     "v_mov_b32_sdwa v20, s44 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_0",
                                     "v_mov_b32_sdwa v13, v20 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD"])).split("float4 main", 1)[1]
    assert "asuint((-0.5f))" in body, body

    program = assembly(False)
    for old, new in (("s_mov_b32 s21, 0", "s_mov_b32 s21, 1"),
                     ("s_mov_b32 s40, 1", "s_mov_b32 s40, 8"),
                     ("attr0.y", "attr1.y"),
                     ("s_wqm_b64 exec, exec", "s_branch 0")):
        must_reject(translate, [line.replace(old, new) for line in program])
    must_reject(translate, program[:-1])
    for wait in ("s_waitcnt vmcnt(0)", "s_waitcnt lgkmcnt(0)"):
        must_reject(translate, [line for line in program if line != wait])
    for exp_inst in ("s_waitcnt_expcnt null, 0x0", "s_waitcnt_expcnt null, 0x3", "s_waitcnt_expcnt null, 0",
                     "s_waitcnt_expcnt s0, 0x0", "s_waitcnt_expcnt 0x3", "s_waitcnt_expcnt 0",
                     "s_waitcnt expcnt(0)", "s_waitcnt expcnt(7)"):
        assert "return float4(" in translate([exp_inst] + program)
        assert translate(program[:-1] + [exp_inst, program[-1]]) == translate(program)
    must_reject(translate, program + ["s_waitcnt_expcnt null, 0"])
    must_reject(translate, ["s_waitcnt_expcnt null, 0x8"] + program)
    must_reject(translate, ["s_waitcnt_expcnt null, 8"] + program)
    must_reject(translate, ["s_waitcnt_expcnt 0x8"] + program)
    must_reject(translate, ["s_waitcnt_expcnt invalid"] + program)
    must_reject(translate, ["s_waitcnt_expcnt null, invalid"] + program)
    must_reject(translate, ["s_waitcnt expcnt(8)"] + program)

    wait = program.index("s_waitcnt vmcnt(0)")
    overwrite = "v_floor_f32_e32 v13, 1.0"
    must_reject(translate, program[:wait] + [overwrite] + program[wait:])
    free = translate(program[:wait] + ["v_floor_f32_e32 v24, 1.0"] + program[wait:])
    late = translate(program[:wait+1] + [overwrite] + program[wait+1:])
    assert "export16(sample15.y)" in free
    assert "export16(alu17)" in late

    program = arithmetic()
    source = translate(program)
    for expected in ("= sample16.x;", "= (params[2].x * alu20);", "= saturate((alu21 + params[2].y));",
                     "= params[2].z;", "= (alu23 - sample16.y);", "= saturate((alu25 - params[2].w));",
                     "= asfloat(1048576000u);", "= alu29;"):
        assert expected in source, expected
    # A move is a bit copy: LLVM prints integer inline constants as integers.
    for operand, expected in (("1", "asfloat(1u)"), ("-4", "asfloat(4294967292u)"),
                              ("-0.5", "(-0.5f)"), ("0.15915494", "0.15915494f"), ("s40", "asfloat(1u)")):
        changed = [line.replace("v20, 0x3e800000", f"v20, {operand}") for line in program]
        assert f"alu28 = {expected};" in translate(changed), operand
    for old, new in (("v18, v12", "v18, v0"), ("v18, v12", "v18, v8"), ("v18, v12", "v18, v30"),
                     ("v19, s46", "v19, s28"), ("v19, s46", "v19, -s46"), ("v19, s46", "v19, s46 clamp"),
                     ("v18, v12", "v18, v12, v13"), ("v14, -0.5, v14", "v14, v14"),
                     ("v19, v13, v19", "v19, v13, v19, v18"), ("s45 clamp", "s45 clamp mul:2")):
        changed = [line.replace(old, new) for line in program]
        assert changed != program
        must_reject(translate, changed)
    first = program.index("v_mov_b32_e32 v18, v12")
    early = program[:first-1] + [program[first], program[first-1]] + program[first+1:]
    assert early[first] == "s_waitcnt vmcnt(0)"
    must_reject(translate, early)  # copies the sample before the wait
    # Until vmcnt(0), the sample may still fill v12 to v15, so a move into one of them has no defined
    # order with it. The same move into a free register, or into v12 after the wait, is accepted.
    wait = program.index("s_waitcnt vmcnt(0)")
    for register in ("v12", "v13", "v14", "v15"):
        must_reject(translate, program[:wait] + [f"v_mov_b32_e32 {register}, 1.0"] + program[wait:])
    must_reject(translate, program[:wait] + ["v_ceil_f32_e32 v12, 1.0"] + program[wait:])
    free = translate(program[:wait] + ["v_mov_b32_e32 v21, 1.0"] + program[wait:])
    assert free.index("tex.Sample") < free.index("= 1.0f;") < free.index("= sample16.x;")
    late = translate(program[:wait+1] + ["v_mov_b32_e32 v12, 1.0"] + program[wait+1:])
    assert "= 1.0f;" in late and "= sample16.x;" not in late
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v17, v14, v15")
    must_reject(translate, program[:pos] + ["v_mov_b32_e32 v20, v16"] + program[pos:])
    pos = program.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, program[:pos] + ["v_mov_b32_e32 v20, 1.0"] + program[pos:])
    pos = program.index("v_interp_p1_f32_e32 v9, v0, attr0.y")
    partial = program[:pos] + ["v_interp_p1_f32_e32 v10, v0, attr0.x"] + program[pos:]
    translate(partial)  # an unpaired first phase is allowed, but its partial value is not numeric
    must_reject(translate, [line.replace("v18, v12", "v18, v10") for line in partial])
    pos = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0x20"))
    must_reject(translate, program[:pos] + ["v_mov_b32_e64 v19, s46"] + program[pos:])

    for old, new in (("0x40", "0x48"), ("0xc", "0xfe"), ("0xc", "0x100"),
                     ("s_waitcnt lgkmcnt(0)", "s_nop 0"), ("s[44:47], 0xc", "s[40:43], 0xc")):
        must_reject(translate, [line.replace(old, new) for line in curve()])
    source = translate(select())
    assert translate(select(True)) == source
    for expected in ("(sample16.x >= sample16.y || isnan(sample16.x) || isnan(sample16.y))",
                     "sample16.x == sample16.y",
                     "params[0].w < sample16.x", "sample16.y >= params[1].x",
                     "? 1.0f : asfloat(0u);", "? sample16.w : params[1].z;"):
        assert expected in source, expected
    for wave64 in (False, True):
        program = select(wave64)
        mask = "s[52:53]" if wave64 else "s52"
        absolute_compare = [line.replace("v13, s45", "v13, |v12|") for line in program]
        assert "abs(" in translate(absolute_compare)
        pos = program.index(f"v_cndmask_b32_e64 v13, v13, s46, {mask}")
        must_reject(translate, program[:pos] + ["s_mov_b32 s52, 0"] + program[pos:])
        for old, new in ((f"s46, {mask}", "s46, s[53:54]" if wave64 else "s46, s50"),
                         ("v_cmp_eq_f32_e32", "v_cmp_eq_f64_e32"),
                         ("v_cmp_ge_f32_e64", "v_cmpx_ge_f32_e64"),
                         (f"v_cmp_ge_f32_e64 {mask}", "v_cmp_ge_f32_e64 " + ("s[56:57]" if wave64 else "s56")),
                         ("v13, s45", "v13, |v12"),
                         ("v15, s47, v15", "v15, s50, v15")):
            changed = [line.replace(old, new) for line in program]
            assert changed != program
            must_reject(translate, changed)
        pos = program.index("s_waitcnt vmcnt(0)")
        must_reject(translate, program[:pos] + program[pos+1:])
    # v_fract_f32 goes through the clamped helper, never through a bare D3D frac.
    body = translate(shaping()).split("float4 main", 1)[1]
    assert "fract32(" in body and "frac(" not in body, body
    # v_mad_f32 and v_mac_f32 use the unfused, denormal-flushing helper, never the fma lowering.
    body = translate(with_constants(MAD_BODY)).split("float4 main", 1)[1]
    assert body.count("mad32(") == 4 and " mad(" not in body, body
    must_reject(translate, with_constants(["v_mad_f32 v12, s44, s45"]))
    must_reject(translate, with_constants(["v_mac_f32_e32 v14, s44, v15, v14"]))

    # 64-bit scalar ALU and 32-bit BFE controls.
    program = scalar64()
    source = translate(program)
    for expected in ("uint2 s_lshl_b64(", "uint2 s_lshr_b64(", "uint2 s_ashr_i64(",
                     "uint2 s_bfm_b64(", "uint s_bfe_u32(", "uint s_bfe_i32(",
                     "reversebits(", "asfloat(salu"):
        assert expected in source, expected
    # Existing programs without shifts/bfm/bfe emit none of the scalar helper functions.
    assert "uint2 s_lshl_b64(" not in translate(assembly(False))
    # Verify 32-bit hex literal sign-extension for s_ashr_i64 vs zero-extension for b64 ops.
    ashr_pos = program.index("s_ashr_i64 s[72:73], s[52:53], s56")
    signed_lit = translate(program[:ashr_pos] + ["s_ashr_i64 s[72:73], 0x80000000, 4"] + program[ashr_pos+1:])
    assert "s_ashr_i64(uint2(2147483648u, 4294967295u), 4u)" in signed_lit
    unsigned_lit = translate(program[:ashr_pos] + ["s_and_b64 s[72:73], s[52:53], 0x80000000"] + program[ashr_pos+1:])
    assert "2147483648u" in unsigned_lit and "0u)" in unsigned_lit
    # Verify all 10 SCC observation bits flip between the two constant sets.
    refs = []
    for values in SCALAR64_SETS:
        constants = bytearray(256)
        struct.pack_into("<12I", constants, 16, *values)
        refs.append(scalar64_reference(program, struct.unpack("<64I", constants)))
    assert (refs[0][1] ^ refs[1][1]) == 0x3ff, (bin(refs[0][1]), bin(refs[1][1]))
    # Negative controls: exec save/restore, #257 ops, out-of-scope 64-bit ops, invalid operands and SCC rules.
    body_start = program.index("s_and_b64 s[72:73], s[52:53], s[54:55]")
    for bad in (
        "s_and_saveexec_b64 s[70:71], s[52:53]",
        "s_or_saveexec_b64 s[70:71], s[52:53]",
        "s_andn2_saveexec_b64 s[70:71], s[52:53]",
        "s_or_b64 exec, exec, s[52:53]",
        "s_and_b64 vcc, s[52:53], s[54:55]",
        "s_cmov_b64 exec, s[52:53]",
        "s_not_b64 vcc, s[52:53]",
        "s_and_b64 s[70:71], exec, s[52:53]",
        "s_xor_b64 s[70:71], vcc, s[52:53]",
        "s_cmov_b64 s[70:71], s[52:53]",  # before any SCC write
        "s_and_b64 s[53:54], s[52:53], s[54:55]",  # unaligned pair
        "s_and_b64 s[106:107], s[52:53], s[54:55]",  # out-of-range SGPR pair
        "s_bfe_u32 s106, s60, 0x80004",  # out-of-range SGPR
        "s_bfe_u32 s70, v12, 0x80004",  # VGPR source
        "s_bfe_u32 s70, m0, 0x80004",  # m0 source
        "s_and_b64 s[70:71], s[28:29], s[52:53]",  # texture descriptor source
        "s_and_b64 s[70:71], s[44:45], s[52:53]",  # CBV descriptor source
        "s_and_b64 s[70:71], s[50:51], s[52:53]",  # saved exec source
        "s_and_b64 s[70:71], s[20:21], s[52:53]",  # texture_ptr source
        "s_and_b64 s[70:71], s[90:91], s[52:53]",  # unset source
        "s_cselect_b64 s[70:71], s[52:53], s[54:55]",
        "s_bfe_u64 s[70:71], s[52:53], s61",
        "s_bfe_i64 s[70:71], s[52:53], s61",
        "s_bfm_b32 s70, 4, 8",
        "s_cselect_b32 s70, s56, s57",
    ):
        must_reject(translate, program[:body_start] + [bad] + program[body_start:])
    for supported in (
        "s_add_u32 s70, s56, s57",
        "s_and_b32 s70, s56, s57",
        "s_not_b32 s70, s56",
        "s_cmp_eq_i32 s56, s57",
    ):
        translate(program[:body_start] + [supported] + program[body_start:])
    # s_cmov_b64 rejects unset or non-numeric destination pairs even after SCC is set.
    after_and = body_start + 1
    for bad_dst in ("s[90:91]", "s[44:45]", "s[50:51]", "s[28:29]"):
        must_reject(translate, program[:after_and] + [f"s_cmov_b64 {bad_dst}, s[52:53]"] + program[after_and:])
    # Pending scalar load read and write overlap must be rejected.
    load_wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[60:63], s[44:47], 0x30"))
    for early in ("s_and_b64 s[70:71], s[52:53], s[54:55]", "s_not_b64 s[52:53], 0", "s_bfe_u32 s60, 1, 0x80004"):
        must_reject(translate, program[:load_wait] + [early] + program[load_wait:])
    # s_wqm_b64 and exec-restoring s_and_b64 clear tracked SCC.
    wqm_pos = program.index("s_wqm_b64 exec, exec")
    exec_restore_pos = program.index("s_and_b64 exec, exec, s[50:51]")
    prefix = ["s_mov_b32 s90, 0", "s_mov_b32 s91, 0", "s_not_b64 s[92:93], 0"]
    must_reject(translate, program[:wqm_pos] + prefix + [program[wqm_pos], "s_cmov_b64 s[90:91], s[92:93]"] + program[wqm_pos+1:])
    must_reject(translate, program[:exec_restore_pos] + prefix + [program[exec_restore_pos], "s_cmov_b64 s[90:91], s[92:93]"] + program[exec_restore_pos+1:])
    # A scalar result cannot supply a sampler word, texture address high word, or v_cndmask condition.
    sample_pos = program.index("image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D")
    must_reject(translate, program[:sample_pos] + ["s_bfe_u32 s40, 1, 0x10000"] + program[sample_pos:])
    sel = select(True)
    cnd_pos = sel.index("v_cndmask_b32_e64 v13, v13, s46, s[52:53]")
    must_reject(translate, sel[:cnd_pos] + ["s_not_b64 s[52:53], 0"] + sel[cnd_pos:])

    # 32-bit integer comparisons: signed vs unsigned casts, wave64 spelling, inline constants, and rejections.
    i32_source = translate(int_predicate_group("i32", "e32", 0))
    u32_source = translate(int_predicate_group("u32", "e64", 4))
    assert "asint(params[0].w) < asint(alu20)" in i32_source, i32_source
    assert "asuint(params[0].w) >= asuint(alu20)" in u32_source, u32_source
    assert translate(int_predicate_group("i32", "e32", 0, wave64=True)) == i32_source
    assert translate(int_predicate_group("u32", "e64", 4, wave64=True)) == u32_source
    for kind, token, expected in (
        ("i32", "-1", "asint(4294967295u)"),
        ("u32", "-1", "4294967295u"),
        ("i32", "0x80000000", "asint(2147483648u)"),
        ("u32", "0x80000000", "2147483648u"),
        ("i32", "1.0", "asint(1065353216u)"),
        ("u32", "1.0", "1065353216u"),
    ):
        prog = with_constants(["v_mov_b32_e32 v18, s45",
                               f"v_cmp_lt_{kind}_e32 vcc_lo, {token}, v18",
                               "v_cndmask_b32_e32 v12, 0, 1.0, vcc_lo"])
        assert f"bool cmp21 = {expected} < " in translate(prog), (kind, token)
    base_int = int_predicate_group("i32", "e64", 0)
    for bad in (
        "v_cmp_lt_i32_e64 s52, -s44, v18",
        "v_cmp_lt_i32_e64 s52, |s44|, v18",
        "v_cmp_lt_i32_e64 s52, abs(s44), v18",
        "v_cmp_lt_i32_e64 s52, s44, -v18",
        "v_cmp_lt_i32_e64 s52, s44, v18 clamp",
        "v_cmpx_lt_i32_e64 s52, s44, v18",
        "v_cmp_lt_i32_e64 s52, 0x100000000, v18",
    ):
        changed = [line.replace("v_cmp_lt_i32_e64 s52, s44, v18", bad) for line in base_int]
        assert changed != base_int
        must_reject(translate, changed)
    pos = base_int.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_int[:pos] + ["v_cmp_eq_u32_e32 vcc_lo, v12, v13"] + base_int[pos:])
    load = next(i for i, line in enumerate(base_int) if line.startswith("s_buffer_load_dwordx4"))
    pos = base_int.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_int[:pos] + ["v_cmp_eq_i32_e32 vcc_lo, s44, 0"] + base_int[pos:])
    pos = base_int.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_int[:pos] + ["v_cmp_eq_i32_e32 vcc_lo, 0, 0"] + base_int[pos:])

    # 32-bit integer min/max/min3/max3/med3 controls and rejections.
    pair_src = translate(int_minmax_pair("e32"))
    assert "asfloat(min(asint(params[0].w), asint(alu20)))" in pair_src, pair_src
    assert "asfloat(max(asuint(params[0].w), asuint(alu20)))" in pair_src, pair_src
    assert translate(int_minmax_pair("e64")) == pair_src
    tri_i32 = translate(int_minmax_triple("i32"))
    tri_u32 = translate(int_minmax_triple("u32"))
    assert "asfloat(min(min(asint(params[0].w), asint(alu20)), asint(alu21)))" in tri_i32, tri_i32
    assert tri_i32.count("int3 med") == 2 and tri_u32.count("uint3 med") == 2
    for bad_pair in (
        "v_min_i32_e64 v12, -s44, v18",
        "v_min_i32_e64 v12, |s44|, v18",
        "v_min_i32_e64 v12, s44, v18 clamp",
        "v_min_i32_e64 v12, s44",
        "v_min_i32_e64 v12, 0x100000000, v18",
    ):
        changed = [line.replace("v_min_i32_e64 v12, s44, v18", bad_pair) for line in int_minmax_pair("e64")]
        assert changed != int_minmax_pair("e64")
        must_reject(translate, changed)
    for bad_tri in (
        "v_med3_i32 v14, -s44, v18, v19",
        "v_med3_i32 v14, s44, |v18|, v19",
        "v_med3_i32 v14, s44, v18, v19 clamp",
        "v_med3_i32 v14, s44, v18",
    ):
        changed = [line.replace("v_med3_i32 v14, s44, v18, v19", bad_tri) for line in int_minmax_triple("i32")]
        assert changed != int_minmax_triple("i32")
        must_reject(translate, changed)
    base_pair = int_minmax_pair("e32")
    wait = base_pair.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_pair[:wait] + ["v_min_i32_e32 v12, 0, 1"] + base_pair[wait:])
    must_reject(translate, base_pair[:wait] + ["v_med3_u32 v13, 0, 1, 2"] + base_pair[wait:])
    load = next(i for i, line in enumerate(base_pair) if line.startswith("s_buffer_load_dwordx4"))
    lgkm = base_pair.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_pair[:lgkm] + ["v_max_u32_e32 v20, s44, 0"] + base_pair[lgkm:])
    wqm = base_pair.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_pair[:wqm] + ["v_min_u32_e32 v20, 0, 1"] + base_pair[wqm:])

    # v_frexp_mant_f32 and v_ldexp_f32 controls and rejections.
    frexp_body = translate(frexp_mant_values()).split("float4 main", 1)[1]
    assert "frexp_mant32(params[0].w)" in frexp_body, frexp_body
    assert "frexp_mant32((-abs(params[1].x)))" in frexp_body, frexp_body
    assert "saturate(frexp_mant32(params[1].z))" in frexp_body, frexp_body
    ldexp_body = translate(ldexp_values()).split("float4 main", 1)[1]
    assert "ldexp32(alu20, asint(params[1].x))" in ldexp_body, ldexp_body
    assert "ldexp32((-abs(params[0].w)), asint(2u))" in ldexp_body, ldexp_body
    assert "ldexp32(params[1].y, asint(4294967294u))" in ldexp_body, ldexp_body
    assert "saturate(ldexp32(params[1].z, asint(3u)))" in ldexp_body, ldexp_body
    for bad in (
        "v_ldexp_f32 v12, v18, -s45",
        "v_ldexp_f32 v12, v18, |s45|",
        "v_ldexp_f32 v12, v18, abs(s45)",
        "v_ldexp_f32 v12, v18",
        "v_ldexp_f32 v12, v18, 0x100000000",
    ):
        changed = [line.replace("v_ldexp_f32 v12, v18, s45", bad) for line in ldexp_values()]
        assert changed != ldexp_values()
        must_reject(translate, changed)
    base_ld = ldexp_values()
    wait = base_ld.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_ld[:wait] + ["v_frexp_mant_f32_e32 v12, 1.0"] + base_ld[wait:])
    must_reject(translate, base_ld[:wait] + ["v_ldexp_f32 v13, 1.0, 2"] + base_ld[wait:])
    load = next(i for i, line in enumerate(base_ld) if line.startswith("s_buffer_load_dwordx4"))
    lgkm = base_ld.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_ld[:lgkm] + ["v_ldexp_f32 v20, 1.0, s45"] + base_ld[lgkm:])
    wqm = base_ld.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_ld[:wqm] + ["v_ldexp_f32 v20, 1.0, 2"] + base_ld[wqm:])


def check_program_extent(directory, library):
    # A conditional jumps over the early exit. The alternate block starts
    # with a two-dword instruction (s_mov_b32 with a literal operand).
    branch = bytes.fromhex("010084bf")
    end = bytes.fromhex("000081bf")
    mov = bytes.fromhex("ff0380be 01000000")
    pad = bytes.fromhex("00009fbf")
    code = branch + end + mov + end
    padding = pad * 5
    base = 288 - 40
    extent = 40 + len(code) + len(padding)
    archive = directory / "bounded.rpf"

    def extract(body, size=extent, limit=base + extent):
        archive.write_bytes(envelope(body, directory_end=(size, limit)))
        return native_program(archive, "fixture", llvm=library)[1]

    jump_over_nop = bytes.fromhex("010082bf 000080bf")
    assert extract(jump_over_nop + code + padding[8:]) == jump_over_nop + code
    must_reject(extract, bytes.fromhex("010082bf 810380be") + code + padding[8:])
    assert extract(code + padding + end) == code  # marker in metadata
    # A terminator in metadata cannot stand in for one missing inside the section.
    must_reject(extract, bytes(len(code)) + padding + end)
    assert extract(code + padding + pad * 4, limit=0xdeadbeefdeadbeef) == code
    must_reject(extract, code + padding, extent, base + extent + 4)
    must_reject(extract, code + padding[:-4])
    must_reject(extract, code + padding, extent, 0xdeadbeefdeadbeef)
    must_reject(extract, code + padding[:4] + bytes(16) + end)
    must_reject(extract, bytes.fromhex("020084bf") + code[4:] + padding)  # target inside literal
    must_reject(extract, bytes.fromhex("000080bf") + code[4:] + padding)  # disconnected block
    must_reject(extract, bytes.fromhex("040084bf") + code[4:] + padding)  # target in padding
    must_reject(extract, branch + bytes.fromhex("ff0380be") + end + bytes.fromhex("000080bf") + end + padding)


def check_binary(directory, assembler, objcopy, library):
    check_program_extent(directory, library)
    fixtures = [(assembly(False), translate), (assembly(True), translate), (curve(), translate), (shaping(), translate), (select(), translate), (absolute_value(), translate),
                (rounding(), translate), (arithmetic(), translate), (with_scalar_constant_reads(), translate), (with_scalar_pair_reads(), translate),
                (truncate_values(), translate), (affine_coordinates(), translate), (ceiling(), translate),
                (scalar64(), translate), (sopk_scalar(), translate)]
    fixtures.append((rgb_sample(), translate))
    fixtures.append((with_constants(CLASS_BODY), translate))
    fixtures.append((with_constants(RECIPROCAL_BODY), translate))
    fixtures.append((bitwise_program(), translate))
    fixtures.append((fract_values(), translate))
    fixtures.extend([(frexp_mant_values(), translate), (ldexp_values(), translate)])
    fixtures.append((three_way(), translate))
    fixtures.extend((predicate_group(first), translate) for first in range(0, 16, 4))
    fixtures.extend((int_predicate_group(kind, encoding, first), translate)
                    for kind in ("i32", "u32") for encoding in ("e32", "e64") for first in (0, 4))

    fixtures.extend((int_minmax_pair(encoding), translate) for encoding in ("e32", "e64"))
    fixtures.extend((int_minmax_triple(kind), translate) for kind in ("i32", "u32"))
    fixtures.extend((predicate_group(first, sdwa=True), translate) for first in range(0, 16, 4))
    fixtures.append((with_constants(CUBE_BODY), translate))
    fixtures.append((with_constants(MAD_BODY), translate))
    fixtures.extend((dual_program(case), translate_dual) for case in DUAL_CASES)
    fixtures.append((with_constants(MOVK_BODY), translate))
    fixtures.append((with_constants(HALF_BODY), translate))
    fixtures.append((with_constants(VOP3P_BODY), translate))
    fixtures.append((with_constants(FMA_MIX_BODY), translate))
    fixtures.append((with_constants(NARROW_BODY), translate))
    fixtures.append((dual_program(swap_weights=True), translate_dual))
    fixtures.extend((with_constants(body), translate) for body, *_ in E64_ARITH)
    fixtures.extend((predicate_group_e64(first), translate) for first in range(0, 16, 4))
    fixtures.append((scalar(), translate))
    fixtures.extend((sopk_program(k), translate) for k in range(len(SOPK_RESULTS)))
    fixtures.append((with_constants(LITERAL_FMA_BODY), translate))
    fixtures.append((unpack_bytes(), translate))
    fixtures.append((reciprocal_iflag(), translate))
    fixtures.append((offset_nibbles(), translate))
    fixtures.append((scheduling_hints(), translate))
    fixtures.extend((buffer_color(rg), translate_buffer_color) for rg in (False, True))
    fixtures.extend((attribute_color(rg), translate_attribute) for rg in (False, True))
    fixtures.append((with_constants(PK_INT16_BODY), translate))
    fixtures.extend((with_constants(body), translate) for body in (SDWA_PRESERVE_BODY, SDWA_PAD_BODY))
    exp_sync_program = ["s_waitcnt_expcnt null, 0"] + assembly(False)[:-1] + ["s_waitcnt_expcnt null, 0", "s_endpgm"]
    fixtures.append((exp_sync_program, translate))
    fixtures.append((with_constants(DOT_BODY), translate, "gfx1030"))
    for index, entry in enumerate(fixtures):
        instructions, lower_fixture = entry[:2]
        cpu = entry[2] if len(entry) > 2 else "gfx1013"
        source = directory / "fixture.s"
        obj = directory / "fixture.o"
        binary = directory / "fixture.bin"
        source.write_text("\n".join(instructions) + "\n", encoding="ascii")
        subprocess.run([assembler, "-triple=amdgcn--amdpal", f"-mcpu={cpu}",
                        "-filetype=obj", source, "-o", obj], check=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, binary], check=True)
        archive = directory / "fixture.rpf"
        archive.write_bytes(envelope(binary.read_bytes()))
        _, native = native_program(archive, "fixture")
        assert native == binary.read_bytes()
        assert lower_fixture(disassemble(native, library, cpu=cpu.encode())) == lower_fixture(instructions)
        if index == 0:
            check_container_selector(directory, native, library)


def check_ceiling_draws(directory, compiler, renderer, backend):
    instructions = ceiling()
    source = directory / "ceiling.hlsl"
    shader = directory / "ceiling.cso"
    case = directory / "ceiling.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]
    source.write_text(translate(instructions), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    for a, b, c, d in ((-1.75, 1.75, -1.75, -1.75), (-2.0, 2.0, 0.0, 0.0),
                       (-0.75, 0.75, -0.75, -0.75), (2.25, -2.75, 2.25, 2.25)):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        # Python's integer results lose the sign of zero. The readback compares float bits.
        expected = struct.pack("<4f", math.copysign(math.ceil(a), a),
                               math.copysign(math.ceil(-abs(b)), -1), math.ceil(abs(c)),
                               min(math.ceil(abs(d)), 1))
        case.write_bytes(constants + expected * 64)
        subprocess.run(draw, check=True)

    floor = [line.replace("v_ceil_f32_e32", "v_floor_f32_e32") for line in instructions]
    source.write_text(translate(floor), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing ceiling with floor escaped detection")


def check_affine_lod_draw(directory: Path, compiler: str, renderer: str, backend: str) -> None:
    program = affine_coordinates()
    coordinate = f"float2(alu{program.index('v_fma_f32 v8, 0.5, v8, 0')}, uv.y)"
    source = translate(program)
    assert source.count(coordinate) == 1
    # Replace values on uncovered lanes with the quad's top-left coordinates before computing LOD.
    poor_gradient = (f"(uv.x + uv.y < 0.96875f ? {coordinate} : "
                     "(((floor(uv * 4.0f) * 2.0f + 0.5f) / 8.0f) + float2(0.5f, 0.0f))"
                     " * float2(params[0].z * 0.5f, 1.0f))")
    hlsl = directory / "quad_lod.hlsl"
    shader = directory / "quad_lod.cso"
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, hlsl]
    draw = [renderer, backend, "--check-pixel-quad-lod", shader]
    hlsl.write_text(source, encoding="ascii")
    subprocess.run(compile_shader, check=True)
    subprocess.run(draw, check=True)
    hlsl.write_text(source.replace(coordinate, poor_gradient), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("The draw accepted a missing helper-lane gradient")


def check_int_minmax_draws(directory, compiler, renderer, backend):
    source = directory / "int_minmax.hlsl"
    shader = directory / "int_minmax.cso"
    case = directory / "int_minmax.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]

    def order_key(value, signed):
        return struct.unpack("<i" if signed else "<I", struct.pack("<f", value))[0]

    def pick_pair(a, b):
        return (
            min(a, b, key=lambda v: order_key(v, True)),
            max(a, b, key=lambda v: order_key(v, True)),
            min(a, b, key=lambda v: order_key(v, False)),
            max(a, b, key=lambda v: order_key(v, False)),
        )

    def pick_triple(a, b, c, signed):
        ordered = sorted((a, b, c), key=lambda v: order_key(v, signed))
        tied = sorted((c, b, b), key=lambda v: order_key(v, signed))[1]
        return ordered[0], ordered[2], ordered[1], tied

    # Exact binary16 floats whose 32-bit patterns reverse order across signed, unsigned, and float.
    for encoding in ("e32", "e64"):
        source.write_text(translate(int_minmax_pair(encoding)), encoding="ascii")
        subprocess.run(compile_shader, check=True)
        for a, b in ((-2.0, 1.0), (-2.0, -1.0)):
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, a, b, 0.0, 0.0)
            case.write_bytes(constants + struct.pack("<4f", *pick_pair(a, b)) * 64)
            subprocess.run(draw, check=True)

    swapped = [line.replace("v_min_i32_e32", "v_min_u32_e32") for line in int_minmax_pair("e32")]
    source.write_text(translate(swapped), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, -2.0, 1.0, 0.0, 0.0)
    case.write_bytes(constants + struct.pack("<4f", *pick_pair(-2.0, 1.0)) * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_min_i32 with v_min_u32 escaped detection")

    for kind in ("i32", "u32"):
        source.write_text(translate(int_minmax_triple(kind)), encoding="ascii")
        subprocess.run(compile_shader, check=True)
        for a, b, c in ((-2.0, -1.0, 1.0), (0.5, -0.5, 4.0)):
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, a, b, c, 0.0)
            case.write_bytes(constants + struct.pack("<4f", *pick_triple(a, b, c, kind == "i32")) * 64)
            subprocess.run(draw, check=True)

    wrong_med = [line.replace("v_med3_i32", "v_med3_u32") for line in int_minmax_triple("i32")]
    source.write_text(translate(wrong_med), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, -2.0, -1.0, 1.0, 0.0)
    case.write_bytes(constants + struct.pack("<4f", *pick_triple(-2.0, -1.0, 1.0, True)) * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_med3_i32 with v_med3_u32 escaped detection")

def check_frexp_ldexp_draws(directory, compiler, renderer, backend):
    source = directory / "frexp_ldexp.hlsl"
    shader = directory / "frexp_ldexp.cso"
    case = directory / "frexp_ldexp.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]

    source.write_text(translate(frexp_mant_values()), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    for a, b, c, d in ((3.0, 6.0, -0.25, -3.0), (-0.0, 0.0, 0.75, 12.0)):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        expected = (
            math.frexp(a)[0],
            math.frexp(-abs(b))[0],
            math.frexp(abs(c))[0],
            min(1.0, max(0.0, math.frexp(d)[0])),
        )
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)

    wrong_frexp = [line.replace("v_frexp_mant_f32_e32", "v_mov_b32_e32") for line in frexp_mant_values()]
    source.write_text(translate(wrong_frexp), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, 3.0, 6.0, -0.25, -3.0)
    case.write_bytes(constants + struct.pack("<4f", 0.75, -0.75, 0.5, 0.0) * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_frexp_mant_f32 with v_mov_b32 escaped detection")

    source.write_text(translate(ldexp_values()), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    for a, exp_i, c, d in ((0.75, 3, 6.0, 0.25), (-0.0, 10, -2.0, -0.5), (1.5, -2, 0.5, 0.0625)):
        constants = bytearray(256)
        struct.pack_into("<fiff", constants, 12, a, exp_i, c, d)
        expected = (
            math.ldexp(a, exp_i),
            math.ldexp(-abs(a), 2),
            math.ldexp(c, -2),
            min(1.0, max(0.0, math.ldexp(d, 3))),
        )
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)

    wrong_ldexp = [line.replace("v_ldexp_f32 v14, s46, -2", "v_ldexp_f32 v14, s46, 2") for line in ldexp_values()]
    source.write_text(translate(wrong_ldexp), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    constants = bytearray(256)
    struct.pack_into("<fiff", constants, 12, 0.75, 3, 6.0, 0.25)
    case.write_bytes(constants + struct.pack("<4f", 6.0, -3.0, 1.5, 1.0) * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Inverting the v_ldexp_f32 exponent sign escaped detection")


def check_three_way_draw(directory, compiler, renderer, backend):
    source = directory / "three_way.hlsl"
    shader = directory / "three_way.cso"
    source.write_text(translate(three_way()), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    for constants in ((.25, .75, .375, .5), (.625, .125, -.3125, .5)):
        case = three_way_case(directory, *constants)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        result = subprocess.run(draw)
        if result.returncode:
            return result.returncode
    without_abs = [line.replace("|s46|", "s46") for line in three_way()]
    source.write_text(translate(without_abs), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("The renderer failed to detect the missing min3 absolute value")
    source.write_text(translate([line.replace("v_med3_f32 v12", "v_min3_f32 v12") for line in three_way()]),
                      encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("A minimum replaced the median without a mismatch")
    return 0


def check_buffer_color_draw(directory, compiler, renderer, backend):
    source, shader, case = (directory / name for name in ("buffer.hlsl", "buffer.cso", "buffer.case"))
    for rg in (False, True):
        source.write_text(translate_buffer_color(buffer_color(rg)), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        draw = [renderer, backend, "--check-pixel-case-rg" if rg else "--check-pixel-case", shader, case]
        for red, green in ((0.25, 0.75), (-0.5, 0.125)):
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 16, green, 2, 3, red)
            blue, alpha = (-1, -1) if rg else (green, red)
            expected = struct.pack("<4f", red, green, blue, alpha) * 64
            case.write_bytes(constants + expected)
            result = subprocess.run(draw)
            if result.returncode:
                return result.returncode
        struct.pack_into("<f", constants, 28, green)
        case.write_bytes(constants + expected)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError("Wrong buffer color escaped detection")
def check_unpack_byte_draws(directory, compiler, renderer, backend):
    instructions = unpack_bytes()
    source = directory / "ubyte.hlsl"
    shader = directory / "ubyte.cso"
    case = directory / "ubyte.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]
    source.write_text(translate(instructions), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    # Every word is a normal binary32 value, so its bits survive the constant buffer, and
    # all four bytes of each word differ, so reading the wrong byte changes the result.
    for words in ((0x40ff8001, 0x3e7f02c3, 0x41a5005a, 0x00102030 | 0x3f000000),
                  (0x4280ff7f, 0x3f11fe22, 0x407f8013, 0x00000000 | 0x40c08070)):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        values = [float((word >> 8 * n) & 0xff) for n, word in enumerate(words)]
        values[3] = min(values[3], 1.0)
        case.write_bytes(constants + struct.pack("<4f", *values) * 64)
        subprocess.run(draw, check=True)

    wrong = [line.replace("ubyte1", "ubyte0") for line in instructions]
    source.write_text(translate(wrong), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    result = subprocess.run(draw, capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Reading the wrong byte escaped detection")
def check_reciprocal_iflag_draws(directory, compiler, renderer, backend):
    instructions = reciprocal_iflag()
    source = directory / "rcp_iflag.hlsl"
    shader = directory / "rcp_iflag.cso"
    case = directory / "rcp_iflag.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]
    source.write_text(translate(instructions), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    inf = math.inf
    # Powers of two have exact reciprocals, so no rounding difference can hide a fault.
    # A zero operand gives an infinity of its sign, and clamp turns +inf into 1.
    for (a, b, c, d), expected in (((2.0, 4.0, -0.125, 0.5), (0.5, -0.25, 8.0, 1.0)),
                                   ((-0.25, 0.5, 16.0, 4.0), (-4.0, -2.0, 0.0625, 0.25)),
                                   ((0.0, 0.0, -0.0, 0.0), (inf, -inf, inf, 1.0))):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)

    wrong = [line.replace("v_rcp_iflag_f32", "v_sqrt_f32") for line in instructions]
    source.write_text(translate(wrong), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    result = subprocess.run(draw, capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Replacing the reciprocal with a square root escaped detection")
def check_offset_nibble_draws(directory, compiler, renderer, backend):
    instructions = offset_nibbles()
    source = directory / "cvt_off.hlsl"
    shader = directory / "cvt_off.cso"
    case = directory / "cvt_off.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]
    source.write_text(translate(instructions), encoding="ascii")
    subprocess.run(compile_shader, check=True)

    def offset(nibble):
        return (nibble - 16 if nibble & 8 else nibble) / 16

    # Each word is a normal binary32 value whose upper bits are set, so a result that used
    # more than bits 3:0 would change. All 16 nibbles appear across the four draws, and
    # the clamped lane sees both signs.
    high = 0x3f5a6b70
    nibbles = list(range(16))
    for start in range(0, 16, 4):
        group = nibbles[start:start + 4]
        words = [high | n for n in group]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        values = [offset(n) for n in group]
        values[3] = min(1.0, max(0.0, values[3]))
        # Python's integer results lose the sign of zero; nibble 0 gives +0.0 either way.
        case.write_bytes(constants + struct.pack("<4f", *values) * 64)
        subprocess.run(draw, check=True)

    # Reading the nibble as unsigned changes every result with bit 3 set.
    unsigned = translate(instructions).replace("asint(", "(")
    source.write_text(unsigned, encoding="ascii")
    subprocess.run(compile_shader, check=True)
    words = [high | 0x8, high | 0x9, high | 0xf, high | 0xc]
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *words)
    case.write_bytes(constants + struct.pack("<4f", -0.5, -0.4375, -0.0625, 0.0) * 64)
    result = subprocess.run(draw, capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Reading the nibble without its sign escaped detection")


def check_rgb_sample_draw(directory: Path, compiler: str, renderer: str, backend: str) -> None:
    hlsl = directory / "rgb-sample.hlsl"
    shader = directory / "rgb-sample.cso"
    case = directory / "rgb-sample.case"
    hlsl.write_text(translate(rgb_sample()), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, hlsl], check=True)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for alpha in (0.5, 1.0):
        colors = (struct.pack("<4f", half_rtz(red / 255), half_rtz(green / 255), half_rtz(128 / 255), alpha)
                  for green in RAMP for red in RAMP)
        case.write_bytes(bytes(256) + b"".join(colors))
        if alpha == 0.5:
            subprocess.run(draw, check=True)
        else:
            wrong = subprocess.run(draw, capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch x=0 y=0" not in wrong.stderr:
                raise AssertionError("RGB sample with an overwritten alpha register escaped detection")


def check_render(directory, compiler, renderer, backend, *, core_only=False):
    for swap in (False, True):
        source = directory / "fixture.hlsl"
        shader = directory / "fixture.cso"
        source.write_text(translate(assembly(swap)), encoding="ascii")
        subprocess.run([compiler, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        draw = [renderer, backend, "--check-pixel", shader]
        result = subprocess.run([*draw, "bgra" if swap else "rgba"])
        if result.returncode:
            return result.returncode
        if swap:
            wrong = subprocess.run([*draw, "rgba"], capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError("The renderer failed to detect the changed channel order")
    check_vop3p_draw(directory, compiler, renderer, backend)
    check_bitwise_draw(directory, compiler, renderer, backend)
    check_fma_mix_draw(directory, compiler, renderer, backend)
    check_dot_draw(directory, compiler, renderer, backend)
    check_pk_int16_draw(directory, compiler, renderer, backend)
    check_narrow_draw(directory, compiler, renderer, backend)
    check_movk_draw(directory, compiler, renderer, backend)
    check_half_draw(directory, compiler, renderer, backend)
    for exp_shader_inst in ("s_waitcnt_expcnt null, 0x3", "s_waitcnt expcnt(0)"):
        exp_program = [exp_shader_inst] + assembly(False)[:-1] + [exp_shader_inst, "s_endpgm"]
        source.write_text(translate(exp_program), encoding="ascii")
        subprocess.run([compiler, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        subprocess.run([renderer, backend, "--check-pixel", shader, "rgba"], check=True)
    check_rgb_sample_draw(directory, compiler, renderer, backend)
    program = arithmetic()
    source.write_text(translate(program), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    for constants in ((4.0, -.0625, .125, -.015625), (2.0, -.03125, .0625, -.0078125)):
        draw = [renderer, backend, "--check-pixel-case", shader, arithmetic_case(directory, *constants)]
        subprocess.run(draw, check=True)
    swapped = [line.replace("v_subrev_f32", "v_sub_f32") for line in program]
    source.write_text(translate(swapped), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapped subtraction operands escaped detection")

    program = with_scalar_constant_reads()
    source.write_text(translate(program), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    case = directory / "scalar-constants.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for red, green, blue in ((.25, -.5, .75), (-.125, .625, .875)):
        constants = bytearray(256)
        for offset, value in ((0, red), (8, green), (252, blue)):
            struct.pack_into("<f", constants, offset, value)
        case.write_bytes(constants + struct.pack("<4f", red, green, blue, 1) * 64)
        subprocess.run(draw, check=True)
    wrong_word = [line.replace("s62, s[44:47], 0x8", "s62, s[44:47], 0x4") for line in program]
    source.write_text(translate(wrong_word), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Reading the neighboring constant word escaped detection")

    check_affine_lod_draw(directory, compiler, renderer, backend)
    program = with_scalar_pair_reads()
    source.write_text(translate(program), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    for constants in ((2.0, -.0625, 1.0, .5, .25, .5), (.5, -.03125, .75, 2.0, .125, .25)):
        case = scalar_pair_case(directory, *constants)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        subprocess.run(draw, check=True)
    # Words 4, 6 and 63 are the second word of each pair; each must reach its own channel.
    reference = case.read_bytes()
    for offset in (16, 24, 252):
        corrupted = bytearray(reference)
        struct.pack_into("<f", corrupted, offset, 1.0)
        case.write_bytes(corrupted)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Changing the constant at byte {offset} escaped detection")
    case.write_bytes(reference)
    shifted = [line.replace("s[62:63], s[44:47], 0x14", "s[62:63], s[44:47], 0x10") for line in program]
    source.write_text(translate(shifted), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Reading the pair one word early escaped detection")

    for program_fn, case_fn in ((curve, curve_case), (shaping, shaping_case)):
        source.write_text(translate(program_fn()), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        for gain in (.75, 1.125):
            case = case_fn(directory, gain)
            draw = [renderer, backend, "--check-pixel-case", shader, case]
            subprocess.run(draw, check=True)
        corrupted = bytearray(case.read_bytes())
        struct.pack_into("<f", corrupted, 16, 0.0)
        case.write_bytes(corrupted)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError("Stale reference accepted after changing the bound constants")

    if core_only:
        return 0
    check_predicate_and_rounding_draws(directory, compiler, renderer, backend)
    return check_blend_and_scalar_draws(directory, compiler, renderer, backend)


def check_predicate_and_rounding_draws(directory, compiler, renderer, backend):
    source = directory / "fixture.hlsl"
    shader = directory / "fixture.cso"
    program = truncate_values()
    source.write_text(translate(program), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    for constants in ((1.75, -1.75, -2.75, 2.75), (-1.75, 2.75, 3.75, -1.75)):
        case = truncate_case(directory, *constants)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        subprocess.run(draw, check=True)
    rounded_down = [line.replace("v_trunc_f32", "v_floor_f32") for line in program]
    source.write_text(translate(rounded_down), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Rounding toward negative infinity escaped detection")

    program = sopk_scalar()
    source.write_text(translate(program), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    for red, green in ((0.5, 0.25), (0.125, 1.0)):
        case = sopk_case(directory, red, green)
        subprocess.run([renderer, backend, "--check-pixel-case", shader, case], check=True)

    case = sopk_case(directory, 0.5, 0.25)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    changed = [line.replace("s_mulk_i32 s50, 0x3", "s_mulk_i32 s50, 0x5") for line in program]
    source.write_text(translate(changed), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)

    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Changing the sopk multiply escaped detection")

    source.write_text(translate(rounding()), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    # Ties go to even (2.5 to 2, -3.5 to -4), unlike rounding half away from zero or half up.
    for constants in ((4.0, 2.5, 1.25), (8.0, -3.5, -0.75)):
        draw = [renderer, backend, "--check-pixel-case", shader, rounding_case(directory, *constants)]
        subprocess.run(draw, check=True)
    for old, new, failure in (("v_rndne_f32_e32", "v_floor_f32_e32", "Flooring instead of rounding"),
                              ("v_ceil_f32_e32", "v_floor_f32_e32", "Flooring instead of ceiling")):
        source.write_text(translate([line.replace(old, new) for line in rounding()]), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"{failure} escaped detection")
    # Rounding half up gives -3 for the -3.5 tie, so the same draw must catch it.
    source.write_text(translate(rounding()).replace("round(", "floor(0.5f + "), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Rounding ties up instead of to even escaped detection")

    program = absolute_value()
    case = absolute_case(directory)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    subprocess.run([compiler, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    subprocess.run(draw, check=True)

    without_abs = [line.replace("|v12|", "v12") for line in program]
    source.write_text(translate(without_abs), encoding="ascii")
    subprocess.run([compiler, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("The renderer failed to detect the missing absolute value")
    check_ceiling_draws(directory, compiler, renderer, backend)
    check_unpack_byte_draws(directory, compiler, renderer, backend)
    check_reciprocal_iflag_draws(directory, compiler, renderer, backend)
    check_offset_nibble_draws(directory, compiler, renderer, backend)
    source.write_text(translate(select()), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    for constants in ((.5, .75, .375, .25), (.125, .5, 1.0, 0.0)):
        case = select_case(directory, *constants)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        subprocess.run(draw, check=True)
    corrupted = bytearray(case.read_bytes())
    struct.pack_into("<f", corrupted, 12, .5)
    case.write_bytes(corrupted)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Stale reference accepted after moving the comparison threshold")
    # Binary32 values on either side of a half-ULP midpoint, in both signs.
    values = (1 + 3 / 4096, 1 + 1 / 4096, -1 - 3 / 4096, -1 - 1 / 4096)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *values)
    program = with_constants([f"v_sub_f32_e64 v{12 + i}, s{44 + i}, 0" for i in range(4)])
    source.write_text(translate(program), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    case = directory / "signed-export.case"
    case.write_bytes(constants + struct.pack("<4f", *map(half_rtz, values)) * 64)
    subprocess.run([renderer, backend, "--check-pixel-case", shader, case], check=True)
    check_class_draw(directory, compiler, renderer, backend)
    check_predicate_draws(directory, compiler, renderer, backend)
    check_int_predicate_draws(directory, compiler, renderer, backend)
    check_sdwa_predicate_draws(directory, compiler, renderer, backend)
    check_literal_fma_draw(directory, compiler, renderer, backend)
    return 0


def check_blend_and_scalar_draws(directory, compiler, renderer, backend):
    source = directory / "fixture.hlsl"
    shader = directory / "fixture.cso"
    capture = directory / "blend.rgba"
    draw = [renderer, backend, "--dual-pixel", shader, capture]
    for case in DUAL_CASES:
        source.write_text(translate_dual(dual_program(case)), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        subprocess.run(draw, check=True)
    wrong = subprocess.run([renderer, backend, "--dual-pixel-swapped", shader, capture], capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapped texture bindings escaped detection")
    source.write_text(translate_dual(dual_program(swap_weights=True)), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapped blend weights escaped detection")
    check_cube_draw(directory, compiler, renderer, backend)
    check_sdwa_draw(directory, compiler, renderer, backend)
    check_fract_draw(directory, compiler, renderer, backend)
    check_reciprocal_draw(directory, compiler, renderer, backend)
    check_int_minmax_draws(directory, compiler, renderer, backend)

    check_frexp_ldexp_draws(directory, compiler, renderer, backend)

    program = scalar64()
    source.write_text(translate(program), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    for values in SCALAR64_SETS:
        case = scalar64_case(directory, values)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        subprocess.run(draw, check=True)
    # Keep the original two sets for the SCC toggle checks and add a draw-only case.
    _, _, *middle, _, alt = SCALAR64_SETS[0]
    a = 0x11234567 << 32 | 0xffffffff
    b = middle[1] << 32 | middle[0]
    # s_not_b64 of a | b sets only bit 63, so its SCC depends on the high word alone.
    assert ~(a | b) & (2**64 - 1) == 1 << 63
    case = scalar64_case(directory, (0xffffffff, 0x11234567, *middle, 0x00400018, alt))
    subprocess.run([renderer, backend, "--check-pixel-case", shader, case], check=True)
    # Mutations verified against Set 1 (where negative signed values and SCC=1 paths are active).
    set1_case = scalar64_case(directory, SCALAR64_SETS[0])
    draw = [renderer, backend, "--check-pixel-case", shader, set1_case]
    for old, new, label in (
        ("s_ashr_i64 s[72:73], s[52:53], s56", "s_lshr_b64 s[72:73], s[52:53], s56", "signed right shift"),
        ("s_bfe_i32 s72, s60, s61", "s_bfe_u32 s72, s60, s61", "signed bitfield extract"),
        ("s_andn2_b64 s[72:73], s[52:53], s[54:55]", "s_and_b64 s[72:73], s[52:53], s[54:55]", "andn2 inversion"),
        ("s_cmov_b64 s[78:79], -1", "s_cmov_b64 s[78:79], 0", "conditional move"),
        ("s_nand_b64 s[72:73], s[52:53], s[54:55]", "s_and_b64 s[72:73], s[52:53], s[54:55]", "nand inversion"),
        ("s_nor_b64 s[72:73], s[52:53], s[54:55]", "s_or_b64 s[72:73], s[52:53], s[54:55]", "nor inversion"),
        ("s_xnor_b64 s[72:73], s[52:53], s[54:55]", "s_xor_b64 s[72:73], s[52:53], s[54:55]", "xnor inversion"),
        ("s_nor_b64 s[72:73], s[82:83], -1", "s_nor_b64 s[72:73], s[82:83], 0", "nor SCC probe"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")
    check_mad_draw(directory, compiler, renderer, backend)
    source.write_text(translate(scalar()), encoding="ascii")
    subprocess.run([compiler, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    for words in SCALAR_WORDS:
        case = scalar_case(directory, words)
        subprocess.run([renderer, backend, "--check-pixel-case", shader, case], check=True)
    for old, new in SCALAR_MUTATIONS:
        case = scalar_case(directory, scalar_detecting_words(old, new))
        source.write_text(translate([line.replace(old, new) for line in scalar()]), encoding="ascii")
        subprocess.run([compiler, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, case], capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Replacing {old} with {new} escaped detection")
    return 0


def check_e64_render(directory, compiler, renderer, backend):
    """The e64 draw cases on their own, so they run as a separate CTest entry per backend. A renderer that reports an
    unavailable adapter (exit 77) makes this a skip, as for the other pixel draws."""
    try:
        check_e64_draws(directory, compiler, renderer, backend)
    except subprocess.CalledProcessError as error:
        if error.returncode == 77:
            return 77
        raise
    return 0


def main():
    if len(sys.argv) == 1:
        with tempfile.TemporaryDirectory() as tmp:
            check_controls(Path(tmp))
        return 0
    mode, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        if mode in ("draw-blend-scalar", "draw-predicate-rounding"):
            try:
                draw = check_blend_and_scalar_draws if mode == "draw-blend-scalar" else check_predicate_and_rounding_draws
                return draw(directory, *args)
            except subprocess.CalledProcessError as error:
                if error.returncode == 77 and str(error.cmd[0]) == str(args[1]):
                    return 77
                raise
        if mode == "draw-e64":
            # The controls already run in the binary and draw modes; this mode only draws.
            return check_e64_render(directory, *args)
        if mode == "draw-sopk":
            # As for draw-e64, the controls run in the binary and draw modes.
            return check_sopk_render(directory, *args)
        check_controls(directory)
        if mode == "binary":
            check_binary(directory, *args)
        elif mode in ("draw", "draw-core"):
            return check_render(directory, *args, core_only=mode == "draw-core")
        elif mode == "three-way":
            return check_three_way_draw(directory, *args)
        elif mode == "buffer-color":
            return check_buffer_color_draw(directory, *args)
        else:
            raise ValueError("Expected binary, draw, draw-e64, draw-sopk, three-way or buffer-color mode")


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:  # also on skip or failure; tools/form_coverage.py reads the test's status from JUnit
        write_record(os.environ.get("FORM_COVERAGE_RECORD"))
