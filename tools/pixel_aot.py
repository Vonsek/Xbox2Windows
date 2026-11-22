"""Offline lowering for a straight-line, texture-sampling gfx1013 pixel shader."""
import argparse
import ctypes as C
import hashlib
import json
import mmap
from pathlib import Path
import re
import struct
import tempfile
import zlib

from resource_inventory import parts


def inflate(data, size, window):
    if not 0 < size <= 16 << 20:
        raise ValueError("Invalid decoded size")
    decoder = zlib.decompressobj(window)
    image = decoder.decompress(data, size + 1)
    if len(image) != size or not decoder.eof:
        raise ValueError("Incomplete or mismatched compressed stream")
    return image


def _read_named_container(data: mmap.mmap, end: int) -> bytes:
    payload = end + 17
    if payload > len(data):
        raise ValueError("Missing or truncated named record")
    stored, _, repeat, size = struct.unpack_from("<4I", data, end + 1)
    if stored != repeat or not 0 < stored <= 8 << 20 or payload + stored > len(data):
        raise ValueError("Invalid record extent")
    return inflate(data[payload:payload + stored], size, -15)


def _select_named_container(data: mmap.mmap, needle: bytes, digest: str) -> bytes:
    selected = None
    cursor = 0
    while (pos := data.find(needle, cursor)) >= 0:
        # Advance past the name, independent of the candidate's claimed payload extent.
        cursor = pos + len(needle)
        try:
            blob = _read_named_container(data, cursor)
        except (ValueError, zlib.error):
            continue
        if hashlib.sha256(blob).hexdigest() == digest:
            if selected is not None:
                raise ValueError("Multiple records match the name and container SHA-256")
            selected = blob
    if selected is None:
        raise ValueError("No record matches the name and container SHA-256")
    return selected


def _bounded_code(image, start):
    """Strip code padding within the FSSM v0x101 section directory."""
    if len(image) < 240 or struct.unpack_from("<I", image, 4)[0] != 0x101:
        raise ValueError("Unknown multi-exit FSSM layout")
    base, size = struct.unpack_from("<QQ", image, 192)
    limit, = struct.unpack_from("<Q", image, 232)
    if limit == 0xdeadbeefdeadbeef:
        limit = base + size
        if image[limit:limit+16] != b"\x00\x00\x9f\xbf" * 4:
            raise ValueError("Unverified FSSM sentinel extent")
    if (base < 240 or base + 40 != start or size % 16 or base + size != limit
            or limit > len(image) or limit <= start):
        raise ValueError("Inconsistent FSSM code extent")
    end = limit
    while end > start and image[end-4:end] == b"\x00\x00\x9f\xbf":
        end -= 4
    if end <= start or end == limit or image[end-4:end] != b"\x00\x00\x81\xbf":
        raise ValueError("Expected termination followed by code padding")
    return image[start:end]


def _check_two_exits(code, llvm):
    """Both decoded exits must be reachable from the single entry."""
    instructions = dict(disassemble(code, llvm, offsets=True))
    exits = [pc for pc, line in instructions.items() if line == "s_endpgm"]
    if len(exits) != 2 or exits[-1] + 4 != len(code):
        raise ValueError("Expected two decoded exits")
    # An aligned word can be a literal or the second half of an instruction.
    # Follow branches only from decoded PCs and reject unresolved transfers.
    pcs = list(instructions)
    following = dict(zip(pcs, pcs[1:] + [len(code)]))
    pending, visited, targets = [0], set(), set()
    while pending:
        pc = pending.pop()
        if pc in visited:
            continue
        if pc not in instructions:
            raise ValueError("Branch outside decoded instruction boundaries")
        visited.add(pc)
        op = instructions[pc].split()[0]
        if op == "s_endpgm":
            continue
        if op == "s_branch" or op in ("s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz",
                                      "s_cbranch_vccnz", "s_cbranch_execz", "s_cbranch_execnz"):
            target = pc + 4 + 4 * struct.unpack_from("<h", code, pc)[0]
            targets.add(target)
            pending.append(target)
            if op == "s_branch":
                continue
        elif not (op.startswith(("v_", "image_", "buffer_", "tbuffer_", "ds_", "flat_", "global_",
                                 "scratch_", "exp", "s_load_", "s_buffer_load_", "s_cmp", "s_bitcmp",
                                 "s_mov_", "s_and", "s_or_", "s_xor_", "s_not_", "s_lsh", "s_ash",
                                 "s_add", "s_sub", "s_mul", "s_min_", "s_max_", "s_cselect", "s_bfe",
                                 "s_bfm", "s_brev", "s_ff", "s_bcnt", "s_wqm", "s_quadmask"))
                  or op in ("s_nop", "s_inst_prefetch", "s_waitcnt", "s_clause", "s_delay_alu",
                            "s_setprio", "s_sleep", "s_movk_i32", "s_mulk_i32", "s_flbit_i32_b32",
                            "s_setreg_imm32_b32")):
            raise ValueError(f"Unverified multi-exit control flow: {op}")
        pending.append(following[pc])
    # A direct branch may skip alignment NOPs. Keep them in the original span.
    unreachable = instructions.keys() - visited
    if exits[0] + 4 not in targets or any(instructions[pc].split()[0] != "s_nop" for pc in unreachable):
        raise ValueError("Unreachable code or unverified alternate exit")


def native_program(archive: Path, record: str, *, container_sha256: str | None = None, llvm=None) -> tuple[bytes, bytes]:
    name = record.encode("ascii")
    if not 0 < len(name) < 255 or b"\0" in name:
        raise ValueError("Invalid length-prefixed record name")
    needle = bytes([len(name) + 1]) + name + b"\0"
    if container_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", container_sha256):
        raise ValueError("Invalid decoded container SHA-256")
    with archive.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if container_sha256 is None:
            pos = data.find(needle)
            end = pos + len(needle)
            if pos < 0 or end + 17 > len(data):
                raise ValueError("Missing or truncated named record")
            if data.find(needle, end) >= 0:
                raise ValueError("Multiple records match the name")
            blob = _read_named_container(data, end)
        else:
            blob = _select_named_container(data, needle, container_sha256.lower())
    return blob, container_code(parts(blob)[b"DXIL"], stage=0, llvm=llvm)


def container_code(program, *, stage, llvm=None):
    """Extract the stage-specific instruction stream from a DXIL program part."""
    if len(program) < 24 or struct.unpack_from("<H", program, 2)[0] != stage:
        raise ValueError("Unexpected shader stage")
    table = program.index(b"XDAT", 24)
    if table + 12 > len(program):
        raise ValueError("Truncated XDAT header")
    count, = struct.unpack_from("<I", program, table+4)
    if table + 12 + count*12 > len(program):
        raise ValueError("Truncated XDAT table")
    entries = struct.iter_unpack("<3I", program[table+12:table+12+count*12])
    native_entries = [(o, n) for tag, o, n in entries if tag == 13]
    if len(native_entries) != 1:
        raise ValueError("Expected one native program entry")
    offset, length = native_entries[0]
    if offset < table + 12 + count * 12 or offset+length > len(program) or length < 24 or program[offset:offset+4] != b"XBCH":
        raise ValueError("Invalid native program entry")
    size, = struct.unpack_from("<I", program, offset+16)
    image = inflate(program[offset+24:offset+length], size, 15)
    if len(image) < 196 or image[:4] != b"FSSM":
        raise ValueError("Invalid native program header")
    start = struct.unpack_from("<I", image, 192)[0] + 40
    if start < 196 or start % 4 or start + 4 > len(image):
        raise ValueError("Invalid native code offset")
    if struct.unpack_from("<I", image, 4)[0] == 0x101:
        code = _bounded_code(image, start)
        markers = sum(word == 0xbf810000 for word, in struct.iter_unpack("<I", code))
        if markers == 1:
            return code
        if stage == 0 and markers == 2 and llvm is not None:
            _check_two_exits(code, llvm)
            return code
        raise ValueError("Unverified multi-exit program")
    # Older fixtures lack the section extent; preserve their unique-marker rule.
    if stage != 0:
        raise ValueError("Compute requires a bounded FSSM section")
    ends = [i+4 for i in range(start, len(image)-3, 4) if image[i:i+4] == b"\0\0\x81\xbf"]
    if len(ends) != 1:
        raise ValueError("Expected one aligned, terminated program")
    return image[start:ends[0]]


def disassemble(code, library, *, cpu=b"gfx1013", offsets=False, features=b""):
    llvm = C.CDLL(library)
    for component in ("TargetInfo", "Target", "TargetMC", "AsmPrinter", "Disassembler"):
        getattr(llvm, "LLVMInitializeAMDGPU" + component)()
    pointer, string = C.c_void_p, C.c_char_p
    llvm.LLVMCreateDisasmCPUFeatures.argtypes = [string, string, string, pointer, C.c_int, pointer, pointer]
    llvm.LLVMCreateDisasmCPUFeatures.restype = pointer
    llvm.LLVMDisasmInstruction.argtypes = [pointer, pointer, C.c_uint64, C.c_uint64, pointer, C.c_size_t]
    llvm.LLVMDisasmInstruction.restype = C.c_size_t
    llvm.LLVMDisasmDispose.argtypes = [pointer]
    ctx = llvm.LLVMCreateDisasmCPUFeatures(b"amdgcn--amdpal", cpu, features, None, 0, None, None)
    if not ctx:
        raise ValueError(f"LLVM has no {cpu.decode()} disassembler")
    raw = C.create_string_buffer(code)
    offset = 0
    instructions = []
    try:
        while offset < len(code):
            line = C.create_string_buffer(1024)
            size = llvm.LLVMDisasmInstruction(ctx, C.byref(raw, offset), len(code)-offset,
                                              offset, line, len(line))
            if not size:
                raise ValueError(f"Unknown instruction at byte {offset}")
            decoded = line.value.decode("ascii").strip()
            instructions.append((offset, decoded) if offsets else decoded)
            offset += size
    finally:
        llvm.LLVMDisasmDispose(ctx)
    return instructions


SAMPLED_RESOURCES = """Texture2D<float4> tex : register(t0);
SamplerState samp : register(s0);"""
PREAMBLE = """// v_cmp_class_f32: the ten binary32 classes of LLVM SIInstrFlags::ClassFlags, from the raw encoding:
// signaling NaN, quiet NaN, -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf.
uint class32(float value) {
    uint bits = asuint(value);
    uint magnitude = bits & 0x7fffffffu;
    bool negative = bits >= 0x80000000u;
    if (magnitude > 0x7f800000u) return (bits & 0x00400000u) != 0u ? 0x2u : 0x1u;
    if (magnitude == 0x7f800000u) return negative ? 0x4u : 0x200u;
    if (magnitude >= 0x00800000u) return negative ? 0x8u : 0x100u;
    if (magnitude != 0u) return negative ? 0x10u : 0x80u;
    return negative ? 0x20u : 0x40u;
}
// v_fract_f32: x - floor(x) clamped to 0x3f7fffff, so a tiny negative input cannot round up to 1.0.
// D3D frac has no clamp and differs between adapters. NaN fails the comparison and propagates.
float fract32(float value) {
    float f = frac(value);
    return f >= 1.0f ? asfloat(0x3f7fffffu) : f;
}
// v_mad_f32 and v_mac_f32 round the product before the add and do not support binary32 denormals.
// Flush each operand, the product and the sum to signed zero; NaN fails the comparison and propagates.
float ftz32(float value) {
    return abs(value) < asfloat(0x00800000u) ? asfloat(asuint(value) & 0x80000000u) : value;
}
float mad32(float a, float b, float c) {
    precise float product = ftz32(ftz32(a) * ftz32(b));
    precise float sum = product + ftz32(c);
    return ftz32(sum);
}
// v_frexp_mant_f32: normalized significand in (-1, -0.5] U [0.5, 1), preserving +/-0, +/-Inf and NaN.
float frexp_mant32(float value) {
    uint bits = asuint(value);
    uint exp = (bits >> 23) & 0xffu;
    uint mant = bits & 0x7fffffu;
    if (exp == 0xffu || (exp == 0u && mant == 0u)) return value;
    uint sign = bits & 0x80000000u;
    if (exp == 0u) {
        uint shift = 23u - firstbithigh(mant);
        mant = (mant << shift) & 0x7fffffu;
    }
    return asfloat(sign | 0x3f000000u | mant);
}
// v_ldexp_f32: exact binary32 scaling by 2^delta, preserving +/-0, +/-Inf and NaN.
float ldexp32(float value, int delta) {
    uint bits = asuint(value);
    uint sign = bits & 0x80000000u;
    uint exp = (bits >> 23) & 0xffu;
    uint mant = bits & 0x7fffffu;
    if (exp == 0xffu || (exp == 0u && mant == 0u)) return value;
    int e = (int)exp;
    if (exp == 0u) {
        uint shift = 23u - firstbithigh(mant);
        mant = (mant << shift) & 0x7fffffu;
        e = 1 - (int)shift;
    }
    e += clamp(delta, -512, 512);
    if (e >= 255) return asfloat(sign | 0x7f800000u);
    if (e <= 0) {
        if (e < -23) return asfloat(sign);
        uint full = mant | 0x800000u;
        uint shift = (uint)(1 - e);
        uint half_bit = 1u << (shift - 1u);
        uint mask = (1u << shift) - 1u;
        uint rem = full & mask;
        uint q = full >> shift;
        if (rem > half_bit || (rem == half_bit && (q & 1u) != 0u)) q++;
        return asfloat(sign | q);
    }
    return asfloat(sign | ((uint)e << 23) | mant);
}
// Finite binary32 to binary16, round toward zero, then widen for the float target.
float export16(float value) {
    uint h = f32tof16(value);
    if (abs(f16tof32(h)) > abs(value)) h--;
    return f16tof32(h);
}
// v_cvt_f32_f16: widen the binary16 in the low half of a word, built from its encoding so that
// desktop denormal handling cannot change it. Subnormal halves stay exact as m * 2^-24; NaN payloads widen.
float widen16(uint word) {
    uint h = word & 0xffffu;
    uint sign = (h & 0x8000u) << 16;
    uint exponent = (h >> 10) & 0x1fu;
    uint mantissa = h & 0x3ffu;
    if (exponent == 0u) {
        precise float magnitude = float(mantissa) * asfloat(0x33800000u);
        return sign != 0u ? -magnitude : magnitude;
    }
    uint wide = exponent == 0x1fu ? 0xffu : exponent + 112u;
    return asfloat(sign | (wide << 23) | (mantissa << 13));
}
// v_cvt_f16_f32: binary32 to the binary16 encoding, rounding to nearest even in integer arithmetic so that
// desktop conversion and denormal handling cannot change it. Overflow gives infinity, a binary16 subnormal
// result is kept, and a NaN keeps its sign and top payload bits with the quiet bit set.
uint narrow16(float value) {
    uint bits = asuint(value);
    uint sign = (bits >> 16) & 0x8000u;
    uint exponent = (bits >> 23) & 0xffu;
    uint mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return sign | 0x7c00u | (mantissa != 0u ? 0x200u | (mantissa >> 13) : 0u);
    }
    if (exponent < 102u) {
        return sign;
    }
    uint full = mantissa | 0x800000u;
    if (exponent < 113u) {
        uint shift = 126u - exponent;
        return sign | ((full + (1u << (shift - 1u)) - 1u + ((full >> shift) & 1u)) >> shift);
    }
    if (exponent > 142u) {
        return sign | 0x7c00u;
    }
    uint rounded = (full + 0xfffu + ((full >> 13) & 1u)) >> 13;
    return sign | min(((exponent - 113u) << 10) + rounded, 0x7c00u);
}
"""


def registers(first, last, count):
    first, last = int(first), int(last)
    if last - first + 1 != count:
        raise ValueError("Descriptor or vector width mismatch")
    return range(first, last+1)


HINT_LIMITS = {"s_clause": 0x3f, "s_setprio": 3, "s_sleep": 0x7f}

ALU = {
    "log": (1, "log2({0})"), "exp": (1, "exp2({0})"), "rcp": (1, "(1.0f / {0})"),
    # Same reciprocal; the integer-divide flag it raises has no desktop equivalent.
    "rcp_iflag": (1, "(1.0f / {0})"),
    "sqrt": (1, "sqrt({0})"), "rsq": (1, "rsqrt({0})"),
    "floor": (1, "floor({0})"), "ceil": (1, "ceil({0})"), "fract": (1, "fract32({0})"),
    "trunc": (1, "trunc({0})"), "frexp_mant": (1, "frexp_mant32({0})"),
    "rndne": (1, "round({0})"),  # HLSL round is DXIL Round_ne: ties to even
    "add": (2, "({0} + {1})"), "mul": (2, "({0} * {1})"),
    "max": (2, "max({0}, {1})"), "min": (2, "min({0}, {1})"), "sub": (2, "({0} - {1})"),
    "subrev": (2, "({1} - {0})"),
    "mul_legacy": (2, "(({0} == 0 || {1} == 0) ? 0.0f : {0} * {1})"),
    "fma": (3, "mad({0}, {1}, {2})"), "fmac": (3, "mad({0}, {1}, {2})"),
    "mad": (3, "mad32({0}, {1}, {2})"), "mac": (3, "mad32({0}, {1}, {2})"),
    # Each result is one of the operands, so finite inputs need no rounding.
    "min3": (3, "min(min({0}, {1}), {2})"), "max3": (3, "max(max({0}, {1}), {2})"),
    "med3": (3, "max(min({0}, {1}), min(max({0}, {1}), {2}))"),
}
# N-prefixed predicates include unordered operands. Keep NLG's NaN case explicit in DXIL.
COMPARE = {
    "f": "false", "tru": "true",
    "lt": "{0} < {1}", "eq": "{0} == {1}", "le": "{0} <= {1}", "gt": "{0} > {1}",
    "lg": "({0} < {1} || {0} > {1})", "ge": "{0} >= {1}",
    "o": "({0} == {0} && {1} == {1})", "u": "({0} != {0} || {1} != {1})",
    "nge": "({0} < {1} || isnan({0}) || isnan({1}))", "nlg": "({0} == {1} || isnan({0}) || isnan({1}))",
    "ngt": "({0} <= {1} || isnan({0}) || isnan({1}))", "nle": "({0} > {1} || isnan({0}) || isnan({1}))",
    "neq": "!({0} == {1})", "nlt": "({0} >= {1} || isnan({0}) || isnan({1}))",
}
INT_COMPARE = {
    "f": "false", "lt": "{0} < {1}", "eq": "{0} == {1}", "le": "{0} <= {1}",
    "gt": "{0} > {1}", "ne": "{0} != {1}", "ge": "{0} >= {1}", "t": "true",
}
# 32-bit integer operations on register bits. Shift counts use their low five bits, as on the target.
INTEGER = {
    "and_b32": (2, "{0} & {1}"), "or_b32": (2, "{0} | {1}"), "xor_b32": (2, "{0} ^ {1}"),
    "not_b32": (1, "~{0}"), "bfi_b32": (3, "({0} & {1}) | (~{0} & {2})"),
    "lshlrev_b32": (2, "{1} << ({0} & 31u)"), "lshrrev_b32": (2, "{1} >> ({0} & 31u)"),
    "ashrrev_i32": (2, "asuint(asint({1}) >> ({0} & 31u))"),
}
LANE_MASK = r"(vcc(?:_lo)?|s\d+|s\[\d+:\d+\])"
SALU_BIT64 = {
    "and": "uint2({0} & {2}, {1} & {3})",
    "or": "uint2({0} | {2}, {1} | {3})",
    "xor": "uint2({0} ^ {2}, {1} ^ {3})",
    "andn2": "uint2({0} & ~{2}, {1} & ~{3})",
    "orn2": "uint2({0} | ~{2}, {1} | ~{3})",
    "nand": "uint2(~({0} & {2}), ~({1} & {3}))",
    "nor": "uint2(~({0} | {2}), ~({1} | {3}))",
    "xnor": "uint2(~({0} ^ {2}), ~({1} ^ {3}))",
}
SALU_HELPERS = {
    "s_lshl_b64": (
        "uint2 s_lshl_b64(uint2 v, uint s) {\n"
        "    s &= 63u;\n"
        "    if (s == 0u) return v;\n"
        "    if (s < 32u) return uint2(v.x << s, (v.y << s) | (v.x >> (32u - s)));\n"
        "    return uint2(0u, v.x << (s - 32u));\n"
        "}"
    ),
    "s_lshr_b64": (
        "uint2 s_lshr_b64(uint2 v, uint s) {\n"
        "    s &= 63u;\n"
        "    if (s == 0u) return v;\n"
        "    if (s < 32u) return uint2((v.x >> s) | (v.y << (32u - s)), v.y >> s);\n"
        "    return uint2(v.y >> (s - 32u), 0u);\n"
        "}"
    ),
    "s_ashr_i64": (
        "uint2 s_ashr_i64(uint2 v, uint s) {\n"
        "    s &= 63u;\n"
        "    if (s == 0u) return v;\n"
        "    if (s < 32u) return uint2((v.x >> s) | (v.y << (32u - s)), asuint(asint(v.y) >> s));\n"
        "    return uint2(asuint(asint(v.y) >> (s - 32u)), asuint(asint(v.y) >> 31));\n"
        "}"
    ),
    "s_bfm_b64": (
        "uint2 s_bfm_b64(uint width, uint offset) {\n"
        "    uint w = width & 63u;\n"
        "    uint2 m = uint2(0u, 0u);\n"
        "    if (w > 0u && w < 32u) m = uint2((1u << w) - 1u, 0u);\n"
        "    else if (w == 32u) m = uint2(0xffffffffu, 0u);\n"
        "    else if (w > 32u) m = uint2(0xffffffffu, (1u << (w - 32u)) - 1u);\n"
        "    return s_lshl_b64(m, offset);\n"
        "}"
    ),
    "s_bfe_u32": (
        "uint s_bfe_u32(uint src, uint ctrl) {\n"
        "    uint off = ctrl & 31u;\n"
        "    uint w = (ctrl >> 16) & 127u;\n"
        "    if (w == 0u) return 0u;\n"
        "    if (off + w < 32u) return (src << (32u - off - w)) >> (32u - w);\n"
        "    return src >> off;\n"
        "}"
    ),
    "s_bfe_i32": (
        "uint s_bfe_i32(uint src, uint ctrl) {\n"
        "    uint off = ctrl & 31u;\n"
        "    uint w = (ctrl >> 16) & 127u;\n"
        "    if (w == 0u) return 0u;\n"
        "    if (off + w < 32u) return asuint(asint(src << (32u - off - w)) >> (32u - w));\n"
        "    return asuint(asint(src) >> off);\n"
        "}"
    ),
    "pack16_rne": (
        "// IEEE binary32 to binary16, round to nearest even, independent of driver f32tof16 rounding.\n"
        "uint pack16_rne(float value) {\n"
        "    uint bits = asuint(value);\n"
        "    uint sign = (bits >> 16) & 0x8000u;\n"
        "    uint abs_bits = bits & 0x7fffffffu;\n"
        "    if (abs_bits > 0x7f800000u) return 0x7e00u;\n"
        "    if (abs_bits >= 0x477ff000u) return sign | 0x7c00u;\n"
        "    if (abs_bits >= 0x38800000u) {\n"
        "        uint shifted = abs_bits - 0x38000000u;\n"
        "        uint round_bit = 0x00000fffu + ((shifted >> 13) & 1u);\n"
        "        return sign | ((shifted + round_bit) >> 13);\n"
        "    }\n"
        "    if (abs_bits < 0x33000000u) return sign;\n"
        "    uint sig = 0x00800000u | (abs_bits & 0x007fffffu);\n"
        "    uint shift = 126u - (abs_bits >> 23);\n"
        "    uint half_m = sig >> shift;\n"
        "    uint rem = sig & ((1u << shift) - 1u);\n"
        "    uint mid = 1u << (shift - 1u);\n"
        "    if (rem > mid || (rem == mid && (half_m & 1u) != 0u)) half_m += 1u;\n"
        "    return sign | half_m;\n"
        "}"
    ),
    "min16_num": (
        "float min16_num(float a, float b) {\n"
        "    return isnan(a) ? b : (isnan(b) ? a : min(a, b));\n"
        "}"
    ),
    "max16_num": (
        "float max16_num(float a, float b) {\n"
        "    return isnan(a) ? b : (isnan(b) ? a : max(a, b));\n"
        "}"
    ),
    "dot_u64_add": (
        "uint2 dot_u64_add(uint2 acc, uint v) {\n"
        "    uint lo = acc.x + v;\n"
        "    return uint2(lo, acc.y + (lo < acc.x ? 1u : 0u));\n"
        "}"
    ),
    "dot_u64_sat": (
        "uint dot_u64_sat(uint2 s) {\n"
        "    return s.y != 0u ? 0xffffffffu : s.x;\n"
        "}"
    ),
    "dot_i64_add": (
        "uint2 dot_i64_add(uint2 acc, int v) {\n"
        "    uint uv = asuint(v);\n"
        "    uint lo = acc.x + uv;\n"
        "    uint hi = acc.y + (v < 0 ? 0xffffffffu : 0u) + (lo < acc.x ? 1u : 0u);\n"
        "    return uint2(lo, hi);\n"
        "}"
    ),
    "dot_i64_sat": (
        "uint dot_i64_sat(uint2 s) {\n"
        "    if (asint(s.y) > 0 || (s.y == 0u && s.x > 0x7fffffffu)) return 0x7fffffffu;\n"
        "    if (asint(s.y) < -1 || (s.y == 0xffffffffu && s.x < 0x80000000u)) return 0x80000000u;\n"
        "    return s.x;\n"
        "}"
    ),
    "v_dot2_i32_i16": (
        "uint v_dot2_i32_i16(uint a0, uint a1, uint b0, uint b1, uint c, bool sat) {\n"
        "    int p0 = (asint(a0 << 16) >> 16) * (asint(b0 << 16) >> 16);\n"
        "    int p1 = (asint(a1 << 16) >> 16) * (asint(b1 << 16) >> 16);\n"
        "    uint2 s = dot_i64_add(dot_i64_add(dot_i64_add(uint2(0u, 0u), asint(c)), p0), p1);\n"
        "    return sat ? dot_i64_sat(s) : s.x;\n"
        "}"
    ),
    "v_dot2_u32_u16": (
        "uint v_dot2_u32_u16(uint a0, uint a1, uint b0, uint b1, uint c, bool sat) {\n"
        "    uint p0 = (a0 & 0xffffu) * (b0 & 0xffffu);\n"
        "    uint p1 = (a1 & 0xffffu) * (b1 & 0xffffu);\n"
        "    uint2 s = dot_u64_add(dot_u64_add(uint2(c, 0u), p0), p1);\n"
        "    return sat ? dot_u64_sat(s) : s.x;\n"
        "}"
    ),
    "v_dot4_i32_i8": (
        "uint v_dot4_i32_i8(uint a, uint b, uint c, bool sat) {\n"
        "    int sum = 0;\n"
        "    for (uint k = 0u; k < 4u; ++k) {\n"
        "        int x = asint(((a >> (k * 8u)) & 0xffu) << 24) >> 24;\n"
        "        int y = asint(((b >> (k * 8u)) & 0xffu) << 24) >> 24;\n"
        "        sum += x * y;\n"
        "    }\n"
        "    uint2 s = dot_i64_add(dot_i64_add(uint2(0u, 0u), asint(c)), sum);\n"
        "    return sat ? dot_i64_sat(s) : s.x;\n"
        "}"
    ),
    "v_dot4_u32_u8": (
        "uint v_dot4_u32_u8(uint a, uint b, uint c, bool sat) {\n"
        "    uint sum = 0u;\n"
        "    for (uint k = 0u; k < 4u; ++k) {\n"
        "        sum += ((a >> (k * 8u)) & 0xffu) * ((b >> (k * 8u)) & 0xffu);\n"
        "    }\n"
        "    uint2 s = dot_u64_add(uint2(c, 0u), sum);\n"
        "    return sat ? dot_u64_sat(s) : s.x;\n"
        "}"
    ),
    "v_dot8_i32_i4": (
        "uint v_dot8_i32_i4(uint a, uint b, uint c, bool sat) {\n"
        "    int sum = 0;\n"
        "    for (uint k = 0u; k < 8u; ++k) {\n"
        "        int x = asint(((a >> (k * 4u)) & 0xfu) << 28) >> 28;\n"
        "        int y = asint(((b >> (k * 4u)) & 0xfu) << 28) >> 28;\n"
        "        sum += x * y;\n"
        "    }\n"
        "    uint2 s = dot_i64_add(dot_i64_add(uint2(0u, 0u), asint(c)), sum);\n"
        "    return sat ? dot_i64_sat(s) : s.x;\n"
        "}"
    ),
    "v_dot8_u32_u4": (
        "uint v_dot8_u32_u4(uint a, uint b, uint c, bool sat) {\n"
        "    uint sum = 0u;\n"
        "    for (uint k = 0u; k < 8u; ++k) {\n"
        "        sum += ((a >> (k * 4u)) & 0xfu) * ((b >> (k * 4u)) & 0xfu);\n"
        "    }\n"
        "    uint2 s = dot_u64_add(uint2(c, 0u), sum);\n"
        "    return sat ? dot_u64_sat(s) : s.x;\n"
        "}"
    ),
}

VOP3P_F16 = {
    "add": (2, "({0} + {1})"),
    "mul": (2, "({0} * {1})"),
    "min": (2, "min16_num({0}, {1})"),
    "max": (2, "max16_num({0}, {1})"),
    "fma": (3, "mad({0}, {1}, {2})"),
}
INLINE_F16 = {
    "0.5": 0x3800, "-0.5": 0xb800, "1.0": 0x3c00, "-1.0": 0xbc00,
    "2.0": 0x4000, "-2.0": 0xc000, "4.0": 0x4400, "-4.0": 0xc400,
    "0.15915494": 0x3118,
}
VOP3P_INT16 = {
    "add_i16": (2, True), "add_u16": (2, True),
    "sub_i16": (2, True), "sub_u16": (2, True),
    "min_i16": (2, False), "min_u16": (2, False),
    "max_i16": (2, False), "max_u16": (2, False),
    "mul_lo_u16": (2, False),
    "mad_i16": (3, True), "mad_u16": (3, True),
    "lshlrev_b16": (2, False), "lshrrev_b16": (2, False), "ashrrev_i16": (2, False),
}


def sgpr_name(token):
    if not (m := re.fullmatch(r"s(\d+)", token)):
        raise ValueError(f"Expected a scalar register, got {token}")
    if not 0 <= int(m.group(1)) <= 105:
        raise ValueError(f"Scalar register out of range: {token}")
    return token


def sgpr_pair(token):
    if not (m := re.fullmatch(r"s\[(\d+):(\d+)\]", token)):
        raise ValueError(f"Expected a 64-bit scalar register pair, got {token}")
    regs = registers(*m.groups(), 2)
    if regs.start < 0 or regs.stop - 1 > 105 or regs.start % 2:
        raise ValueError(f"Invalid 64-bit scalar register pair: {token}")
    return [f"s{n}" for n in regs]


def scalar_u32(token, scalar, pending):
    if re.fullmatch(r"s\d+", token):
        sgpr_name(token)
        if token in pending:
            raise ValueError("Constant read before load completion")
        value = scalar[token]
        if isinstance(value, int):
            return f"{value}u"
        if isinstance(value, tuple) and len(value) == 2 and value[0] == "uint":
            return value[1]
        if isinstance(value, str) and value.startswith("params["):
            return f"asuint({value})"
        raise ValueError(f"Expected a numeric scalar register: {token}")
    if re.fullmatch(r"-?\d+", token):
        value = int(token)
        if not -0x80000000 <= value <= 0xffffffff:
            raise ValueError(f"32-bit scalar immediate out of range: {token}")
        return f"{value & 0xffffffff}u"
    if re.fullmatch(r"0x[0-9a-f]{1,8}", token):
        return f"{int(token, 16)}u"
    if re.fullmatch(r"-?\d+\.\d+", token):
        bits, = struct.unpack("<I", struct.pack("<f", float(token)))
        return f"{bits}u"
    raise ValueError(f"Unsupported 32-bit scalar operand {token}")


def scalar_u64(token, scalar, pending, *, signed=False):
    if token.startswith("s["):
        lo, hi = sgpr_pair(token)
        return scalar_u32(lo, scalar, pending), scalar_u32(hi, scalar, pending)
    if re.fullmatch(r"-?\d+", token):
        value = int(token)
        if not -0x8000000000000000 <= value <= 0xffffffffffffffff:
            raise ValueError(f"64-bit scalar immediate out of range: {token}")
        bits = value & 0xffffffffffffffff
        return f"{bits & 0xffffffff}u", f"{bits >> 32}u"
    if re.fullmatch(r"0x[0-9a-f]{1,8}", token):
        value = int(token, 16)
        high = 0xffffffff if signed and (value & 0x80000000) else 0
        return f"{value}u", f"{high}u"
    raise ValueError(f"Unsupported 64-bit scalar operand {token}")


def lane_mask(token):
    """Name the registers of a per-lane condition for either wavefront width."""
    if token.startswith("vcc"):
        return ["vcc"]
    if m := re.fullmatch(r"s\[(\d+):(\d+)\]", token):
        return [f"s{n}" for n in registers(*m.groups(), 2)]
    return [token]


# 32-bit scalar ALU on uniform register bits as (source count, result, SCC); SCC None leaves it unchanged.
# {r} names the result. Shift counts use their low five bits; signed overflow sets SCC for _i32 add/sub.
NONZERO = "{r} != 0u"
SCALAR = {
    "add_u32": (2, "{0} + {1}", "{r} < {0}"), "sub_u32": (2, "{0} - {1}", "{0} < {1}"),
    "add_i32": (2, "{0} + {1}", "(~({0} ^ {1}) & ({0} ^ {r})) >> 31 != 0u"),
    "sub_i32": (2, "{0} - {1}", "(({0} ^ {1}) & ({0} ^ {r})) >> 31 != 0u"),
    "mul_i32": (2, "{0} * {1}", None),
    "min_u32": (2, "min({0}, {1})", "{0} < {1}"), "max_u32": (2, "max({0}, {1})", "{0} > {1}"),
    "min_i32": (2, "asuint(min(asint({0}), asint({1})))", "asint({0}) < asint({1})"),
    "max_i32": (2, "asuint(max(asint({0}), asint({1})))", "asint({0}) > asint({1})"),
    "and_b32": (2, "{0} & {1}", NONZERO), "or_b32": (2, "{0} | {1}", NONZERO),
    "xor_b32": (2, "{0} ^ {1}", NONZERO), "andn2_b32": (2, "{0} & ~{1}", NONZERO),
    "orn2_b32": (2, "{0} | ~{1}", NONZERO), "not_b32": (1, "~{0}", NONZERO),
    "lshl_b32": (2, "{0} << ({1} & 31u)", NONZERO), "lshr_b32": (2, "{0} >> ({1} & 31u)", NONZERO),
    "ashr_i32": (2, "asuint(asint({0}) >> ({1} & 31u))", NONZERO),
}


def fp_operand(token, scalar, vector, pending, sample_dest, *, quad_values: dict[str, bool] | None = None):
    if re.fullmatch(r"-\d+", token):
        return f"asfloat({int(token) & 0xffffffff}u)"
    if token.startswith("-"):
        return "(-" + fp_operand(token[1:], scalar, vector, pending, sample_dest, quad_values=quad_values) + ")"
    if absolute := re.fullmatch(r"abs\(([^()|]+)\)|\|([^()|]+)\|", token):
        inner = absolute.group(1) or absolute.group(2)
        return f"abs({fp_operand(inner, scalar, vector, pending, sample_dest, quad_values=quad_values)})"
    if re.fullmatch(r"[vs]\d+", token):
        if token in pending:
            raise ValueError("Constant read before load completion")
        value = (vector if token.startswith("v") else scalar)[token]
        if quad_values is not None and token.startswith("v") and value not in quad_values:
            raise ValueError("Operand lacks a helper-lane definition")
        if isinstance(value, tuple) and value[0] == "uint":
            return f"asfloat({value[1]})"
        if isinstance(value, int):
            return f"asfloat({value}u)"
        if isinstance(value, tuple) and len(value) == 2 and value[0] == "uint":
            return f"asfloat({value[1]})"
        if not isinstance(value, str) or not (value.startswith(("sample", "alu", "params["))
                or (quad_values is not None and value in quad_values)):
            raise ValueError("Expected a numeric register")
        if token in sample_dest:
            raise ValueError("Sample used before vmcnt wait")
        return value
    if re.fullmatch(r"0x[0-9a-f]{1,8}|\d+", token):
        return f"asfloat({int(token, 0)}u)"
    if re.fullmatch(r"\d+\.\d+", token):
        return token + "f"
    raise ValueError(f"Unsupported arithmetic operand {token}")


def int_operand(token, scalar, vector, pending, sample_dest, *, signed):
    if re.fullmatch(r"v\d+", token):
        value = vector[token]
        if not isinstance(value, str) or not value.startswith(("sample", "alu", "params[")):
            raise ValueError("Expected a numeric register")
        if token in sample_dest:
            raise ValueError("Sample used before vmcnt wait")
        return f"asint({value})" if signed else f"asuint({value})"
    bits = scalar_u32(token, scalar, pending)
    if signed and bits.startswith("asuint(") and bits.endswith(")"):
        return f"asint({bits[7:-1]})"
    return f"asint({bits})" if signed else bits
def vop3p_f16_word(token, scalar, vector, pending, sample_dest):
    """Return a 32-bit uint HLSL expression for one packed binary16 VOP3P source operand.

    On gfx1013 (RDNA 2), inline f16 float constants occupy bits [15:0] with bits [31:16] zeroed.
    """
    if token in INLINE_F16:
        return f"{INLINE_F16[token]}u"
    if m := re.fullmatch(r"([vs])(\d+)", token):
        limit = 255 if m.group(1) == "v" else 105
        if not 0 <= int(m.group(2)) <= limit:
            raise ValueError(f"Register out of range: {token}")
        if token in pending:
            raise ValueError("Constant read before load completion")
        value = (vector if m.group(1) == "v" else scalar)[token]
        if isinstance(value, int):
            return f"{value}u"
        if isinstance(value, tuple) and len(value) == 2 and value[0] == "uint":
            return value[1]
        if not isinstance(value, str) or not value.startswith(("sample", "alu", "params[")):
            raise ValueError("Expected a numeric register")
        if token in sample_dest:
            raise ValueError("Sample used before vmcnt wait")
        return f"asuint({value})"
    if re.fullmatch(r"0x[0-9a-f]{1,8}", token):
        return f"{int(token, 16)}u"
    if re.fullmatch(r"-?\d+", token):
        value = int(token)
        if not -16 <= value <= 64:
            raise ValueError(f"Unsupported packed integer inline constant {token}")
        return f"{value & 0xffffffff}u"
    raise ValueError(f"Unsupported packed operand {token}")


def parse_vop3p_f16(rest, arity):
    parts_ = rest.split(", ")
    if len(parts_) != arity:
        raise ValueError("Invalid packed operand count")
    last_op, _, mod_text = parts_[-1].partition(" ")
    operands = [*parts_[:-1], last_op]
    opts = {}
    clamp = False
    for mod in mod_text.split():
        if mod == "clamp":
            if clamp:
                raise ValueError("Duplicate clamp modifier")
            clamp = True
        elif m := re.fullmatch(r"(op_sel|op_sel_hi|neg_lo|neg_hi):\[([01](?:,[01])*)\]", mod):
            name = m.group(1)
            bits = [int(b) for b in m.group(2).split(",")]
            if name in opts or len(bits) != arity:
                raise ValueError(f"Invalid packed modifier {mod}")
            opts[name] = bits
        else:
            raise ValueError(f"Unsupported packed modifier {mod}")
    return operands, opts, clamp
def fma_mix_reg(token, scalar, vector, pending, sample_dest):
    m = re.fullmatch(r"([vs])(\d+)", token)
    if not m:
        raise ValueError(f"Expected register, got {token}")
    limit = 255 if m.group(1) == "v" else 105
    if not 0 <= int(m.group(2)) <= limit:
        raise ValueError(f"Register out of range: {token}")
    if token in pending:
        raise ValueError("Constant read before load completion")
    value = (vector if m.group(1) == "v" else scalar)[token]
    if isinstance(value, int):
        return f"asfloat({value}u)", f"{value}u"
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "uint":
        return f"asfloat({value[1]})", value[1]
    if not isinstance(value, str) or not value.startswith(("sample", "alu", "params[")):
        raise ValueError("Expected a numeric register")
    if token in sample_dest:
        raise ValueError("Sample used before vmcnt wait")
    return value, f"asuint({value})"


def fma_mix_operand(token, is_f16, is_hi, scalar, vector, pending, sample_dest):
    """Lower one v_fma_mix* source operand, widening f16 sources before abs/neg."""
    if token in INLINE_F16:
        if is_f16:
            word = f"{INLINE_F16[token]}u"
            return f"widen16({word} >> 16)" if is_hi else f"widen16({word})"
        return f"({token}f)"
    neg = False
    if not re.fullmatch(r"-\d+", token) and token.startswith("-"):
        neg = True
        token = token[1:]
    use_abs = False
    if m := re.fullmatch(r"abs\(([^()|]+)\)|\|([^()|]+)\|", token):
        use_abs = True
        token = m.group(1) or m.group(2)
    if is_f16:
        if token in INLINE_F16:
            word = f"{INLINE_F16[token]}u"
        elif re.fullmatch(r"[vs]\d+", token):
            _, word = fma_mix_reg(token, scalar, vector, pending, sample_dest)
        elif re.fullmatch(r"0x[0-9a-f]{1,8}", token):
            word = f"{int(token, 16)}u"
        elif re.fullmatch(r"-?\d+", token):
            val = int(token)
            if not -16 <= val <= 64:
                raise ValueError(f"Unsupported mixed-precision inline constant {token}")
            word = f"{val & 0xffffffff}u"
        else:
            raise ValueError(f"Unsupported mixed-precision operand {token}")
        expr = f"widen16({word} >> 16)" if is_hi else f"widen16({word})"
    else:
        if token in INLINE_F16:
            expr = f"({token}f)"
        elif re.fullmatch(r"[vs]\d+", token):
            expr, _ = fma_mix_reg(token, scalar, vector, pending, sample_dest)
        elif re.fullmatch(r"0x[0-9a-f]{1,8}", token):
            expr = f"asfloat({int(token, 16)}u)"
        elif re.fullmatch(r"-?\d+", token):
            val = int(token)
            if not -16 <= val <= 64:
                raise ValueError(f"Unsupported mixed-precision inline constant {token}")
            expr = f"asfloat({val & 0xffffffff}u)"
        else:
            raise ValueError(f"Unsupported mixed-precision operand {token}")
    if use_abs:
        expr = f"abs({expr})"
    if neg:
        expr = f"(-{expr})"
    return expr


def parse_fma_mix(rest):
    parts_ = rest.split(", ")
    if len(parts_) != 3:
        raise ValueError("Invalid mixed-precision operand count")
    last_op, _, mod_text = parts_[-1].partition(" ")
    operands = [*parts_[:-1], last_op]
    opts = {}
    clamp = False
    for mod in mod_text.split():
        if mod == "clamp":
            if clamp:
                raise ValueError("Duplicate clamp modifier")
            clamp = True
        elif m := re.fullmatch(r"(op_sel|op_sel_hi):\[([01],[01],[01])\]", mod):
            name = m.group(1)
            if name in opts:
                raise ValueError(f"Duplicate modifier {mod}")
            opts[name] = [int(b) for b in m.group(2).split(",")]
        else:
            raise ValueError(f"Unsupported mixed-precision modifier {mod}")
    op_sel = opts.get("op_sel", [0, 0, 0])
    op_sel_hi = opts.get("op_sel_hi", [0, 0, 0])
    if any(hi and not f16 for f16, hi in zip(op_sel, op_sel_hi)):
        raise ValueError("High-half selection requires an f16 operand")
    return operands, op_sel, op_sel_hi, clamp
def dot_word(token, scalar, vector, pending, sample_dest, *, allow_f16=False):
    """Return a 32-bit uint HLSL expression for one VOP3P dot-product source operand."""
    if allow_f16 and token in INLINE_F16:
        return f"{INLINE_F16[token]}u"
    if m := re.fullmatch(r"([vs])(\d+)", token):
        limit = 255 if m.group(1) == "v" else 105
        if not 0 <= int(m.group(2)) <= limit:
            raise ValueError(f"Register out of range: {token}")
        if token in pending:
            raise ValueError("Constant read before load completion")
        value = (vector if m.group(1) == "v" else scalar)[token]
        if isinstance(value, int):
            return f"{value}u"
        if isinstance(value, tuple) and len(value) == 2 and value[0] == "uint":
            return value[1]
        if not isinstance(value, str) or not value.startswith(("sample", "alu", "params[")):
            raise ValueError("Expected a numeric register")
        if token in sample_dest:
            raise ValueError("Sample used before vmcnt wait")
        return f"asuint({value})"
    if re.fullmatch(r"0x[0-9a-f]{1,8}", token):
        return f"{int(token, 16)}u"
    if re.fullmatch(r"-?\d+", token):
        val = int(token)
        if not -16 <= val <= 64:
            raise ValueError(f"Unsupported dot-product inline constant {token}")
        return f"{val & 0xffffffff}u"
    raise ValueError(f"Unsupported dot-product operand {token}")


def dot_f32_accum(token, scalar, vector, pending, sample_dest):
    if token in INLINE_F16:
        return f"({token}f)"
    return f"asfloat({dot_word(token, scalar, vector, pending, sample_dest)})"


def parse_dot(rest, *, allow_opsel, allow_neg):
    parts_ = rest.split(", ")
    if len(parts_) != 3:
        raise ValueError("Invalid dot-product operand count")
    last_op, _, mod_text = parts_[-1].partition(" ")
    operands = [*parts_[:-1], last_op]
    opts = {}
    clamp = False
    for mod in mod_text.split():
        if mod == "clamp":
            if clamp:
                raise ValueError("Duplicate clamp modifier")
            clamp = True
        elif m := re.fullmatch(r"(op_sel|op_sel_hi|neg_lo|neg_hi):\[([01],[01],[01])\]", mod):
            name = m.group(1)
            if name in opts:
                raise ValueError(f"Duplicate modifier {mod}")
            if name in ("op_sel", "op_sel_hi") and not allow_opsel:
                raise ValueError(f"Unsupported dot-product modifier {mod}")
            if name in ("neg_lo", "neg_hi") and not allow_neg:
                raise ValueError(f"Unsupported dot-product modifier {mod}")
            opts[name] = [int(b) for b in m.group(2).split(",")]
        else:
            raise ValueError(f"Unsupported dot-product modifier {mod}")
    op_sel = opts.get("op_sel", [0, 0, 0])
    op_sel_hi = opts.get("op_sel_hi", [1, 1, 1])
    neg_lo = opts.get("neg_lo", [0, 0, 0])
    neg_hi = opts.get("neg_hi", [0, 0, 0])
    if op_sel[2] != 0 or op_sel_hi[2] != 1 or neg_hi[2] != 0:
        raise ValueError("Unsupported accumulator lane modifier on dot product")
    return operands, op_sel, op_sel_hi, neg_lo, neg_hi, clamp
def vop3p_int_word(token, scalar, vector, pending, sample_dest):
    """Return a 32-bit uint HLSL expression for one packed 16-bit integer VOP3P source operand."""
    if m := re.fullmatch(r"([vs])(\d+)", token):
        limit = 255 if m.group(1) == "v" else 105
        if not 0 <= int(m.group(2)) <= limit:
            raise ValueError(f"Register out of range: {token}")
        if token in pending:
            raise ValueError("Constant read before load completion")
        value = (vector if m.group(1) == "v" else scalar)[token]
        if isinstance(value, int):
            return f"{value}u"
        if isinstance(value, tuple) and len(value) == 2 and value[0] == "uint":
            return value[1]
        if not isinstance(value, str) or not value.startswith(("sample", "alu", "params[")):
            raise ValueError("Expected a numeric register")
        if token in sample_dest:
            raise ValueError("Sample used before vmcnt wait")
        return f"asuint({value})"
    if re.fullmatch(r"0x[0-9a-f]{1,8}", token):
        return f"{int(token, 16)}u"
    if re.fullmatch(r"-?\d+", token):
        value = int(token)
        if not -16 <= value <= 64:
            raise ValueError(f"Unsupported packed integer inline constant {token}")
        return f"{value & 0xffffffff}u"
    raise ValueError(f"Unsupported packed integer operand {token}")


def parse_vop3p_int16(rest, arity, *, allow_clamp):
    parts_ = rest.split(", ")
    if len(parts_) != arity:
        raise ValueError("Invalid packed integer operand count")
    last_op, _, mod_text = parts_[-1].partition(" ")
    operands = [*parts_[:-1], last_op]
    opts = {}
    clamp = False
    for mod in mod_text.split():
        if mod == "clamp":
            if clamp or not allow_clamp:
                raise ValueError("Invalid clamp modifier")
            clamp = True
        elif m := re.fullmatch(r"(op_sel|op_sel_hi):\[([01](?:,[01])*)\]", mod):
            name = m.group(1)
            bits = [int(b) for b in m.group(2).split(",")]
            if name in opts or len(bits) != arity:
                raise ValueError(f"Invalid packed integer modifier {mod}")
            opts[name] = bits
        else:
            raise ValueError(f"Unsupported packed integer modifier {mod}")
    return operands, opts, clamp


def pk_int16_lane_expr(op, word_vars, sel, clamp):
    def u16(w, hi):
        return f"(({w} >> 16) & 0xffffu)" if hi else f"({w} & 0xffffu)"

    def i16(w, hi):
        return f"(asint({w}) >> 16)" if hi else f"(asint({w} << 16) >> 16)"

    if op == "add_i16":
        r = f"({i16(word_vars[0], sel[0])} + {i16(word_vars[1], sel[1])})"
        return f"(asuint(clamp({r}, -32768, 32767)) & 0xffffu)" if clamp else f"(asuint({r}) & 0xffffu)"
    if op == "add_u16":
        r = f"({u16(word_vars[0], sel[0])} + {u16(word_vars[1], sel[1])})"
        return f"min({r}, 65535u)" if clamp else f"({r} & 0xffffu)"
    if op == "sub_i16":
        r = f"({i16(word_vars[0], sel[0])} - {i16(word_vars[1], sel[1])})"
        return f"(asuint(clamp({r}, -32768, 32767)) & 0xffffu)" if clamp else f"(asuint({r}) & 0xffffu)"
    if op == "sub_u16":
        a, b = u16(word_vars[0], sel[0]), u16(word_vars[1], sel[1])
        return f"({a} < {b} ? 0u : ({a} - {b}))" if clamp else f"(({a} - {b}) & 0xffffu)"
    if op == "min_i16":
        return f"(asuint(min({i16(word_vars[0], sel[0])}, {i16(word_vars[1], sel[1])})) & 0xffffu)"
    if op == "min_u16":
        return f"min({u16(word_vars[0], sel[0])}, {u16(word_vars[1], sel[1])})"
    if op == "max_i16":
        return f"(asuint(max({i16(word_vars[0], sel[0])}, {i16(word_vars[1], sel[1])})) & 0xffffu)"
    if op == "max_u16":
        return f"max({u16(word_vars[0], sel[0])}, {u16(word_vars[1], sel[1])})"
    if op == "mul_lo_u16":
        return f"(({u16(word_vars[0], sel[0])} * {u16(word_vars[1], sel[1])}) & 0xffffu)"
    if op == "mad_i16":
        r = f"({i16(word_vars[0], sel[0])} * {i16(word_vars[1], sel[1])} + {i16(word_vars[2], sel[2])})"
        return f"(asuint(clamp({r}, -32768, 32767)) & 0xffffu)" if clamp else f"(asuint({r}) & 0xffffu)"
    if op == "mad_u16":
        r = f"({u16(word_vars[0], sel[0])} * {u16(word_vars[1], sel[1])} + {u16(word_vars[2], sel[2])})"
        return f"min({r}, 65535u)" if clamp else f"({r} & 0xffffu)"
    if op == "lshlrev_b16":
        return f"(({u16(word_vars[1], sel[1])} << ({u16(word_vars[0], sel[0])} & 15u)) & 0xffffu)"
    if op == "lshrrev_b16":
        return f"({u16(word_vars[1], sel[1])} >> ({u16(word_vars[0], sel[0])} & 15u))"
    return f"(asuint({i16(word_vars[1], sel[1])} >> ({u16(word_vars[0], sel[0])} & 15u)) & 0xffffu)"
# SDWA selections: (bit offset, width). A source selection is zero-extended, or sign-extended with sext();
# a destination selection writes the low bits of the result into that field. Following the equivalences in
# LLVM's SIPeepholeSDWA, UNUSED_PAD zeroes the other bits and UNUSED_PRESERVE keeps them from the destination.
SDWA_SEL = {"BYTE_0": (0, 8), "BYTE_1": (8, 8), "BYTE_2": (16, 8), "BYTE_3": (24, 8),
            "WORD_0": (0, 16), "WORD_1": (16, 16), "DWORD": (0, 32)}


def sdwa_source(word, sel, sext):
    offset, width = SDWA_SEL[sel]
    if width == 32:
        return word
    if sext:
        return f"asuint(asint({word} << {32 - offset - width}u) >> {32 - width}u)"
    return f"(({word} >> {offset}u) & {(1 << width) - 1:#x}u)"


def sdwa_dest(result, sel, unused, old):
    offset, width = SDWA_SEL[sel]
    if width == 32:
        return result
    if unused == "UNUSED_SEXT":
        raise ValueError("UNUSED_SEXT is not lowered")
    field = f"(({result} & {(1 << width) - 1:#x}u) << {offset}u)"
    if unused == "UNUSED_PAD":
        return field
    # The kept bits must come from a tracked numeric value; hardware would keep whatever an untracked register holds.
    if not (isinstance(old, str) and old.startswith(("sample", "alu", "params["))):
        raise ValueError("UNUSED_PRESERVE needs a tracked numeric destination")
    keep = ~(((1 << width) - 1) << offset) & 0xffffffff
    return f"((asuint({old}) & {keep:#x}u) | {field})"


def lower(instructions, texture_sgpr, interp_sgpr, sampler, constant_descriptor=None, *, second_texture_offset=None):
    """Track descriptor provenance, interpolation and packed exports through register writes."""
    if texture_sgpr == interp_sgpr or not 0 <= texture_sgpr <= 105 or not 0 <= interp_sgpr <= 105:
        raise ValueError("Invalid scalar inputs")
    if sampler is not None and (len(sampler) != 4 or any(not 0 <= word <= 0xffffffff for word in sampler)):
        raise ValueError("Invalid sampler descriptor words")
    scalar = {f"s{texture_sgpr}": "texture_ptr", f"s{interp_sgpr}": "interp_base"}
    vector = {"v0": "bary_i", "v1": "bary_j"}
    # Each emitted value has a unique identity. Entry-only writes never gain a helper definition.
    # The boolean records whether the quad value depends on interpolated coordinates.
    quad_values = {"uv.x": True, "uv.y": True}
    lines = [PREAMBLE] if sampler is None else [SAMPLED_RESOURCES, PREAMBLE]
    if constant_descriptor is not None:
        if constant_descriptor < 0 or constant_descriptor % 4:
            raise ValueError("Invalid constant descriptor offset")
        lines.insert(0, "cbuffer Constants : register(b0) { float4 params[16]; };")
    bindings = {} if sampler is None else {0: "tex"}
    if second_texture_offset is not None:
        if sampler is None:
            raise ValueError("Second texture requires a sampler")
        if second_texture_offset < 32 or second_texture_offset % 32:
            raise ValueError("Second texture descriptor must have a distinct aligned offset")
        bindings[second_texture_offset] = "tex1"
        lines.insert(0, "Texture2D<float4> tex1 : register(t1);")
    if constant_descriptor is not None and any(
            start < constant_descriptor + 16 and constant_descriptor < start + 32 for start in bindings):
        raise ValueError("Texture and constant descriptors overlap")
    pending = set()
    scc = None
    helpers = set()
    mask = "entry"
    samples = 0
    sample_dest = {}  # VGPR -> issue number of its outstanding image_sample
    exported = False
    ended = False
    entry_line = len(lines)
    lines.append("float4 main(float2 uv : TEXCOORD0) : SV_Target0 {")
    attribute_input = False
    for i, line in enumerate(instructions):
        def match(pattern):
            return re.fullmatch(pattern, line)

        if ended:
            raise ValueError("Instruction follows program end")
        if match(r"(?:s_inst_prefetch|s_nop) (?:0x[0-9a-f]+|\d+)"):
            continue
        if match(r"s_waitcnt_expcnt\s+(?:(?:null|s\d+),\s*)?(?:0x[0-7]|[0-7])"):
            continue
        if match(r"s_waitcnt\s+expcnt\([0-7]\)"):
            continue
        if exported and line != "s_endpgm":
            raise ValueError("Unexpected instruction after color export")
        # Scheduling hints with no effect on results. Each operand is limited to the width
        # of its RDNA 2 field: clause length minus one, wave priority and sleep duration.
        if m := match(r"(s_clause|s_setprio|s_sleep) (0x[0-9a-f]+|\d+)"):
            if int(m[2], 0) > HINT_LIMITS[m[1]]:
                raise ValueError(f"Invalid {m[1]} operand")
            continue
        if line == "s_waitcnt lgkmcnt(0)":
            pending.clear()
        elif m := match(r"s_waitcnt vmcnt\((\d+)\)"):
            limit = int(m[1])
            if limit > 63:
                raise ValueError("VM wait count exceeds the gfx10 six-bit field")
            # Only the newest limit samples can remain outstanding. Filtering
            # the current set also preserves guarantees made by earlier waits.
            sample_dest = {reg: issue for reg, issue in sample_dest.items() if issue > samples - limit}
        elif match(r"s_waitcnt (?:vmcnt\(0\) lgkmcnt\(0\)|lgkmcnt\(0\) vmcnt\(0\))"):
            pending.clear()
            sample_dest.clear()
        elif m := match(r"s_mov_b64 s\[(\d+):(\d+)\], exec"):
            regs = registers(*m.groups(), 2)
            for n in regs:
                if f"s{n}" in pending:
                    raise ValueError("Write overlaps an outstanding scalar load")
                scalar[f"s{n}"] = (mask, n-regs.start)
        elif line == "s_wqm_b64 exec, exec" and mask == "entry":
            mask = "quad"
            scc = None
        elif m := match(r"s_and_b64 exec, exec, s\[(\d+):(\d+)\]"):
            pair = registers(*m.groups(), 2)
            if mask != "quad" or [scalar.get(f"s{n}") for n in pair] != [("entry",0),("entry",1)]:
                raise ValueError("Unsupported execution mask")
            mask = "entry"
            scc = None
        elif m := match(r"s_(cmov|not|brev)_b64 (s\[\d+:\d+\]), ([^,]+)"):
            op, dst, src = m.groups()
            regs = sgpr_pair(dst)
            if pending.intersection(regs):
                raise ValueError("Write overlaps an outstanding scalar load")
            s0, s1 = scalar_u64(src, scalar, pending)
            if op == "cmov":
                if scc is None:
                    raise ValueError("Conditional move requires a tracked SCC")
                d0, d1 = scalar_u64(dst, scalar, pending)
                expr = f"({scc} ? uint2({s0}, {s1}) : uint2({d0}, {d1}))"
            elif op == "not":
                expr = f"uint2(~{s0}, ~{s1})"
            else:
                expr = f"uint2(reversebits({s1}), reversebits({s0}))"
            lines.append(f"    uint2 salu{i} = {expr};")
            scalar[regs[0]], scalar[regs[1]] = ("uint", f"salu{i}.x"), ("uint", f"salu{i}.y")
            if op == "not":
                scc = f"((salu{i}.x | salu{i}.y) != 0u)"
        elif m := match(rf"s_({'|'.join(SALU_BIT64)})_b64 (s\[\d+:\d+\]), ([^,]+), ([^,]+)"):
            op, dst, src0, src1 = m.groups()
            regs = sgpr_pair(dst)
            if pending.intersection(regs):
                raise ValueError("Write overlaps an outstanding scalar load")
            a0, a1 = scalar_u64(src0, scalar, pending)
            b0, b1 = scalar_u64(src1, scalar, pending)
            lines.append(f"    uint2 salu{i} = {SALU_BIT64[op].format(a0, a1, b0, b1)};")
            scalar[regs[0]], scalar[regs[1]] = ("uint", f"salu{i}.x"), ("uint", f"salu{i}.y")
            scc = f"((salu{i}.x | salu{i}.y) != 0u)"
        elif m := match(r"(s_lshl_b64|s_lshr_b64|s_ashr_i64|s_bfm_b64) (s\[\d+:\d+\]), ([^,]+), ([^,]+)"):
            op, dst, src0, src1 = m.groups()
            regs = sgpr_pair(dst)
            if pending.intersection(regs):
                raise ValueError("Write overlaps an outstanding scalar load")
            shift = scalar_u32(src1, scalar, pending)
            if op == "s_bfm_b64":
                width = scalar_u32(src0, scalar, pending)
                helpers.update(("s_lshl_b64", "s_bfm_b64"))
                expr = f"s_bfm_b64({width}, {shift})"
            else:
                a0, a1 = scalar_u64(src0, scalar, pending, signed=(op == "s_ashr_i64"))
                helpers.add(op)
                expr = f"{op}(uint2({a0}, {a1}), {shift})"
            lines.append(f"    uint2 salu{i} = {expr};")
            scalar[regs[0]], scalar[regs[1]] = ("uint", f"salu{i}.x"), ("uint", f"salu{i}.y")
            if op != "s_bfm_b64":
                scc = f"((salu{i}.x | salu{i}.y) != 0u)"
        elif m := match(r"(s_bfe_u32|s_bfe_i32) (s\d+), ([^,]+), ([^,]+)"):
            op, dst, src0, src1 = m.groups()
            sgpr_name(dst)
            if dst in pending:
                raise ValueError("Write overlaps an outstanding scalar load")
            a = scalar_u32(src0, scalar, pending)
            b = scalar_u32(src1, scalar, pending)
            helpers.add(op)
            lines.append(f"    uint salu{i} = {op}({a}, {b});")
            scalar[dst] = ("uint", f"salu{i}")
            scc = f"(salu{i} != 0u)"
        elif m := match(r"(s_cmp_eq_u64|s_cmp_lg_u64) ([^,]+), ([^,]+)"):
            op, src0, src1 = m.groups()
            a0, a1 = scalar_u64(src0, scalar, pending)
            b0, b1 = scalar_u64(src1, scalar, pending)
            cond = f"({a0} == {b0} && {a1} == {b1})" if op == "s_cmp_eq_u64" else f"({a0} != {b0} || {a1} != {b1})"
            lines.append(f"    bool scc{i} = {cond};")
            scc = f"scc{i}"
        elif m := match(r"(s_bitcmp0_b64|s_bitcmp1_b64) ([^,]+), ([^,]+)"):
            op, src0, src1 = m.groups()
            a0, a1 = scalar_u64(src0, scalar, pending)
            b = scalar_u32(src1, scalar, pending)
            bit_test = f"((({b} & 32u) != 0u ? ({a1} & (1u << ({b} & 31u))) : ({a0} & (1u << ({b} & 31u)))) {'==' if op == 's_bitcmp0_b64' else '!='} 0u)"
            lines.append(f"    bool scc{i} = {bit_test};")
            scc = f"scc{i}"
        elif m := match(r"(s_bitcmp0_b32|s_bitcmp1_b32) ([^,]+), ([^,]+)"):
            op, src0, src1 = m.groups()
            a = scalar_u32(src0, scalar, pending)
            b = scalar_u32(src1, scalar, pending)
            bit_test = f"(({a} & (1u << ({b} & 31u))) {'==' if op == 's_bitcmp0_b32' else '!='} 0u)"
            lines.append(f"    bool scc{i} = {bit_test};")
            scc = f"scc{i}"
        elif m := match(r"(s_cmp_(?:eq|lg|gt|ge|lt|le)_(?:i32|u32)) ([^,]+), ([^,]+)"):
            op, src0, src1 = m.groups()
            signed = op.endswith("_i32")
            comp = op.split("_")[2]
            op_sym = {"eq": "==", "lg": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}[comp]
            a = scalar_u32(src0, scalar, pending)
            b = scalar_u32(src1, scalar, pending)
            cond = f"(asint({a}) {op_sym} asint({b}))" if signed else f"({a} {op_sym} {b})"
            lines.append(f"    bool scc{i} = {cond};")
            scc = f"scc{i}"
        elif m := match(r"(s_mov_b32|s_movk_i32) (s\d+|m0), (s\d+|0x[0-9a-f]+|\d+)"):
            op, dst, src = m.groups()
            if src in pending or dst in pending:
                raise ValueError("Unresolved scalar memory dependency")
            val = scalar[src] if src.startswith("s") else int(src, 0)
            if op == "s_movk_i32":
                if not isinstance(val, int) or not 0 <= val <= 0xffff:
                    raise ValueError("Invalid movk immediate")
                val = (val if val < 0x8000 else val - 0x10000) & 0xffffffff
            scalar[dst] = val
        elif m := match(rf"s_({'|'.join(SCALAR)}) (s\d+), (.+)"):
            # Uniform values become HLSL uint locals; an exec mask does not affect them.
            op, dst, operands = m.groups()
            arity, template, carry = SCALAR[op]
            tokens = operands.split(", ")
            if dst in pending:
                raise ValueError("Write overlaps an outstanding scalar load")
            if len(tokens) != arity:
                raise ValueError("Invalid scalar operation")
            values = [scalar_u32(t, scalar, pending) for t in tokens]
            lines.append(f"    uint salu{i} = {template.format(*values)};")
            if carry:
                scc = f"({carry.format(*values, r=f'salu{i}')})"
            scalar[dst] = ("uint", f"salu{i}")
        elif m := match(r"s_cselect_b32 (s\d+), ([^,]+), ([^,]+)"):
            dst, *tokens = m.groups()
            if scc is None:
                raise ValueError("Select condition is not a tracked scalar result")
            if dst in pending:
                raise ValueError("Write overlaps an outstanding scalar load")
            values = [scalar_u32(t, scalar, pending) for t in tokens]
            lines.append(f"    uint salu{i} = {scc} ? {values[0]} : {values[1]};")
            scalar[dst] = ("uint", f"salu{i}")
        elif m := match(r"(s_addk_i32|s_mulk_i32) (s\d+), (-?\d+|0x[0-9a-f]+)"):
            op, dst, imm = m.groups()
            operand = scalar_u32(dst, scalar, pending)
            word = int(imm, 16) if imm.startswith("0x") else int(imm)
            if not -0x8000 <= word <= 0xffff:
                raise ValueError("Invalid sopk immediate")
            signed = word - 0x10000 if word >= 0 and word & 0x8000 else word
            addend = signed & 0xffffffff
            operation = "+" if op == "s_addk_i32" else "*"
            scalar[dst] = ("uint", f"alu{i}")
            lines.append(f"    uint alu{i} = ({operand} {operation} {addend}u);")

            if op == "s_addk_i32":
                scc = f"((({operand} ^ alu{i}) & ({addend}u ^ alu{i}) & 0x80000000u) != 0u)"
        elif m := match(r"s_cmpk_(eq|lg|gt|ge|lt|le)_(i32|u32) (s\d+), (0x[0-9a-f]{1,4}|\d+)"):
            # SCC = S0 against SIMM16, sign-extended for _i32 and zero-extended for _u32. S0 is the SDST field.
            relation, kind, src, imm = m.groups()
            value = scalar_u32(sgpr_name(src), scalar, pending)
            word = int(imm, 0)
            if word > 0xffff:
                raise ValueError("Invalid sopk immediate")
            op_sym = {"eq": "==", "lg": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}[relation]
            if kind == "i32":
                cond = f"(asint({value}) {op_sym} {word - 0x10000 if word & 0x8000 else word})"
            else:
                cond = f"({value} {op_sym} {word}u)"
            lines.append(f"    bool scc{i} = {cond};")
            scc = f"scc{i}"
        elif m := match(r"s_cmovk_i32 (s\d+), (0x[0-9a-f]{1,4}|\d+)"):
            # D = SCC ? sign-extended SIMM16 : D. The old value must be a known scalar result.
            dst, imm = m.groups()
            word = int(imm, 0)
            if word > 0xffff:
                raise ValueError("Invalid sopk immediate")
            if scc is None:
                raise ValueError("Conditional move requires a tracked SCC")
            old = scalar_u32(sgpr_name(dst), scalar, pending)
            lines.append(f"    uint salu{i} = {scc} ? {(word - 0x10000 if word & 0x8000 else word) & 0xffffffff}u : {old};")
            scalar[dst] = ("uint", f"salu{i}")
        elif m := match(r"s_load_dwordx8 s\[(\d+):(\d+)\], s\[(\d+):(\d+)\], (null|0x[0-9a-f]+)"):
            dst = registers(*m.groups()[:2], 8)
            addr = registers(*m.groups()[2:4], 2)
            offset = 0 if m[5] == "null" else int(m[5], 16)
            if offset not in bindings:
                raise ValueError("Unbound texture descriptor offset")
            if any(f"s{n}" in pending for n in (*addr, *dst)) or [scalar.get(f"s{n}") for n in addr] != ["texture_ptr", 0]:
                raise ValueError("Texture address differs from binding contract")
            for j, n in enumerate(dst):
                scalar[f"s{n}"] = (bindings[offset], j)
                pending.add(f"s{n}")
        elif m := match(r"(?:(s_load_dwordx4|s_buffer_load_dwordx[24]) s\[(\d+):(\d+)\]|s_buffer_load_dword s(\d+))"
                        r", s\[(\d+):(\d+)\], (0x[0-9a-f]+|null)"):
            op, first, last, single, low, high, imm = m.groups()
            offset = 0 if imm == "null" else int(imm, 16)
            descriptor = op == "s_load_dwordx4"
            width = 1 if single is not None else 2 if op == "s_buffer_load_dwordx2" else 4
            if single is not None:
                first = last = single
            # LLVM names a 64-bit scalar destination only as an even-aligned pair.
            if width < 4 and (int(last) > 105 or int(first) % width or int(low) % 4 or int(high) > 105):
                raise ValueError("Invalid scalar buffer registers")
            dst = registers(first, last, width)
            base = [f"s{n}" for n in registers(low, high, 2 if descriptor else 4)]
            expected = ["texture_ptr", 0] if descriptor else [("cbv", n) for n in range(4)]
            if constant_descriptor is None or pending.intersection(base) or [scalar.get(r) for r in base] != expected:
                raise ValueError("Unresolved constant buffer binding")
            if descriptor and offset != constant_descriptor:
                raise ValueError("Constant descriptor differs from binding contract")
            if not descriptor and (offset % 4 or offset + 4 * len(dst) > 256):
                raise ValueError("Constant read exceeds binding or is unaligned")
            for j, n in enumerate(dst):
                if f"s{n}" in pending:
                    raise ValueError("Write overlaps an outstanding scalar load")
                word = offset // 4 + j
                scalar[f"s{n}"] = ("cbv", j) if descriptor else f"params[{word // 4}].{'xyzw'[word % 4]}"
                pending.add(f"s{n}")
        elif m := match(r"v_cube(id|sc|tc|ma)_f32(?:_e64)? (v\d+), ([^,]+), ([^,]+), ([^,]+?)( clamp)?"):
            # LLVM ConstantFoldAMDGCNCubeIntrinsic: z, then y, then x is the major axis; a face is negative
            # only for a nonzero, non-NaN negative coordinate.
            op, dst, *tokens, clamp = m.groups()
            if mask != "entry":
                raise ValueError("Invalid cube operation")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            x, y, z = [fp_operand(t, scalar, vector, pending, sample_dest) for t in tokens]
            lines.append(f"    precise float3 cube{i} = float3({x}, {y}, {z});")
            c, a = f"cube{i}", f"abs(cube{i})"
            zmajor = f"({a}.z >= {a}.x && {a}.z >= {a}.y)"
            ymajor = f"({a}.y >= {a}.x)"
            expression = {
                "id": f"({zmajor} ? ({c}.z < 0.0f ? 5.0f : 4.0f) : {ymajor} ? ({c}.y < 0.0f ? 3.0f : 2.0f) : ({c}.x < 0.0f ? 1.0f : 0.0f))",
                "sc": f"({zmajor} ? ({c}.z < 0.0f ? -{c}.x : {c}.x) : {ymajor} ? {c}.x : ({c}.x < 0.0f ? {c}.z : -{c}.z))",
                "tc": f"({zmajor} ? -{c}.y : {ymajor} ? ({c}.y < 0.0f ? -{c}.z : {c}.z) : -{c}.y)",
                "ma": f"(({zmajor} ? {c}.z : {ymajor} ? {c}.y : {c}.x) * 2.0f)",
            }[op]
            if clamp:
                expression = f"saturate({expression})"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {expression};")
        elif m := match(r"v_interp_p([12])_f32_e32 (v\d+), (v\d+), attr0\.([xyzw])"):
            phase, dst, src, component = m.groups()
            entry_interp = mask == "entry" and sampler is None
            if ((not entry_interp and (mask != "quad" or component not in "xy"))
                    or scalar.get("m0") != "interp_base"):
                raise ValueError("Unknown interpolation context")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            if phase == "1" and vector.get(src) == "bary_i":
                vector[dst] = ("partial", component, mask)
            elif phase == "2" and vector.get(src) == "bary_j" and vector.get(dst) == ("partial", component, mask):
                if entry_interp:
                    attribute_input = True
                    vector[dst] = f"alu{i}"
                    lines.append(f"    float alu{i} = uv.{component};")
                else:
                    vector[dst] = f"uv.{component}"
            else:
                raise ValueError("Unpaired interpolation")
        elif m := match(r"image_sample v\[(\d+):(\d+)\], v\[(\d+):(\d+)\], s\[(\d+):(\d+)\], s\[(\d+):(\d+)\] dmask:(0x[7f]) dim:SQ_RSRC_IMG_2D"):
            *operands, dmask = m.groups()
            channel_count = 3 if dmask == "0x7" else 4
            dest, coords, image, state = [registers(*operands[k:k+2], width)
                                          for k, width in zip((0,2,4,6), (channel_count,2,8,4))]
            if any(f"v{n}" in sample_dest for n in dest):
                raise ValueError("Write overlaps an outstanding sample")
            descriptor = [scalar.get(f"s{n}") for n in image]
            texture = next((name for name in bindings.values() if descriptor == [(name, j) for j in range(8)]), None)
            if (mask != "entry" or any(f"s{n}" in pending for n in (*image, *state))
                    or texture is None
                    or [scalar.get(f"s{n}") for n in state] != list(sampler)):
                raise ValueError("Unsupported image or sampler binding")
            uv = [vector[f"v{n}"] for n in coords]
            if any(v not in quad_values for v in uv):
                raise ValueError("Unsupported texture coordinate")
            lines.append(f"    float4 sample{i} = {texture}.Sample(samp, float2({', '.join(uv)}));")
            samples += 1
            for j, n in enumerate(dest):
                vector[f"v{n}"] = f"sample{i}.{'xyzw'[j]}"
                sample_dest[f"v{n}"] = samples
        elif m := match(r"v_(fmaak|fmamk)_f32 (v\d+), ([^,]+), ([^,]+), ([^,]+)"):
            op, dst, *tokens = m.groups()
            literal = tokens[2] if op == "fmaak" else tokens[1]
            if mask != "entry" or not re.fullmatch(r"0x[0-9a-f]{1,8}", literal):
                raise ValueError("Invalid literal multiply-add")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            values = [fp_operand(t, scalar, vector, pending, sample_dest) for t in tokens]
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {ALU['fma'][1].format(*values)};")
        elif m := match(r"v_mov_b32_sdwa (v\d+), (sext\()?([vs]\d+|-?\d+(?:\.\d+)?)(\))? dst_sel:(BYTE_[0-3]|WORD_[01]|DWORD) dst_unused:(UNUSED_PAD|UNUSED_SEXT|UNUSED_PRESERVE) src0_sel:(BYTE_[0-3]|WORD_[01]|DWORD)"):
            dst, sext, src, close, dst_sel, unused, src_sel = m.groups()
            if mask != "entry" or bool(sext) != bool(close):
                raise ValueError("Invalid register move")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            word = f"asuint({fp_operand(src, scalar, vector, pending, sample_dest)})"
            result = sdwa_dest(sdwa_source(word, src_sel, bool(sext)), dst_sel, unused, vector.get(dst))
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = asfloat({result});")
        elif m := match(r"v_cvt_f32_f16_sdwa (v\d+), (-?)(?:\|([vs]\d+)\||([vs]\d+))( clamp)? dst_sel:DWORD dst_unused:UNUSED_(?:PAD|PRESERVE) src0_sel:(WORD_[01])"):
            # The SDWA form widens the selected half; modifiers apply after widening, as in the e64 form.
            dst, negate, bars, plain, clamp, src_sel = m.groups()
            if mask != "entry":
                raise ValueError("Invalid conversion")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            word = f"asuint({fp_operand(bars or plain, scalar, vector, pending, sample_dest)})"
            expression = f"widen16({sdwa_source(word, src_sel, False)})"
            if bars:
                expression = f"abs({expression})"
            if negate:
                expression = f"(-{expression})"
            if clamp:
                expression = f"saturate({expression})"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {expression};")
        elif m := match(r"v_mov_b32(?:_e32|_e64)? (v\d+), (-?\d+(?:\.\d+)?|0x[0-9a-f]{1,8}|[vs]\d+)"):
            # A bit copy of a numeric value; integer inline constants are bit patterns.
            dst, src = m.groups()
            if mask != "entry":
                raise ValueError("Invalid register move")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            value = fp_operand(src, scalar, vector, pending, sample_dest)
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {value};")
        elif m := match(rf"v_({'|'.join(ALU)})_f32(?:_e32|_e64)? (v\d+), (.+?)( clamp)?"):
            op, dst, operands, clamp = m.groups()
            tokens = operands.split(", ")
            if op in ("fmac", "mac"):
                tokens.append(dst)
            arity, template = ALU[op]
            quad = mask == "quad"
            if (mask not in ("entry", "quad") or len(tokens) != arity
                    or (quad and (samples or second_texture_offset is not None
                                  or op not in ("add", "fma", "mul_legacy")))):
                raise ValueError("Invalid arithmetic operation")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            if quad and (clamp or any("|" in t or "abs(" in t for t in tokens)):
                raise ValueError("Whole-quad coordinate arithmetic must be affine")
            provenance = quad_values if quad else None
            values = [fp_operand(t, scalar, vector, pending, sample_dest, quad_values=provenance) for t in tokens]
            if quad:
                varying = [quad_values.get(vector.get(t.lstrip("-"), ""), False) for t in tokens]
                if op != "add" and all(varying[:2]):
                    raise ValueError("Whole-quad coordinate arithmetic must be affine")
                quad_values[f"alu{i}"] = any(varying)
            expression = template.format(*values)
            if clamp:
                expression = f"saturate({expression})"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {expression};")
        elif m := match(rf"v_({'|'.join(INTEGER)})(?:_e32|_e64)? (v\d+), (.+)"):
            op, dst, sources = m.groups()
            tokens = sources.split(", ")
            arity, template = INTEGER[op]
            if mask != "entry" or dst in sample_dest or len(tokens) != arity:
                raise ValueError("Invalid integer operation")
            args = [int_operand(t, scalar, vector, pending, sample_dest, signed=False) for t in tokens]
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = asfloat({template.format(*args)});")
        elif m := match(rf"v_cmp_class_f32(?:_e32|_e64)? {LANE_MASK}, ([^,]+), ([^,]+)"):
            dst, value, classes = m.groups()
            regs = lane_mask(dst)
            if mask != "entry" or pending.intersection(regs):
                raise ValueError("Invalid comparison")
            if re.fullmatch(r"-?\d+|0x[0-9a-f]{1,8}", classes):
                bits = f"{int(classes, 0) & 0xffffffff}u"
            elif re.fullmatch(r"[vs]\d+", classes):
                bits = f"asuint({fp_operand(classes, scalar, vector, pending, sample_dest)})"
            else:
                raise ValueError("Unsupported class mask")
            operand = fp_operand(value, scalar, vector, pending, sample_dest)
            for j, reg in enumerate(regs):
                scalar[reg] = ("condition", f"cmp{i}", j, len(regs))
            lines.append(f"    bool cmp{i} = (class32({operand}) & {bits}) != 0u;")

        elif m := match(r"v_(add|sub)_nc_u32(?:_e32)? (v\d+), ([^,]+), ([^,]+)"):
            op, dst, src0, src1 = m.groups()
            if mask != "entry" or dst in sample_dest:
                raise ValueError("Invalid integer operation")
            a = int_operand(src0, scalar, vector, pending, sample_dest, signed=False)
            b = int_operand(src1, scalar, vector, pending, sample_dest, signed=False)
            operator = "+" if op == "add" else "-"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = asfloat({a} {operator} {b});")
        elif m := match(r"v_(min|max)_(i32|u32)(?:_e32|_e64)? (v\d+), ([^,]+), ([^,]+)"):
            op, kind, dst, *tokens = m.groups()
            if mask != "entry":
                raise ValueError("Invalid integer arithmetic operation")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            values = [int_operand(t, scalar, vector, pending, sample_dest, signed=kind == "i32")
                      for t in tokens]
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = asfloat({op}({values[0]}, {values[1]}));")
        elif m := match(r"v_(min3|max3|med3)_(i32|u32)(?:_e64)? (v\d+), ([^,]+), ([^,]+), ([^,]+)"):
            op, kind, dst, *tokens = m.groups()
            if mask != "entry":
                raise ValueError("Invalid integer arithmetic operation")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            values = [int_operand(t, scalar, vector, pending, sample_dest, signed=kind == "i32")
                      for t in tokens]
            if op == "med3":
                vec_type = "int3" if kind == "i32" else "uint3"
                lines.append(f"    {vec_type} med{i} = {vec_type}({values[0]}, {values[1]}, {values[2]});")
                expression = f"max(min(med{i}.x, med{i}.y), min(max(med{i}.x, med{i}.y), med{i}.z))"
            else:
                fn = op[:-1]
                expression = f"{fn}({fn}({values[0]}, {values[1]}), {values[2]})"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = asfloat({expression});")

        elif m := match(r"v_ldexp_f32(?:_e64)? (v\d+), ([^,]+), ([^,]+?)( clamp)?"):
            dst, src0, src1, clamp = m.groups()
            if mask != "entry":
                raise ValueError("Invalid arithmetic operation")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            value = fp_operand(src0, scalar, vector, pending, sample_dest)
            exponent = int_operand(src1, scalar, vector, pending, sample_dest, signed=True)
            expression = f"ldexp32({value}, {exponent})"
            if clamp:
                expression = f"saturate({expression})"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {expression};")
        elif m := match(rf"v_cmp_({'|'.join(COMPARE)})_f32(?:_e32|_e64)? {LANE_MASK}, ([^,]+), ([^,]+)"):
            op, dst, *tokens = m.groups()
            regs = lane_mask(dst)
            if mask != "entry" or pending.intersection(regs):
                raise ValueError("Invalid comparison")
            values = [fp_operand(t, scalar, vector, pending, sample_dest) for t in tokens]
            for j, reg in enumerate(regs):
                scalar[reg] = ("condition", f"cmp{i}", j, len(regs))
            lines.append(f"    bool cmp{i} = {COMPARE[op].format(*values)};")
        elif m := match(rf"v_cmp_({'|'.join(INT_COMPARE)})_(i32|u32)(?:_e32|_e64)? {LANE_MASK}, ([^,]+), ([^,]+)"):
            op, kind, dst, *tokens = m.groups()
            regs = lane_mask(dst)
            if mask != "entry" or pending.intersection(regs):
                raise ValueError("Invalid comparison")
            values = [int_operand(t, scalar, vector, pending, sample_dest, signed=kind == "i32")
                      for t in tokens]
            for j, reg in enumerate(regs):
                scalar[reg] = ("condition", f"cmp{i}", j, len(regs))
            lines.append(f"    bool cmp{i} = {INT_COMPARE[op].format(*values)};")
        elif m := match(rf"v_cmp_({'|'.join(COMPARE)})_f32_sdwa {LANE_MASK}, ([^,]+), ([^,]+) src0_sel:DWORD src1_sel:DWORD"):
            # With whole-dword selections the SDWA comparison is the e32/e64 one; byte and word selections
            # would compare zero-extended fields as binary32 and are refused.
            op, dst, *tokens = m.groups()
            regs = lane_mask(dst)
            if mask != "entry" or pending.intersection(regs):
                raise ValueError("Invalid comparison")
            values = [fp_operand(t, scalar, vector, pending, sample_dest) for t in tokens]
            for j, reg in enumerate(regs):
                scalar[reg] = ("condition", f"cmp{i}", j, len(regs))
            lines.append(f"    bool cmp{i} = {COMPARE[op].format(*values)};")
        elif m := match(rf"v_cndmask_b32(?:_e32|_e64)? (v\d+), ([^,]+), ([^,]+), {LANE_MASK}"):
            dst, unset, chosen, source = m.groups()
            regs = lane_mask(source)
            condition = scalar.get(regs[0])
            if (mask != "entry" or pending.intersection(regs) or not isinstance(condition, tuple)
                    or condition[0] != "condition"
                    or [scalar.get(r) for r in regs] != [(*condition[:2], j, len(regs)) for j in range(len(regs))]):
                raise ValueError("Select mask is not a tracked comparison")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            values = [fp_operand(t, scalar, vector, pending, sample_dest) for t in (chosen, unset)]
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {condition[1]} ? {values[0]} : {values[1]};")
        elif m := match(r"v_cvt_f32_f16(?:_e32|_e64)? (v\d+), (-?)(?:\|([vs]\d+|0x[0-9a-f]{1,8}|\d+)\||abs\(([vs]\d+|0x[0-9a-f]{1,8}|\d+)\)|([vs]\d+|0x[0-9a-f]{1,8}|\d+))( clamp)?"):
            dst, negate, bars, call, plain, clamp = m.groups()
            if mask != "entry":
                raise ValueError("Invalid conversion")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            source = bars or call or plain
            if re.fullmatch(r"0x[0-9a-f]{1,8}|\d+", source):
                word = f"{int(source, 0) & 0xffffffff}u"
            else:
                word = f"asuint({fp_operand(source, scalar, vector, pending, sample_dest)})"
            expression = f"widen16({word})"
            if bars or call:
                expression = f"abs({expression})"
            if negate:
                expression = f"(-{expression})"
            if clamp:
                expression = f"saturate({expression})"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {expression};")
        elif m := match(r"v_pk_(add|mul|min|max|fma)_f16 (v\d+), (.+)"):
            op, dst, rest = m.groups()
            arity, template = VOP3P_F16[op]
            if mask != "entry" or not 0 <= int(dst[1:]) <= 255:
                raise ValueError("Invalid packed operation")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            tokens, opts, clamp = parse_vop3p_f16(rest, arity)
            words = [vop3p_f16_word(t, scalar, vector, pending, sample_dest) for t in tokens]
            word_vars = []
            for k, expr in enumerate(words):
                wvar = f"pk{i}_{k}"
                lines.append(f"    uint {wvar} = {expr};")
                word_vars.append(wvar)
            packed_lanes = []
            for lane, sel_key, neg_key, default_sel in ((0, "op_sel", "neg_lo", 0), (1, "op_sel_hi", "neg_hi", 1)):
                sel = opts.get(sel_key, [default_sel] * arity)
                neg = opts.get(neg_key, [0] * arity)
                lane_args = []
                for wvar, high, negate in zip(word_vars, sel, neg):
                    val = f"widen16({wvar} >> 16)" if high else f"widen16({wvar})"
                    lane_args.append(f"(-{val})" if negate else val)
                expr = template.format(*lane_args)
                if clamp:
                    expr = f"saturate({expr})"
                lvar = f"pk{i}_l{lane}"
                lines.append(f"    precise float {lvar} = {expr};")
                packed_lanes.append(f"pack16_rne({lvar})")
            helpers.add("pack16_rne")
            if op in ("min", "max"):
                helpers.add(f"{op}16_num")
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = asfloat({packed_lanes[0]} | ({packed_lanes[1]} << 16));")
        elif m := match(r"v_cvt_f32_ubyte([0-3])(?:_e32|_e64)? (v\d+), ([vs]\d+|0x[0-9a-f]{1,8}|\d+)( clamp)?"):
            # Byte N of the source word as an unsigned integer. The source is read as bits,
            # so float modifiers do not apply and are rejected by the pattern.
            byte, dst, source, clamp = m.groups()
            if mask != "entry":
                raise ValueError("Invalid conversion")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            if re.fullmatch(r"0x[0-9a-f]{1,8}|\d+", source):
                word = f"{int(source, 0) & 0xffffffff}u"
            else:
                word = f"asuint({fp_operand(source, scalar, vector, pending, sample_dest)})"
            expression = f"float(({word} >> {8 * int(byte)}u) & 0xffu)"
            if clamp:
                expression = f"saturate({expression})"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {expression};")
        elif m := match(r"v_fma_mix(_f32|lo_f16|hi_f16) (v\d+), (.+)"):
            kind, dst, rest = m.groups()
            if mask != "entry" or not 0 <= int(dst[1:]) <= 255:
                raise ValueError("Invalid mixed-precision FMA")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            tokens, op_sel, op_sel_hi, clamp = parse_fma_mix(rest)
            args = [fma_mix_operand(t, f16, hi, scalar, vector, pending, sample_dest)
                    for t, f16, hi in zip(tokens, op_sel, op_sel_hi)]
            expr = f"mad({args[0]}, {args[1]}, {args[2]})"
            if clamp:
                expr = f"saturate({expr})"
            if kind == "_f32":
                vector[dst] = f"alu{i}"
                lines.append(f"    precise float alu{i} = {expr};")
            else:
                _, prev_word = fma_mix_reg(dst, scalar, vector, pending, sample_dest)
                helpers.add("pack16_rne")
                lines.append(f"    precise float mix{i} = {expr};")
                lines.append(f"    uint mix{i}_p = {prev_word};")
                if kind == "lo_f16":
                    packed = f"(mix{i}_p & 0xffff0000u) | pack16_rne(mix{i})"
                else:
                    packed = f"(mix{i}_p & 0x0000ffffu) | (pack16_rne(mix{i}) << 16)"
                vector[dst] = f"alu{i}"
                lines.append(f"    precise float alu{i} = asfloat({packed});")
        elif m := match(r"(v_dot2_f32_f16|v_dot2_i32_i16|v_dot2_u32_u16|v_dot4_i32_i8|v_dot4_u32_u8|v_dot8_i32_i4|v_dot8_u32_u4) (v\d+), (.+)"):
            op, dst, rest = m.groups()
            if mask != "entry" or not 0 <= int(dst[1:]) <= 255:
                raise ValueError("Invalid dot-product operation")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            is_f16 = op == "v_dot2_f32_f16"
            is_dot2 = op.startswith("v_dot2_")
            tokens, op_sel, op_sel_hi, neg_lo, neg_hi, clamp = parse_dot(
                rest, allow_opsel=is_dot2, allow_neg=is_f16,
            )
            if is_f16:
                w0 = dot_word(tokens[0], scalar, vector, pending, sample_dest, allow_f16=True)
                w1 = dot_word(tokens[1], scalar, vector, pending, sample_dest, allow_f16=True)
                c_expr = dot_f32_accum(tokens[2], scalar, vector, pending, sample_dest)
                lines.append(f"    uint dot{i}_0 = {w0};")
                lines.append(f"    uint dot{i}_1 = {w1};")
                a0 = f"widen16(dot{i}_0 >> 16)" if op_sel[0] else f"widen16(dot{i}_0)"
                b0 = f"widen16(dot{i}_1 >> 16)" if op_sel[1] else f"widen16(dot{i}_1)"
                a1 = f"widen16(dot{i}_0 >> 16)" if op_sel_hi[0] else f"widen16(dot{i}_0)"
                b1 = f"widen16(dot{i}_1 >> 16)" if op_sel_hi[1] else f"widen16(dot{i}_1)"
                if neg_lo[0]: a0 = f"(-{a0})"
                if neg_lo[1]: b0 = f"(-{b0})"
                if neg_hi[0]: a1 = f"(-{a1})"
                if neg_hi[1]: b1 = f"(-{b1})"
                if neg_lo[2]: c_expr = f"(-{c_expr})"
                expr = f"(({a0} * {b0}) + ({a1} * {b1}) + {c_expr})"
                if clamp:
                    expr = f"saturate({expr})"
                vector[dst] = f"alu{i}"
                lines.append(f"    precise float alu{i} = {expr};")
            else:
                w0 = dot_word(tokens[0], scalar, vector, pending, sample_dest)
                w1 = dot_word(tokens[1], scalar, vector, pending, sample_dest)
                w2 = dot_word(tokens[2], scalar, vector, pending, sample_dest)
                signed = "_i32_" in op
                helpers.update(("dot_i64_add", "dot_i64_sat", op) if signed else ("dot_u64_add", "dot_u64_sat", op))
                sat = "true" if clamp else "false"
                if is_dot2:
                    lines.append(f"    uint dot{i}_0 = {w0};")
                    lines.append(f"    uint dot{i}_1 = {w1};")
                    a0 = f"(dot{i}_0 >> 16)" if op_sel[0] else f"dot{i}_0"
                    a1 = f"(dot{i}_0 >> 16)" if op_sel_hi[0] else f"dot{i}_0"
                    b0 = f"(dot{i}_1 >> 16)" if op_sel[1] else f"dot{i}_1"
                    b1 = f"(dot{i}_1 >> 16)" if op_sel_hi[1] else f"dot{i}_1"
                    call = f"{op}({a0}, {a1}, {b0}, {b1}, {w2}, {sat})"
                else:
                    call = f"{op}({w0}, {w1}, {w2}, {sat})"
                vector[dst] = f"alu{i}"
                lines.append(f"    precise float alu{i} = asfloat({call});")
        elif m := match(rf"v_pk_({'|'.join(VOP3P_INT16)}) (v\d+), (.+)"):
            op, dst, rest = m.groups()
            arity, allow_clamp = VOP3P_INT16[op]
            if mask != "entry" or not 0 <= int(dst[1:]) <= 255:
                raise ValueError("Invalid packed integer operation")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            tokens, opts, clamp = parse_vop3p_int16(rest, arity, allow_clamp=allow_clamp)
            words = [vop3p_int_word(t, scalar, vector, pending, sample_dest) for t in tokens]
            word_vars = []
            for k, expr in enumerate(words):
                wvar = f"pki{i}_{k}"
                lines.append(f"    uint {wvar} = {expr};")
                word_vars.append(wvar)
            lo_expr = pk_int16_lane_expr(op, word_vars, opts.get("op_sel", [0] * arity), clamp)
            hi_expr = pk_int16_lane_expr(op, word_vars, opts.get("op_sel_hi", [1] * arity), clamp)
            lines.append(f"    uint pki{i}_l0 = {lo_expr};")
            lines.append(f"    uint pki{i}_l1 = {hi_expr};")
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = asfloat(pki{i}_l0 | (pki{i}_l1 << 16));")
        elif m := match(r"v_cvt_off_f32_i4(?:_e32|_e64)? (v\d+), ([vs]\d+|0x[0-9a-f]{1,8}|\d+)( clamp)?"):
            # Bits 3:0 of the source as a signed nibble in units of 1/16, so 0x7 gives 0.4375
            # and 0x8 gives -0.5. The source is read as bits, so float modifiers do not apply.
            dst, source, clamp = m.groups()
            if mask != "entry":
                raise ValueError("Invalid conversion")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            if re.fullmatch(r"0x[0-9a-f]{1,8}|\d+", source):
                word = f"{int(source, 0) & 0xffffffff}u"
            else:
                word = f"asuint({fp_operand(source, scalar, vector, pending, sample_dest)})"
            expression = f"(float(asint({word} << 28u) >> 28) * 0.0625f)"
            if clamp:
                expression = f"saturate({expression})"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {expression};")
        elif m := match(r"v_cvt_f16_f32(?:_e32|_e64)? (v\d+), ([^,\s]+)"):
            # GFX10 16-bit results replace only the low half; the high half of the destination is kept
            # (LLVM GCNSubtarget::zeroesHigh16BitsOfDest). A destination with no tracked value contributes zero.
            dst, src = m.groups()
            if mask != "entry":
                raise ValueError("Invalid conversion")
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            old = vector.get(dst)
            if old is not None and not (isinstance(old, str) and old.startswith(("sample", "alu", "params["))):
                raise ValueError("Unsupported conversion destination")
            # fp_operand reads modifiers and inline constants, so -1 stays the bit pattern 0xffffffff.
            value = fp_operand(src, scalar, vector, pending, sample_dest)
            high = f"(asuint({old}) & 0xffff0000u)" if old is not None else "0u"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = asfloat({high} | narrow16({value}));")
        elif m := match(r"v_cvt_pkrtz_f16_f32_e32 (v\d+), (v\d+), (v\d+)"):
            dst, lo, hi = m.groups()
            if dst in sample_dest:
                raise ValueError("Write overlaps an outstanding sample")
            pair = (vector[lo], vector[hi])
            if mask != "entry" or lo in sample_dest or hi in sample_dest or not all(isinstance(v, str) and v.startswith(("sample", "alu")) for v in pair):
                raise ValueError("Unsupported pack operand")
            vector[dst] = pair
        elif m := match(r"exp mrt0,? (v\d+), \1, (v\d+|off), \2 done compr vm"):
            pairs = [vector[v] for v in m.groups() if v != "off"]
            if mask != "entry" or exported or not all(isinstance(p, tuple) and len(p)==2 and
                    all(v.startswith(("sample", "alu")) for v in p) for p in pairs):
                raise ValueError("Unsupported export")
            args = [f"export16({v})" for pair in pairs for v in pair]
            input_type = "float4" if attribute_input else "float2"
            lines[entry_line] = f"float{len(args)} main({input_type} uv : TEXCOORD0) : SV_Target0 {{"
            lines.append(f"    return float{len(args)}({', '.join(args)});")
            exported = True
        elif line == "s_endpgm" and exported:
            ended = True
        else:
            raise ValueError(f"Unsupported instruction {i}: {line}")
    if not ended or samples != len(bindings):
        raise ValueError("Unexpected sample count or missing color export")
    if helpers:
        defs = [body for name, body in SALU_HELPERS.items() if name in helpers]
        lines[:0] = defs
    return "\n".join([*lines, "}",""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("record")
    parser.add_argument("--container-sha256", help="SHA-256 of the full decoded DXBC container to select by name")
    parser.add_argument("--llvm", required=True, help="LLVM C library with AMDGPU support")
    parser.add_argument("--texture-sgpr", type=int, required=True, help="Entry register for the low descriptor-table address")
    parser.add_argument("--interp-sgpr", type=int, required=True, help="Entry register for the interpolation base")
    profile = parser.add_mutually_exclusive_group(required=True)
    profile.add_argument("--no-texture", action="store_true", help="Require zero texture samples")
    profile.add_argument("--sampler-words", nargs=4, type=lambda s: int(s, 0),
                        help="Four descriptor words to bind to the supplied desktop sampler at s0")
    parser.add_argument("--output", type=Path, required=True, help="Destination .hlsl in a local output directory")
    parser.add_argument("--constant-descriptor", type=lambda s: int(s, 0),
                        help="Byte offset of a descriptor mapped to the 256-byte desktop buffer b0")
    parser.add_argument("--second-texture-offset", type=lambda s: int(s, 0), help="Byte offset of the t1 descriptor for a two-sample shader")
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.suffix.lower() != ".hlsl" or dest == args.archive.resolve():
        parser.error("Choose an HLSL destination distinct from the input")
    try:
        blob, code = native_program(args.archive, args.record, container_sha256=args.container_sha256, llvm=args.llvm)
        instructions = disassemble(code, args.llvm)
        source = lower(instructions, args.texture_sgpr, args.interp_sgpr, args.sampler_words, args.constant_descriptor, second_texture_offset=args.second_texture_offset)
    except (ValueError, KeyError, struct.error, OSError, AttributeError, zlib.error) as exc:
        parser.exit(1, f"Translation failed: {exc}\n")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, mode="wb") as f:
        f.write(source.encode("utf-8"))
    Path(f.name).replace(dest)
    print(json.dumps({"record":args.record, "container_sha256":hashlib.sha256(blob).hexdigest(),
                      "native_sha256":hashlib.sha256(code).hexdigest(), "instructions":len(instructions),
                      "hlsl_sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
