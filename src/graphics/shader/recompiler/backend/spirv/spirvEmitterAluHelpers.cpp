#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

#include <array>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

uint32_t EmitAndConstant(EmitterState& state, uint32_t value, uint32_t mask) {
	const auto ret = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), ret, value,
	                          ConstantU32(state, mask));
	return ret;
}

uint32_t EmitCompareU32Constant(EmitterState& state, spv::Op opcode, uint32_t value,
                                uint32_t constant) {
	const auto ret = state.builder.AllocateId();
	state.builder.AddFunction(opcode, TypeBool(state), ret, value, ConstantU32(state, constant));
	return ret;
}

F32Class EmitClassifyF32Bits(EmitterState& state, uint32_t bits) {
	F32Class cls;
	cls.bits            = bits;
	const auto abs_bits = EmitAndConstant(state, cls.bits, 0x7fffffffu);
	cls.nan  = EmitCompareU32Constant(state, spv::OpUGreaterThan, abs_bits, 0x7f800000u);
	cls.zero = EmitCompareU32Constant(state, spv::OpIEqual, abs_bits, 0);
	return cls;
}

F32Class EmitClassifyF32(EmitterState& state, uint32_t value) {
	return EmitClassifyF32Bits(state, EmitBitcastF32ToU32(state, value));
}

uint32_t EmitClassMaskBitMatch(EmitterState& state, uint32_t mask, uint32_t bit,
                               uint32_t class_match) {
	const auto selected =
	    EmitCompareU32Constant(state, spv::OpINotEqual, EmitAndConstant(state, mask, 1u << bit), 0);
	return EmitLogicalAndBool(state, selected, class_match);
}

static uint32_t EmitClassMaskBits(EmitterState& state, uint32_t bits, uint32_t mask, bool half) {
	const uint32_t sign_mask     = half ? 0x8000u : 0x80000000u;
	const uint32_t exponent_mask = half ? 0x7c00u : 0x7f800000u;
	const uint32_t mantissa_mask = half ? 0x03ffu : 0x007fffffu;
	const uint32_t quiet_mask    = half ? 0x0200u : 0x00400000u;
	const auto     sign_bits     = EmitAndConstant(state, bits, sign_mask);
	const auto     abs_bits      = EmitAndConstant(state, bits, sign_mask - 1u);
	const auto     exponent_bits = EmitAndConstant(state, abs_bits, exponent_mask);
	const auto     mantissa_bits = EmitAndConstant(state, abs_bits, mantissa_mask);
	const auto     quiet_bits    = EmitAndConstant(state, mantissa_bits, quiet_mask);

	const auto sign             = EmitCompareU32Constant(state, spv::OpINotEqual, sign_bits, 0);
	const auto positive         = EmitLogicalNotBool(state, sign);
	const auto exponent_zero    = EmitCompareU32Constant(state, spv::OpIEqual, exponent_bits, 0);
	const auto exponent_nonzero = EmitLogicalNotBool(state, exponent_zero);
	const auto exponent_inf =
	    EmitCompareU32Constant(state, spv::OpIEqual, exponent_bits, exponent_mask);
	const auto finite_exponent  = EmitLogicalNotBool(state, exponent_inf);
	const auto mantissa_zero    = EmitCompareU32Constant(state, spv::OpIEqual, mantissa_bits, 0);
	const auto mantissa_nonzero = EmitLogicalNotBool(state, mantissa_zero);
	const auto quiet            = EmitCompareU32Constant(state, spv::OpINotEqual, quiet_bits, 0);

	const auto nan_common = EmitLogicalAndBool(state, exponent_inf, mantissa_nonzero);
	const auto snan       = EmitLogicalAndBool(state, nan_common, EmitLogicalNotBool(state, quiet));
	const auto qnan       = EmitLogicalAndBool(state, nan_common, quiet);
	const auto inf        = EmitLogicalAndBool(state, exponent_inf, mantissa_zero);
	const auto normal     = EmitLogicalAndBool(state, exponent_nonzero, finite_exponent);
	const auto denorm     = EmitLogicalAndBool(state, exponent_zero, mantissa_nonzero);
	const auto zero       = EmitCompareU32Constant(state, spv::OpIEqual, abs_bits, 0);

	uint32_t match = EmitClassMaskBitMatch(state, mask, 0, snan);
	match          = EmitLogicalOrBool(state, match, EmitClassMaskBitMatch(state, mask, 1, qnan));
	match          = EmitLogicalOrBool(
	    state, match, EmitClassMaskBitMatch(state, mask, 2, EmitLogicalAndBool(state, inf, sign)));
	match = EmitLogicalOrBool(
	    state, match,
	    EmitClassMaskBitMatch(state, mask, 3, EmitLogicalAndBool(state, normal, sign)));
	match = EmitLogicalOrBool(
	    state, match,
	    EmitClassMaskBitMatch(state, mask, 4, EmitLogicalAndBool(state, denorm, sign)));
	match = EmitLogicalOrBool(
	    state, match, EmitClassMaskBitMatch(state, mask, 5, EmitLogicalAndBool(state, zero, sign)));
	match = EmitLogicalOrBool(
	    state, match,
	    EmitClassMaskBitMatch(state, mask, 6, EmitLogicalAndBool(state, zero, positive)));
	match = EmitLogicalOrBool(
	    state, match,
	    EmitClassMaskBitMatch(state, mask, 7, EmitLogicalAndBool(state, denorm, positive)));
	match = EmitLogicalOrBool(
	    state, match,
	    EmitClassMaskBitMatch(state, mask, 8, EmitLogicalAndBool(state, normal, positive)));
	return EmitLogicalOrBool(
	    state, match,
	    EmitClassMaskBitMatch(state, mask, 9, EmitLogicalAndBool(state, inf, positive)));
}

uint32_t EmitClassMaskF32(EmitterState& state, uint32_t value, uint32_t mask) {
	return EmitClassMaskBits(state, EmitBitcastF32ToU32(state, value), mask, false);
}

uint32_t EmitClassMaskF16(EmitterState& state, uint32_t bits, uint32_t mask) {
	return EmitClassMaskBits(state, bits, mask, true);
}

uint32_t EmitMinMaxF32Value(EmitterState& state, uint32_t lhs, uint32_t rhs, bool max_value) {
	const auto lhs_class = EmitClassifyF32(state, lhs);
	const auto rhs_class = EmitClassifyF32(state, rhs);

	const auto numeric_cond = state.builder.AllocateId();
	state.builder.AddFunction(max_value ? spv::OpFOrdGreaterThanEqual : spv::OpFOrdLessThan,
	                          TypeBool(state), numeric_cond, lhs, rhs);
	const auto ordered_bits =
	    EmitSelectValueU32(state, numeric_cond, lhs_class.bits, rhs_class.bits);

	const auto both_zero    = EmitLogicalAndBool(state, lhs_class.zero, rhs_class.zero);
	const auto zero_bits    = max_value ? EmitAndU32(state, lhs_class.bits, rhs_class.bits)
	                                    : EmitOrU32(state, lhs_class.bits, rhs_class.bits);
	const auto numeric_bits = EmitSelectValueU32(state, both_zero, zero_bits, ordered_bits);

	const auto rhs_nan_bits =
	    EmitSelectValueU32(state, rhs_class.nan, lhs_class.bits, numeric_bits);
	const auto result_bits = EmitSelectValueU32(state, lhs_class.nan, rhs_class.bits, rhs_nan_bits);
	return EmitBitcastU32ToF32(state, result_bits);
}

uint32_t EmitFlushF32DenormToSignedZero(EmitterState& state, uint32_t value) {
	const auto bits      = state.builder.AllocateId();
	const auto abs_bits  = state.builder.AllocateId();
	const auto sign_bits = state.builder.AllocateId();
	const auto subnormal = state.builder.AllocateId();
	const auto selected  = state.builder.AllocateId();
	const auto ret       = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), abs_bits, bits,
	                          ConstantU32(state, 0x7fffffffu));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), sign_bits, bits,
	                          ConstantU32(state, 0x80000000u));
	state.builder.AddFunction(spv::OpULessThan, TypeBool(state), subnormal, abs_bits,
	                          ConstantU32(state, 0x00800000u));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, subnormal, sign_bits, bits);
	state.builder.AddFunction(spv::OpBitcast, TypeF32(state), ret, selected);
	return ret;
}

namespace {

// One half, rounded to nearest even from the f32 bits with integer arithmetic (the driver's pack
// may truncate): normal halves round their 13 dropped mantissa bits, halves below 2^-14 become
// subnormals or zero, and overflow (including the round-up from 65504) gives infinity. NaN keeps
// the driver's encoding h.
uint32_t RoundHalfToNearestEven(EmitterState& state, uint32_t x, uint32_t h) {
	const auto u32     = TypeU32(state);
	const auto boolean = TypeBool(state);
	const auto C       = [&](uint32_t value) { return ConstantU32(state, value); };
	const auto Op      = [&](spv::Op op, uint32_t type, uint32_t a, uint32_t b) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(op, type, result, a, b);
		return result;
	};
	const auto Sel = [&](uint32_t condition, uint32_t a, uint32_t b) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, u32, result, condition, a, b);
		return result;
	};
	// Rounds value >> shift to nearest even (shift 1..31, as a value).
	const auto RoundShift = [&](uint32_t value, uint32_t shift) {
		const auto kept    = Op(spv::OpShiftRightLogical, u32, value, shift);
		const auto mask    = Op(spv::OpISub, u32, Op(spv::OpShiftLeftLogical, u32, C(1), shift), C(1));
		const auto rest    = Op(spv::OpBitwiseAnd, u32, value, mask);
		const auto halfway = Op(spv::OpShiftLeftLogical, u32, C(1), Op(spv::OpISub, u32, shift, C(1)));
		const auto odd     = Op(spv::OpINotEqual, boolean, Op(spv::OpBitwiseAnd, u32, kept, C(1)), C(0));
		const auto up      = Op(spv::OpLogicalOr, boolean, Op(spv::OpUGreaterThan, boolean, rest, halfway),
		                        Op(spv::OpLogicalAnd, boolean, Op(spv::OpIEqual, boolean, rest, halfway), odd));
		return Op(spv::OpIAdd, u32, kept, Sel(up, C(1), C(0)));
	};
	const auto bits     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, u32, bits, x);
	const auto sign     = Op(spv::OpBitwiseAnd, u32, Op(spv::OpShiftRightLogical, u32, bits, C(16)), C(0x8000u));
	const auto exponent = Op(spv::OpBitwiseAnd, u32, Op(spv::OpShiftRightLogical, u32, bits, C(23)), C(0xffu));
	const auto mantissa = Op(spv::OpBitwiseAnd, u32, bits, C(0x007fffffu));
	// Normal: biased half exponent e = exponent - 112 in 1..30; the carry of the rounding may
	// reach 0x7c00 (infinity), which is the RNE result.
	const auto normal   = RoundShift(Op(spv::OpBitwiseOr, u32,
	                                    Op(spv::OpShiftLeftLogical, u32, Op(spv::OpISub, u32, exponent, C(112)), C(23)),
	                                    mantissa),
	                                 C(13));
	// Subnormal half (exponent 102..112): the significand with its implicit bit, shifted by
	// 126 - exponent (14..24).
	const auto subnormal =
	    RoundShift(Op(spv::OpBitwiseOr, u32, mantissa, C(0x00800000u)), Op(spv::OpISub, u32, C(126), exponent));
	const auto is_normal    = Op(spv::OpLogicalAnd, boolean, Op(spv::OpUGreaterThan, boolean, exponent, C(112)),
	                             Op(spv::OpULessThan, boolean, exponent, C(143)));
	const auto is_subnormal = Op(spv::OpLogicalAnd, boolean, Op(spv::OpUGreaterThanEqual, boolean, exponent, C(102)),
	                             Op(spv::OpULessThanEqual, boolean, exponent, C(112)));
	const auto overflow = Op(spv::OpLogicalAnd, boolean, Op(spv::OpUGreaterThanEqual, boolean, exponent, C(143)),
	                         Op(spv::OpULessThan, boolean, exponent, C(255)));
	const auto is_nan   = Op(spv::OpLogicalAnd, boolean, Op(spv::OpIEqual, boolean, exponent, C(255)),
	                         Op(spv::OpINotEqual, boolean, mantissa, C(0)));
	const auto is_inf   = Op(spv::OpLogicalAnd, boolean, Op(spv::OpIEqual, boolean, exponent, C(255)),
	                         Op(spv::OpIEqual, boolean, mantissa, C(0)));
	auto magnitude = Sel(is_normal, normal, C(0));
	magnitude      = Sel(is_subnormal, subnormal, magnitude);
	magnitude      = Sel(Op(spv::OpLogicalOr, boolean, overflow, is_inf), C(0x7c00u), magnitude);
	return Sel(is_nan, h, Op(spv::OpBitwiseOr, u32, sign, magnitude));
}

} // namespace

uint32_t EmitPackHalf2x16(EmitterState& state, uint32_t pair) {
	const auto u32 = TypeU32(state);
	const auto raw = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, u32, raw, GlslStd450(state), GLSLstd450PackHalf2x16,
	                          pair);
	std::array<uint32_t, 2> halves {};
	for (uint32_t component = 0; component < 2u; component++) {
		const auto x = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), x, pair, component);
		const auto shifted = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpShiftRightLogical, u32, shifted, raw,
		                          ConstantU32(state, 16u * component));
		const auto h = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitwiseAnd, u32, h, shifted, ConstantU32(state, 0xffffu));
		halves[component] = RoundHalfToNearestEven(state, x, h);
	}
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpShiftLeftLogical, u32, high, halves[1], ConstantU32(state, 16u));
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseOr, u32, result, halves[0], high);
	return result;
}

uint32_t EmitTrigCycleF32(EmitterState& state, uint32_t src, bool preserve_signed_zero) {
	const auto fract        = state.builder.AllocateId();
	const auto bits         = state.builder.AllocateId();
	const auto abs_bits     = state.builder.AllocateId();
	const auto large        = state.builder.AllocateId();
	const auto finite       = state.builder.AllocateId();
	const auto large_finite = state.builder.AllocateId();
	const auto reduced      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), fract, GlslStd450(state),
	                          GLSLstd450Fract, src);
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, src);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), abs_bits, bits,
	                          ConstantU32(state, 0x7fffffffu));
	state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), large, abs_bits,
	                          ConstantU32(state, 0x4b000000u));
	state.builder.AddFunction(spv::OpULessThan, TypeBool(state), finite, abs_bits,
	                          ConstantU32(state, 0x7f800000u));
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), large_finite, large, finite);
	state.builder.AddFunction(spv::OpSelect, TypeF32(state), reduced, large_finite,
	                          ConstantF32(state, 0), fract);
	if (!preserve_signed_zero) {
		return reduced;
	}
	const auto zero = state.builder.AllocateId();
	const auto ret  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), zero, abs_bits,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, TypeF32(state), ret, zero, src, reduced);
	return ret;
}

uint32_t EmitF16BitsToF32(EmitterState& state, uint32_t bits) {
	const auto unpacked = state.builder.AllocateId();
	const auto ret      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked, GlslStd450(state),
	                          GLSLstd450UnpackHalf2x16, bits);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), ret, unpacked, 0);
	return ret;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
