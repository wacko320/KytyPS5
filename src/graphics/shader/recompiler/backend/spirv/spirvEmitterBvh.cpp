#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

using Vec3 = std::array<uint32_t, 3>;

uint32_t Extract(EmitterState& s, uint32_t type, uint32_t vector, uint32_t index) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeExtract, type, result, vector, index);
	return result;
}

uint32_t Vector(EmitterState& s, uint32_t type, std::span<const uint32_t> components) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeConstruct, type, result, components);
	return result;
}

// A box32 is 28 dwords: the first 64-byte block, then 12 dwords of the second.
constexpr uint32_t NodeWords = 28;
using Words = std::array<uint32_t, NodeWords>;

// Loads count dwords from a host pointer as 16-byte vectors; nodes are 64-byte aligned.
// BVH memory is read-only to the traversal: a node whose address is uniform then loads through
// the scalar cache.
void LoadBlock(EmitterState& s, uint32_t host, uint32_t count, uint32_t* words) {
	const auto vec4 = TypeU32Vector(s, 4);
	const auto block = s.builder.DecoratedType(
	    spv::OpTypeStruct, {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                        {spv::OpMemberDecorate, {0, spv::DecorationNonWritable}},
	                        {spv::OpDecorate, {spv::DecorationBlock}}}, vec4);
	const auto block_pointer = TypePointer(s, spv::StorageClassPhysicalStorageBuffer, block);
	const auto pointer_type = TypePointer(s, spv::StorageClassPhysicalStorageBuffer, vec4);
	for (uint32_t offset = 0; offset < count; offset += 4) {
		const auto address = offset == 0 ? host
		    : Binary(s, spv::OpIAdd, TypeU64(s), host, ConstantU64(s, offset * 4u));
		const auto base = Unary(s, spv::OpConvertUToPtr, block_pointer, address);
		const auto pointer = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpAccessChain, pointer_type, pointer, base, ConstantU32(s, 0));
		const auto value = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpLoad, vec4, value, pointer, spv::MemoryAccessAlignedMask, 16u);
		for (uint32_t i = 0; i < 4; ++i) words[offset + i] = Extract(s, TypeU32(s), value, i);
	}
}

uint32_t Triangle(EmitterState& s, const Words& words, uint32_t kind, uint32_t bary,
                  Vec3 origin, Vec3 direction) {
	const auto f = TypeF32(s), u = TypeU32(s), b = TypeBool(s);
	const auto zero = ConstantF32Value(s, 0.0f);
	const auto op = [&](spv::Op code, uint32_t x, uint32_t y) {
		const auto value = Binary(s, code, f, x, y);
		s.builder.AddAnnotation(spv::OpDecorate, value, spv::DecorationNoContraction);
		return value;
	};
	const auto cmp = [&](spv::Op code, uint32_t x, uint32_t y) { return Binary(s, code, b, x, y); };
	const auto either = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpLogicalOr, b, x, y); };
	const auto both = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpLogicalAnd, b, x, y); };
	const auto abs = [&](uint32_t x) { return EmitGlsl<GLSLstd450FAbs, IR::Type::F32>(s, x); };
	const auto is_kind = [&](uint32_t n) { return cmp(spv::OpIEqual, kind, ConstantU32(s, n)); };
	const auto vertex = [&](uint32_t index, uint32_t axis) {
		return Unary(s, spv::OpBitcast, f, words[index * 3 + axis]);
	};
	// The four triangles are (0,1,2), (1,3,2), (2,3,4), (2,4,0): each corner picks one of three
	// vertices of the loaded node.
	const auto kind0 = is_kind(0), kind1 = is_kind(1), kind2 = is_kind(2), kind3 = is_kind(3);
	const std::array<std::array<uint32_t, 3>, 3> corners {{{0, 1, 2}, {1, 4, 3}, {4, 0, 2}}};
	const std::array<std::array<uint32_t, 2>, 3> conditions {{{kind0, kind1}, {kind0, kind3},
	                                                         {kind2, kind3}}};
	std::array<Vec3, 3> positions;
	for (uint32_t i = 0; i < 3; ++i) {
		const auto& [first, second, other] = corners[i];
		for (uint32_t axis = 0; axis < 3; ++axis)
			positions[i][axis] = Select(s, f, conditions[i][0], vertex(first, axis),
			    Select(s, f, conditions[i][1], vertex(second, axis), vertex(other, axis)));
	}
	const auto flag = words[15];
	const auto y_largest = cmp(spv::OpFOrdLessThan, abs(direction[0]), abs(direction[1]));
	const auto z_largest = cmp(spv::OpFOrdLessThan,
	    Select(s, f, y_largest, abs(direction[1]), abs(direction[0])), abs(direction[2]));
	const auto rotate = [&](Vec3 value) {
		Vec3 result;
		for (uint32_t i = 0; i < 3; ++i)
			result[i] = Select(s, f, z_largest, value[i],
			    Select(s, f, y_largest, value[(i + 2) % 3], value[(i + 1) % 3]));
		return result;
	};
	origin = rotate(origin);
	direction = rotate(direction);
	std::array<Vec3, 3> projected;
	for (uint32_t i = 0; i < 3; ++i) {
		const auto position = rotate(positions[i]);
		Vec3 relative;
		for (uint32_t axis = 0; axis < 3; ++axis)
			relative[axis] = op(spv::OpFSub, position[axis], origin[axis]);
		projected[i] = {op(spv::OpFSub, op(spv::OpFMul, relative[0], direction[2]),
		                              op(spv::OpFMul, direction[0], relative[2])),
		                op(spv::OpFSub, op(spv::OpFMul, relative[1], direction[2]),
		                              op(spv::OpFMul, direction[1], relative[2])), relative[2]};
	}
	Vec3 edge, weighted;
	for (uint32_t i = 0; i < 3; ++i) {
		const auto& first = projected[(i + 1) % 3];
		const auto& second = projected[(i + 2) % 3];
		edge[i] = op(spv::OpFSub, op(spv::OpFMul, second[0], first[1]),
		                        op(spv::OpFMul, second[1], first[0]));
		weighted[i] = op(spv::OpFMul, edge[i], direction[2]);
	}
	auto numerator = op(spv::OpFAdd, op(spv::OpFAdd,
	    op(spv::OpFMul, edge[0], projected[0][2]), op(spv::OpFMul, edge[1], projected[1][2])),
	    op(spv::OpFMul, edge[2], projected[2][2]));
	auto denominator = op(spv::OpFAdd, op(spv::OpFAdd, weighted[0], weighted[1]), weighted[2]);
	uint32_t negative = ConstantBool(s, false), positive = negative;
	for (auto value: edge) {
		negative = either(negative, cmp(spv::OpFOrdLessThan, value, zero));
		positive = either(positive, cmp(spv::OpFOrdGreaterThan, value, zero));
	}
	const auto winding = cmp(spv::OpFOrdGreaterThan, denominator, zero);
	auto missed = either(both(negative, positive), cmp(spv::OpFOrdEqual, denominator, zero));
	missed = either(missed, Unary(s, spv::OpIsNan, b, numerator));
	missed = either(missed, cmp(spv::OpFOrdLessThan,
	    Select(s, f, winding, numerator, Unary(s, spv::OpFNegate, f, numerator)), zero));
	for (uint32_t i = 0; i < 3; ++i) {
		const auto first = projected[(i + 1) % 3][1], second = projected[(i + 2) % 3][1];
		const auto first_zero = cmp(spv::OpFOrdEqual, first, zero);
		const auto horizontal = both(first_zero, cmp(spv::OpFOrdEqual, second, zero));
		const auto below = either(cmp(spv::OpFOrdLessThan, first, zero),
		    both(first_zero, cmp(spv::OpFOrdGreaterThan, second, zero)));
		const auto right = Binary(s, spv::OpLogicalNotEqual, b, below, winding);
		const auto excluded = Select(s, b, horizontal,
		    cmp(spv::OpFOrdGreaterThan, projected[i][1], zero), right);
		missed = either(missed, both(cmp(spv::OpFOrdEqual, edge[i], zero), excluded));
	}
	const auto remap = Binary(s, spv::OpShiftRightLogical, u, flag,
	    Binary(s, spv::OpIMul, u, kind, ConstantU32(s, 8)));
	const auto coordinate = [&](uint32_t shift) {
		const auto index = Binary(s, spv::OpBitwiseAnd, u,
		    Binary(s, spv::OpShiftRightLogical, u, remap, ConstantU32(s, shift)), ConstantU32(s, 3));
		return Unary(s, spv::OpBitcast, u, Select(s, f,
		    cmp(spv::OpIEqual, index, ConstantU32(s, 1)), weighted[1],
		    Select(s, f, cmp(spv::OpIEqual, index, ConstantU32(s, 2)), weighted[2], weighted[0])));
	};
	const std::array result {
		Unary(s, spv::OpBitcast, u, Select(s, f, missed, ConstantF32(s, 0x7f800000), numerator)),
		Unary(s, spv::OpBitcast, u, Select(s, f, missed, ConstantF32Value(s, 1.0f), denominator)),
		Select(s, u, bary, coordinate(0), Binary(s, spv::OpIAdd, u, flag, kind)),
		Select(s, u, bary, coordinate(2), Select(s, u, missed, ConstantU32(s, 0), ConstantU32(s, 1))) };
	return Vector(s, TypeU32Vector(s, 4), result);
}

// The six bounds of each child: FP16 pairs for a box16, FP32 for a box32. A wave usually
// walks one box format, so the branch skips the other decode.
std::array<std::array<uint32_t, 6>, 4> Bounds(EmitterState& s, const Words& words, uint32_t half) {
	const auto f = TypeF32(s);
	const auto half_label = s.builder.AllocateId(), full_label = s.builder.AllocateId();
	const auto merge = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
	s.builder.AddFunction(spv::OpBranchConditional, half, half_label, full_label);
	EmitLabel(s, half_label);
	std::array<std::array<uint32_t, 6>, 4> packed, full, result;
	for (uint32_t child = 0; child < 4; ++child) {
		for (uint32_t pair = 0; pair < 3; ++pair) {
			const auto unpacked =
			    EmitGlsl<GLSLstd450UnpackHalf2x16, IR::Type::F32x2>(s, words[4 + child * 3 + pair]);
			packed[child][pair * 2] = Extract(s, f, unpacked, 0);
			packed[child][pair * 2 + 1] = Extract(s, f, unpacked, 1);
		}
	}
	s.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(s, full_label);
	for (uint32_t child = 0; child < 4; ++child)
		for (uint32_t component = 0; component < 6; ++component)
			full[child][component] = Unary(s, spv::OpBitcast, f, words[4 + child * 6 + component]);
	s.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(s, merge);
	for (uint32_t child = 0; child < 4; ++child) {
		for (uint32_t component = 0; component < 6; ++component) {
			result[child][component] = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpPhi, f, result[child][component], packed[child][component],
			                      half_label, full[child][component], full_label);
		}
	}
	return result;
}

// The children's slab tests: their entry distances, and whether the ray hits each. Without
// growth (grow 0, the common case) the exit distance needs no rounding.
struct ChildHits {
	std::array<uint32_t, 4> entry, hit;
};
ChildHits SlabTests(EmitterState& s, const std::array<std::array<uint32_t, 6>, 4>& bounds,
                    uint32_t grow, uint32_t extent, const Vec3& origin, const Vec3& inverse,
                    const Vec3& forward) {
	const auto f = TypeF32(s), u = TypeU32(s), b = TypeBool(s);
	const auto zero = ConstantF32Value(s, 0.0f);
	ChildHits result;
	for (uint32_t child = 0; child < 4; ++child) {
		const auto& box = bounds[child];
		Vec3 near, far;
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto lo = Binary(s, spv::OpFMul, f,
			    Binary(s, spv::OpFSub, f, box[axis], origin[axis]), inverse[axis]);
			const auto hi = Binary(s, spv::OpFMul, f,
			    Binary(s, spv::OpFSub, f, box[axis + 3], origin[axis]), inverse[axis]);
			near[axis] = Select(s, f, forward[axis], lo, hi);
			far[axis] = Select(s, f, forward[axis], hi, lo);
		}
		const auto entry = EmitGlsl<GLSLstd450FMax, IR::Type::F32>(s,
		    EmitGlsl<GLSLstd450FMax, IR::Type::F32>(s, near[0], near[1]), near[2]);
		const auto exit = EmitGlsl<GLSLstd450FMin, IR::Type::F32>(s,
		    EmitGlsl<GLSLstd450FMin, IR::Type::F32>(s, far[0], far[1]), far[2]);
		auto hit = Binary(s, spv::OpLogicalAnd, b,
		    Binary(s, spv::OpFOrdGreaterThanEqual, b, exit, zero),
		    Binary(s, spv::OpFOrdLessThan, b, entry, extent));
		uint32_t limit = exit;
		if (grow != 0) {
			// Valid hits have nonnegative exit distance. Saturate ULP growth at +infinity.
			const auto bits = Unary(s, spv::OpBitcast, u, Select(s, f,
			    Binary(s, spv::OpFOrdEqual, b, exit, zero), zero, exit));
			const auto grown_bits = EmitGlsl<GLSLstd450UMin, IR::Type::U32>(s,
			    Binary(s, spv::OpIAdd, u, bits, grow), ConstantU32(s, 0x7f800000));
			limit = Unary(s, spv::OpBitcast, f, grown_bits);
		}
		// With no growth, entry <= exit is the same test: for a nonnegative exit the grown
		// bound is the exit itself (-0 compares as +0).
		result.hit[child] = Binary(s, spv::OpLogicalAnd, b, hit,
		    Binary(s, spv::OpFOrdLessThanEqual, b, entry, limit));
		result.entry[child] = entry;
	}
	return result;
}

uint32_t Boxes(EmitterState& s, const Words& words, uint32_t half, uint32_t grow, uint32_t sort,
               uint32_t extent, const Vec3& origin, const Vec3& inverse) {
	const auto f = TypeF32(s), u = TypeU32(s), b = TypeBool(s);
	const auto vec4 = TypeU32Vector(s, 4);
	const auto zero = ConstantF32Value(s, 0.0f);
	Vec3 forward;
	for (uint32_t axis = 0; axis < 3; ++axis)
		forward[axis] = Binary(s, spv::OpFOrdGreaterThanEqual, b, inverse[axis], zero);
	const auto bounds = Bounds(s, words, half);
	// Two versions of the tests, by the descriptor's (uniform) growth.
	const auto grown_label = s.builder.AllocateId(), plain_label = s.builder.AllocateId();
	const auto tests_merge = s.builder.AllocateId();
	const auto grows = Binary(s, spv::OpINotEqual, b, grow, ConstantU32(s, 0));
	s.builder.AddFunction(spv::OpSelectionMerge, tests_merge, spv::SelectionControlMaskNone);
	s.builder.AddFunction(spv::OpBranchConditional, grows, grown_label, plain_label);
	EmitLabel(s, grown_label);
	const auto grown = SlabTests(s, bounds, grow, extent, origin, inverse, forward);
	const auto grown_exit = s.current_label;
	s.builder.AddFunction(spv::OpBranch, tests_merge);
	EmitLabel(s, plain_label);
	const auto plain = SlabTests(s, bounds, 0, extent, origin, inverse, forward);
	const auto plain_exit = s.current_label;
	s.builder.AddFunction(spv::OpBranch, tests_merge);
	EmitLabel(s, tests_merge);
	std::array<uint32_t, 4> entry, hit, indices;
	for (uint32_t child = 0; child < 4; ++child) {
		entry[child] = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpPhi, f, entry[child], grown.entry[child], grown_exit,
		                      plain.entry[child], plain_exit);
		hit[child] = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpPhi, b, hit[child], grown.hit[child], grown_exit,
		                      plain.hit[child], plain_exit);
	}
	for (uint32_t child = 0; child < 4; ++child)
		indices[child] = Select(s, u, hit[child], words[child], ConstantU32(s, ~0u));
	const auto unsorted = Vector(s, vec4, indices);
	// Sorting is a descriptor bit: the depths and the network are only built when it is set.
	return EmitValueOrDefaultIfCondition(s, sort, vec4, unsorted, [&] {
		std::array<uint32_t, 4> depths, sorted = indices;
		for (uint32_t child = 0; child < 4; ++child) {
			depths[child] = Unary(s, spv::OpBitcast, u, Select(s, f, hit[child],
			    Select(s, f, Binary(s, spv::OpFOrdGreaterThan, b, entry[child], zero), entry[child],
			           zero),
			    ConstantF32(s, 0x7f800000)));
		}
		constexpr std::array<std::pair<uint32_t, uint32_t>, 5> network {{{0,1}, {2,3}, {0,2}, {1,3}, {1,2}}};
		for (size_t step = 0; step < network.size(); ++step) {
			const auto [a, c] = network[step];
			const auto swap = Binary(s, spv::OpUGreaterThan, b, depths[a], depths[c]);
			const auto later = [&](uint32_t position) {
				return std::ranges::any_of(std::span(network).subspan(step + 1), [=](auto pair) {
					return pair.first == position || pair.second == position;
				});
			};
			const auto depth = depths[a];
			if (later(a)) depths[a] = EmitGlsl<GLSLstd450UMin, IR::Type::U32>(s, depth, depths[c]);
			if (later(c)) depths[c] = EmitGlsl<GLSLstd450UMax, IR::Type::U32>(s, depth, depths[c]);
			const auto index = sorted[a];
			sorted[a] = Select(s, u, swap, sorted[c], index);
			sorted[c] = Select(s, u, swap, index, sorted[c]);
		}
		return Vector(s, vec4, sorted);
	});
}

// The intersection of each lane's ray with `node` of the BVH `descriptor`.
uint32_t Intersect(EmitterState& s, uint32_t descriptor, uint32_t node, uint32_t extent,
                   const Vec3& origin, const Vec3& direction, const Vec3& inverse) {
	const auto u = TypeU32(s), b = TypeBool(s), wide = TypeU64(s);
	const auto vec4 = TypeU32Vector(s, 4);
	std::array<uint32_t, 4> words;
	for (uint32_t i = 0; i < 4; ++i) words[i] = Extract(s, u, descriptor, i);
	const auto and_bits = [&](uint32_t value, uint32_t mask) {
		return Binary(s, spv::OpBitwiseAnd, u, value, ConstantU32(s, mask));
	};
	const auto shr = [&](uint32_t value, uint32_t shift) {
		return Binary(s, spv::OpShiftRightLogical, u, value, ConstantU32(s, shift));
	};
	const auto kind = and_bits(Unary(s, spv::OpUConvert, u, node), 7);
	const auto triangle = Binary(s, spv::OpULessThan, b, kind, ConstantU32(s, 4));
	const auto full = Binary(s, spv::OpIEqual, b, kind, ConstantU32(s, 5));
	const auto half = Binary(s, spv::OpIEqual, b, kind, ConstantU32(s, 4));
	const auto bary = Binary(s, spv::OpINotEqual, b, and_bits(words[3], 0x01000000), ConstantU32(s, 0));
	const auto sort = Binary(s, spv::OpINotEqual, b, and_bits(words[1], 0x80000000), ConstantU32(s, 0));
	const auto grow = and_bits(shr(words[1], 23), 0xff);
	const auto base  = Binary(s, spv::OpShiftLeftLogical, wide,
	                          PackU64(s, words[0], and_bits(words[1], 0xff)), ConstantU32(s, 8));
	const auto index = Binary(s, spv::OpShiftRightLogical, wide, node, ConstantU32(s, 3));
	const auto last_index = Binary(s, spv::OpIAdd, wide, index,
	                               Select(s, wide, full, ConstantU64(s, 1), ConstantU64(s, 0)));
	const auto address = Binary(s, spv::OpIAdd, wide, base,
	                            Binary(s, spv::OpShiftLeftLogical, wide, index, ConstantU32(s, 6)));
	const auto last_address =
	    Binary(s, spv::OpIAdd, wide, base,
	           Binary(s, spv::OpShiftLeftLogical, wide, last_index, ConstantU32(s, 6)));
	// The node's checks are kept apart from the ray's: the node is often the same in every lane
	// (wave-wide traversal), and its fetch should then branch on a uniform condition.
	auto node_valid = Binary(s, spv::OpIEqual, b, shr(words[3], 28), ConstantU32(s, 8));
	const auto require = [&](uint32_t condition) {
		node_valid = Binary(s, spv::OpLogicalAnd, b, node_valid, condition);
	};
	require(Binary(s, spv::OpULessThanEqual, b, kind, ConstantU32(s, 5)));
	// The size field bounds box nodes only: GTA V sizes its BLAS descriptors to the box nodes and
	// stores the triangles after them, so bounding triangles too made every triangle a miss and
	// every ray walk its whole BLAS (three ray-tracing kernels ran past the GPU timeout).
	require(Binary(s, spv::OpLogicalOr, b, triangle,
	               Binary(s, spv::OpULessThanEqual, b, last_index,
	                      PackU64(s, words[2], and_bits(words[3], 0x3ff)))));
	// A null BVH reports misses instead of reading near address 0.
	require(Binary(s, spv::OpINotEqual, b, base, ConstantU64(s, 0)));
	require(Binary(s, spv::OpULessThan, b, last_address, ConstantU64(s, uint64_t {1} << 48)));
	// NaN inputs and an infinite origin are invalid; |origin| < inf rejects both at once.
	auto ray_valid = Unary(s, spv::OpLogicalNot, b, Unary(s, spv::OpIsNan, b, extent));
	for (uint32_t i = 0; i < 3; ++i) {
		for (const auto value: {direction[i], inverse[i]})
			ray_valid = Binary(s, spv::OpLogicalAnd, b, ray_valid,
			                   Unary(s, spv::OpLogicalNot, b, Unary(s, spv::OpIsNan, b, value)));
		ray_valid = Binary(s, spv::OpLogicalAnd, b, ray_valid,
		                   Binary(s, spv::OpFOrdLessThan, b,
		                          EmitGlsl<GLSLstd450FAbs, IR::Type::F32>(s, origin[i]),
		                          ConstantF32(s, 0x7f800000)));
	}
	const std::array triangle_invalid {ConstantU32(s, 0x7f800000), ConstantU32(s, 0x3f800000),
	                                  ConstantU32(s, ~0u), ConstantU32(s, 0)};
	const std::array box_invalid {ConstantU32(s, ~0u), ConstantU32(s, ~0u),
	                             ConstantU32(s, ~0u), ConstantU32(s, ~0u)};
	std::array<uint32_t, 4> invalid_words;
	for (uint32_t i = 0; i < 4; ++i)
		invalid_words[i] = Select(s, u, triangle, triangle_invalid[i], box_invalid[i]);
	const auto invalid = Vector(s, vec4, invalid_words);
	const auto result = EmitValueOrDefaultIfCondition(s, node_valid, vec4, invalid, [&] {
		// One page lookup serves a whole node, and its words load as 16-byte vectors before any
		// branch, so a box costs one memory round trip instead of one per child. A node is
		// 64-byte aligned: only a box32's second block can start a new page.
		const auto first = GetBdaPointer(s, address);
		const auto page_end = Binary(s, spv::OpIEqual, b,
		    and_bits(Unary(s, spv::OpUConvert, u, address),
		             static_cast<uint32_t>(BufferCache::CACHING_PAGESIZE - 1)),
		    ConstantU32(s, static_cast<uint32_t>(BufferCache::CACHING_PAGESIZE - 64)));
		const auto crosses = Binary(s, spv::OpLogicalAnd, b, full, page_end);
		const auto next = EmitValueOrDefaultIfCondition(s, crosses, wide, ConstantU64(s, 0), [&] {
			return GetBdaPointer(s, Binary(s, spv::OpIAdd, wide, address, ConstantU64(s, 64)));
		});
		const auto present = Binary(
		    s, spv::OpLogicalAnd, b, Binary(s, spv::OpINotEqual, b, first, ConstantU64(s, 0)),
		    Binary(s, spv::OpLogicalOr, b, Unary(s, spv::OpLogicalNot, b, crosses),
		           Binary(s, spv::OpINotEqual, b, next, ConstantU64(s, 0))));
		// Other nodes reload their first block in place of a second one.
		const auto second = Select(s, wide, full,
		    Select(s, wide, crosses, next, Binary(s, spv::OpIAdd, wide, first, ConstantU64(s, 64))),
		    first);
		return EmitValueOrDefaultIfCondition(s, present, vec4, invalid, [&] {
			Words words;
			LoadBlock(s, first, 16, words.data());
			LoadBlock(s, second, NodeWords - 16, words.data() + 16);
			const auto tri_label = s.builder.AllocateId(), box_label = s.builder.AllocateId();
			const auto merge = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
			s.builder.AddFunction(spv::OpBranchConditional, triangle, tri_label, box_label);
			EmitLabel(s, tri_label);
			const auto tri_value = Triangle(s, words, kind, bary, origin, direction);
			const auto tri_exit = s.current_label;
			s.builder.AddFunction(spv::OpBranch, merge);
			EmitLabel(s, box_label);
			const auto box_value = Boxes(s, words, half, grow, sort, extent, origin, inverse);
			const auto box_exit = s.current_label;
			s.builder.AddFunction(spv::OpBranch, merge);
			EmitLabel(s, merge);
			const auto value = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpPhi, vec4, value, tri_value, tri_exit, box_value, box_exit);
			return value;
		});
	});
	const auto bvec4 = s.builder.Type(spv::OpTypeVector, b, 4u);
	const auto keep = Vector(s, bvec4, std::array {ray_valid, ray_valid, ray_valid, ray_valid});
	const auto value = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpSelect, vec4, value, keep, result, invalid);
	return value;
}

} // namespace

void DefineBvhIntersect(EmitterState& s) {
	if (!s.program.info.bvh) return;
	const auto u = TypeU32(s), f = TypeF32(s), b = TypeBool(s), wide = TypeU64(s);
	const auto vec4 = TypeU32Vector(s, 4), vec3 = TypeF32Vector(s, 3);
	const auto signature = s.builder.Type(spv::OpTypeFunction, vec4, vec4, wide, f, vec3, vec3, vec3);
	s.bvh_intersect_function = s.builder.AllocateId();
	s.builder.AddName(s.bvh_intersect_function, "bvh_intersect");
	s.builder.AddFunction(spv::OpFunction, vec4, s.bvh_intersect_function,
	                      spv::FunctionControlMaskNone, signature);
	std::array<uint32_t, 6> args;
	const std::array types {vec4, wide, f, vec3, vec3, vec3};
	for (uint32_t i = 0; i < args.size(); ++i) {
		args[i] = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpFunctionParameter, types[i], args[i]);
	}
	EmitLabel(s, s.builder.AllocateId());
	const auto descriptor = args[0], node = args[1], extent = args[2];
	Vec3 origin, direction, inverse;
	for (uint32_t i = 0; i < 3; ++i) {
		origin[i] = Extract(s, f, args[3], i);
		direction[i] = Extract(s, f, args[4], i);
		inverse[i] = Extract(s, f, args[5], i);
	}
	// A wave that walks its BVH together tests every lane's ray against one node: then the node
	// is fetched and decoded once, as uniform values (scalar loads), instead of per lane.
	s.builder.RequireCapability(spv::CapabilityGroupNonUniform);
	s.builder.RequireCapability(spv::CapabilityGroupNonUniformBallot);
	s.builder.RequireCapability(spv::CapabilityGroupNonUniformVote);
	const auto scope = ConstantU32(s, spv::ScopeSubgroup);
	const auto first = [&](uint32_t type, uint32_t value) {
		const auto result = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpGroupNonUniformBroadcastFirst, type, result, scope, value);
		return result;
	};
	const auto first_descriptor = first(vec4, descriptor), first_node = first(wide, node);
	auto same = Binary(s, spv::OpIEqual, b, node, first_node);
	for (uint32_t i = 0; i < 4; ++i)
		same = Binary(s, spv::OpLogicalAnd, b, same,
		              Binary(s, spv::OpIEqual, b, Extract(s, u, descriptor, i),
		                     Extract(s, u, first_descriptor, i)));
	const auto uniform = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformAll, b, uniform, scope, same);
	const auto uniform_label = s.builder.AllocateId(), lane_label = s.builder.AllocateId();
	const auto merge = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
	s.builder.AddFunction(spv::OpBranchConditional, uniform, uniform_label, lane_label);
	EmitLabel(s, uniform_label);
	const auto uniform_value =
	    Intersect(s, first_descriptor, first_node, extent, origin, direction, inverse);
	const auto uniform_exit = s.current_label;
	s.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(s, lane_label);
	const auto lane_value = Intersect(s, descriptor, node, extent, origin, direction, inverse);
	const auto lane_exit = s.current_label;
	s.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(s, merge);
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpPhi, vec4, result, uniform_value, uniform_exit, lane_value, lane_exit);
	s.builder.AddFunction(spv::OpReturnValue, result);
	s.builder.AddFunction(spv::OpFunctionEnd);
}

uint32_t EmitBvhIntersect(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	const auto* ray = ctx.ImageAddress(inst.Arg(1));
	const auto node_words = inst.Flags<uint32_t>();
	if (ray == nullptr || (node_words != 1u && node_words != 2u) ||
	    ray->NumArgs() < node_words + 10u || s.bvh_intersect_function == 0)
		ctx.Fail(inst, "BVH intersection requires its node pointer, ray components and shared function");
	const auto vector = [&](uint32_t first) {
		Vec3 values;
		for (uint32_t i = 0; i < 3; ++i)
			values[i] = Unary(s, spv::OpBitcast, TypeF32(s), ctx.Arg(*ray, first + i));
		return Vector(s, TypeF32Vector(s, 3), values);
	};
	const auto descriptor = ctx.Arg(inst, 0);
	const auto node   = node_words == 2u ? PackU64(s, ctx.Arg(*ray, 0), ctx.Arg(*ray, 1))
	                                     : Unary(s, spv::OpUConvert, TypeU64(s), ctx.Arg(*ray, 0));
	const auto extent = Unary(s, spv::OpBitcast, TypeF32(s), ctx.Arg(*ray, node_words));
	const auto origin = vector(node_words + 1u), direction = vector(node_words + 4u),
	           inverse = vector(node_words + 7u);
	return EmitValueOrDefaultIfCondition(s, ctx.Arg(inst, 2), TypeU32Vector(s, 4),
	    ConstantU32CompositeZero(s, 4), [&] {
		const auto result = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpFunctionCall, TypeU32Vector(s, 4), result,
		    s.bvh_intersect_function, descriptor, node, extent, origin, direction, inverse);
		return result;
	});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
