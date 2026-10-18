// GPU tests for IMAGE_BVH_INTERSECT_RAY / IMAGE_BVH64_INTERSECT_RAY, run against the shared
// bvh_intersect function (backend/spirv/spirvEmitterBvh.cpp).
//
// Each case places one BVH node in guest memory behind a BDA mapping, runs a single intersection
// and stores the four result dwords. Expected values are derived by hand from the RDNA2 node
// formats. Miss and invalid outputs follow the shared function: a box reports 0xffffffff per
// child, a triangle (inf, 1.0, ...).

namespace RayTracingCases {

constexpr uint64_t BvhGuestBase = 0x0000000110000000ull;
constexpr u32 BvhBackingDword = 64;
constexpr u32 BvhResourceSgpr = 40;
// s[48:51] holds the harness output buffer descriptor.
constexpr u32 BvhCounterSgpr = 60;
constexpr u32 BvhResultVgpr = 10;
constexpr u32 BvhAddressVgpr = 40;
constexpr u32 BvhMiss = 0xffffffffu;
// Dword 15 of a triangle node: the low byte of each 8-bit field is the identity barycentric
// selector (i = 1, j = 2 << 2) for triangle types 0 and 1; the high half is only an id marker.
constexpr u32 BvhTriangleDword15 = 0xabcd0909u;
constexpr u32 BvhTriangleOffset = 128;

u32 F(float value) { return std::bit_cast<u32>(value); }

u32 H(float value) {
  switch (std::bit_cast<u32>(value)) {
  case 0x00000000u: return 0x0000u;
  case 0x3f800000u: return 0x3c00u;
  case 0xbf800000u: return 0xbc00u;
  case 0x40000000u: return 0x4000u;
  case 0x40800000u: return 0x4400u;
  case 0x40a00000u: return 0x4500u;
  case 0x40c00000u: return 0x4600u;
  case 0x7f800000u: return 0x7c00u;
  default: return 0x7e00u;
  }
}

struct Box {
  float min[3];
  float max[3];
};

struct Ray {
  float extent = 1000.0f;
  float origin[3] = {0.0f, 0.0f, -10.0f};
  float dir[3] = {0.0f, 0.0f, 1.0f};
  float inv_dir[3] = {INFINITY, INFINITY, 1.0f};
};

// Children at increasing z; the ray travels +z from z=-10 along the z axis.
constexpr std::array<u32, 4> BoxChildren{0x11u, 0x22u, 0x33u, 0x44u};
constexpr std::array<Box, 4> Boxes{{
    {{-1, -1, 4}, {1, 1, 5}}, // t = 14
    {{-1, -1, -1}, {1, 1, 1}}, // t = 9
    {{5, -1, -1}, {6, 1, 1}}, // miss: offset in x
    {{-1, -1, 0}, {1, 1, 2}}, // t = 10
}};

std::vector<u32> Box32Node() {
  std::vector<u32> node(32, 0);
  for (u32 i = 0; i < 4; i++) {
    node[i] = BoxChildren[i];
    for (u32 axis = 0; axis < 3; axis++) {
      node[4 + i * 6 + axis] = F(Boxes[i].min[axis]);
      node[7 + i * 6 + axis] = F(Boxes[i].max[axis]);
    }
  }
  return node;
}

std::vector<u32> Box16Node() {
  std::vector<u32> node(16, 0);
  for (u32 i = 0; i < 4; i++) {
    node[i] = BoxChildren[i];
    const std::array<u32, 6> halves{
        H(Boxes[i].min[0]), H(Boxes[i].min[1]), H(Boxes[i].min[2]),
        H(Boxes[i].max[0]), H(Boxes[i].max[1]), H(Boxes[i].max[2])};
    for (u32 word = 0; word < 3; word++) {
      node[4 + i * 3 + word] = halves[word * 2] | (halves[word * 2 + 1] << 16u);
    }
  }
  return node;
}

// Counter-clockwise triangle in the z = 0 plane around the origin. Dword 12 is V4.x (unused).
std::vector<u32> TriangleNode(u32 dword15 = BvhTriangleDword15) {
  std::vector<u32> node(16, 0);
  const float vertices[9] = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
  for (u32 i = 0; i < 9; i++) {
    node[i] = F(vertices[i]);
  }
  node[15] = dword15;
  return node;
}

// A pair-compressed node holding the unit quad in the z = 0 plane: V0 (-1,-1), V1 (1,-1),
// V2 (-1,1), V3 (1,1) at dwords 0, 3, 6, 9. Type 0 is (V0, V1, V2), the half with x + y < 0;
// type 1 is (V1, V3, V2), the half with x + y > 0.
std::vector<u32> QuadNode(u32 dword15) {
  std::vector<u32> node(16, 0);
  const float vertices[12] = {-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0};
  for (u32 i = 0; i < 12; i++) {
    node[i] = F(vertices[i]);
  }
  node[15] = dword15;
  return node;
}

struct Options {
  const char *name = "";
  bool bvh64 = false;
  bool a16 = false;
  bool nsa = false;
  bool sort = true;
  bool return_ij = true;
  u32 node_pointer = 5;
  // High dword of the node pointer; used only with bvh64.
  u32 node_pointer_hi = 0;
  // Descriptor size field: the last valid 64-byte node index (dword 2, plus dword 3 bits
  // [9:0] as bits [41:32]). The default covers every node the test images hold.
  uint64_t size_field = 0xffffffffu;
  // A descriptor whose base is 0.
  bool null_base = false;
  Ray ray;
  std::vector<u32> nodes;
  std::array<u32, 4> expected{};
};

// The T# that points IMAGE_BVH_INTERSECT_RAY at BvhGuestBase.
std::array<u32, 4> Descriptor(const Options &options) {
  const uint64_t base = options.null_base ? 0u : BvhGuestBase;
  return {
      static_cast<u32>(base >> 8u),
      static_cast<u32>(base >> 40u) | (options.sort ? 1u << 31u : 0u),
      static_cast<u32>(options.size_field),
      (static_cast<u32>(options.size_field >> 32u) & 0x3ffu) |
          (options.return_ij ? 1u << 24u : 0u) | (8u << 28u),
  };
}

TestCase Make(const Options &options) {
  using O = ShaderOpcode;

  std::vector<u32> address;
  address.push_back(options.node_pointer);
  if (options.bvh64) {
    address.push_back(options.node_pointer_hi);
  }
  address.push_back(F(options.ray.extent));
  for (float value : options.ray.origin) {
    address.push_back(F(value));
  }
  if (options.a16) {
    const auto &d = options.ray.dir;
    const auto &inv = options.ray.inv_dir;
    address.push_back(H(d[0]) | (H(d[1]) << 16u));
    address.push_back(H(d[2]) | (H(inv[0]) << 16u));
    address.push_back(H(inv[1]) | (H(inv[2]) << 16u));
  } else {
    for (float value : options.ray.dir) {
      address.push_back(F(value));
    }
    for (float value : options.ray.inv_dir) {
      address.push_back(F(value));
    }
  }

  // NSA scatters the components in reverse order over every other register.
  std::vector<u32> registers(address.size());
  for (u32 i = 0; i < address.size(); i++) {
    registers[i] = options.nsa ? BvhAddressVgpr + 2 * (static_cast<u32>(address.size()) - 1 - i)
                               : BvhAddressVgpr + i;
  }

  std::vector<u32> code;
  const auto descriptor = Descriptor(options);
  for (u32 i = 0; i < 4; i++) {
    AppendSMovLiteral(&code, BvhResourceSgpr + i, descriptor[i]);
  }
  for (u32 i = 0; i < address.size(); i++) {
    AppendVMovU32(&code, registers[i], address[i]);
  }
  const u32 nsa_dwords = options.nsa ? static_cast<u32>((address.size() - 1 + 3) / 4) : 0u;
  code.push_back(EncodeMimg0(options.bvh64 ? 0xe7u : 0xe6u, 0xf, nsa_dwords, false, 0, true));
  code.push_back(EncodeMimg1(BvhResultVgpr, registers[0], BvhResourceSgpr / 4, 0, options.a16));
  for (u32 word = 0; word < nsa_dwords; word++) {
    u32 packed = 0;
    for (u32 byte = 0; byte < 4; byte++) {
      const u32 index = 1 + word * 4 + byte;
      if (index < registers.size()) {
        packed |= registers[index] << (byte * 8u);
      }
    }
    code.push_back(packed);
  }
  for (u32 i = 0; i < 4; i++) {
    AppendStoreVgpr(&code, BvhResultVgpr + i, i);
  }
  AppendEnd(&code);

  TestCase test;
  test.name = options.name;
  test.code = std::move(code);
  test.initial.assign(BvhBackingDword, 0);
  test.initial.insert(test.initial.end(), options.nodes.begin(), options.nodes.end());
  test.expected.assign(options.expected.begin(), options.expected.end());
  test.bda_mappings = {{BvhGuestBase, BvhBackingDword * 4u}};
  test.opcodes = {O::S_MOV_B32, O::V_MOV_B32,
                  O::IMAGE_BVH_INTERSECT_RAY,
                  O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  return test;
}

std::vector<u32> BoxAndTriangleNodes(std::vector<u32> box,
                                     const std::vector<u32> &triangle = TriangleNode()) {
  box.resize(BvhTriangleOffset / 4u, 0);
  box.insert(box.end(), triangle.begin(), triangle.end());
  return box;
}

constexpr u32 TrianglePointer = BvhTriangleOffset >> 3u;

} // namespace RayTracingCases

TestCase BvhBox32SortsHitsByEntryDistance() {
  RayTracingCases::Options options;
  options.name = "BvhBox32SortsHitsByEntryDistance";
  options.nodes = RayTracingCases::Box32Node();
  options.expected = {0x22u, 0x44u, 0x11u, RayTracingCases::BvhMiss};
  return RayTracingCases::Make(options);
}

TestCase BvhBox32UnsortedKeepsSlotsAndHonoursExtent() {
  RayTracingCases::Options options;
  options.name = "BvhBox32UnsortedKeepsSlotsAndHonoursExtent";
  options.sort = false;
  options.ray.extent = 13.5f;
  options.nodes = RayTracingCases::Box32Node();
  options.expected = {RayTracingCases::BvhMiss, 0x22u, RayTracingCases::BvhMiss, 0x44u};
  return RayTracingCases::Make(options);
}

TestCase BvhBox16DecodesHalfBounds() {
  RayTracingCases::Options options;
  options.name = "BvhBox16DecodesHalfBounds";
  options.node_pointer = 4;
  options.nodes = RayTracingCases::Box16Node();
  options.expected = {0x22u, 0x44u, 0x11u, RayTracingCases::BvhMiss};
  return RayTracingCases::Make(options);
}

TestCase BvhTriangleReturnsBarycentricNumerators() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhTriangleReturnsBarycentricNumerators";
  options.node_pointer = RayTracingCases::TrianglePointer;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node());
  // t = -40 / -4 = 10, i = -1 / -4 = 0.25, j = -2 / -4 = 0.5.
  options.expected = {F(-40.0f), F(-4.0f), F(-1.0f), F(-2.0f)};
  return RayTracingCases::Make(options);
}

TestCase BvhTriangleReturnsIdAndHitStatus() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhTriangleReturnsIdAndHitStatus";
  options.return_ij = false;
  options.node_pointer = RayTracingCases::TrianglePointer;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node());
  // Without barycentrics: dword 15 plus the triangle type (0) as the id, and the hit flag.
  options.expected = {F(-40.0f), F(-4.0f), RayTracingCases::BvhTriangleDword15, 1u};
  return RayTracingCases::Make(options);
}

TestCase Bvh64A16NsaIntersectsBox32() {
  RayTracingCases::Options options;
  options.name = "Bvh64A16NsaIntersectsBox32";
  options.bvh64 = true;
  options.a16 = true;
  options.nsa = true;
  options.nodes = RayTracingCases::Box32Node();
  options.expected = {0x22u, 0x44u, 0x11u, RayTracingCases::BvhMiss};
  return RayTracingCases::Make(options);
}

TestCase BvhNsaIntersectsTriangle() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhNsaIntersectsTriangle";
  options.nsa = true;
  options.node_pointer = RayTracingCases::TrianglePointer;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node());
  options.expected = {F(-40.0f), F(-4.0f), F(-1.0f), F(-2.0f)};
  return RayTracingCases::Make(options);
}

TestCase BvhInstanceNodeReportsMiss() {
  RayTracingCases::Options options;
  options.name = "BvhInstanceNodeReportsMiss";
  options.node_pointer = 6;
  options.nodes = RayTracingCases::Box32Node();
  options.expected = {RayTracingCases::BvhMiss, RayTracingCases::BvhMiss,
                      RayTracingCases::BvhMiss, RayTracingCases::BvhMiss};
  return RayTracingCases::Make(options);
}

// Derivation for the triangle cases below: the ray travels +z, so the watertight test projects
// onto x/y with kz = z, kx = x, ky = y and no shear; with p = vertex - origin, the edge
// functions are u = p2.x*p1.y - p2.y*p1.x, v = p0.x*p2.y - p0.y*p2.x, w = p1.x*p0.y - p1.y*p0.x,
// det = u + v + w and t = 10 * det (every vertex is 10 units ahead). i_num = v, j_num = w and
// k_num = u = det - v - w.

// Ray at (0.5, 0.25): type 1 is (V1, V3, V2) with p = (0.5,-1.25), (0.5,0.75), (-1.5,0.75),
// so u = -1.5, v = -1.5, w = -1, det = -4.
TestCase BvhTrianglePairSelectsSecondTriangle() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhTrianglePairSelectsSecondTriangle";
  options.ray.origin[0] = 0.5f;
  options.ray.origin[1] = 0.25f;
  options.node_pointer = RayTracingCases::TrianglePointer | 1u;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node(),
                                                       RayTracingCases::QuadNode(0x0909u));
  options.expected = {F(-40.0f), F(-4.0f), F(-1.5f), F(-1.0f)};
  return RayTracingCases::Make(options);
}

// Ray at (-0.5, -0.25) hits type 0 and not V3: p = (-0.5,-0.75), (1.5,-0.75), (-0.5,1.25)
// gives u = -1.5, v = -1, w = -1.5, det = -4.
TestCase BvhTrianglePairSelectsFirstTriangle() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhTrianglePairSelectsFirstTriangle";
  options.ray.origin[0] = -0.5f;
  options.ray.origin[1] = -0.25f;
  options.node_pointer = RayTracingCases::TrianglePointer;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node(),
                                                       RayTracingCases::QuadNode(0x0909u));
  options.expected = {F(-40.0f), F(-4.0f), F(-1.0f), F(-1.5f)};
  return RayTracingCases::Make(options);
}

// Type 1 reads its selectors from bits [15:8]: 0x06 picks j for i and i for j, while the
// type-0 byte (0x09) is the identity.
TestCase BvhTrianglePairUsesSecondSelectorByte() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhTrianglePairUsesSecondSelectorByte";
  options.ray.origin[0] = 0.5f;
  options.ray.origin[1] = 0.25f;
  options.node_pointer = RayTracingCases::TrianglePointer | 1u;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node(),
                                                       RayTracingCases::QuadNode(0x0609u));
  options.expected = {F(-40.0f), F(-4.0f), F(-1.0f), F(-1.5f)};
  return RayTracingCases::Make(options);
}

// As BvhTriangleReturnsBarycentricNumerators (i_num = -1, j_num = -2), with dw15 = 0x06:
// sel_i = 2 returns j_num and sel_j = 1 returns i_num.
TestCase BvhTriangleRemapSwapsBarycentrics() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhTriangleRemapSwapsBarycentrics";
  options.node_pointer = RayTracingCases::TrianglePointer;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node(),
                                                       RayTracingCases::TriangleNode(0x06u));
  options.expected = {F(-40.0f), F(-4.0f), F(-2.0f), F(-1.0f)};
  return RayTracingCases::Make(options);
}

// Ray at (0.25, 0) through TriangleNode: p = (-1.25,-1), (0.75,-1), (-0.25,1) gives
// u = -0.5, v = -1.5, w = -2, det = -4. dw15 = 0x04: sel_i = 0 returns k_num = -0.5,
// sel_j = 1 returns i_num = -1.5.
TestCase BvhTriangleRemapSelectsThirdBarycentric() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhTriangleRemapSelectsThirdBarycentric";
  options.ray.origin[0] = 0.25f;
  options.node_pointer = RayTracingCases::TrianglePointer;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node(),
                                                       RayTracingCases::TriangleNode(0x04u));
  options.expected = {F(-40.0f), F(-4.0f), F(-0.5f), F(-1.5f)};
  return RayTracingCases::Make(options);
}

// The triangle is node 2 (pointer 16), past a size field of 1. Only box nodes are range checked
// (GTA V stores its triangles after the box nodes its descriptors count), so it is tested.
TestCase BvhTrianglePastSizeFieldIsTested() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhTrianglePastSizeFieldIsTested";
  options.size_field = 1;
  options.node_pointer = RayTracingCases::TrianglePointer;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node());
  options.expected = {F(-40.0f), F(-4.0f), F(-1.0f), F(-2.0f)};
  return RayTracingCases::Make(options);
}

// A null BVH reports an invalid node instead of reading near address 0 (the backing holds a
// triangle that the ray would hit at node 2 of a mapped BVH).
TestCase BvhNullBaseReturnsInvalid() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhNullBaseReturnsInvalid";
  options.null_base = true;
  options.node_pointer = RayTracingCases::TrianglePointer;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node());
  options.expected = {F(INFINITY), F(1.0f), RayTracingCases::BvhMiss, 0u};
  return RayTracingCases::Make(options);
}

// Index == size field is the last valid node, so a 64-byte triangle there is in range.
TestCase BvhTriangleAtLastNodeIsInRange() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhTriangleAtLastNodeIsInRange";
  options.size_field = 2;
  options.node_pointer = RayTracingCases::TrianglePointer;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node());
  options.expected = {F(-40.0f), F(-4.0f), F(-1.0f), F(-2.0f)};
  return RayTracingCases::Make(options);
}

// A box32 spans two nodes, so one starting at index == size field runs past the range.
TestCase BvhBox32AtLastNodeReturnsInvalid() {
  RayTracingCases::Options options;
  options.name = "BvhBox32AtLastNodeReturnsInvalid";
  options.size_field = 0;
  options.nodes = RayTracingCases::Box32Node();
  options.expected = {RayTracingCases::BvhMiss, RayTracingCases::BvhMiss,
                      RayTracingCases::BvhMiss, RayTracingCases::BvhMiss};
  return RayTracingCases::Make(options);
}

TestCase BvhBox32EndingAtLastNodeIsInRange() {
  RayTracingCases::Options options;
  options.name = "BvhBox32EndingAtLastNodeIsInRange";
  options.size_field = 1;
  options.nodes = RayTracingCases::Box32Node();
  options.expected = {0x22u, 0x44u, 0x11u, RayTracingCases::BvhMiss};
  return RayTracingCases::Make(options);
}

// Size bits [41:32] extend the size field: the low dword alone (1) would reject node 2.
TestCase BvhHighSizeBitsExtendSizeField() {
  using RayTracingCases::F;
  RayTracingCases::Options options;
  options.name = "BvhHighSizeBitsExtendSizeField";
  options.size_field = (uint64_t{1} << 32u) | 1u;
  options.node_pointer = RayTracingCases::TrianglePointer;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node());
  options.expected = {F(-40.0f), F(-4.0f), F(-1.0f), F(-2.0f)};
  return RayTracingCases::Make(options);
}

// A box16 is one 64-byte node, so unlike a box32 it is in range at index == size field.
TestCase BvhBox16AtLastNodeIsInRange() {
  RayTracingCases::Options options;
  options.name = "BvhBox16AtLastNodeIsInRange";
  options.size_field = 0;
  options.node_pointer = 4;
  options.nodes = RayTracingCases::Box16Node();
  options.expected = {0x22u, 0x44u, 0x11u, RayTracingCases::BvhMiss};
  return RayTracingCases::Make(options);
}

// A child box with min > max holds no points: it misses even when the ray crosses the region
// it would span if its bounds were swapped. Child 0x33 straddles the ray here.
TestCase BvhInvertedBoxChildIsMiss() {
  RayTracingCases::Options options;
  options.name = "BvhInvertedBoxChildIsMiss";
  auto node = RayTracingCases::Box32Node();
  for (u32 axis = 0; axis < 3; axis++) {
    node[4 + 2 * 6 + axis] = RayTracingCases::F(1.0f);
    node[7 + 2 * 6 + axis] = RayTracingCases::F(-1.0f);
  }
  options.nodes = std::move(node);
  options.expected = {0x22u, 0x44u, 0x11u, RayTracingCases::BvhMiss};
  return RayTracingCases::Make(options);
}

// The empty-slot encoding min = +inf, max = -inf misses without its bounds turning into NaN.
TestCase BvhInfiniteEmptyBoxChildIsMiss() {
  RayTracingCases::Options options;
  options.name = "BvhInfiniteEmptyBoxChildIsMiss";
  auto node = RayTracingCases::Box32Node();
  for (u32 axis = 0; axis < 3; axis++) {
    node[4 + 2 * 6 + axis] = RayTracingCases::F(INFINITY);
    node[7 + 2 * 6 + axis] = RayTracingCases::F(-INFINITY);
  }
  options.nodes = std::move(node);
  options.expected = {0x22u, 0x44u, 0x11u, RayTracingCases::BvhMiss};
  return RayTracingCases::Make(options);
}

// A 64-bit box32 pointer with bit 61 set: the node address ((pointer & ~7) << 3) wraps modulo
// 2^64 to the same box at byte 0, but the node index pointer >> 3 is 2^58. A size field of
// 0xffffffff (bits [41:32] clear) is checked, so the pointer is out of range.
TestCase Bvh64HighPointerReturnsInvalid() {
  RayTracingCases::Options options;
  options.name = "Bvh64HighPointerReturnsInvalid";
  options.bvh64 = true;
  options.node_pointer_hi = 1u << 29u;
  options.nodes = RayTracingCases::BoxAndTriangleNodes(RayTracingCases::Box32Node());
  options.expected = {RayTracingCases::BvhMiss, RayTracingCases::BvhMiss,
                      RayTracingCases::BvhMiss, RayTracingCases::BvhMiss};
  return RayTracingCases::Make(options);
}

// Children 1 and 3 contain the ray origin (t_min -2 and -5), child 2 is hit at t = 9 and
// child 0 is missed. The key max(t_min, 0) ties children 1 and 3 at 0; the comparator swaps
// only on a strict order, so tracing the (0,1)(2,3)(0,2)(1,3)(1,2) network gives
// 0x22, 0x44, 0x33, miss. The unclamped key would put 0x44 (-5) before 0x22 (-2).
TestCase BvhSortPutsMissesLastAndClampsKey() {
  using namespace RayTracingCases;
  constexpr std::array<Box, 4> boxes{{
      {{5, -1, -1}, {6, 1, 1}},    // miss: offset in x
      {{-1, -1, -12}, {1, 1, -8}}, // contains the origin: t_min = -2
      {{-1, -1, -1}, {1, 1, 1}},   // t_min = 9
      {{-1, -1, -15}, {1, 1, -5}}, // contains the origin: t_min = -5
  }};
  std::vector<u32> node(32, 0);
  for (u32 i = 0; i < 4; i++) {
    node[i] = BoxChildren[i];
    for (u32 axis = 0; axis < 3; axis++) {
      node[4 + i * 6 + axis] = F(boxes[i].min[axis]);
      node[7 + i * 6 + axis] = F(boxes[i].max[axis]);
    }
  }
  Options options;
  options.name = "BvhSortPutsMissesLastAndClampsKey";
  options.nodes = std::move(node);
  options.expected = {0x22u, 0x44u, 0x33u, BvhMiss};
  return Make(options);
}

// The T# is fetched with s_load_dwordx4 through a pointer that the loop advances, as a guest
// traversal kernel walking a descriptor table does: the load's base is a loop-carried phi, so it
// has no pre-dispatch value and must stay a GPU read (upstream PR #640, fix A). Only the last of
// the three table entries (the one loaded on the final iteration) is valid; the first two have
// a size field of 0, which puts the box32 node out of range and returns all misses.
TestCase BvhDescriptorLoadedThroughLoopCarriedPointer() {
  using namespace RayTracingCases;
  using O = ShaderOpcode;

  constexpr u32 TableOffset = 0x100;
  constexpr uint64_t TableGuest = BvhGuestBase + TableOffset;
  constexpr u32 PointerSgpr = 2;
  constexpr u32 Entries = 3;

  Options options;
  options.name = "BvhDescriptorLoadedThroughLoopCarriedPointer";
  options.nodes = Box32Node();
  options.expected = {0x22u, 0x44u, 0x11u, BvhMiss};
  auto test = Make(options);

  // Drop Make's four s_mov descriptor writes (two words each) and load the T# instead.
  std::vector<u32> code;
  AppendSMovLiteral(&code, PointerSgpr, static_cast<u32>(TableGuest));
  AppendSMovLiteral(&code, PointerSgpr + 1, static_cast<u32>(TableGuest >> 32u));
  AppendSMovLiteral(&code, BvhCounterSgpr, Entries);
  const auto loop = static_cast<int32_t>(code.size());
  code.push_back(EncodeSmem0(0x02u, BvhResourceSgpr, PointerSgpr / 2u)); // s_load_dwordx4
  code.push_back(EncodeSmem1(0, 125)); // soffset = null
  code.push_back(EncodeSop2(0x00, PointerSgpr, PointerSgpr, InlineU32(16))); // s_add_u32
  code.push_back(EncodeSop2(0x04, PointerSgpr + 1, PointerSgpr + 1, InlineU32(0))); // s_addc_u32
  code.push_back(EncodeSop2(0x01, BvhCounterSgpr, BvhCounterSgpr, InlineU32(1))); // s_sub_u32
  code.push_back(EncodeSopc(0x07, BvhCounterSgpr, InlineU32(0)));                // s_cmp_lg_u32
  const auto next = static_cast<int32_t>(code.size()) + 1;
  code.push_back(EncodeSopp(0x05, static_cast<u32>(loop - next) & 0xffffu)); // s_cbranch_scc1
  // Guard the splice: the preamble must be exactly the four descriptor s_mov writes.
  for (u32 word = 0; word < 4; word++) {
    Require("BvhDescriptorLoadedThroughLoopCarriedPointer", "preamble",
            test.code.size() > 8 &&
                test.code[word * 2u] == EncodeSMovB32(BvhResourceSgpr + word, 255u),
            "Make no longer starts with four s_mov literal T# writes");
  }
  code.insert(code.end(), test.code.begin() + 8, test.code.end());
  test.code = std::move(code);

  auto invalid = options;
  invalid.size_field = 0;
  test.initial.resize(BvhBackingDword + TableOffset / 4u, 0);
  for (u32 entry = 0; entry < Entries; entry++) {
    const auto descriptor = Descriptor(entry + 1u == Entries ? options : invalid);
    test.initial.insert(test.initial.end(), descriptor.begin(), descriptor.end());
  }
  test.opcodes = {O::S_MOV_B32, O::S_LOAD_DWORDX4, O::S_ADD_U32, O::S_ADDC_U32,
                  O::S_SUB_U32, O::S_CMP_LG_U32, O::S_CBRANCH_SCC1, O::V_MOV_B32,
                  O::IMAGE_BVH_INTERSECT_RAY, O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  return test;
}

// A node whose page has no host backing reports an invalid node and records the fault. The
// mapping covers only the first guest page; a box32 ending that page reads its second block
// from the missing one, and a triangle at the next node lies wholly inside it.
TestCase BvhNodeInMissingPage(bool box) {
  using namespace RayTracingCases;
  constexpr u32 PageNodes = static_cast<u32>(BufferCache::CACHING_PAGESIZE / 64u);
  Options options;
  options.name = box ? "BvhBox32EndingMappedPageFaultsOnNext" : "BvhTriangleInMissingPageFaults";
  options.node_pointer = box ? ((PageNodes - 1u) << 3u) | 5u : PageNodes << 3u;
  options.nodes.assign(PageNodes * 16u, 0);
  if (box) {
    const auto node = Box32Node();
    std::copy_n(node.begin(), 16, options.nodes.end() - 16);
  }
  options.expected = box ? std::array<u32, 4>{BvhMiss, BvhMiss, BvhMiss, BvhMiss}
                         : std::array<u32, 4>{F(INFINITY), F(1.0f), BvhMiss, 0u};
  auto test = Make(options);
  test.expected_fault_pages = {BvhGuestBase + BufferCache::CACHING_PAGESIZE};
  return test;
}

// A box16 is one 64-byte node: ending the mapped page, it needs nothing from the missing next.
TestCase BvhBox16EndingMappedPageIsTested() {
  using namespace RayTracingCases;
  constexpr u32 PageNodes = static_cast<u32>(BufferCache::CACHING_PAGESIZE / 64u);
  Options options;
  options.name = "BvhBox16EndingMappedPageIsTested";
  options.node_pointer = ((PageNodes - 1u) << 3u) | 4u;
  options.nodes.assign(PageNodes * 16u, 0);
  const auto node = Box16Node();
  std::copy(node.begin(), node.end(), options.nodes.end() - 16);
  options.expected = {0x22u, 0x44u, 0x11u, BvhMiss};
  return Make(options);
}

void AddRayTracingCases(std::vector<TestCase> &cases) {
  cases.push_back(BvhBox32SortsHitsByEntryDistance());
  cases.push_back(BvhBox32UnsortedKeepsSlotsAndHonoursExtent());
  cases.push_back(BvhBox16DecodesHalfBounds());
  cases.push_back(BvhTriangleReturnsBarycentricNumerators());
  cases.push_back(BvhTriangleReturnsIdAndHitStatus());
  cases.push_back(Bvh64A16NsaIntersectsBox32());
  cases.push_back(BvhNsaIntersectsTriangle());
  cases.push_back(BvhInstanceNodeReportsMiss());
  cases.push_back(BvhTrianglePairSelectsSecondTriangle());
  cases.push_back(BvhTrianglePairSelectsFirstTriangle());
  cases.push_back(BvhTrianglePairUsesSecondSelectorByte());
  cases.push_back(BvhTriangleRemapSwapsBarycentrics());
  cases.push_back(BvhTriangleRemapSelectsThirdBarycentric());
  cases.push_back(BvhTrianglePastSizeFieldIsTested());
  cases.push_back(BvhNullBaseReturnsInvalid());
  cases.push_back(BvhTriangleAtLastNodeIsInRange());
  cases.push_back(BvhBox32AtLastNodeReturnsInvalid());
  cases.push_back(BvhBox32EndingAtLastNodeIsInRange());
  cases.push_back(BvhHighSizeBitsExtendSizeField());
  cases.push_back(BvhSortPutsMissesLastAndClampsKey());
  cases.push_back(BvhBox16AtLastNodeIsInRange());
  cases.push_back(BvhInvertedBoxChildIsMiss());
  cases.push_back(BvhInfiniteEmptyBoxChildIsMiss());
  cases.push_back(Bvh64HighPointerReturnsInvalid());
  cases.push_back(BvhDescriptorLoadedThroughLoopCarriedPointer());
  cases.push_back(BvhNodeInMissingPage(true));
  cases.push_back(BvhNodeInMissingPage(false));
  cases.push_back(BvhBox16EndingMappedPageIsTested());
}
