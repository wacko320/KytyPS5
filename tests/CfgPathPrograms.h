#ifndef KYTY_TESTS_CFG_PATH_PROGRAMS_H_
#define KYTY_TESTS_CFG_PATH_PROGRAMS_H_

// Random control-flow shapes for checking goto elimination: arbitrary graphs biased toward a few
// shared targets, and linked-call ladders (tests, each skipping a callee that returns to a
// shared join), optionally inside loops and with callees holding their own control flow.
// Blocks are abstract: each test encodes them with the instructions it needs.

#include <cstdint>
#include <vector>

struct CfgPathBlock {
  enum Kind { Next, Cond, Jump, End } kind = Next;
  uint32_t target = 0;
};

struct CfgPathRandom {
  uint64_t state;
  uint32_t operator()() {
    state += 0x9e3779b97f4a7c15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
    return static_cast<uint32_t>(z ^ (z >> 31u));
  }
  uint32_t Below(uint32_t bound) { return bound == 0 ? 0 : (*this)() % bound; }
  bool Chance(uint32_t percent) { return Below(100) < percent; }
};

// Arbitrary blocks, with gotos biased toward a few shared targets. The last block ends.
inline std::vector<CfgPathBlock> RandomCfgPath(CfgPathRandom &random) {
  const uint32_t count = 4 + random.Below(28);
  std::vector<uint32_t> hot;
  for (uint32_t index = 0; index < 1 + random.Below(3); index++)
    hot.push_back(1 + random.Below(count - 1));
  std::vector<CfgPathBlock> blocks(count);
  for (uint32_t index = 0; index + 1 < count; index++) {
    auto &block = blocks[index];
    const auto choice = random.Below(100);
    block.kind = choice < 40   ? CfgPathBlock::Cond
                 : choice < 65 ? CfgPathBlock::Jump
                 : choice < 72 ? CfgPathBlock::End
                               : CfgPathBlock::Next;
    if (random.Chance(50)) block.target = hot[random.Below(static_cast<uint32_t>(hot.size()))];
    else if (random.Chance(20)) block.target = random.Below(index + 1);
    else block.target = index + 1 + random.Below(count - index - 1);
  }
  blocks.back().kind = CfgPathBlock::End;
  return blocks;
}

inline void EmitCfgPathLadder(CfgPathRandom &random, std::vector<CfgPathBlock> &blocks,
                              uint32_t exit, uint32_t depth) {
  constexpr uint32_t Pending = UINT32_MAX;
  const bool loop = random.Chance(60);
  const auto header = static_cast<uint32_t>(blocks.size());
  blocks.push_back({CfgPathBlock::Next});
  std::vector<uint32_t> returns;
  const uint32_t callees = 4 + random.Below(7);
  for (uint32_t callee = 0; callee < callees; callee++) {
    const auto test = static_cast<uint32_t>(blocks.size());
    blocks.push_back({CfgPathBlock::Cond});
    const auto body = static_cast<uint32_t>(blocks.size());
    if (depth < 2 && random.Chance(15)) EmitCfgPathLadder(random, blocks, exit, depth + 1);
    const uint32_t size = 1 + random.Below(4);
    for (uint32_t index = 0; index < size; index++) {
      const auto self = static_cast<uint32_t>(blocks.size());
      CfgPathBlock block{CfgPathBlock::Next};
      const auto choice = random.Below(100);
      if (choice < 15) {
        block = {CfgPathBlock::Cond, Pending}; // an early return
        returns.push_back(self);
      } else if (choice < 25 && self > body) {
        block = {CfgPathBlock::Cond, body + random.Below(self - body + 1)}; // a loop in the callee
      } else if (choice < 32) {
        block = {CfgPathBlock::Cond, exit}; // leave the program
      } else if (choice < 42 && index + 1 < size) {
        block = {CfgPathBlock::Cond, self + 2}; // a skip within the callee
      }
      blocks.push_back(block);
    }
    returns.push_back(static_cast<uint32_t>(blocks.size()));
    blocks.push_back({CfgPathBlock::Jump, Pending});
    blocks[test].target = static_cast<uint32_t>(blocks.size()); // a mismatch skips the callee
  }
  const auto join = static_cast<uint32_t>(blocks.size());
  for (const auto index : returns) blocks[index].target = join;
  blocks.push_back({CfgPathBlock::Next});
  if (loop) blocks.push_back({CfgPathBlock::Cond, header});
}

// A ladder, perhaps inside a loop around everything; the last block ends.
inline std::vector<CfgPathBlock> RandomCfgLadder(CfgPathRandom &random) {
  constexpr uint32_t Exit = UINT32_MAX - 1;
  std::vector<CfgPathBlock> blocks{{CfgPathBlock::Next}};
  EmitCfgPathLadder(random, blocks, Exit, 0);
  if (random.Chance(50)) blocks.push_back({CfgPathBlock::Cond, 1u});
  const auto exit = static_cast<uint32_t>(blocks.size());
  blocks.push_back({CfgPathBlock::End});
  for (auto &block : blocks) {
    if (block.target == Exit) block.target = exit;
  }
  return blocks;
}

inline std::vector<CfgPathBlock> RandomCfgProgram(uint64_t seed) {
  CfgPathRandom random{0x5eed0000ull + seed};
  return seed % 3 == 0 ? RandomCfgPath(random) : RandomCfgLadder(random);
}

#endif
