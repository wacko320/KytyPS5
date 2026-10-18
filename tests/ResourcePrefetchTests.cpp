// ResourcePrefetcher: worker-thread resource materialization must give the serial result, and
// fall back to it whenever the memory it read changed. `--bench` times a synthetic GPU-thread
// loop with and without the workers.

#include "graphics/host_gpu/renderer/cache/srtPageWatch.h"
#include "graphics/host_gpu/renderer/pipeline/resourcePrefetch.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <span>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ResourcePrefetcher;
using Libs::Graphics::SrtPageWatch;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourcePrefetchTests: failed: %s\n", text);
    std::abort();
  }
}

Block &AddValueBlock(Program &program) {
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
  program.block_storage.push_back(std::move(block));
  return *result;
}

// user data s[0:1] points at a root table whose first two words point at a descriptor table:
// `buffers` V#s, loaded one dword at a time, the way SRT walks read guest memory. Every
// `clean_every`th flattened SRT slot is read through the strict reader.
struct TablePlanShape {
  uint32_t buffers = 8;
  uint32_t clean_every = 0;
  // Extra arithmetic per dword, to make one evaluation cost what a game shader's does.
  uint32_t arithmetic = 0;
  // A uniform-buffer dword read over the descriptor table (LoadBufferU32) sizes one more,
  // shader-written buffer whose address comes from the root table: ExtractResourcePlan
  // captures the plan's reads, and reads that address through the clean walker.
  bool capture = false;
};

// Tests run again with every table plan capturing (TablePlanShape::capture).
bool g_capture_plans = false;

ResourcePlan TablePlan(const TablePlanShape &shape) {
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);

  auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
  const auto load = [&](Value low, Value high, uint32_t offset) {
    auto &handle =
        block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
    auto &raw = block.AppendNewInst(
        ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(offset), Value(0u), Value(true)});
    raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
    const auto slot = static_cast<uint32_t>(program.srt_reads.size());
    program.srt_reads.push_back({Value(&raw), slot});
    return Value(&block.AppendNewInst(ValueOpcode::ReadConst,
                                      {Value(&srt), Value(slot)}));
  };
  const auto low = Value(&block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))}));
  const auto high = Value(&block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(1))}));
  const auto table_low = load(low, high, 0);
  const auto table_high = load(low, high, 4);
  const auto salt = Value(&block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(2))}));
  for (uint32_t buffer = 0; buffer < shape.buffers; buffer++) {
    DescriptorSource source;
    source.dword_count = 4;
    for (uint32_t dword = 0; dword < 4; dword++) {
      auto value = load(table_low, table_high, (buffer * 4u + dword) * 4u);
      for (uint32_t i = 0; i < shape.arithmetic; i++) {
        value = Value(&block.AppendNewInst(ValueOpcode::IAdd32, {value, salt}));
        value = Value(&block.AppendNewInst(ValueOpcode::ISub32, {value, salt}));
      }
      source.dwords[dword] = value;
    }
    program.descriptor_sources.push_back(source);
    program.info.buffers.push_back(
        {.source = static_cast<uint32_t>(program.descriptor_sources.size() - 1u)});
  }
  if (shape.capture || g_capture_plans) {
    MemoryInfo buffer;
    buffer.kind = ResourceKind::Buffer;
    program.memory_info.push_back(buffer);
    const auto stride_high = Value(&block.AppendNewInst(
        ValueOpcode::IAdd32, {table_high, Value(4u << 16u)}));
    auto &handle = block.AppendNewInst(
        ValueOpcode::GetBufferResource,
        {table_low, stride_high, Value(4u), Value(0x16204u)});
    auto &records = block.AppendNewInst(
        ValueOpcode::LoadBufferU32,
        {Value(&handle), Value(0u), Value(0u), Value(0u), Value(true)});
    records.SetFlags(MemoryFlags{.index = 1});
    DescriptorSource source;
    source.dword_count = 4;
    source.dwords = {table_low, table_high, Value(&records), Value(0x16204u)};
    program.descriptor_sources.push_back(source);
    program.info.buffers.push_back(
        {.source = static_cast<uint32_t>(program.descriptor_sources.size() - 1u),
         .written = true});
  }
  auto plan = ExtractResourcePlan(program);
  Check(plan.capture_specialization_reads == (shape.capture || g_capture_plans),
        "the fixture's capturing plan was not captured");
  if (shape.clean_every != 0) {
    // Strict reads: the plan's flat-slot reads (ExtractResourcePlan resets their flags) go
    // through the specialization reader, which a clean evaluator serves.
    for (size_t slot = 0; slot < plan.srt_reads.size(); slot += shape.clean_every) {
      auto *read = plan.srt_reads[slot].value.ResolveInstruction();
      read->SetFlags(SrtReadFlags{.index = read->Flags<SrtReadFlags>().index, .clean = 1});
    }
    plan.requires_specialization_memory = true;
    // Real plans mark clean reads before CompileSrtWalker, which then compiles no walker for
    // them (they capture); the raw JIT walk would bypass the strict reader.
    plan.srt_code.reset();
  }
  return plan;
}

// A plan and its per-worker copies, as the pipeline cache keeps them.
struct PlanSet {
  PlanSet(const TablePlanShape &shape, uint32_t workers) : plan(TablePlan(shape)) {
    for (uint32_t i = 0; i < workers; i++) {
      replicas.push_back(std::make_unique<ResourcePlan>(TablePlan(shape)));
      views.push_back(replicas.back().get());
    }
  }

  ResourcePlan plan;
  std::vector<std::unique_ptr<ResourcePlan>> replicas;
  std::vector<const ResourcePlan *> views;
};

// Guest memory for the tests: host words, with a set of "GPU-written" ranges the strict
// reader refuses, like TryReadGpuCleanBacking.
struct Memory {
  std::vector<uint32_t> words;
  std::vector<std::pair<uint64_t, uint64_t>> dirty;

  uint64_t Address(size_t word) const {
    return reinterpret_cast<uint64_t>(words.data() + word);
  }
};

Memory *g_memory = nullptr;
std::atomic<uint64_t> g_unreadable_page{0};
std::atomic<bool> g_block_views{false};
std::atomic<bool> g_view_entered{false};
// Pages the workers viewed: a job that ran viewed at least one.
std::atomic<uint64_t> g_views{0};

const void *ViewPage(uint64_t page) {
  g_views.fetch_add(1);
  if (g_block_views.load()) {
    g_view_entered.store(true);
    while (g_block_views.load()) {
      std::this_thread::yield();
    }
  }
  return page == g_unreadable_page.load() ? nullptr
                                          : reinterpret_cast<const void *>(page);
}

bool StrictRead(void *userdata, uint64_t address, std::span<uint32_t> values) {
  const auto &memory = *static_cast<const Memory *>(userdata);
  const auto end = address + values.size_bytes();
  for (const auto &[begin, size] : memory.dirty) {
    if (address < begin + size && begin < end) {
      return false;
    }
  }
  if (values.empty()) {
    return false;
  }
  std::memcpy(values.data(), reinterpret_cast<const void *>(address),
              values.size_bytes());
  return true;
}

// The pipeline cache's runtime also has an ordinary reader: the strict read when the bytes are
// clean, else the guest memory itself.
bool g_serial_ordinary_reader = false;

bool OrdinaryRead(void *userdata, uint64_t address, std::span<uint32_t> values) {
  if (!StrictRead(userdata, address, values)) {
    std::memcpy(values.data(), reinterpret_cast<const void *>(address), values.size_bytes());
  }
  return true;
}

SrtRuntime SerialRuntime(Memory &memory, std::span<const uint32_t> user_data) {
  return {.user_data = user_data,
          .shader_base = 0x1000,
          .read_memory = g_serial_ordinary_reader ? OrdinaryRead : nullptr,
          .userdata = &memory,
          .read_specialization_memory = StrictRead};
}

bool SameSnapshot(const ResourceSnapshot &a, const ResourceSnapshot &b) {
  return a.buffers == b.buffers && a.images == b.images &&
         a.samplers == b.samplers && a.flattened_srt == b.flattened_srt &&
         a.user_data == b.user_data && a.dma_prefetch == b.dma_prefetch &&
         a.uniform_fill == b.uniform_fill &&
         a.specialization_reads == b.specialization_reads &&
         a.gpu_flat_copies == b.gpu_flat_copies;
}

struct Result {
  bool ok = false;
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
};

Result Serial(const ResourcePlan &plan, Memory &memory,
              std::span<const uint32_t> user_data) {
  Result result;
  result.ok = MaterializeResources(plan, SerialRuntime(memory, user_data),
                                   result.snapshot, result.specialization);
  return result;
}

// The pipeline cache's use: a worker result when one is accepted, the serial path otherwise.
Result Prepared(ResourcePrefetcher &prefetcher, const PlanSet &plans,
                Memory &memory, std::span<const uint32_t> user_data,
                bool *hit = nullptr) {
  Result result;
  const auto runtime = SerialRuntime(memory, user_data);
  const bool used = prefetcher.TryMaterialize(&plans, runtime, result.snapshot,
                                              result.specialization);
  if (hit != nullptr) {
    *hit = used;
  }
  result.ok = used || MaterializeResources(plans.plan, runtime, result.snapshot,
                                           result.specialization);
  return result;
}

void CheckSame(const Result &a, const Result &b, const char *text) {
  Check(a.ok && b.ok, text);
  Check(SameSnapshot(a.snapshot, b.snapshot) &&
            a.specialization == b.specialization,
        text);
}

// Root tables at word 0 of each 64-word block; descriptor tables after them.
constexpr uint32_t Tables = 64;
constexpr uint32_t TableWords = 256;

Memory MakeMemory(uint32_t seed) {
  Memory memory;
  memory.words.resize(Tables * TableWords * 2);
  std::mt19937 random(seed);
  for (auto &word : memory.words) {
    word = random();
  }
  for (uint32_t table = 0; table < Tables; table++) {
    const auto descriptors = memory.Address((Tables + table) * TableWords);
    memory.words[table * TableWords] = static_cast<uint32_t>(descriptors);
    memory.words[table * TableWords + 1] = static_cast<uint32_t>(descriptors >> 32u);
  }
  return memory;
}

std::array<uint32_t, 4> UserData(const Memory &memory, uint32_t table, uint32_t salt) {
  const auto root = memory.Address(table * TableWords);
  return {static_cast<uint32_t>(root), static_cast<uint32_t>(root >> 32u), salt, 0};
}

void TestReplicaEvaluatesLikeThePlan() {
  auto memory = MakeMemory(1);
  const TablePlanShape shape{.buffers = 6, .clean_every = 3};
  const auto plan = TablePlan(shape);
  const auto replica = TablePlan(shape);
  for (uint32_t table = 0; table < 8; table++) {
    const auto user_data = UserData(memory, table, table);
    CheckSame(Serial(plan, memory, user_data), Serial(replica, memory, user_data),
              "a replica evaluated differently from its plan");
  }
}

// Workers write into outputs that last held another draw's result.
void TestOutputsDoNotDependOnPriorContents() {
  auto memory = MakeMemory(2);
  const auto plan = TablePlan({.buffers = 6, .clean_every = 2});
  auto reused = Serial(plan, memory, UserData(memory, 5, 1));
  const auto user_data = UserData(memory, 9, 2);
  reused.ok = MaterializeResources(plan, SerialRuntime(memory, user_data),
                                   reused.snapshot, reused.specialization);
  CheckSame(reused, Serial(plan, memory, user_data),
            "materialization depended on its outputs' earlier contents");
}

void TestSpeculationMatchesSerial() {
  auto memory = MakeMemory(3);
  g_memory = &memory;
  constexpr uint32_t Workers = 3;
  const PlanSet vertex({.buffers = 6, .clean_every = 4}, Workers);
  const PlanSet pixel({.buffers = 10, .clean_every = 0}, Workers);
  std::mt19937 random(7);
  constexpr uint32_t Draws = 3000;
  constexpr uint32_t Ahead = 8;
  std::vector<std::array<uint32_t, 4>> vertex_data(Draws + Ahead);
  std::vector<std::array<uint32_t, 4>> pixel_data(Draws + Ahead);
  for (uint32_t i = 0; i < Draws + Ahead; i++) {
    vertex_data[i] = UserData(memory, random() % Tables, random());
    pixel_data[i] = UserData(memory, random() % Tables, random());
  }
  uint32_t submitted = 0;
  uint32_t hits = 0;
  ResourcePrefetcher prefetcher(Workers, {.view_page = ViewPage});
  for (uint32_t draw = 1; draw <= Draws; draw++) {
    for (; submitted < draw + Ahead; submitted++) {
      const auto index = submitted;
      prefetcher.Submit(index + 1, &pixel, pixel.views, pixel_data[index], 0x1000);
      prefetcher.Submit(index + 1, &vertex, vertex.views, vertex_data[index], 0x1000);
    }
    prefetcher.BeginDraw(draw);
    bool hit = false;
    CheckSame(Prepared(prefetcher, pixel, memory, pixel_data[draw - 1], &hit),
              Serial(pixel.plan, memory, pixel_data[draw - 1]),
              "a pixel draw's prepared resources differ from the serial ones");
    hits += hit ? 1u : 0u;
    CheckSame(Prepared(prefetcher, vertex, memory, vertex_data[draw - 1], &hit),
              Serial(vertex.plan, memory, vertex_data[draw - 1]),
              "a vertex draw's prepared resources differ from the serial ones");
    hits += hit ? 1u : 0u;
  }
  const auto &stats = prefetcher.GetStats();
  Check(hits == stats.hits && hits > 0, "no draw used a worker result");
  Check(stats.stale == 0 && stats.failed == 0 && stats.mismatched == 0,
        "unchanged memory was reported stale");
  std::printf("  speculation: %u of %u evaluations from workers (%llu cancelled)\n",
              hits, Draws * 2, static_cast<unsigned long long>(stats.cancelled));
}

void WaitUntilEvaluated() {
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

// Until every handed-over job has finished or was cancelled: a loaded machine can delay the
// workers well past a fixed sleep.
void WaitUntilQuiescent(const ResourcePrefetcher &prefetcher) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!prefetcher.Quiescent()) {
    Check(std::chrono::steady_clock::now() < deadline, "the workers never finished their jobs");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void TestChangedMemoryFallsBack() {
  auto memory = MakeMemory(4);
  g_memory = &memory;
  const PlanSet plans({.buffers = 4, .clean_every = 0}, 1);
  const auto user_data = UserData(memory, 3, 0);
  const auto descriptor_word = Tables * TableWords + 3 * TableWords + 5;

  // A packet between speculation and execution rewrote a descriptor dword.
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  prefetcher.Submit(1, &plans, plans.views, user_data, 0x1000);
  WaitUntilEvaluated();
  memory.words[descriptor_word] ^= 0x10u;
  prefetcher.BeginDraw(1);
  bool hit = true;
  CheckSame(Prepared(prefetcher, plans, memory, user_data, &hit),
            Serial(plans.plan, memory, user_data),
            "rewritten descriptor memory was not re-read");
  Check(!hit && prefetcher.GetStats().stale == 1, "a stale read was accepted");

  // Rewritten with the same bytes: still valid.
  prefetcher.Submit(2, &plans, plans.views, user_data, 0x1000);
  WaitUntilEvaluated();
  const auto word = memory.words[descriptor_word];
  memory.words[descriptor_word] = word;
  prefetcher.BeginDraw(2);
  CheckSame(Prepared(prefetcher, plans, memory, user_data, &hit),
            Serial(plans.plan, memory, user_data), "an unchanged result differs");
  Check(hit, "unchanged memory was not accepted");

  // The root pointer moved: the speculation followed the old one.
  prefetcher.Submit(3, &plans, plans.views, user_data, 0x1000);
  WaitUntilEvaluated();
  const auto moved = memory.Address((Tables + 9) * TableWords);
  memory.words[3 * TableWords] = static_cast<uint32_t>(moved);
  prefetcher.BeginDraw(3);
  CheckSame(Prepared(prefetcher, plans, memory, user_data, &hit),
            Serial(plans.plan, memory, user_data),
            "a moved descriptor table was not followed");
  Check(!hit, "a result read through a stale pointer was accepted");
}

void TestStrictReadOutcomeIsReplayed() {
  auto memory = MakeMemory(5);
  g_memory = &memory;
  const PlanSet plans({.buffers = 4, .clean_every = 1}, 1);
  const auto user_data = UserData(memory, 2, 0);
  const auto serial_clean = Serial(plans.plan, memory, user_data);
  Check(serial_clean.ok, "the strict-read plan did not materialize");

  // The GPU wrote the table after the worker read it: the strict reader now refuses it.
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  prefetcher.Submit(1, &plans, plans.views, user_data, 0x1000);
  WaitUntilEvaluated();
  memory.dirty.push_back({memory.Address(2 * TableWords), 8});
  prefetcher.BeginDraw(1);
  Result prepared;
  const bool used = prefetcher.TryMaterialize(
      &plans, SerialRuntime(memory, user_data), prepared.snapshot, prepared.specialization);
  Check(!used, "a strict read the GPU thread refuses was accepted");
  Check(!Serial(plans.plan, memory, user_data).ok,
        "the fixture's dirty range did not fail the serial evaluation");
  memory.dirty.clear();

  // Clean at execution: the replay accepts reads the strict reader serves.
  prefetcher.Submit(2, &plans, plans.views, user_data, 0x1000);
  WaitUntilEvaluated();
  prefetcher.BeginDraw(2);
  bool hit = false;
  CheckSame(Prepared(prefetcher, plans, memory, user_data, &hit), serial_clean,
            "a strict-read plan's result differs");
  Check(hit, "a clean strict-read result was not accepted");
}

void TestMispredictionFallsBack() {
  auto memory = MakeMemory(6);
  g_memory = &memory;
  const PlanSet plans({.buffers = 4}, 1);
  const PlanSet other({.buffers = 2}, 1);
  const auto guessed = UserData(memory, 1, 0);
  const auto actual = UserData(memory, 2, 0);
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  prefetcher.Submit(1, &plans, plans.views, guessed, 0x1000);
  prefetcher.Submit(1, &other, other.views, actual, 0x2000);
  WaitUntilEvaluated();
  prefetcher.BeginDraw(1);
  bool hit = true;
  CheckSame(Prepared(prefetcher, plans, memory, actual, &hit),
            Serial(plans.plan, memory, actual), "mispredicted user data was used");
  Check(!hit && prefetcher.GetStats().mismatched == 1,
        "mispredicted user data was not reported");
  CheckSame(Prepared(prefetcher, other, memory, actual, &hit),
            Serial(other.plan, memory, actual), "a mispredicted shader base was used");
  Check(!hit && prefetcher.GetStats().mismatched == 2,
        "a mispredicted shader base was not reported");
  // Earlier draws' speculation is dropped; later draws' is kept.
  prefetcher.Submit(2, &plans, plans.views, actual, 0x1000);
  prefetcher.Submit(3, &plans, plans.views, actual, 0x1000);
  WaitUntilEvaluated();
  prefetcher.BeginDraw(3);
  CheckSame(Prepared(prefetcher, plans, memory, actual, &hit),
            Serial(plans.plan, memory, actual), "a later draw's result differs");
  Check(hit, "the current draw's speculation was not found");
  CheckSame(Prepared(prefetcher, plans, memory, actual, &hit),
            Serial(plans.plan, memory, actual), "a second evaluation differs");
  Check(!hit, "one speculation was used twice");
}

void TestUnstartedJobIsCancelled() {
  auto memory = MakeMemory(7);
  g_memory = &memory;
  const PlanSet plans({.buffers = 4}, 1);
  const PlanSet other({.buffers = 3}, 1);
  const auto user_data = UserData(memory, 4, 0);
  // The only worker is stuck in the first job, so the second never starts.
  g_view_entered.store(false);
  g_block_views.store(true);
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  prefetcher.Submit(1, &other, other.views, user_data, 0x1000);
  while (!g_view_entered.load()) {
    std::this_thread::yield();
  }
  prefetcher.Submit(1, &plans, plans.views, user_data, 0x1000);
  prefetcher.BeginDraw(1);
  bool hit = true;
  CheckSame(Prepared(prefetcher, plans, memory, user_data, &hit),
            Serial(plans.plan, memory, user_data), "a cancelled job's result differs");
  Check(!hit && prefetcher.GetStats().cancelled == 1,
        "an unstarted job was waited for");
  g_block_views.store(false);
  CheckSame(Prepared(prefetcher, other, memory, user_data, &hit),
            Serial(other.plan, memory, user_data), "a running job's result differs");
  Check(hit, "a running job's result was not waited for");
  // The cancelled job's slot is reused while its stale queue entry is still pending.
  for (uint64_t draw = 2; draw < 200; draw++) {
    const auto data = UserData(memory, static_cast<uint32_t>(draw % Tables), 0);
    prefetcher.Submit(draw, &plans, plans.views, data, 0x1000);
    prefetcher.BeginDraw(draw);
    CheckSame(Prepared(prefetcher, plans, memory, data),
              Serial(plans.plan, memory, data), "a reused job slot's result differs");
  }
}

void TestUnreadablePageFallsBack() {
  auto memory = MakeMemory(8);
  g_memory = &memory;
  const PlanSet plans({.buffers = 4}, 1);
  const auto user_data = UserData(memory, 6, 0);
  g_unreadable_page.store(memory.Address(6 * TableWords) & ~uint64_t{4095});
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  prefetcher.Submit(1, &plans, plans.views, user_data, 0x1000);
  WaitUntilEvaluated();
  g_unreadable_page.store(0);
  prefetcher.BeginDraw(1);
  bool hit = true;
  CheckSame(Prepared(prefetcher, plans, memory, user_data, &hit),
            Serial(plans.plan, memory, user_data),
            "an unreadable page's result differs");
  Check(!hit && prefetcher.GetStats().failed == 1,
        "a worker read of an unreadable page was accepted");
}

void TestStalePlansAreThrottled() {
  auto memory = MakeMemory(9);
  g_memory = &memory;
  const PlanSet plans({.buffers = 2}, 1);
  const auto user_data = UserData(memory, 1, 0);
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  for (uint64_t draw = 1; draw < 64; draw++) {
    prefetcher.Submit(draw, &plans, plans.views, user_data, 0x1000);
    WaitUntilEvaluated();
    memory.words[Tables * TableWords + TableWords + 1]++;
    prefetcher.BeginDraw(draw);
    CheckSame(Prepared(prefetcher, plans, memory, user_data),
              Serial(plans.plan, memory, user_data), "a throttled plan's result differs");
    if (prefetcher.GetStats().throttled != 0) {
      break;
    }
  }
  Check(prefetcher.GetStats().throttled != 0,
        "a plan whose inputs always change kept being speculated");
}

// Batched submissions (BeginBatch) reach the workers only at Flush, at the batch cap or when a
// draw looks for one, and give the results unbatched ones do.
void TestPrefetchBatchFlush() {
  auto memory = MakeMemory(10);
  g_memory = &memory;
  const PlanSet plans({.buffers = 4}, 1);
  const auto user_data = UserData(memory, 5, 0);
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  const auto views = g_views.load();
  prefetcher.BeginBatch();
  prefetcher.Submit(1, &plans, plans.views, user_data, 0x1000);
  WaitUntilEvaluated();
  Check(g_views.load() == views && prefetcher.GetStats().flushes == 0,
        "a batched submission reached the workers before the flush");
  prefetcher.Flush();
  WaitUntilQuiescent(prefetcher);
  Check(g_views.load() != views, "a flushed submission was not evaluated");
  prefetcher.BeginDraw(1);
  bool hit = false;
  CheckSame(Prepared(prefetcher, plans, memory, user_data, &hit),
            Serial(plans.plan, memory, user_data), "a batched result differs");
  Check(hit, "a flushed evaluation was not used");
  Check(prefetcher.GetStats().flushes == 1 && prefetcher.GetStats().flushed_jobs == 1,
        "one flush of one job was not counted");

  // A long batch is handed over at the cap, before it ends.
  prefetcher.BeginBatch();
  for (uint64_t draw = 2; draw < 2 + 40; draw++) {
    prefetcher.Submit(draw, &plans, plans.views, user_data, 0x1000);
  }
  Check(prefetcher.GetStats().flushes == 2 && prefetcher.GetStats().flushed_jobs == 33,
        "a full batch was not handed over");
  prefetcher.Flush();
  Check(prefetcher.GetStats().flushed_jobs == 41, "the rest of a batch was not handed over");
  WaitUntilQuiescent(prefetcher);
  prefetcher.BeginDraw(41);
  CheckSame(Prepared(prefetcher, plans, memory, user_data, &hit),
            Serial(plans.plan, memory, user_data), "the last result of a long batch differs");
  Check(hit, "the last evaluation of a long batch was not used");

  // Many workers, one batch per draw, as the lookahead submits.
  constexpr uint32_t Workers = 3;
  const PlanSet vertex({.buffers = 6, .clean_every = 4}, Workers);
  const PlanSet pixel({.buffers = 10}, Workers);
  std::mt19937 random(11);
  constexpr uint32_t Draws = 2000;
  constexpr uint32_t Ahead = 8;
  std::vector<std::array<uint32_t, 4>> vertex_data(Draws + Ahead);
  std::vector<std::array<uint32_t, 4>> pixel_data(Draws + Ahead);
  for (uint32_t i = 0; i < Draws + Ahead; i++) {
    vertex_data[i] = UserData(memory, random() % Tables, random());
    pixel_data[i] = UserData(memory, random() % Tables, random());
  }
  ResourcePrefetcher batched(Workers, {.view_page = ViewPage});
  uint32_t submitted = 0;
  uint32_t hits = 0;
  for (uint32_t draw = 1; draw <= Draws; draw++) {
    batched.BeginBatch();
    for (; submitted < draw + Ahead; submitted++) {
      batched.Submit(submitted + 1, &pixel, pixel.views, pixel_data[submitted], 0x1000);
      batched.Submit(submitted + 1, &vertex, vertex.views, vertex_data[submitted], 0x1000);
    }
    batched.Flush();
    batched.BeginDraw(draw);
    CheckSame(Prepared(batched, pixel, memory, pixel_data[draw - 1], &hit),
              Serial(pixel.plan, memory, pixel_data[draw - 1]),
              "a batched pixel draw's resources differ from the serial ones");
    hits += hit ? 1u : 0u;
    CheckSame(Prepared(batched, vertex, memory, vertex_data[draw - 1], &hit),
              Serial(vertex.plan, memory, vertex_data[draw - 1]),
              "a batched vertex draw's resources differ from the serial ones");
    hits += hit ? 1u : 0u;
  }
  const auto &stats = batched.GetStats();
  Check(hits == stats.hits && hits > 0, "no batched draw used a worker result");
  Check(stats.stale == 0 && stats.failed == 0 && stats.mismatched == 0,
        "unchanged memory was reported stale in a batch");
  Check(stats.flushed_jobs == stats.submitted && stats.flushes == Draws,
        "batched submissions were lost or flushed one by one");
  std::printf("  batched: %u of %u evaluations from workers (%llu cancelled)\n", hits,
              Draws * 2, static_cast<unsigned long long>(stats.cancelled));
}

// A draw that looks for a speculation still waiting in a batch hands the batch over first: its
// own job has not started (the serial path evaluates it), later draws' jobs then run.
void TestPrefetchConsumeBeforeFlush() {
  auto memory = MakeMemory(12);
  g_memory = &memory;
  const PlanSet plans({.buffers = 4}, 1);
  const PlanSet other({.buffers = 3}, 1);
  const auto first = UserData(memory, 2, 0);
  const auto second = UserData(memory, 7, 1);
  // The only worker is stuck in an earlier job until the draw has looked.
  g_view_entered.store(false);
  g_block_views.store(true);
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  prefetcher.Submit(1, &other, other.views, first, 0x1000);
  while (!g_view_entered.load()) {
    std::this_thread::yield();
  }
  prefetcher.BeginBatch();
  prefetcher.Submit(1, &plans, plans.views, first, 0x1000);
  prefetcher.Submit(2, &plans, plans.views, second, 0x1000);
  prefetcher.BeginDraw(1);
  bool hit = true;
  CheckSame(Prepared(prefetcher, plans, memory, first, &hit),
            Serial(plans.plan, memory, first), "a draw found in a batch was prepared wrongly");
  Check(!hit && prefetcher.GetStats().cancelled == 1,
        "an unstarted batched job was used or waited for");
  Check(prefetcher.GetStats().flushes == 2 && prefetcher.GetStats().flushed_jobs == 3,
        "a draw looking for a batched job did not hand the batch over");
  g_block_views.store(false);
  WaitUntilQuiescent(prefetcher);
  prefetcher.BeginDraw(2);
  CheckSame(Prepared(prefetcher, plans, memory, second, &hit),
            Serial(plans.plan, memory, second), "the next batched draw's result differs");
  Check(hit, "the next batched draw's job did not run after the hand-over");
  prefetcher.Flush();
}

// Retiring batched jobs before the flush (BeginDraw, Clear): the flush skips them, also when their
// records already hold later submissions, and no record is lost.
void TestPrefetchRetirePending() {
  auto memory = MakeMemory(13);
  g_memory = &memory;
  const PlanSet plans({.buffers = 4}, 1);
  const auto data = [&](uint64_t draw) {
    return UserData(memory, static_cast<uint32_t>(draw % Tables), static_cast<uint32_t>(draw));
  };
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  prefetcher.BeginBatch();
  for (uint64_t draw = 1; draw <= 4; draw++) {
    prefetcher.Submit(draw, &plans, plans.views, data(draw), 0x1000);
  }
  // Draws 1 and 2 retire unqueued; draws 5 and 6 reuse their records.
  prefetcher.BeginDraw(3);
  prefetcher.Submit(5, &plans, plans.views, data(5), 0x1000);
  prefetcher.Submit(6, &plans, plans.views, data(6), 0x1000);
  Check(prefetcher.GetStats().jobs == 4, "retired batched records were not reused");
  prefetcher.Flush();
  Check(prefetcher.GetStats().flushed_jobs == 4,
        "a batched job retired before the flush was queued");
  WaitUntilQuiescent(prefetcher);
  for (uint64_t draw = 3; draw <= 6; draw++) {
    prefetcher.BeginDraw(draw);
    bool hit = false;
    const auto user_data = data(draw);
    CheckSame(Prepared(prefetcher, plans, memory, user_data, &hit),
              Serial(plans.plan, memory, user_data), "a reused batched record's result differs");
    Check(hit, "a batched job after retired ones was not used");
  }

  // Cleared while batched: nothing runs, every record returns.
  const auto views = g_views.load();
  const auto flushed = prefetcher.GetStats().flushed_jobs;
  prefetcher.BeginBatch();
  prefetcher.Submit(7, &plans, plans.views, data(7), 0x1000);
  prefetcher.Submit(8, &plans, plans.views, data(8), 0x1000);
  prefetcher.Clear();
  prefetcher.Flush();
  WaitUntilEvaluated();
  Check(prefetcher.GetStats().flushed_jobs == flushed && g_views.load() == views,
        "a batched job cleared before the flush ran");
  const auto jobs = prefetcher.GetStats().jobs;
  for (uint64_t draw = 9; draw < 300; draw++) {
    prefetcher.BeginBatch();
    prefetcher.Submit(draw, &plans, plans.views, data(draw), 0x1000);
    prefetcher.Submit(draw + 1, &plans, plans.views, data(draw + 1), 0x1000);
    if (draw % 3 == 0) {
      prefetcher.Clear();
    }
    prefetcher.BeginDraw(draw);
    const auto user_data = data(draw);
    CheckSame(Prepared(prefetcher, plans, memory, user_data),
              Serial(plans.plan, memory, user_data), "a retired batch's draw differs");
    prefetcher.Flush();
  }
  Check(prefetcher.GetStats().jobs <= jobs + 8, "batched records retired unqueued were lost");
}

// Compute plans also read the dispatch's workgroup counts: a result evaluated for other counts
// is not used.
void TestWorkgroupCountsMustMatch() {
  auto memory = MakeMemory(14);
  g_memory = &memory;
  const PlanSet plans({.buffers = 3}, 1);
  const auto user_data = UserData(memory, 4, 0);
  const std::array<uint32_t, 3> guessed{8, 1, 1};
  const std::array<uint32_t, 3> actual{4, 1, 1};
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  prefetcher.Submit(1, &plans, plans.views, user_data, 0x1000, guessed);
  WaitUntilEvaluated();
  prefetcher.BeginDraw(1);
  auto runtime = SerialRuntime(memory, user_data);
  runtime.workgroup_counts = actual;
  Result prepared;
  Check(!prefetcher.TryMaterialize(&plans, runtime, prepared.snapshot, prepared.specialization) &&
            prefetcher.GetStats().mismatched == 1,
        "a result evaluated for other workgroup counts was used");
  prefetcher.Submit(2, &plans, plans.views, user_data, 0x1000, actual);
  WaitUntilEvaluated();
  prefetcher.BeginDraw(2);
  Check(prefetcher.TryMaterialize(&plans, runtime, prepared.snapshot, prepared.specialization),
        "a result evaluated for the same workgroup counts was not used");
  prepared.ok = true;
  CheckSame(prepared, Serial(plans.plan, memory, user_data),
            "a workgroup-count result differs from the serial one");
}

// Before a plan's replicas are replaced, no worker may still evaluate them.
void TestClearAndWaitWaitsForRunningJobs() {
  auto memory = MakeMemory(15);
  g_memory = &memory;
  const PlanSet plans({.buffers = 2}, 1);
  const auto user_data = UserData(memory, 2, 0);
  g_view_entered.store(false);
  g_block_views.store(true);
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  prefetcher.Submit(1, &plans, plans.views, user_data, 0x1000);
  while (!g_view_entered.load()) {
    std::this_thread::yield();
  }
  std::atomic<bool> done{false};
  std::thread clear([&] {
    prefetcher.ClearAndWait();
    done.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  Check(!done.load(), "ClearAndWait returned while a worker evaluated a plan");
  g_block_views.store(false);
  clear.join();
  Check(done.load(), "ClearAndWait did not return");
}

// A worker read of bytes the GPU wrote (and the CPU has not read back) is stale in guest memory:
// the result is rejected, so the serial path can leave data-only ones to a GPU copy.
std::atomic<uint64_t> g_gpu_written_page{0};

bool GpuWrittenPage(uint64_t address, uint64_t size) {
  const auto page = g_gpu_written_page.load();
  return page != 0 && address < page + 4096 && page < address + size;
}

void TestGpuWrittenReadsAreRejected() {
  auto memory = MakeMemory(16);
  g_memory = &memory;
  const PlanSet plans({.buffers = 2}, 1);
  const auto user_data = UserData(memory, 3, 0);
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  prefetcher.Submit(1, &plans, plans.views, user_data, 0x1000);
  WaitUntilQuiescent(prefetcher);
  prefetcher.BeginDraw(1);
  auto runtime = SerialRuntime(memory, user_data);
  runtime.gpu_written = GpuWrittenPage;
  g_gpu_written_page.store(memory.Address(3 * TableWords) & ~uint64_t{4095});
  Result prepared;
  Check(!prefetcher.TryMaterialize(&plans, runtime, prepared.snapshot, prepared.specialization) &&
            prefetcher.GetStats().stale == 1,
        "a result that read GPU-written bytes was used");
  g_gpu_written_page.store(0);
  prefetcher.Submit(2, &plans, plans.views, user_data, 0x1000);
  WaitUntilQuiescent(prefetcher);
  prefetcher.BeginDraw(2);
  Check(prefetcher.TryMaterialize(&plans, runtime, prepared.snapshot, prepared.specialization),
        "a result without GPU-written reads was not used");
}

void TestCanSpeculate() {
  auto plan = TablePlan({.buffers = 2});
  Check(ResourcePrefetcher::CanSpeculate(plan), "a plain table plan was not speculated");
  Check(ResourcePrefetcher::CanSpeculate(TablePlan({.buffers = 2, .capture = true})),
        "a capturing plan was not speculated");
  plan.source_reads.push_back({0x1000, 4});
  Check(!ResourcePrefetcher::CanSpeculate(plan), "a plan with call-source reads was speculated");
  plan.source_reads.clear();
  plan.srt_plan_complete = false;
  Check(!ResourcePrefetcher::CanSpeculate(plan), "an incomplete SRT plan was speculated");
}

// Page generations. The fixture's GPU thread settles a page unless a "GPU-written" range is on
// it, as RenderContext::IsGpuSettled does.
std::atomic<uint32_t> g_settled_checks{0};

bool SettledPage(uint64_t page) {
  g_settled_checks.fetch_add(1);
  for (const auto &[begin, size] : g_memory->dirty) {
    if (begin < page + SrtPageWatch::PageSize && page < begin + size) {
      return false;
    }
  }
  return true;
}

ResourcePrefetcher::MemoryOps WatchedOps(SrtPageWatch &watch) {
  return {.view_page = ViewPage, .watch = &watch, .page_settled = SettledPage,
          .replay_check = true};
}

void TestPageWatchRanges() {
  auto watch = std::make_unique<SrtPageWatch>();
  constexpr uint64_t Page = SrtPageWatch::PageSize;
  // Pages on both sides of a word and of a leaf (2^20 pages) boundary.
  const uint64_t leaf = (uint64_t{1} << 20u) * Page;
  const std::array<uint64_t, 6> pages = {63 * Page,     64 * Page,     leaf - Page,
                                         leaf,          leaf + Page,   (uint64_t{1} << 48u) - Page};
  const auto generation = watch->Generation();
  for (const auto page : pages) {
    Check(!watch->IsSettled(page), "a page was settled before anything settled it");
    const auto count = watch->SettleCount();
    watch->Settle(page);
    Check(watch->IsSettled(page) && watch->SettleCount() == count + 1 &&
              watch->IsSettledBy(page, count + 1) && !watch->IsSettledBy(page, count),
          "a settled page did not take the next settle count");
  }
  watch->Settle(pages[0]);
  Check(watch->SettleCount() == pages.size() && watch->IsSettledBy(pages[0], 1),
        "settling a settled page counted again");
  Check(watch->Generation() == generation, "settling a page moved the generation");
  // A byte at the end of one page unsettles that page only.
  watch->Unsettle(64 * Page - 1, 1);
  Check(!watch->IsSettled(63 * Page) && watch->IsSettled(64 * Page),
        "a one-byte range unsettled the wrong pages");
  // A range across a leaf boundary.
  watch->Unsettle(leaf - 8, 16);
  Check(!watch->IsSettled(leaf - Page) && !watch->IsSettled(leaf) && watch->IsSettled(leaf + Page),
        "a range across leaves unsettled the wrong pages");
  // Ranges reaching past the address space, and pages beyond it.
  watch->Unsettle((uint64_t{1} << 48u) - 4, UINT64_MAX / 2);
  Check(!watch->IsSettled(pages[5]), "a range past the address space kept its first page");
  watch->Settle(uint64_t{1} << 48u);
  Check(!watch->IsSettled(uint64_t{1} << 48u), "a page past the address space settled");
  watch->Unsettle(uint64_t{1} << 47u, uint64_t{1} << 40u);
  Check(watch->IsSettled(leaf + Page), "an unsettle outside its range reached a page");
  watch->Unsettle(0, uint64_t{1} << 48u);
  Check(!watch->IsSettled(leaf + Page), "an unsettle of everything kept a page");
}

// One draw through the fixture's pipeline: speculate, let the worker finish, run the
// `between` packets, execute. Returns whether a worker result was used; a used one must be the
// serial result.
template <typename Between>
bool Draw(ResourcePrefetcher &prefetcher, const PlanSet &plans, Memory &memory,
          std::span<const uint32_t> user_data, uint64_t draw, Between &&between) {
  prefetcher.Submit(draw, &plans, plans.views, user_data, 0x1000);
  WaitUntilEvaluated();
  between();
  prefetcher.BeginDraw(draw);
  const auto serial = Serial(plans.plan, memory, user_data);
  Result prepared;
  const bool used = prefetcher.TryMaterialize(&plans, SerialRuntime(memory, user_data),
                                              prepared.snapshot, prepared.specialization);
  if (used) {
    prepared.ok = true;
    CheckSame(prepared, serial, "an accepted result differs from the serial one");
  }
  return used;
}

void TestSettledPagesSkipTheReplay() {
  auto memory = MakeMemory(20);
  g_memory = &memory;
  auto watch = std::make_unique<SrtPageWatch>();
  const PlanSet plans({.buffers = 4, .clean_every = 2}, 1);
  const auto user_data = UserData(memory, 3, 0);
  ResourcePrefetcher prefetcher(1, WatchedOps(*watch));
  const auto &stats = prefetcher.GetStats();
  const auto descriptor = Tables * TableWords + 3 * TableWords + 5;
  uint64_t draw = 0;
  const auto nothing = [] {};

  // The first result is replayed, which settles its pages; the next one is not.
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) && stats.unreplayed == 0,
        "a result on unsettled pages was not replayed");
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) && stats.unreplayed == 1,
        "a result on settled pages was replayed");

  // A command-processor store between speculation and execution unsettles its page.
  Check(!Draw(prefetcher, plans, memory, user_data, ++draw,
              [&] {
                memory.words[descriptor] ^= 0x40u;
                watch->Unsettle(memory.Address(descriptor), 4);
              }) &&
            stats.unreplayed == 1 && stats.stale == 1,
        "a store to a settled page did not replay its reader");
  // The failed replay settled nothing; the next one does.
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) && stats.unreplayed == 1,
        "an unsettled page was trusted");
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) && stats.unreplayed == 2,
        "a page did not settle again");

  // A CPU write published by an epoch acquire: the generation moved, so the replay runs.
  Check(!Draw(prefetcher, plans, memory, user_data, ++draw,
              [&] {
                memory.words[descriptor] ^= 0x80u;
                watch->Advance();
              }) &&
            stats.unreplayed == 2,
        "a result from before an acquire was not replayed");

  // A write binding next to a root table: the replay still accepts the result, but the page
  // cannot settle, and is not checked again until the generation moves.
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing),
        "an unchanged result was not accepted");
  const auto unreplayed = stats.unreplayed;
  const auto root = memory.Address(3 * TableWords);
  const auto root_page = root & ~(SrtPageWatch::PageSize - 1u);
  Check(Draw(prefetcher, plans, memory, user_data, ++draw,
             [&] {
               memory.dirty.push_back({root + 8, 4});
               watch->Unsettle(root + 8, 4);
             }) &&
            stats.unreplayed == unreplayed && !watch->IsSettled(root_page),
        "a page under GPU-written bytes settled");
  const auto checks = g_settled_checks.load();
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) &&
            stats.unreplayed == unreplayed && g_settled_checks.load() == checks,
        "an unsettled page was checked again in the same generation");
  // GPU-written bytes the reads cover: the strict reader refuses them.
  Check(!Draw(prefetcher, plans, memory, user_data, ++draw,
              [&] {
                memory.dirty.push_back({root, 8});
                watch->Unsettle(root, 8);
              }),
        "a strict read the GPU thread refuses was accepted");
  // Read back: the page settles after the next replay.
  memory.dirty.clear();
  watch->Advance();
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) &&
            stats.unreplayed == unreplayed && watch->IsSettled(root_page),
        "a page read back did not settle");
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) &&
            stats.unreplayed == unreplayed + 1,
        "a page read back was not trusted once settled");
  Check(stats.replay_mismatches == 0, "an accepted result differed from its replay");
}

// A store that does not unsettle its page (a missing report) is a result the replay rejects:
// MemoryOps::replay_check replays every result accepted without one and counts it.
void TestReplayCheckCatchesAMissedUnsettle() {
  auto memory = MakeMemory(21);
  g_memory = &memory;
  auto watch = std::make_unique<SrtPageWatch>();
  const PlanSet plans({.buffers = 4, .clean_every = 0}, 1);
  const auto user_data = UserData(memory, 5, 0);
  ResourcePrefetcher prefetcher(1, WatchedOps(*watch));
  const auto &stats = prefetcher.GetStats();
  (void)Draw(prefetcher, plans, memory, user_data, 1, [] {});
  (void)Draw(prefetcher, plans, memory, user_data, 2, [] {});
  Check(stats.unreplayed == 1 && stats.replay_mismatches == 0,
        "the fixture did not settle its pages");
  prefetcher.Submit(3, &plans, plans.views, user_data, 0x1000);
  WaitUntilEvaluated();
  memory.words[Tables * TableWords + 5 * TableWords + 2] ^= 0x100u;
  prefetcher.BeginDraw(3);
  Result prepared;
  const bool used = prefetcher.TryMaterialize(&plans, SerialRuntime(memory, user_data),
                                              prepared.snapshot, prepared.specialization);
  Check(used && stats.unreplayed == 2, "a result on settled pages was replayed");
  Check(stats.replay_mismatches == 1, "the replay check missed a stale accepted result");
}

// A page settles with the next settle count: a result is replayed when one of its own pages
// settled, or settled again, after its submit, and not when another page did.
void TestSettlesAreCountedPerPage() {
  auto memory = MakeMemory(24);
  g_memory = &memory;
  auto watch = std::make_unique<SrtPageWatch>();
  const PlanSet plans({.buffers = 4, .clean_every = 2}, 1);
  const auto user_data = UserData(memory, 2, 0);
  ResourcePrefetcher prefetcher(1, WatchedOps(*watch));
  const auto &stats = prefetcher.GetStats();
  const auto nothing = [] {};
  uint64_t draw = 0;
  (void)Draw(prefetcher, plans, memory, user_data, ++draw, nothing);
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) && stats.unreplayed == 1,
        "the fixture did not settle its pages");
  // Another page settles between submit and execution.
  const auto elsewhere = memory.Address(0) + 64 * SrtPageWatch::PageSize;
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, [&] { watch->Settle(elsewhere); }) &&
            stats.unreplayed == 2,
        "a page the result did not read sent it back to the replay");
  // One of its pages is unsettled and settles again: the bytes it read may be older.
  const auto root_page = memory.Address(2 * TableWords) & ~(SrtPageWatch::PageSize - 1u);
  Check(Draw(prefetcher, plans, memory, user_data, ++draw,
             [&] {
               watch->Unsettle(root_page, 4);
               watch->Settle(root_page);
             }) &&
            stats.unreplayed == 2,
        "a page that settled again after the submit was trusted");
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) && stats.unreplayed == 3,
        "a page that settled before the submit was not trusted");
  Check(stats.replay_mismatches == 0, "an accepted result differed from its replay");
}

// A capturing plan (its reads checked against the shader's buffer writes) is speculated, and
// a used result is the serial one, its captured read ranges included.
void TestCapturingPlanMatchesSerial() {
  auto memory = MakeMemory(25);
  g_memory = &memory;
  const PlanSet plans({.buffers = 4, .clean_every = 3, .capture = true}, 1);
  const auto user_data = UserData(memory, 6, 0);
  const auto serial = Serial(plans.plan, memory, user_data);
  // The two root-table words, each descriptor dword, and the uniform-buffer dword.
  Check(serial.ok && serial.snapshot.specialization_reads.size() == 2u + 4u * 4u + 1u &&
            serial.snapshot.specialization_reads.back() ==
                std::pair<uint64_t, uint64_t>{memory.Address((Tables + 6) * TableWords), 4u},
        "the fixture's capturing plan did not capture its reads");
  ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
  for (uint64_t draw = 1; draw <= 4; draw++) {
    prefetcher.Submit(draw, &plans, plans.views, user_data, 0x1000);
    WaitUntilQuiescent(prefetcher);
    prefetcher.BeginDraw(draw);
    bool hit = false;
    CheckSame(Prepared(prefetcher, plans, memory, user_data, &hit), serial,
              "a capturing plan's prepared resources differ from the serial ones");
    Check(hit, "a capturing plan's result was not used");
  }
}

// A captured clean read whose bytes the GPU wrote between speculation and execution: the
// draw's strict reader refuses them, so the result is not used, with or without settled pages.
void TestCapturedCleanRangeTurnsDirty() {
  auto memory = MakeMemory(26);
  g_memory = &memory;
  const PlanSet plans({.buffers = 4, .capture = true}, 1);
  const auto user_data = UserData(memory, 4, 0);
  // The uniform-buffer dword, read only through the clean walker.
  const auto records = memory.Address((Tables + 4) * TableWords);
  const auto root = memory.Address(4 * TableWords);
  for (const auto range : {std::pair<uint64_t, uint64_t>{records, 4}, {root, 8}}) {
    ResourcePrefetcher prefetcher(1, {.view_page = ViewPage});
    prefetcher.Submit(1, &plans, plans.views, user_data, 0x1000);
    WaitUntilQuiescent(prefetcher);
    memory.dirty = {range};
    prefetcher.BeginDraw(1);
    bool hit = true;
    const auto prepared = Prepared(prefetcher, plans, memory, user_data, &hit);
    Check(!hit && prefetcher.GetStats().stale == 1,
          "a capturing plan's result was used over GPU-written clean bytes");
    Check(!prepared.ok, "the fixture's dirty clean range did not fail the serial evaluation");
    memory.dirty.clear();
  }

  auto watch = std::make_unique<SrtPageWatch>();
  ResourcePrefetcher prefetcher(1, WatchedOps(*watch));
  const auto &stats = prefetcher.GetStats();
  const auto nothing = [] {};
  uint64_t draw = 0;
  (void)Draw(prefetcher, plans, memory, user_data, ++draw, nothing);
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) && stats.unreplayed == 1,
        "a capturing plan's settled pages were not trusted");
  // A write binding over the uniform-buffer dword unsettles its page.
  Check(!Draw(prefetcher, plans, memory, user_data, ++draw,
              [&] {
                memory.dirty = {{records, 4}};
                watch->Unsettle(records, 4);
              }) &&
            stats.unreplayed == 1,
        "a capturing plan's result was used after its clean range was written");
  memory.dirty.clear();
  watch->Advance();
  (void)Draw(prefetcher, plans, memory, user_data, ++draw, nothing);
  Check(Draw(prefetcher, plans, memory, user_data, ++draw, nothing) && stats.unreplayed == 2,
        "a capturing plan's pages did not settle again");
  Check(stats.replay_mismatches == 0, "an accepted capturing result differed from its replay");
}

// As TestRandomMutationsStayExact, with every mutation reported the way the caches report it.
void TestRandomReportedMutationsStayExact() {
  auto memory = MakeMemory(22);
  g_memory = &memory;
  constexpr uint32_t Workers = 3;
  auto watch = std::make_unique<SrtPageWatch>();
  const PlanSet plans({.buffers = 5, .clean_every = 3}, Workers);
  std::mt19937 random(23);
  std::vector<std::array<uint32_t, 4>> data;
  constexpr uint32_t Draws = 3000;
  for (uint32_t i = 0; i < Draws + 8; i++) {
    data.push_back(UserData(memory, random() % 8, random() % 4));
  }
  uint32_t submitted = 0;
  ResourcePrefetcher prefetcher(Workers, WatchedOps(*watch));
  for (uint32_t draw = 1; draw <= Draws; draw++) {
    for (; submitted < draw + 6; submitted++) {
      prefetcher.Submit(submitted + 1, &plans, plans.views, data[submitted], 0x1000);
    }
    if (random() % 5 == 0) {
      const auto word = (Tables + random() % 8) * TableWords + random() % 20;
      auto &target = reinterpret_cast<std::atomic<uint32_t> &>(memory.words[word]);
      target.store(random() % 2 == 0 ? target.load() : static_cast<uint32_t>(random()),
                   std::memory_order_relaxed);
      watch->Unsettle(memory.Address(word), 4);
    }
    if (random() % 11 == 0) {
      const auto root = memory.Address((random() % 8) * TableWords);
      memory.dirty = {{root, 4}};
      watch->Unsettle(root, 4);
    } else if (random() % 5 == 0) {
      memory.dirty.clear();
    }
    if (random() % 50 == 0) {
      watch->Advance();
    }
    prefetcher.BeginDraw(draw);
    const auto serial = Serial(plans.plan, memory, data[draw - 1]);
    Result prepared;
    const bool used = prefetcher.TryMaterialize(&plans, SerialRuntime(memory, data[draw - 1]),
                                                prepared.snapshot, prepared.specialization);
    if (used) {
      prepared.ok = true;
      CheckSame(prepared, serial, "an accepted result differs from the serial one");
    }
  }
  const auto &stats = prefetcher.GetStats();
  std::printf("  reported mutations: %llu hits (%llu unreplayed), %llu stale\n",
              static_cast<unsigned long long>(stats.hits),
              static_cast<unsigned long long>(stats.unreplayed),
              static_cast<unsigned long long>(stats.stale));
  Check(stats.unreplayed > 0 && stats.hits > stats.unreplayed && stats.stale > 0,
        "the reported-mutation fixture exercised one outcome only");
  Check(stats.replay_mismatches == 0, "an accepted result differed from its replay");
}

// Many draws against memory other threads never touch while the GPU thread mutates it between
// speculation and execution: every result must still be the serial one.
void TestRandomMutationsStayExact() {
  auto memory = MakeMemory(10);
  g_memory = &memory;
  constexpr uint32_t Workers = 3;
  const PlanSet plans({.buffers = 5, .clean_every = 3}, Workers);
  std::mt19937 random(11);
  std::vector<std::array<uint32_t, 4>> data;
  constexpr uint32_t Draws = 2000;
  for (uint32_t i = 0; i < Draws + 8; i++) {
    data.push_back(UserData(memory, random() % 8, random() % 4));
  }
  uint32_t submitted = 0;
  ResourcePrefetcher prefetcher(Workers, {.view_page = ViewPage});
  for (uint32_t draw = 1; draw <= Draws; draw++) {
    for (; submitted < draw + 6; submitted++) {
      prefetcher.Submit(submitted + 1, &plans, plans.views, data[submitted], 0x1000);
    }
    // The workers may be reading these words right now; that is the race the replay covers.
    // Only this thread writes them, and only between whole-word stores, as guest memory is.
    if (random() % 3 == 0) {
      const auto table = random() % 8;
      auto &target = reinterpret_cast<std::atomic<uint32_t> &>(
          memory.words[(Tables + table) * TableWords + random() % 20]);
      target.store(random() % 2 == 0 ? target.load() : static_cast<uint32_t>(random()),
                   std::memory_order_relaxed);
    }
    if (random() % 7 == 0) {
      memory.dirty = {{memory.Address((random() % 8) * TableWords), 4}};
    } else if (random() % 5 == 0) {
      memory.dirty.clear();
    }
    prefetcher.BeginDraw(draw);
    const auto serial = Serial(plans.plan, memory, data[draw - 1]);
    Result prepared;
    const auto runtime = SerialRuntime(memory, data[draw - 1]);
    const bool used = prefetcher.TryMaterialize(&plans, runtime, prepared.snapshot,
                                                prepared.specialization);
    if (used) {
      prepared.ok = true;
      CheckSame(prepared, serial, "an accepted result differs from the serial one");
    }
  }
  const auto &stats = prefetcher.GetStats();
  std::printf("  mutations: %llu hits, %llu stale, %llu cancelled, %llu throttled\n",
              static_cast<unsigned long long>(stats.hits),
              static_cast<unsigned long long>(stats.stale),
              static_cast<unsigned long long>(stats.cancelled),
              static_cast<unsigned long long>(stats.throttled));
  Check(stats.hits > 0 && stats.stale > 0, "the mutation fixture exercised one outcome only");
}

// A GPU thread that materializes two stages per draw and then spends `record_us` recording
// commands, as RenderExecutor does, against the same loop with the evaluations on workers.
void Bench() {
  auto memory = MakeMemory(12);
  g_memory = &memory;
  const TablePlanShape shape{.buffers = 24, .clean_every = 5, .arithmetic = 2};
  constexpr uint32_t Draws = 20000;
  constexpr uint32_t Ahead = 8;
  std::mt19937 random(13);
  std::vector<std::array<uint32_t, 4>> data;
  for (uint32_t i = 0; i < Draws + Ahead; i++) {
    data.push_back(UserData(memory, random() % Tables, random()));
  }
  const auto spin = [](double us) {
    const auto end = std::chrono::steady_clock::now() +
                     std::chrono::duration<double, std::micro>(us);
    while (std::chrono::steady_clock::now() < end) {
    }
  };
  using Clock = std::chrono::steady_clock;
  const auto micros = [](Clock::duration duration) {
    return std::chrono::duration<double, std::micro>(duration).count();
  };

  PlanSet serial_plans(shape, 0);
  Result out;
  auto begin = Clock::now();
  for (uint32_t draw = 0; draw < 2000; draw++) {
    (void)MaterializeResources(serial_plans.plan, SerialRuntime(memory, data[draw]),
                               out.snapshot, out.specialization);
  }
  const auto evaluation_us = micros(Clock::now() - begin) / 2000.0;
  std::printf("bench: one evaluation %.2f us (%zu SRT reads)\n", evaluation_us,
              serial_plans.plan.srt_reads.size());

  for (const double record_us : {0.0, 10.0, 30.0}) {
    begin = Clock::now();
    for (uint32_t draw = 0; draw < Draws; draw++) {
      for (int stage = 0; stage < 2; stage++) {
        (void)MaterializeResources(serial_plans.plan, SerialRuntime(memory, data[draw]),
                                   out.snapshot, out.specialization);
      }
      spin(record_us);
    }
    const auto serial_us = micros(Clock::now() - begin) / Draws;
    for (const uint32_t workers : {1u, 2u, 3u}) {
      PlanSet stage_plans[2] = {PlanSet(shape, workers), PlanSet(shape, workers)};
      uint32_t submitted = 0;
      Clock::duration prepare{};
      ResourcePrefetcher prefetcher(workers, {.view_page = ViewPage});
      begin = Clock::now();
      for (uint32_t draw = 0; draw < Draws; draw++) {
        for (; submitted < std::min(draw + Ahead, Draws); submitted++) {
          for (auto &plans : stage_plans) {
            prefetcher.Submit(submitted + 1, &plans, plans.views, data[submitted], 0x1000);
          }
        }
        prefetcher.BeginDraw(draw + 1);
        const auto prepare_begin = Clock::now();
        for (auto &plans : stage_plans) {
          (void)Prepared(prefetcher, plans, memory, data[draw]);
        }
        prepare += Clock::now() - prepare_begin;
        spin(record_us);
      }
      const auto prefetch_us = micros(Clock::now() - begin) / Draws;
      const auto &stats = prefetcher.GetStats();
      std::printf("bench: record %4.1f us/draw, %u workers: serial %.2f us/draw, "
                  "prefetched %.2f us/draw (%.2fx), %.0f%% from workers, "
                  "%.2f us/evaluation on the GPU thread\n",
                  record_us, workers, serial_us, prefetch_us, serial_us / prefetch_us,
                  100.0 * static_cast<double>(stats.hits) / (2.0 * Draws),
                  micros(prepare) / (2.0 * Draws));
    }
  }
}

} // namespace

namespace Common {

int DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view) {
  std::abort();
}

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

void DbgExit(int) { std::abort(); }

} // namespace Common

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
    Bench();
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "--bench-ordinary") == 0) {
    g_serial_ordinary_reader = true;
    Bench();
    return 0;
  }
  TestReplicaEvaluatesLikeThePlan();
  TestOutputsDoNotDependOnPriorContents();
  TestSpeculationMatchesSerial();
  TestChangedMemoryFallsBack();
  TestStrictReadOutcomeIsReplayed();
  TestMispredictionFallsBack();
  TestUnstartedJobIsCancelled();
  TestUnreadablePageFallsBack();
  TestStalePlansAreThrottled();
  TestPrefetchBatchFlush();
  TestPrefetchConsumeBeforeFlush();
  TestPrefetchRetirePending();
  TestRandomMutationsStayExact();
  TestWorkgroupCountsMustMatch();
  TestCanSpeculate();
  TestGpuWrittenReadsAreRejected();
  TestClearAndWaitWaitsForRunningJobs();
  TestPageWatchRanges();
  TestSettledPagesSkipTheReplay();
  TestReplayCheckCatchesAMissedUnsettle();
  TestSettlesAreCountedPerPage();
  TestRandomReportedMutationsStayExact();
  TestCapturingPlanMatchesSerial();
  TestCapturedCleanRangeTurnsDirty();
  // Again with every plan capturing its reads.
  g_capture_plans = true;
  TestSpeculationMatchesSerial();
  TestChangedMemoryFallsBack();
  TestStrictReadOutcomeIsReplayed();
  TestRandomMutationsStayExact();
  TestSettledPagesSkipTheReplay();
  TestRandomReportedMutationsStayExact();
  g_capture_plans = false;
  // Again with an ordinary reader, as the pipeline cache's runtime has one.
  g_serial_ordinary_reader = true;
  TestSpeculationMatchesSerial();
  TestChangedMemoryFallsBack();
  TestStrictReadOutcomeIsReplayed();
  TestRandomMutationsStayExact();
  TestSettledPagesSkipTheReplay();
  TestRandomReportedMutationsStayExact();
  g_serial_ordinary_reader = false;
  std::puts("ResourcePrefetchTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
