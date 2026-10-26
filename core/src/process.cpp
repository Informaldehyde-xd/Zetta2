#include "zb/process.h"

#include <android/log.h>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <unwind.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>

#include <cerrno>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iterator>
#include <string_view>
#include <unordered_map>
#include <thread>

#include <dynarmic/interface/exclusive_monitor.h>

#include "zb/elf_loader.h"
#include "zb/hang_watchdog.h"
#include "zb/initial_stack.h"
#include "zb/log.h"
#include "zb/host_egl.h"
#include "zb/host_jni.h"
#include "zb/runtime_report.h"
#include "zb/syscalls.h"

namespace zb {

namespace {

// arch/arm/include/uapi/asm/hwcap.h
constexpr std::uint32_t kHwcapHalf = 1u << 1;
constexpr std::uint32_t kHwcapThumb = 1u << 2;
constexpr std::uint32_t kHwcapFastMult = 1u << 4;
constexpr std::uint32_t kHwcapVfp = 1u << 6;
constexpr std::uint32_t kHwcapEdsp = 1u << 7;
constexpr std::uint32_t kHwcapNeon = 1u << 12;
constexpr std::uint32_t kHwcapVfpv3 = 1u << 13;
constexpr std::uint32_t kHwcapTls = 1u << 15;
constexpr std::uint32_t kHwcapVfpv4 = 1u << 16;
constexpr std::uint32_t kHwcapIdiva = 1u << 17;
constexpr std::uint32_t kHwcapIdivt = 1u << 18;
constexpr std::uint32_t kHwcapVfpd32 = 1u << 19;
constexpr std::uint32_t kHwcapLpae = 1u << 20;
constexpr std::uint32_t kGuestHwcap = kHwcapHalf | kHwcapThumb | kHwcapFastMult | kHwcapVfp | kHwcapEdsp | kHwcapNeon |
                                      kHwcapVfpv3 | kHwcapTls | kHwcapVfpv4 | kHwcapIdiva | kHwcapIdivt |
                                      kHwcapVfpd32 | kHwcapLpae;

constexpr std::uint32_t kCpsrUserMode = 0x10;
constexpr std::uint32_t kCpsrThumb = 0x20;

// Linux clone flags (identical on arm and arm64).
constexpr std::uint32_t kCloneSetTls = 0x00080000;
constexpr std::uint32_t kCloneParentSetTid = 0x00100000;
constexpr std::uint32_t kCloneChildClearTid = 0x00200000;
constexpr std::uint32_t kCloneChildSetTid = 0x01000000;

constexpr std::uint64_t kSeenUnexpectedSvc = 1ULL << 40;
constexpr std::uint64_t kSeenHostCall = 1ULL << 41;

// svc #(0x5A0000 | index) from the generated stub libraries (tools/gen_stubs.py).
constexpr std::uint32_t kHostCallBase = 0x5A0000;

struct HostCallName {
    std::uint32_t index;
    const char* library;
    const char* name;
};

constexpr HostCallName kHostCallNames[] = {
#include "gen/hostcalls.inc"
};

std::pair<const char*, const char*> host_call_name(std::uint32_t index) {
    for (const auto& host_call : kHostCallNames) {
        if (host_call.index == index) return {host_call.library, host_call.name};
    }
    return {"?", "?"};
}

}  // namespace

// ---- host call profiler -------------------------------------------------------------------------
// Where does the time of a translated game go: in the translated code itself, or in the host calls
// it makes (OpenGL, EGL, JNI, ...)? Every host call is timed here, and the busiest ones are written
// to zb-profile.txt every few seconds (the file is replaced each time, so it always holds the
// latest busy window). Waiting counts too: a call that blocks (eglSwapBuffers waiting for the
// display) shows its whole wait, which is not CPU time. A host call that is cheap and very
// frequent can still cost a lot, because every one of them leaves translated code and comes back.
namespace {

// Same directory as zb-runtime-report.txt. Change the package name if you rename it.
constexpr const char* kProfileFile = "/storage/emulated/0/Android/data/com.zettabridge.launcher/files/zb-profile.txt";
constexpr const char* kFreezeFile = "/storage/emulated/0/Android/data/com.zettabridge.launcher/files/zb-freeze.txt";
constexpr const char* kPcProfileFile = "/storage/emulated/0/Android/data/com.zettabridge.launcher/files/zb-pcprofile.txt";
// The latest busy window is replaced as the game goes on (a pause or a menu would otherwise
// overwrite the interesting one), so the busiest window seen so far is kept in a file of its own.
constexpr const char* kProfileBestFile = "/storage/emulated/0/Android/data/com.zettabridge.launcher/files/zb-profile-busiest.txt";
constexpr const char* kPcProfileBestFile = "/storage/emulated/0/Android/data/com.zettabridge.launcher/files/zb-pcprofile-busiest.txt";
constexpr std::size_t kProfileSlots = 0x10000;  // a host call index is 16 bits
constexpr std::chrono::seconds kProfileWindow{5};
constexpr std::size_t kProfileRows = 18;

std::atomic<std::uint64_t> g_host_call_count[kProfileSlots];
std::atomic<std::uint64_t> g_host_call_ticks[kProfileSlots];

inline std::uint64_t read_ticks() {
#if defined(__aarch64__)
    std::uint64_t value;
    asm volatile("mrs %0, cntvct_el0" : "=r"(value));
    return value;
#else
    return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

inline std::uint64_t tick_frequency() {
#if defined(__aarch64__)
    std::uint64_t value;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(value));
    return value != 0 ? value : 19200000;
#else
    return 1000000000ull;  // steady_clock counts nanoseconds where this is tested
#endif
}

inline void write_text_file(const char* path, const std::string& text) {
    const int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd >= 0) {
        [[maybe_unused]] const ssize_t written = ::write(fd, text.data(), text.size());
        ::close(fd);
    }
}

inline void account_host_call(std::uint32_t index, std::uint64_t started) {
    const std::uint64_t elapsed = read_ticks() - started;
    g_host_call_count[index & 0xFFFF].fetch_add(1, std::memory_order_relaxed);
    g_host_call_ticks[index & 0xFFFF].fetch_add(elapsed, std::memory_order_relaxed);
}

}  // namespace

namespace {

struct PathMapping {
    std::string_view guest_prefix;
    std::string_view sysroot_prefix;
};

// The runtime APEX paths collapse onto the bootstrap copies extracted into system/.
constexpr PathMapping kPathMappings[] = {
    {"/apex/com.android.runtime/lib/bionic/", "/system/lib/"},
    {"/apex/com.android.runtime/bin/", "/system/bin/"},
    {"/system/", "/system/"},
    {"/vendor/", "/vendor/"},
    {"/odm/", "/odm/"},
    {"/product/", "/product/"},
    {"/system_ext/", "/system_ext/"},
    {"/apex/", "/apex/"},
    {"/linkerconfig/", "/linkerconfig/"},
};

const char* exception_name(Dynarmic::A32::Exception e) {
    using E = Dynarmic::A32::Exception;
    switch (e) {
    case E::UndefinedInstruction: return "undefined instruction";
    case E::UnpredictableInstruction: return "unpredictable instruction";
    case E::DecodeError: return "decode error";
    case E::SendEvent: return "SEV";
    case E::SendEventLocal: return "SEVL";
    case E::WaitForInterrupt: return "WFI";
    case E::WaitForEvent: return "WFE";
    case E::Yield: return "YIELD";
    case E::Breakpoint: return "breakpoint";
    case E::PreloadData: return "PLD";
    case E::PreloadDataWithIntentToWrite: return "PLDW";
    case E::PreloadInstruction: return "PLI";
    case E::NoExecuteFault: return "jump to non-executable memory";
    }
    return "unknown exception";
}

void write_guest_u32(GuestMemory& mem, std::uint32_t addr, std::uint32_t value) {
    if (std::uint8_t* p = mem.host_ptr(addr, 4, kPageWrite)) std::memcpy(p, &value, 4);
}

// Linux ARM kernel user helpers (arch/arm/kernel/entry-armv.S, SMP variants), ARM mode and
// position independent. Encodings from the NDK assembler. Real ARMv5 (armeabi) code calls
// these for atomics and TLS. Version 3 advertises get_tls, cmpxchg and memory_barrier;
// cmpxchg64 (version 5) is not provided.
constexpr std::uint32_t kKuserPage = 0xFFFF0000;
constexpr std::uint32_t kKuserHostReturn[] = {
    0xef5affff,  // svc #0x5affff
};
constexpr std::uint32_t kKuserMemoryBarrier[] = {
    0xe12fff1e,  // bx lr
};
constexpr std::uint32_t kKuserCmpxchg[] = {
    0xe1923f9f,  // 1: ldrex r3, [r2]
    0xe0533000,  //    subs r3, r3, r0
    0x01823f91,  //    strexeq r3, r1, [r2]
    0x03330001,  //    teqeq r3, #1
    0x0afffffa,  //    beq 1b
    0xe2730000,  //    rsbs r0, r3, #0
    0xe12fff1e,  //    bx lr
};
constexpr std::uint32_t kKuserGetTls[] = {
    0xee1d0f70,  // mrc p15, 0, r0, c13, c0, 3
    0xe12fff1e,  // bx lr
};
constexpr std::uint32_t kKuserHelperVersion = 3;

bool map_kuser_page(GuestMemory& mem) {
    if (!mem.map_anon(kKuserPage, kPageSize, PROT_READ | PROT_WRITE)) return false;
    const auto put = [&](std::uint32_t offset, const std::uint32_t* words, std::size_t count) {
        std::memcpy(mem.base() + kKuserPage + offset, words, count * sizeof(std::uint32_t));
    };
    put(0xfa0, kKuserMemoryBarrier, std::size(kKuserMemoryBarrier));
    put(0xfc0, kKuserCmpxchg, std::size(kKuserCmpxchg));
    put(0xfe0, kKuserGetTls, std::size(kKuserGetTls));
    put(0xf00, kKuserHostReturn, std::size(kKuserHostReturn));
    put(0xffc, &kKuserHelperVersion, 1);
    return mem.protect(kKuserPage, kPageSize, PROT_READ | PROT_EXEC);
}

}  // namespace

Process::Process() : monitor_(std::make_unique<Dynarmic::ExclusiveMonitor>(kMaxThreads)) {
    if (const char* precise = std::getenv("ZB_PRECISE_FAULTS")) precise_faults_ = precise[0] == '1';
}

Process::~Process() = default;

void Process::request_exit(int status) {
    exit_status_ = status;
    exiting_ = true;
    runtime_report().note_guest_exit("guest exited with status " + std::to_string(status));
}

void Process::invalidate(std::uint32_t addr, std::uint32_t len) {
    std::lock_guard<std::mutex> lock(threads_mutex_);
    for (GuestThread* t : threads_) t->invalidate(addr, len);
    for (GuestThread* t : borrowers_) t->invalidate(addr, len);
}

bool Process::first_time(std::uint64_t key) {
    std::lock_guard<std::mutex> lock(seen_mutex_);
    return seen_.insert(key).second;
}

void Process::record_file_mapping(std::uint32_t start, std::uint32_t length, std::uint64_t offset, std::string path,
                                  bool offset_is_vaddr) {
    forget_mappings(start, length);
    file_mappings_.push_back({start, length, offset, std::move(path), offset_is_vaddr});
}

std::vector<std::string> Process::mapped_file_paths() const {
    std::vector<std::string> paths;
    for (auto it = file_mappings_.rbegin(); it != file_mappings_.rend(); ++it) {
        if (it->path.empty()) continue;
        if (std::find(paths.begin(), paths.end(), it->path) == paths.end()) paths.push_back(it->path);
    }
    return paths;
}

void Process::forget_mappings(std::uint32_t start, std::uint64_t length) {
    const std::uint64_t end = static_cast<std::uint64_t>(start) + length;
    std::erase_if(file_mappings_, [&](const FileMapping& m) {
        return m.start < end && static_cast<std::uint64_t>(m.start) + m.length > start;
    });
    std::erase_if(textrel_ranges_, [&](const std::pair<std::uint32_t, std::uint32_t>& r) {
        return r.first < end && static_cast<std::uint64_t>(r.first) + r.second > start;
    });
}

std::string Process::describe_address(std::uint32_t addr) const {
    for (const auto& m : file_mappings_) {
        if (addr >= m.start && addr - m.start < m.length) {
            char where[64];
            std::snprintf(where, sizeof where, " %s 0x%llx", m.offset_is_vaddr ? "vaddr" : "offset",
                          static_cast<unsigned long long>(m.offset + (addr - m.start)));
            return m.path + where;
        }
    }
    return "?";
}

void Process::add_textrel_range(std::uint32_t start, std::uint32_t length) {
    textrel_ranges_.emplace_back(start, length);
}

bool Process::overlaps_textrel_range(std::uint32_t start, std::uint64_t length) const {
    const std::uint64_t end = static_cast<std::uint64_t>(start) + length;
    for (const auto& [range_start, range_length] : textrel_ranges_) {
        if (range_start < end && static_cast<std::uint64_t>(range_start) + range_length > start) return true;
    }
    return false;
}

void Process::add_path_alias(std::string guest_prefix, std::string host_prefix) {
    if (guest_prefix.empty() || host_prefix.empty()) return;
    std::lock_guard<std::mutex> lock(path_aliases_mutex_);
    for (auto& [from, to] : path_aliases_) {
        if (from == guest_prefix) {
            to = std::move(host_prefix);
            return;
        }
    }
    path_aliases_.emplace_back(std::move(guest_prefix), std::move(host_prefix));
}

std::string Process::translate_path(const char* guest_path) const {
    const std::string_view path(guest_path);
    if (path == "/proc/self/exe") return exe_path_;
    
    // --- STUB LIBS PATH FIX (NON-ROOT UNIVERSAL) ---
    if (path == "/system/lib/libEGL.so" || path == "/system/lib/libGLESv2.so" ||
        path == "/system/lib/libGLESv1_CM.so" || path == "/system/lib/libandroid.so" ||
        path == "/system/lib/libjnigraphics.so") {
        
        // Find the folder where the current game is located dynamically
        std::string game_dir = exe_path_.substr(0, exe_path_.find_last_of('/'));
        
        // Look for the library in the game's local lib folder
        std::string stub = game_dir + "/" + std::string(path.substr(12));
        
        // If it exists in the game's lib folder, use it!
        if (::access(stub.c_str(), F_OK) == 0) return stub;
    }
    // -----------------------------------------------

    {
        std::lock_guard<std::mutex> lock(path_aliases_mutex_);
        const std::string* best_from = nullptr;
        const std::string* best_to = nullptr;
        for (const auto& [from, to] : path_aliases_) {
            if (path.substr(0, from.size()) != from) continue;
            if (best_from == nullptr || from.size() > best_from->size()) {
                best_from = &from;
                best_to = &to;
            }
        }
        if (best_from != nullptr) return *best_to + std::string(path.substr(best_from->size()));
    }
    if (!sysroot_.empty()) {
        for (const auto& m : kPathMappings) {
            if (path.substr(0, m.guest_prefix.size()) == m.guest_prefix) {
                std::string out = sysroot_;
                out += m.sysroot_prefix;
                out += path.substr(m.guest_prefix.size());
                if (::access(out.c_str(), F_OK) == 0) return out;
                return std::string(path);
            }
        }
    }
    return std::string(path);
}

std::size_t Process::thread_count() const {
    std::lock_guard<std::mutex> lock(threads_mutex_);
    return threads_.size();
}

bool Process::is_borrower(const GuestThread& thread) const {
    std::lock_guard<std::mutex> lock(threads_mutex_);
    for (const GuestThread* t : borrowers_) {
        if (t == &thread) return true;
    }
    return false;
}

int Process::allocate_processor_id() {
    std::lock_guard<std::mutex> lock(threads_mutex_);
    for (std::size_t i = 0; i < processor_ids_.size(); ++i) {
        if (!processor_ids_.test(i)) {
            processor_ids_.set(i);
            return static_cast<int>(i);
        }
    }
    return -1;
}

GuestThread* Process::find_thread(std::int32_t tid) {
    std::lock_guard<std::mutex> lock(threads_mutex_);
    for (GuestThread* t : borrowers_) {
        if (t->tid == tid) return t;
    }
    for (GuestThread* t : threads_) {
        if (t->tid == tid) return t;
    }
    return nullptr;
}

bool Process::post_signal_to(std::int32_t tid, const g::siginfo32& info) {
    // post_signal takes no Process lock (atomics and a futex wake), so holding threads_mutex_
    // here cannot deadlock.
    std::lock_guard<std::mutex> lock(threads_mutex_);
    for (GuestThread* t : borrowers_) {
        if (t->tid == tid) {
            t->post_signal(info);
            return true;
        }
    }
    for (GuestThread* t : threads_) {
        if (t->tid == tid) {
            t->post_signal(info);
            return true;
        }
    }
    return false;
}

std::unique_ptr<GuestThread> Process::create_borrower(GuestThread& carrier) {
    const int processor_id = allocate_processor_id();
    if (processor_id < 0) return nullptr;
    auto borrower =
        std::make_unique<GuestThread>(mem_, monitor_.get(), static_cast<std::size_t>(processor_id), precise_faults_);
    borrower->regs().fill(0);
    borrower->regs()[13] = carrier.regs()[13] & ~7u;
    borrower->ext_regs().fill(0);
    borrower->set_cpsr(kCpsrUserMode);
    borrower->set_fpscr(carrier.fpscr());
    borrower->set_tls(carrier.tls());
    borrower->tid = carrier.tid;
    borrower->sigmask = carrier.sigmask;
    borrower->altstack = carrier.altstack;
    std::lock_guard<std::mutex> lock(threads_mutex_);
    borrowers_.push_back(borrower.get());
    return borrower;
}

// Borrower counterpart of unregister_thread(), in this order: tid routing is removed first, then
// the exclusive monitor slot is cleared and the JIT freed, and only then is the processor id
// released, so a concurrent create_borrower/clone_thread cannot reuse the id while this JIT still
// exists. A borrower is never published as the process signal target, and only the borrowing host
// thread names it as current (cleared by the caller before this), so no forwarding reader can
// still hold it.
void Process::destroy_borrower(std::unique_ptr<GuestThread> borrower, GuestThread& carrier) {
    if (!borrower) return;
    carrier.sigmask = borrower->sigmask;
    carrier.altstack = borrower->altstack;
    {
        std::lock_guard<std::mutex> lock(threads_mutex_);
        std::erase(borrowers_, borrower.get());
    }
    g::siginfo32 info;
    while (borrower->take_signal(0, info)) carrier.post_signal(info);
    const std::size_t processor_id = borrower->processor_id();
    monitor_->ClearProcessor(processor_id);
    borrower.reset();
    std::lock_guard<std::mutex> lock(threads_mutex_);
    processor_ids_.reset(processor_id);
}

void Process::register_thread(GuestThread* thread) {
    std::lock_guard<std::mutex> lock(threads_mutex_);
    threads_.push_back(thread);
}

int Process::run(const std::string& path, const std::vector<std::string>& argv, const std::vector<std::string>& envp) {
    if (!mem_.ok()) return 1;

    char resolved[PATH_MAX];
    exe_path_ = realpath(path.c_str(), resolved) != nullptr ? resolved : path;

    LoadedElf exe;
    std::string error;
    if (!load_elf(mem_, path, kExecutableLimit, exe, error)) {
        log("%s", error.c_str());
        return 1;
    }
    record_file_mapping(exe.load_start, exe.load_end - exe.load_start, exe.load_start - exe.bias, exe_path_, true);

    std::uint32_t start_pc = exe.entry;
    std::uint32_t interp_base = 0;
    if (!exe.interp.empty()) {
        const std::string interp_path = translate_path(exe.interp.c_str());
        LoadedElf interp;
        if (!load_elf(mem_, interp_path, kMmapLimit, interp, error)) {
            log("cannot load the dynamic linker %s: %s", exe.interp.c_str(), error.c_str());
            if (sysroot_.empty()) log("pass --sysroot or set ZB_SYSROOT");
            return 1;
        }
        record_file_mapping(interp.load_start, interp.load_end - interp.load_start, interp.load_start - interp.bias,
                            interp_path, true);
        interp_base = interp.bias;
        start_pc = interp.entry;
        linker_start_ = interp.load_start;
        linker_end_ = interp.load_end;
    }

    if (!mem_.map_anon(kStackTop - kStackSize, kStackSize, PROT_READ | PROT_WRITE)) {
        log("cannot map the guest stack");
        return 1;
    }
    if (!map_kuser_page(mem_)) {
        log("cannot map the kuser helper page");
        return 1;
    }
    brk_start = brk_current = exe.load_end;

    const std::vector<AuxEntry> auxv = {
        {AT_PHDR, exe.phdr},
        {AT_PHENT, sizeof(Elf32_Phdr)},
        {AT_PHNUM, exe.phnum},
        {AT_PAGESZ, kPageSize},
        {AT_BASE, interp_base},
        {AT_FLAGS, 0},
        {AT_ENTRY, exe.entry},
        {AT_UID, static_cast<std::uint32_t>(getuid())},
        {AT_EUID, static_cast<std::uint32_t>(geteuid())},
        {AT_GID, static_cast<std::uint32_t>(getgid())},
        {AT_EGID, static_cast<std::uint32_t>(getegid())},
        {AT_HWCAP, kGuestHwcap},
        {AT_HWCAP2, 0},
        {AT_CLKTCK, 100},
        {AT_SECURE, 0},
    };
    const std::uint32_t sp = build_initial_stack(mem_, kStackTop, argv, envp, auxv, exe_path_);
    if (sp == 0) {
        log("arguments and environment do not fit on the guest stack");
        return 1;
    }
    initial_sp_ = sp;

    main_ = std::make_unique<GuestThread>(mem_, monitor_.get(), static_cast<std::size_t>(allocate_processor_id()),
                                          precise_faults_);
    auto& regs = main_->regs();
    regs.fill(0);
    regs[13] = sp;
    regs[15] = start_pc & ~1u;
    main_->set_cpsr(kCpsrUserMode | ((start_pc & 1) ? kCpsrThumb : 0));
    main_->tid = static_cast<std::int32_t>(::syscall(SYS_gettid));
    register_thread(main_.get());
    set_process_signal_target(main_.get());
    install_host_signal_forwarding();
    install_terminate_reporter();

    thread_loop(*main_);
    // Every return below retires what run() published, so destroying the Process afterwards cannot
    // leave a signal handler with a freed main_. The target is retired while t_current_thread still
    // names main_, then this host thread stops naming it.
    if (exiting_) {
        // exit_group/fatal exit: other live threads would already have ended the host process, so
        // main_ is the only registered thread. Its CLONE_CHILD_CLEARTID is not honoured on a
        // process-wide exit.
        unregister_thread(*main_);
        set_current_thread(nullptr);
        return exit_status_;
    }

    // The main thread called exit() while other threads may still run: wait for them.
    const int main_status = main_->exit_status;
    finish_thread(*main_);
    set_current_thread(nullptr);
    wait_for_threads();
    return exiting_ ? exit_status_.load() : main_status;
}

bool Process::dispatch_stop(GuestThread& thread, const Stop& stop) {
    switch (stop.kind) {
    case StopKind::Svc:
        if (stop.swi == 0) {
            if (handle_syscall(*this, thread)) return true;
            if (!exiting_ && thread.call_depth > 0) {
                // bionic has already released this thread's TLS and stack; nothing may run on it.
                log("guest thread exited inside a host-to-guest call");
                request_exit(1);
                exit_host_process();
            }
            if (exiting_ && thread_count() > 1) exit_host_process();
            return false;
        }
        if (stop.swi == kHostReturnSwi) {
            // GuestThread::call consumes its own return; any other one is an illegal instruction.
            if (thread.call_depth == 0) log("host return svc outside a host-to-guest call at pc 0x%08x", stop.pc - 4);
            Stop illegal;
            illegal.kind = StopKind::Exception;
            illegal.exception = Dynarmic::A32::Exception::UndefinedInstruction;
            illegal.pc = stop.pc - 4;
            return fault_or_crash(thread, illegal);
        }
        if ((stop.swi & 0xFF0000u) == kHostCallBase) {
    const std::uint32_t index = stop.swi & 0xFFFFu;
    
    #ifndef ZB_STOCK_BUILD
    record_thread_activity(thread.tid, ThreadActivityKind::kHostCall, index);
    #endif
    
    // A host call is host C++ called from translated guest code, and an exception that
    // escapes it has no frame above to catch it: the process dies with "terminating" and
    // nothing else. The name of the call and the reason are worth far more than the
    // abort, so they go into the report and the guest is stopped instead.
    bool served = false;
    
    #ifndef ZB_STOCK_BUILD
    const std::uint64_t host_call_started = read_ticks();
    #endif
    
    try {
        served = host_call_handler_ && host_call_handler_(index, thread);
    } catch (const std::exception& failure) {
        const auto [failed_library, failed_name] = host_call_name(index);
        std::string detail = std::string(failed_library) + ":" + failed_name + ": " + failure.what();
        log("host call %s threw: %s", failed_name, failure.what());
        runtime_report().note_jni_detail("host-call-threw", detail, true);
        
        #ifndef ZB_STOCK_BUILD
        record_thread_activity_done(thread.tid);
        #endif
        
        request_exit(70);
        return false;
    } catch (...) {
        const auto [failed_library, failed_name] = host_call_name(index);
        log("host call %s threw a non-standard exception", failed_name);
        runtime_report().note_jni_detail("host-call-threw",
                                         std::string(failed_library) + ":" + failed_name, true);
        
        #ifndef ZB_STOCK_BUILD
        record_thread_activity_done(thread.tid);
        #endif
        
        request_exit(70);
        return false;
bool Process::dispatch_stop(GuestThread& thread, const Stop& stop) {
    switch (stop.kind) {
    case StopKind::Svc:
        if (stop.swi == 0) {
            if (handle_syscall(*this, thread)) return true;
            if (!exiting_ && thread.call_depth > 0) {
                // bionic has already released this thread's TLS and stack; nothing may run on it.
                log("guest thread exited inside a host-to-guest call");
                request_exit(1);
                exit_host_process();
            }
            if (exiting_ && thread_count() > 1) exit_host_process();
            return false;
        }
        if (stop.swi == kHostReturnSwi) {
            // GuestThread::call consumes its own return; any other one is an illegal instruction.
            if (thread.call_depth == 0) log("host return svc outside a host-to-guest call at pc 0x%08x", stop.pc - 4);
            Stop illegal;
            illegal.kind = StopKind::Exception;
            illegal.exception = Dynarmic::A32::Exception::UndefinedInstruction;
            illegal.pc = stop.pc - 4;
            return fault_or_crash(thread, illegal);
        }
        if ((stop.swi & 0xFF0000u) == kHostCallBase) {
            const std::uint32_t index = stop.swi & 0xFFFFu;
            
            #ifndef ZB_STOCK_BUILD
            record_thread_activity(thread.tid, ThreadActivityKind::kHostCall, index);
            #endif
            
            // A host call is host C++ called from translated guest code, and an exception that
            // escapes it has no frame above to catch it: the process dies with "terminating" and
            // nothing else. The name of the call and the reason are worth far more than the
            // abort, so they go into the report and the guest is stopped instead.
            bool served = false;
            
            #ifndef ZB_STOCK_BUILD
            const std::uint64_t host_call_started = read_ticks();
            #endif
            
            try {
                served = host_call_handler_ && host_call_handler_(index, thread);
            } catch (const std::exception& failure) {
                const auto [failed_library, failed_name] = host_call_name(index);
                std::string detail = std::string(failed_library) + ":" + failed_name + ": " + failure.what();
                log("host call %s threw: %s", failed_name, failure.what());
                runtime_report().note_jni_detail("host-call-threw", detail, true);
                
                #ifndef ZB_STOCK_BUILD
                record_thread_activity_done(thread.tid);
                #endif
                
                request_exit(70);
                return false;
            } catch (...) {
                const auto [failed_library, failed_name] = host_call_name(index);
                log("host call %s threw a non-standard exception", failed_name);
                runtime_report().note_jni_detail("host-call-threw",
                                                 std::string(failed_library) + ":" + failed_name, true);
                
                #ifndef ZB_STOCK_BUILD
                record_thread_activity_done(thread.tid);
                #endif
                
                request_exit(70);
                return false;
            }
            
            #ifndef ZB_STOCK_BUILD
            account_host_call(index, host_call_started);
            #endif
            
            if (served) {
                #ifndef ZB_STOCK_BUILD
                record_thread_activity_done(thread.tid);
                #endif
                return !exiting_;
            }
            const auto [library, name] = host_call_name(index);
            if (first_time(kSeenHostCall | index)) {
                log("host call %s:%s is not implemented yet", library, name);
            }
            // The guest keeps running with r0 = 0; the report is the only record that survives a
            // device run, because OxygenOS drops our logcat output.
            runtime_report().note_unimplemented_host_call(index, library, name);
            thread.regs()[0] = 0;
            return true;
        }
        if (first_time(kSeenUnexpectedSvc | stop.swi)) {
            log("unexpected svc #0x%x at pc 0x%08x", stop.swi, stop.pc);
        }
        thread.regs()[0] = static_cast<std::uint32_t>(-ENOSYS);
        return true;
    case StopKind::Interrupted:
        return true;
    case StopKind::MemoryFault:
    case StopKind::Exception:
        return fault_or_crash(thread, stop);
    case StopKind::None:
        log("guest stopped without a reason at pc 0x%08x", stop.pc);
        request_exit(1);
        if (thread_count() > 1) exit_host_process();
        return false;
    }
    return false;
}

bool Process::fault_or_crash(GuestThread& thread, const Stop& stop) {
    if (deliver_fault(thread, stop)) return true;
    crash_report(stop, thread);
    request_exit(128 + (stop.kind == StopKind::MemoryFault ? SIGSEGV : SIGILL));
    if (thread_count() > 1) exit_host_process();
    return false;
}

bool Process::after_stop(GuestThread& thread) {
    if (!thread.has_pending_signals(thread.sigmask) || dispatch_pending_signals(thread)) return true;
    if (thread_count() > 1) exit_host_process();
    return false;
}

Process* Process::reporter_process = nullptr;

// ---- abort recorder ----------------------------------------------------------------------------
// The process can also die from a plain SIGABRT that the terminate reporter below never sees: the
// message is "libc++abi: terminating" and nothing else, and Android's own crash dump may fail to
// parse the guest mappings. This handler records, from the thread that aborts:
//   - the host call stack (library, symbol, offset), which says whether the abort came from the
//     translator, the GL driver, the JNI bridge or somewhere else;
//   - the guest registers of the guest thread running on this host thread, its pc and lr resolved
//     to library + offset, and every word on its stack that points into executable file-backed
//     guest memory (a poor man's backtrace);
//   - the recent JNI and EGL calls.
// Every line goes to logcat (tag "zb-abort", priority F) and is appended to
// <external files dir>/zb-abort.txt. Then control goes to the handler that was installed before
// (the system's crash handler), so the usual crash dump and the exit status are unchanged.
namespace {

constexpr const char* kAbortTag = "zb-abort";
// Same directory as zb-runtime-report.txt and zb-errors.txt. Change the package name if you rename it.
constexpr const char* kAbortFile = "/storage/emulated/0/Android/data/com.zettabridge.launcher/files/zb-abort.txt";
constexpr int kMaxHostFrames = 40;
constexpr int kMaxGuestStackHits = 40;
constexpr std::uint32_t kGuestStackScanWords = 1024;  // 4 KiB above sp

int g_abort_fd = -1;
struct sigaction g_previous_abort;
std::atomic<bool> g_abort_seen{false};

void abort_line(const char* text) {
    __android_log_write(ANDROID_LOG_FATAL, kAbortTag, text);
    if (g_abort_fd >= 0) {
        [[maybe_unused]] const ssize_t a = ::write(g_abort_fd, text, std::strlen(text));
        [[maybe_unused]] const ssize_t b = ::write(g_abort_fd, "\n", 1);
    }
}

struct HostTrace {
    void* frames[kMaxHostFrames];
    int count = 0;
};

_Unwind_Reason_Code collect_host_frame(_Unwind_Context* context, void* arg) {
    auto* trace = static_cast<HostTrace*>(arg);
    const std::uintptr_t pc = _Unwind_GetIP(context);
    if (pc == 0) return _URC_NO_REASON;
    if (trace->count >= kMaxHostFrames) return _URC_END_OF_STACK;
    trace->frames[trace->count++] = reinterpret_cast<void*>(pc);
    return _URC_NO_REASON;
}

}  // namespace

// The translator does not always throw when it cannot go on: a failed check inside it calls
// std::terminate directly, and the process dies with the message "terminating" and a stripped
// backtrace. Nothing can catch that, but the guest program counter of the thread it happened on
// is the whole answer, and this thread still knows it.
void Process::install_terminate_reporter() {
    static std::terminate_handler previous = nullptr;
    static std::once_flag once;
    // One guest process has one Process, and it outlives everything: the handler needs it only to
    // name the library an address belongs to.
    reporter_process = this;
    std::call_once(once, [] {
        previous = std::set_terminate([] {
            if (GuestThread* thread = Process::current_thread()) {
                // The bare address says nothing: what is needed is the library and the offset in
                // it, which is exactly what the crash report already resolves.
                const std::uint32_t pc = thread->regs()[15];
                const std::string where =
                    reporter_process != nullptr ? reporter_process->describe_address(pc) : std::string();
                char detail[192];
                std::snprintf(detail, sizeof detail, "pc 0x%08x %s", pc, where.c_str());
                log("terminating while running guest code at %s", detail);
                runtime_report().note_jni_detail("translator-gave-up", detail, true);
            } else {
                runtime_report().note_jni_detail("translator-gave-up", "outside guest code", true);
            }
            if (previous != nullptr) previous();
            std::abort();
        });

        // The abort recorder (see the block above). Installed after the system's own handler, so
        // that one is called last.
        g_abort_fd = ::open(kAbortFile, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        struct sigaction action;
        std::memset(&action, 0, sizeof action);
        action.sa_sigaction = +[](int sig, siginfo_t* info, void* context) {
            if (!g_abort_seen.exchange(true)) {
                char line[512];
                std::snprintf(line, sizeof line, "---- SIGABRT on host tid %ld, si_code %d, si_pid %d ----",
                              static_cast<long>(::syscall(SYS_gettid)), info != nullptr ? info->si_code : 0,
                              info != nullptr ? static_cast<int>(info->si_pid) : 0);
                abort_line(line);

                HostTrace trace;
                _Unwind_Backtrace(collect_host_frame, &trace);
                for (int i = 0; i < trace.count; ++i) {
                    const auto pc = reinterpret_cast<std::uintptr_t>(trace.frames[i]);
                    Dl_info dl;
                    std::memset(&dl, 0, sizeof dl);
                    if (dladdr(trace.frames[i], &dl) != 0 && dl.dli_fname != nullptr) {
                        const auto base = reinterpret_cast<std::uintptr_t>(dl.dli_fbase);
                        const auto sym = reinterpret_cast<std::uintptr_t>(dl.dli_saddr);
                        std::snprintf(line, sizeof line, "host #%02d pc %llx %s (%s+%llx)", i,
                                      static_cast<unsigned long long>(pc - base), dl.dli_fname,
                                      dl.dli_sname != nullptr ? dl.dli_sname : "?",
                                      static_cast<unsigned long long>(sym != 0 ? pc - sym : 0));
                    } else {
                        std::snprintf(line, sizeof line, "host #%02d pc %llx", i, static_cast<unsigned long long>(pc));
                    }
                    abort_line(line);
                }

                GuestThread* thread = Process::current_thread();
                Process* process = reporter_process;
                if (thread == nullptr || process == nullptr) {
                    abort_line("guest: this host thread was running no guest code");
                } else {
                    const auto& r = thread->regs();
                    auto where = [process](std::uint32_t address) { return process->describe_address(address & ~1u); };
                    std::snprintf(line, sizeof line, "guest pc %08x in %s", r[15], where(r[15]).c_str());
                    abort_line(line);
                    std::snprintf(line, sizeof line, "guest lr %08x in %s", r[14], where(r[14]).c_str());
                    abort_line(line);
                    std::snprintf(line, sizeof line,
                                  "guest r0=%08x r1=%08x r2=%08x r3=%08x r4=%08x r5=%08x r6=%08x r7=%08x",
                                  r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
                    abort_line(line);
                    std::snprintf(line, sizeof line,
                                  "guest r8=%08x r9=%08x r10=%08x r11=%08x r12=%08x sp=%08x cpsr=%08x tls=%08x",
                                  r[8], r[9], r[10], r[11], r[12], r[13], thread->cpsr(), thread->tls());
                    abort_line(line);

                    const std::uint32_t sp = r[13];
                    int shown = 0;
                    for (std::uint32_t i = 0; i < kGuestStackScanWords && shown < kMaxGuestStackHits; ++i) {
                        const std::uint32_t at = sp + 4 * i;
                        const std::uint8_t* bytes = process->mem_.host_ptr(at, 4, kPageRead);
                        if (bytes == nullptr) break;
                        std::uint32_t word;
                        std::memcpy(&word, bytes, sizeof word);
                        const std::uint32_t code = word & ~1u;
                        if (code < kPageSize || !process->mem_.accessible(code, 2, kPageExec)) continue;
                        const std::string in = process->describe_address(code);
                        if (in == "?") continue;
                        std::snprintf(line, sizeof line, "guest stack sp+0x%x: %08x in %s", 4 * i, word, in.c_str());
                        abort_line(line);
                        ++shown;
                    }

                    std::string recent = "recent-jni-calls: " + jni_recent_calls();
                    abort_line(recent.substr(0, 1500).c_str());
                    recent = "recent-egl-calls: " + egl_recent_calls();
                    abort_line(recent.substr(0, 1500).c_str());
                }
                abort_line("---- end of abort record ----");
            }
            // Hand over to the handler that was installed before this one (the system's crash
            // handler). If there was none, returning lets abort() reset the action and raise again.
            if ((g_previous_abort.sa_flags & SA_SIGINFO) != 0 && g_previous_abort.sa_sigaction != nullptr) {
                g_previous_abort.sa_sigaction(sig, info, context);
            } else if (g_previous_abort.sa_handler != SIG_DFL && g_previous_abort.sa_handler != SIG_IGN &&
                       g_previous_abort.sa_handler != nullptr) {
                g_previous_abort.sa_handler(sig);
            }
        };
        action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGABRT, &action, &g_previous_abort) != 0) {
            log("cannot install the abort recorder: %s", std::strerror(errno));
        }

        // The host call profiler: a background thread that turns the counters above into a file.
        std::thread([] {
            const std::uint64_t frequency = tick_frequency();
            std::vector<std::uint64_t> seen_count(kProfileSlots, 0);
            std::vector<std::uint64_t> seen_ticks(kProfileSlots, 0);
            auto window_start = std::chrono::steady_clock::now();
            std::uint64_t busiest_calls = 0;
            for (;;) {
                std::this_thread::sleep_for(kProfileWindow);
                const auto now = std::chrono::steady_clock::now();
                const double window_ms = std::chrono::duration<double, std::milli>(now - window_start).count();
                window_start = now;

                struct Row {
                    std::uint32_t index;
                    std::uint64_t calls;
                    std::uint64_t ticks;
                };
                std::vector<Row> rows;
                std::uint64_t total_calls = 0;
                std::uint64_t total_ticks = 0;
                for (std::size_t i = 0; i < kProfileSlots; ++i) {
                    const std::uint64_t calls = g_host_call_count[i].load(std::memory_order_relaxed);
                    const std::uint64_t ticks = g_host_call_ticks[i].load(std::memory_order_relaxed);
                    const std::uint64_t delta_calls = calls - seen_count[i];
                    const std::uint64_t delta_ticks = ticks - seen_ticks[i];
                    seen_count[i] = calls;
                    seen_ticks[i] = ticks;
                    if (delta_calls == 0) continue;
                    rows.push_back({static_cast<std::uint32_t>(i), delta_calls, delta_ticks});
                    total_calls += delta_calls;
                    total_ticks += delta_ticks;
                }
                if (total_calls < 100) continue;  // an idle window: keep the last busy one on disk

                std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.ticks > b.ticks; });
                const double total_ms = 1000.0 * static_cast<double>(total_ticks) / static_cast<double>(frequency);
                std::string text;
                char line[256];
                std::snprintf(line, sizeof line,
                              "window %.2f s: %llu host calls (%.0f per second), %.1f ms inside them,\n"
                              "= %.1f%% of the window summed over all threads (100%% = one thread doing nothing else)\n\n",
                              window_ms / 1000.0, static_cast<unsigned long long>(total_calls),
                              1000.0 * static_cast<double>(total_calls) / window_ms, total_ms,
                              100.0 * total_ms / window_ms);
                text += line;
                std::snprintf(line, sizeof line, "%-4s %-34s %9s %9s %9s %7s\n", "#", "host call", "calls", "ms", "avg us",
                              "% win");
                text += line;
                for (std::size_t r = 0; r < rows.size() && r < kProfileRows; ++r) {
                    const auto [library, name] = host_call_name(rows[r].index);
                    char label[64];
                    std::snprintf(label, sizeof label, "%s:%s", library, name);
                    const double ms = 1000.0 * static_cast<double>(rows[r].ticks) / static_cast<double>(frequency);
                    std::snprintf(line, sizeof line, "%-4zu %-34.34s %9llu %9.1f %9.2f %6.1f%%\n", r + 1, label,
                                  static_cast<unsigned long long>(rows[r].calls), ms,
                                  1000.0 * ms / static_cast<double>(rows[r].calls), 100.0 * ms / window_ms);
                    text += line;
                }
                write_text_file(kProfileFile, text);
                if (total_calls > busiest_calls) {
                    busiest_calls = total_calls;
                    write_text_file(kProfileBestFile, text);
                }
            }
        }).detach();

        // The guest PC sampler: where does a busy guest thread spend its time? Every 5 ms it reads
        // the program counter that each guest thread last committed (block granularity: translated
        // code only writes it back when a block ends) and whether the thread is inside a host call
        // or a syscall. Every 10 seconds it writes zb-pcprofile.txt, replacing the previous one:
        // for the three threads that used the most CPU in the window (taken from /proc, so sleeping
        // threads never show up), the split between translated code, host calls and syscalls, then
        // the libraries the translated code was in and its hottest 64-byte blocks. A guest thread
        // that is parked sits in the guest's own start-up program ("zbhost"); those samples are
        // left out of the percentages.
        std::thread([] {
            struct Counters {
                std::uint32_t samples = 0;
                std::uint32_t in_host_call = 0;
                std::uint32_t in_syscall = 0;
            };
            std::unordered_map<std::int32_t, Counters> per_thread;
            std::unordered_map<std::uint64_t, std::uint32_t> hits;  // (tid << 32) | (pc >> 6)
            std::unordered_map<std::int32_t, std::uint64_t> cpu_before;
            // user + system time of a thread so far, in clock ticks (usually 10 ms each)
            auto thread_cpu_ticks = [](std::int32_t tid) -> std::uint64_t {
                char path[64];
                std::snprintf(path, sizeof path, "/proc/self/task/%d/stat", static_cast<int>(tid));
                char buffer[1024];
                std::FILE* file = std::fopen(path, "re");
                if (file == nullptr) return 0;
                const std::size_t got = std::fread(buffer, 1, sizeof buffer - 1, file);
                std::fclose(file);
                buffer[got] = 0;
                const char* close_paren = std::strrchr(buffer, ')');
                unsigned long user = 0, system = 0;
                if (close_paren == nullptr ||
                    std::sscanf(close_paren + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &user, &system) != 2)
                    return 0;
                return static_cast<std::uint64_t>(user) + system;
            };
            const long ticks_per_second = sysconf(_SC_CLK_TCK) > 0 ? sysconf(_SC_CLK_TCK) : 100;
            auto window_start = std::chrono::steady_clock::now();
            std::uint64_t busiest_cpu = 0;
            std::uint64_t calls_before = 0;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                Process* process = reporter_process;
                if (process == nullptr) continue;
                const std::vector<ThreadActivitySample> activity = snapshot_thread_activity();
                {
                    std::lock_guard<std::mutex> lock(process->threads_mutex_);
                    auto sample_one = [&](GuestThread* thread) {
                        const std::int32_t tid = thread->tid;
                        Counters& counters = per_thread[tid];
                        ++counters.samples;
                        for (const ThreadActivitySample& a : activity) {
                            if (a.tid != tid || !a.in_progress) continue;
                            if (a.kind == ThreadActivityKind::kHostCall) {
                                ++counters.in_host_call;
                                return;
                            }
                            if (a.kind == ThreadActivityKind::kSyscall) {
                                ++counters.in_syscall;
                                return;
                            }
                        }
                        const std::uint32_t pc = thread->regs()[15];
                        ++hits[(static_cast<std::uint64_t>(static_cast<std::uint32_t>(tid)) << 32) | (pc >> 6)];
                    };
                    for (GuestThread* thread : process->threads_) sample_one(thread);
                    for (GuestThread* thread : process->borrowers_) {
                        if (std::find(process->threads_.begin(), process->threads_.end(), thread) == process->threads_.end())
                            sample_one(thread);
                    }
                }

                const auto now = std::chrono::steady_clock::now();
                if (now - window_start < std::chrono::seconds(10)) continue;
                const double seconds = std::chrono::duration<double>(now - window_start).count();
                window_start = now;

                // The three threads that used the most CPU since the last report.
                std::vector<std::pair<std::uint64_t, std::int32_t>> busiest;  // (cpu ticks, tid)
                for (const auto& [tid, counters] : per_thread) {
                    (void)counters;
                    const std::uint64_t total = thread_cpu_ticks(tid);
                    const std::uint64_t used = total - std::min(total, cpu_before[tid]);
                    cpu_before[tid] = total;
                    if (used > 0) busiest.emplace_back(used, tid);
                }
                std::sort(busiest.begin(), busiest.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                if (busiest.size() > 3) busiest.resize(3);
                const double window_ticks = seconds * static_cast<double>(ticks_per_second);

                // A window with next to no host calls is a game that has stopped (or sits in a menu
                // that draws nothing): write what every guest thread is doing and waiting for.
                std::uint64_t calls_now = 0;
                for (std::size_t i = 0; i < kProfileSlots; ++i) calls_now += g_host_call_count[i].load(std::memory_order_relaxed);
                const std::uint64_t calls_in_window = calls_now - std::min(calls_now, calls_before);
                calls_before = calls_now;
                if (calls_in_window < 50 && calls_now > 1000) {
                    struct Snapshot {
                        std::int32_t tid;
                        std::uint32_t pc, lr, sp;
                        std::uint64_t pending, blocked;  // guest signals queued for the thread, and its mask
                        bool in_host_syscall;            // asleep in a host syscall a guest signal can break
                    };
                    std::vector<Snapshot> snapshots;
                    {
                        std::lock_guard<std::mutex> lock(process->threads_mutex_);
                        auto take = [&](GuestThread* thread) {
                            const auto& r = thread->regs();
                            snapshots.push_back({thread->tid, r[15], r[14], r[13], thread->pending_signals(),
                                                 thread->sigmask, thread->blocking()});
                        };
                        for (GuestThread* thread : process->threads_) take(thread);
                        for (GuestThread* thread : process->borrowers_) {
                            if (std::find(process->threads_.begin(), process->threads_.end(), thread) == process->threads_.end())
                                take(thread);
                        }
                    }
                    const std::vector<ThreadActivitySample> waiting = snapshot_thread_activity();
                    std::string freeze;
                    char row[400];
                    std::snprintf(row, sizeof row, "only %llu host calls in the last %.0f s: the game looks stopped\n",
                                  static_cast<unsigned long long>(calls_in_window), seconds);
                    freeze += row;
                    {
                        std::lock_guard<std::mutex> mm(process->mm_mutex());
                        for (const Snapshot& snap : snapshots) {
                            char comm_path[64];
                            std::snprintf(comm_path, sizeof comm_path, "/proc/self/task/%d/comm", static_cast<int>(snap.tid));
                            char name[64] = "?";
                            if (std::FILE* comm = std::fopen(comm_path, "re")) {
                                if (std::fgets(name, sizeof name, comm) != nullptr) name[std::strcspn(name, "\n")] = 0;
                                std::fclose(comm);
                            }
                            char state = '?';
                            char stat_path[64];
                            std::snprintf(stat_path, sizeof stat_path, "/proc/self/task/%d/stat", static_cast<int>(snap.tid));
                            if (std::FILE* stat = std::fopen(stat_path, "re")) {
                                char buffer[512];
                                const std::size_t got = std::fread(buffer, 1, sizeof buffer - 1, stat);
                                std::fclose(stat);
                                buffer[got] = 0;
                                if (const char* close_paren = std::strrchr(buffer, ')')) state = close_paren[2];
                            }
                            std::string doing = "running translated code";
                            for (const ThreadActivitySample& a : waiting) {
                                if (a.tid != snap.tid || !a.in_progress) continue;
                                if (a.kind == ThreadActivityKind::kHostCall) {
                                    const auto [library, call] = host_call_name(a.id);
                                    char text[96];
                                    std::snprintf(text, sizeof text, "inside host call 0x%x (%s:%s)", a.id, library, call);
                                    doing = text;
                                } else if (a.kind == ThreadActivityKind::kSyscall) {
                                    char text[64];
                                    std::snprintf(text, sizeof text, "inside syscall %u", a.id);
                                    doing = text;
                                }
                                break;
                            }
                            std::snprintf(row, sizeof row, "\nthread %d %s [%c]: %s\n  pc %08x in %s\n  lr %08x in %s\n",
                                          static_cast<int>(snap.tid), name, state, doing.c_str(), snap.pc,
                                          process->describe_address(snap.pc & ~1u).c_str(), snap.lr,
                                          process->describe_address(snap.lr & ~1u).c_str());
                            freeze += row;
                            // Bit n-1 of these masks is guest signal n. A signal that is pending and not
                            // blocked for a thread that sleeps in a syscall is one that never arrived.
                            std::snprintf(row, sizeof row, "  signals: pending %016llx, blocked %016llx, undelivered %016llx, in host syscall %d\n",
                                          static_cast<unsigned long long>(snap.pending),
                                          static_cast<unsigned long long>(snap.blocked),
                                          static_cast<unsigned long long>(snap.pending & ~snap.blocked),
                                          snap.in_host_syscall ? 1 : 0);
                            freeze += row;
                            int shown = 0;
                            for (std::uint32_t i = 0; i < 512 && shown < 6; ++i) {
                                const std::uint8_t* bytes = process->mem_.host_ptr(snap.sp + 4 * i, 4, kPageRead);
                                if (bytes == nullptr) break;
                                std::uint32_t word;
                                std::memcpy(&word, bytes, sizeof word);
                                const std::uint32_t code = word & ~1u;
                                if (code < kPageSize || !process->mem_.accessible(code, 2, kPageExec)) continue;
                                const std::string in = process->describe_address(code);
                                if (in == "?") continue;
                                std::snprintf(row, sizeof row, "  stack sp+0x%x: %08x in %s\n", 4 * i, word, in.c_str());
                                freeze += row;
                                ++shown;
                            }
                        }
                    }
                    freeze += "\nrecent-jni-calls: " + jni_recent_calls().substr(0, 1500) + "\n";
                    write_text_file(kFreezeFile, freeze);
                }

                if (busiest.empty() || static_cast<double>(busiest.front().first) < 0.05 * window_ticks) {
                    per_thread.clear();
                    hits.clear();
                    continue;  // nothing used the CPU: keep the last report on disk
                }

                std::string text;
                char line[320];
                std::snprintf(line, sizeof line, "window %.1f s, one sample every 5 ms\n", seconds);
                text += line;
                for (const auto& [cpu_used, tid] : busiest) {
                    const Counters& counters = per_thread[tid];
                    char comm_path[64];
                    std::snprintf(comm_path, sizeof comm_path, "/proc/self/task/%d/comm", static_cast<int>(tid));
                    char name[64] = "?";
                    if (std::FILE* comm = std::fopen(comm_path, "re")) {
                        if (std::fgets(name, sizeof name, comm) != nullptr) name[std::strcspn(name, "\n")] = 0;
                        std::fclose(comm);
                    }
                    const double total = static_cast<double>(counters.samples);
                    std::snprintf(line, sizeof line,
                                  "\nthread %d %s: used %.0f%% of one core; in host calls %.1f%%, in syscalls %.1f%% of its samples\n",
                                  static_cast<int>(tid), name, 100.0 * static_cast<double>(cpu_used) / window_ticks,
                                  total > 0 ? 100.0 * counters.in_host_call / total : 0.0,
                                  total > 0 ? 100.0 * counters.in_syscall / total : 0.0);
                    text += line;

                    // Resolve this thread's hit blocks to libraries (the mapping table is guarded by mm_mutex).
                    std::vector<std::pair<std::uint32_t, std::uint32_t>> blocks;  // (hits, pc)
                    for (const auto& [key, count] : hits) {
                        if (static_cast<std::int32_t>(key >> 32) != tid) continue;
                        blocks.emplace_back(count, static_cast<std::uint32_t>(key & 0xFFFFFFFFu) << 6);
                    }
                    std::sort(blocks.begin(), blocks.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                    std::unordered_map<std::string, std::uint32_t> by_library;
                    std::vector<std::pair<std::uint32_t, std::string>> hot;
                    std::uint32_t counted = 0;  // translated-code samples outside the parked start-up program
                    {
                        std::lock_guard<std::mutex> mm(process->mm_mutex());
                        for (const auto& [count, pc] : blocks) {
                            const std::string where = process->describe_address(pc);
                            std::string library = where;
                            const std::size_t cut = where.rfind(" offset 0x") != std::string::npos
                                                        ? where.rfind(" offset 0x")
                                                        : where.rfind(" vaddr 0x");
                            if (cut != std::string::npos) library = where.substr(0, cut);
                            const std::size_t slash = library.rfind('/');
                            if (slash != std::string::npos) library = library.substr(slash + 1);
                            if (library == "zbhost") continue;
                            if (library == "?") library = "(no file: Mono JIT code, heap, stacks)";
                            by_library[library] += count;
                            counted += count;
                            if (hot.size() < 20) hot.emplace_back(count, where);
                        }
                    }
                    if (counted == 0) {
                        text += "  no translated-code samples (parked or waiting)\n";
                        continue;
                    }
                    std::vector<std::pair<std::uint32_t, std::string>> libraries;
                    for (const auto& [library, count] : by_library) libraries.emplace_back(count, library);
                    std::sort(libraries.begin(), libraries.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                    for (std::size_t i = 0; i < libraries.size() && i < 8; ++i) {
                        std::snprintf(line, sizeof line, "  %5.1f%%  %s\n",
                                      100.0 * libraries[i].first / static_cast<double>(counted), libraries[i].second.c_str());
                        text += line;
                    }
                    text += "  hottest 64-byte blocks of translated code:\n";
                    for (const auto& [count, where] : hot) {
                        std::snprintf(line, sizeof line, "    %5.1f%%  %s\n", 100.0 * count / static_cast<double>(counted),
                                      where.c_str());
                        text += line;
                    }
                }
                per_thread.clear();
                hits.clear();
                write_text_file(kPcProfileFile, text);
                if (busiest.front().first > busiest_cpu) {
                    busiest_cpu = busiest.front().first;
                    write_text_file(kPcProfileBestFile, text);
                }
            }
        }).detach();
    });
}

void Process::thread_loop(GuestThread& thread) {
    set_current_thread(&thread);
    // The translator throws for what it cannot translate (an instruction its backend does not
    // implement), and nothing above this loop can catch it: the process aborts with "terminating"
    // and no hint of which instruction it was. The program counter and the reason are the whole
    // answer, so they go into the report and this guest thread ends instead.
    try {
        while (dispatch_stop(thread, thread.run()) && after_stop(thread)) {
        }
    } catch (const std::exception& failure) {
        char where[96];
        std::snprintf(where, sizeof where, "pc 0x%08x: %s", thread.regs()[15], failure.what());
        log("the translator gave up at %s", where);
        runtime_report().note_jni_detail("translator-gave-up", where, true);
        request_exit(71);
    } catch (...) {
        char where[64];
        std::snprintf(where, sizeof where, "pc 0x%08x", thread.regs()[15]);
        log("the translator gave up at %s", where);
        runtime_report().note_jni_detail("translator-gave-up", where, true);
        request_exit(71);
    }
}

std::optional<GuestResult> Process::call_guest(GuestThread& thread, std::uint32_t target, const GuestCall& args) {
    return thread.call(target, args, [&](const Stop& stop) { return dispatch_stop(thread, stop) && after_stop(thread); });
}

std::int32_t Process::clone_thread(GuestThread& parent, std::uint32_t flags, std::uint32_t stack,
                                   std::uint32_t parent_tid_addr, std::uint32_t tls, std::uint32_t child_tid_addr) {
    const int processor_id = allocate_processor_id();
    if (processor_id < 0) return -EAGAIN;

    const std::size_t code_cache_size =
        parent.child_code_cache_size != 0 ? parent.child_code_cache_size : kDefaultCodeCacheSize;
    auto child = std::make_unique<GuestThread>(mem_, monitor_.get(), static_cast<std::size_t>(processor_id),
                                               precise_faults_, code_cache_size);
    child->regs() = parent.regs();
    child->regs()[0] = 0;
    if (stack != 0) child->regs()[13] = stack;
    child->ext_regs() = parent.ext_regs();
    child->set_cpsr(parent.cpsr());
    child->set_fpscr(parent.fpscr());
    child->set_tls((flags & kCloneSetTls) ? tls : parent.tls());
    child->sigmask = parent.sigmask;
    if (flags & kCloneChildClearTid) child->clear_child_tid = child_tid_addr;

    std::promise<pid_t> tid_promise;
    std::future<pid_t> tid_future = tid_promise.get_future();
    register_thread(child.get());
    std::thread([this, owned = std::move(child), promise = &tid_promise, flags, child_tid_addr]() mutable {
        const auto tid = static_cast<pid_t>(::syscall(SYS_gettid));
        owned->tid = tid;
        if (flags & kCloneChildSetTid) write_guest_u32(mem_, child_tid_addr, static_cast<std::uint32_t>(tid));
        promise->set_value(tid);
        thread_main(std::move(owned));
    }).detach();

    const pid_t tid = tid_future.get();
    if (flags & kCloneParentSetTid) write_guest_u32(mem_, parent_tid_addr, static_cast<std::uint32_t>(tid));
    return tid;
}

void Process::thread_main(std::unique_ptr<GuestThread> thread) {
    thread_loop(*thread);
    // Stop naming the thread before finish_thread() and the unique_ptr free it: a host signal
    // landing here afterwards must not take the forwarding fast path to a freed GuestThread.
    // Handlers on this host thread run synchronously with this code, so there is no race.
    set_current_thread(nullptr);
    finish_thread(*thread);
}

void Process::finish_thread(GuestThread& thread) {
    if (thread.clear_child_tid != 0) {
        if (std::uint8_t* p = mem_.host_ptr(thread.clear_child_tid, 4, kPageWrite)) {
            const std::uint32_t zero = 0;
            std::memcpy(p, &zero, 4);
            ::syscall(SYS_futex, p, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
        }
    }
    unregister_thread(thread);
}

void Process::unregister_thread(GuestThread& thread) {
    // Waits for in-flight forwarding handlers, so the caller may free the thread afterwards.
    clear_process_signal_target(&thread);
    monitor_->ClearProcessor(thread.processor_id());
    std::lock_guard<std::mutex> lock(threads_mutex_);
    std::erase(threads_, &thread);
    processor_ids_.reset(thread.processor_id());
    threads_cv_.notify_all();
}

void Process::wait_for_threads() {
    std::unique_lock<std::mutex> lock(threads_mutex_);
    threads_cv_.wait(lock, [&] { return threads_.empty(); });
}

void Process::exit_host_process() {
    std::fflush(stderr);
    std::_Exit(exit_status_);
}

void Process::crash_report(const Stop& stop, GuestThread& thread) const {
    char summary[192];
    if (stop.kind == StopKind::MemoryFault) {
        std::snprintf(summary, sizeof summary, "guest SIGSEGV: %s of 0x%08x, pc 0x%08x",
                      stop.fault_write ? "write" : "read", stop.fault_addr, stop.pc);
    } else {
        std::snprintf(summary, sizeof summary, "guest SIGILL: %s at pc 0x%08x", exception_name(stop.exception),
                      stop.pc);
    }
    log("%s", summary);
    // Recorded before request_exit, so the crash and not the exit status reaches the report.
    runtime_report().note_guest_exit(std::string(summary) + " in " + describe_address(stop.pc));
    const auto& r = thread.regs();
    for (int i = 0; i < 16; i += 4) {
        log("  r%-2d %08x  r%-2d %08x  r%-2d %08x  r%-2d %08x", i, r[i], i + 1, r[i + 1], i + 2, r[i + 2], i + 3, r[i + 3]);
    }
    const long host_tid = ::syscall(SYS_gettid);
    if (thread.tid != 0 && thread.tid != host_tid) {
        log("  cpsr %08x  tls %08x  tid %ld  guest tid %d", thread.cpsr(), thread.tls(), host_tid, thread.tid);
    } else {
        log("  cpsr %08x  tls %08x  tid %ld", thread.cpsr(), thread.tls(), host_tid);
    }
    runtime_report().note_crash_detail("recent-jni-calls", jni_recent_calls());
    runtime_report().note_crash_detail("recent-egl-calls", egl_recent_calls());
    log("  pc in %s", describe_address(stop.pc).c_str());
    log("  lr in %s", describe_address(r[14]).c_str());

    // Bounded, allocation-light crash detail for RuntimeReport: this must never crash inside
    // the crash path, since it runs while the process is already dying.
    char regs_line[256];
    std::snprintf(regs_line, sizeof regs_line,
                  "r0=%08x r1=%08x r2=%08x r3=%08x r4=%08x r5=%08x r6=%08x r7=%08x "
                  "r8=%08x r9=%08x r10=%08x r11=%08x r12=%08x r13=%08x r14=%08x r15=%08x",
                  r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12],
                  r[13], r[14], r[15]);
    runtime_report().note_crash_detail("registers", regs_line);

    char state_line[128];
    if (thread.tid != 0 && thread.tid != host_tid) {
        std::snprintf(state_line, sizeof state_line, "cpsr=%08x tls=%08x tid=%ld guest-tid=%d",
                      thread.cpsr(), thread.tls(), host_tid, thread.tid);
    } else {
        std::snprintf(state_line, sizeof state_line, "cpsr=%08x tls=%08x tid=%ld", thread.cpsr(),
                      thread.tls(), host_tid);
    }
    runtime_report().note_crash_detail("state", state_line);

    runtime_report().note_crash_detail("pc", describe_address(stop.pc));
    runtime_report().note_crash_detail("lr", describe_address(r[14]));

    const std::uint32_t sp = r[13];
    if (mem_.accessible(sp, 32, kPageRead)) {
        const std::uint8_t* stack_bytes = mem_.host_ptr(sp, 32, kPageRead);
        std::uint32_t words[8];
        std::memcpy(words, stack_bytes, sizeof words);
        char stack_line[192];
        std::snprintf(stack_line, sizeof stack_line, "%08x %08x %08x %08x %08x %08x %08x %08x",
                      words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
        runtime_report().note_crash_detail("stack", stack_line);
    } else {
        runtime_report().note_crash_detail("stack", "(unreadable)");
    }

    runtime_report().note_crash_detail("precise", precise_faults_ ? "yes" : "no");
}

}  // namespace zb
