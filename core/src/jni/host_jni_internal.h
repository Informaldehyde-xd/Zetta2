#pragma once

// Shared state of HostJni.
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "zb/host_jni.h"
#include "zb/jni_handles.h"
#include "zb/jni_hostcalls.h"
#include "zb/jni_protocol.h"

namespace zb {

// JNI state of one host thread.
struct JniThread {
    HostJni::Impl* owner = nullptr;
    JniBackend::Env env = 0;
    std::uint32_t guest_env = 0;
    bool attached = false;
    int native_depth = 0;
    int user_frames = 0;
    LocalHandles locals;
    std::unique_ptr<LibraryRuntime::Carrier> carrier;

    ~JniThread();
};

// One flat JNI host call.
class JniCall {
public:
    JniCall(HostJni::Impl& jni, GuestThread& thread, JniThread& state, std::uint32_t index)
        : jni_(jni), thread_(thread), state_(state), index_(index),
          regs_{thread.regs()[0], thread.regs()[1], thread.regs()[2], thread.regs()[3]} {
        thread.regs()[0] = 0;
        thread.regs()[1] = 0;
    }

    std::uint32_t index() const { return index_; }
    std::uint32_t arg(unsigned position) const;
    void set(std::uint32_t value) { thread_.regs()[0] = value; }
    void set64(std::uint64_t value) {
        thread_.regs()[0] = static_cast<std::uint32_t>(value);
        thread_.regs()[1] = static_cast<std::uint32_t>(value >> 32);
    }
    JniThread& state() { return state_; }
    JniBackend::Env env();

private:
    HostJni::Impl& jni_;
    GuestThread& thread_;
    JniThread& state_;
    std::uint32_t index_;
    std::uint32_t regs_[4];
};

struct HostJni::Impl {
    Impl(LibraryRuntime& runtime, JniBackend& backend, std::size_t slot_capacity)
        : runtime(runtime), backend(backend), slots(slot_capacity) {}

    std::string last_string_utf;
    HostJni* owner = nullptr;
    LibraryRuntime& runtime;
    JniBackend& backend;
    std::atomic<bool> ready{false};
    zb_jni_guest_api api{};

    GlobalHandles globals{HandleKind::Global};
    GlobalHandles weaks{HandleKind::WeakGlobal};
    IdTable methods;
    IdTable fields;
    
    // --- DPI OVERRIDE TRACKER ---
    std::atomic<std::uint32_t> dpi_field_id{0};
    // ----------------------------

    std::mutex shorty_mutex;
    std::vector<std::string> shorties;
    NativeSlots slots;
    std::atomic<bool> logged_foreign_buffer{false};

    struct BufferMirror {
        std::uint32_t guest = 0;   
        std::uint64_t size = 0;    
        JniBackend::Ref global = 0;
        const void* host = nullptr; // Added for universal buffer tracking
    };
    static constexpr std::uint64_t kMirrorCapBytes = 64u * 1024u * 1024u;
    std::mutex mirror_mutex;
    
    // FIXED: Keyed by Java object reference, not host pointer, to prevent aliasing
    std::unordered_map<JniBackend::Ref, BufferMirror> mirrors;
    
    std::uint64_t mirrored_bytes = 0;
    unsigned mirror_failures = 0;

    JniThread& thread();
    [[noreturn]] void fatal(JniBackend::Env env, const char* fmt, ...) __attribute__((format(printf, 3, 4)));

    JniBackend::Ref resolve(JniThread& state, std::uint32_t handle, const char* function);
    std::uint32_t local(JniThread& state, JniBackend::Ref ref) { return state.locals.add(ref); }
    std::uint32_t intern_method(JniBackend::Id id, const std::string& shorty);
    std::optional<std::string> method_shorty(std::uint32_t id);
    JniBackend::Id method_id(JniBackend::Env env, std::uint32_t id, const char* function);
    JniBackend::Id field_id(JniBackend::Env env, std::uint32_t id, const char* function);

    std::string read_string(JniBackend::Env env, std::uint32_t address, const char* function);
    const std::uint8_t* readable(JniBackend::Env env, std::uint32_t address, std::uint64_t size, const char* function);
    std::uint8_t* writable(JniBackend::Env env, std::uint32_t address, std::uint64_t size, const char* function);
    void write_shorty(JniBackend::Env env, std::uint32_t address, const std::string& shorty, const char* function);

    std::optional<GuestResult> invoke(JniThread& state, std::uint32_t function, const GuestCall& args);
    bool ensure_guest_env(JniThread& state);

    std::uint32_t mirror_direct_buffer(JniBackend::Env env, JniBackend::Ref buffer, const void* host, std::int64_t capacity, const char*& failure);
    void flush_buffer_mirrors(JniBackend::Env env);

    bool serve_objects(JniCall& call);
    bool serve_values(JniCall& call);
    bool serve_data(JniCall& call);
    bool serve_natives(JniCall& call);
    bool serve_vm(JniCall& call);
};

const char* jni_host_call_name(std::uint32_t index);
void install_native_dispatcher(HostJni::Impl* jni);

}  // namespace zb
