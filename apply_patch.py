import os

def patch_file(filepath, target, replacement):
    if not os.path.exists(filepath):
        print(f"[-] File not found: {filepath}")
        return
    with open(filepath, 'r') as f:
        content = f.read()
    
    if replacement in content:
        print(f"[*] Already patched: {filepath}")
        return

    if target in content:
        content = content.replace(target, replacement)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"[+] Successfully patched {filepath}")
    else:
        print(f"[-] Could not find target string in {filepath}. You may need to patch manually.")

# --- Patch 1: host_jni_objects.cpp (FindClass) ---
target_find = """        const std::string class_name = read_string(env, call.arg(0), name);
        call.set(local(state, backend.find_class(env, class_name.c_str())));"""
replacement_find = """        const std::string class_name = read_string(env, call.arg(0), name);
        JniBackend::Ref class_ref = backend.find_class(env, class_name.c_str());
        zb::log("[ZB-CLASS] FindClass: %s -> ref: %lu", class_name.c_str(), class_ref);
        call.set(local(state, class_ref));"""
patch_file("core/src/jni/host_jni_objects.cpp", target_find, replacement_find)

# --- Patch 2: host_jni_objects.cpp (ThrowNew) ---
target_throw = """        const std::string message = call.arg(1) != 0 ? read_string(env, call.arg(1), name) : std::string();"""
replacement_throw = """        const std::string message = call.arg(1) != 0 ? read_string(env, call.arg(1), name) : std::string();
        zb::log("[ZB-THROW] ThrowNew: %s", message.c_str());"""
patch_file("core/src/jni/host_jni_objects.cpp", target_throw, replacement_throw)

# --- Patch 3: host_jni_values.cpp (CallMethodA) ---
target_call = """        const std::optional<std::string> shorty = method_shorty(method);
        if (!shorty) fatal(env, "CallMethodA: jmethodID 0x%08x has no shorty", method);"""
replacement_call = """        const std::optional<std::string> shorty = method_shorty(method);
        if (!shorty) fatal(env, "CallMethodA: jmethodID 0x%08x has no shorty", method);
        zb::log("[ZB-CALL] Method Call: kind=%u type=%c shorty=%s", kind, type, shorty->c_str());"""
patch_file("core/src/jni/host_jni_values.cpp", target_call, replacement_call)

# --- Patch 4: host_jni_values.cpp (Fix the LL and null crash) ---
target_if = """        if (type == 'L') {
            call.set(local(state, result.l));
        } else if (type != 'V') {"""
replacement_if = """        if (type == 'L') {
            // Generic intercept for static String calls (like System.getProperty)
            if (kind == ZB_JNI_CALL_STATIC && *shorty == "LL") {
                zb::log("[ZB-FIX] Intercepted static String call! Injecting correct APK path.");
                JniBackend::Ref fake_path = backend.new_string_utf(env, "/data/user/0/com.zettabridge.launcher/files/plugins/com.galapagossoft.trialdemo/base.apk");
                call.set(local(state, fake_path));
            } else {
                call.set(local(state, result.l));
            }
        } else if (type != 'V') {"""
patch_file("core/src/jni/host_jni_values.cpp", target_if, replacement_if)

print("\\n[+] All patches applied. You can now commit and push.")
