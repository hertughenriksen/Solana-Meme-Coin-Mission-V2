fn main() {
    // Allow undefined glibc symbols that are provided by the system at runtime
    println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");
    
    // Link against the C library
    println!("cargo:rustc-link-lib=c");
    
    // For WSL/Linux systems, ensure we're linking against the system's glibc
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,--as-needed");
    }
}
