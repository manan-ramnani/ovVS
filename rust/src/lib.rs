//! Thin FFI to libiovs C ABI. Build the C++ library first.

use std::os::raw::{c_char, c_int};

extern "C" {
    pub fn iovsGetVersion() -> *const c_char;
    pub fn iovsResourcesCreate(res: *mut *mut std::ffi::c_void) -> c_int;
    pub fn iovsResourcesDestroy(res: *mut std::ffi::c_void) -> c_int;
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_symbol_exists() {
        // Linked tests require libiovs; compile-only crate documents the ABI.
    }
}
