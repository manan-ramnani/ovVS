//! Loads libiovs at runtime and calls the C ABI.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::Path;

#[cfg(windows)]
extern "system" {
    fn LoadLibraryA(name: *const u8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
}

pub struct Lib {
    gemm: unsafe extern "C" fn(*mut c_void, *const f32, *const f32, *mut f32, i64, i64, i64, i32) -> c_int,
    create: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    destroy: unsafe extern "C" fn(*mut c_void) -> c_int,
    version: unsafe extern "C" fn() -> *const c_char,
}

impl Lib {
    pub fn load(path: &Path) -> Result<Self, String> {
        #[cfg(windows)]
        unsafe {
            let mut bytes = path.to_string_lossy().into_owned().into_bytes();
            bytes.push(0);
            let h = LoadLibraryA(bytes.as_ptr());
            if h.is_null() {
                return Err(format!("LoadLibrary failed for {}", path.display()));
            }
            let sym = |name: &[u8]| {
                let p = GetProcAddress(h, name.as_ptr());
                if p.is_null() {
                    Err(format!("missing symbol {}", String::from_utf8_lossy(name)))
                } else {
                    Ok(p)
                }
            };
            Ok(Self {
                version: std::mem::transmute(sym(b"iovsGetVersion\0")?),
                create: std::mem::transmute(sym(b"iovsResourcesCreate\0")?),
                destroy: std::mem::transmute(sym(b"iovsResourcesDestroy\0")?),
                gemm: std::mem::transmute(sym(b"iovsGemm\0")?),
            })
        }
        #[cfg(not(windows))]
        {
            let _ = path;
            Err("libloading implemented for Windows in this crate".into())
        }
    }

    pub fn version(&self) -> String {
        unsafe { CStr::from_ptr((self.version)()).to_string_lossy().into_owned() }
    }

    pub fn gemm(&self, a: &[f32], b: &[f32], m: i64, n: i64, k: i64) -> Result<Vec<f32>, String> {
        let mut res = std::ptr::null_mut();
        unsafe {
            if (self.create)(&mut res) != 0 {
                return Err("create".into());
            }
            let mut c = vec![0f32; (m * n) as usize];
            let rc = (self.gemm)(res, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, 1);
            (self.destroy)(res);
            if rc != 0 {
                return Err(format!("gemm rc={rc}"));
            }
            Ok(c)
        }
    }
}
