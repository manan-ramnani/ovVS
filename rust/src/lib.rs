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
    bf_build: unsafe extern "C" fn(*mut c_void, *const f32, i64, i64, i32, *mut *mut c_void) -> c_int,
    bf_search: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *const f32,
        i64,
        i64,
        *const u8,
        *mut i64,
        *mut f32,
    ) -> c_int,
    bf_destroy: unsafe extern "C" fn(*mut c_void) -> c_int,
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
                bf_build: std::mem::transmute(sym(b"iovsBruteForceBuild\0")?),
                bf_search: std::mem::transmute(sym(b"iovsBruteForceSearch\0")?),
                bf_destroy: std::mem::transmute(sym(b"iovsBruteForceDestroy\0")?),
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

    pub fn brute_search(
        &self,
        data: &[f32],
        n: i64,
        dim: i64,
        query: &[f32],
        k: i64,
    ) -> Result<(Vec<i64>, Vec<f32>), String> {
        let mut res = std::ptr::null_mut();
        unsafe {
            if (self.create)(&mut res) != 0 {
                return Err("create".into());
            }
            let mut ix = std::ptr::null_mut();
            let rc = (self.bf_build)(res, data.as_ptr(), n, dim, 0, &mut ix);
            if rc != 0 {
                (self.destroy)(res);
                return Err(format!("build rc={rc}"));
            }
            let mut nb = vec![0i64; k as usize];
            let mut ds = vec![0f32; k as usize];
            let rc = (self.bf_search)(
                res,
                ix,
                query.as_ptr(),
                1,
                k,
                std::ptr::null(),
                nb.as_mut_ptr(),
                ds.as_mut_ptr(),
            );
            (self.bf_destroy)(ix);
            (self.destroy)(res);
            if rc != 0 {
                return Err(format!("search rc={rc}"));
            }
            Ok((nb, ds))
        }
    }
}
