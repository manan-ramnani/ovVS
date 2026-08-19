package ai.iovs;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;

/** JDK 21 FFM binding over libiovs. Set IOVS_LIBRARY to the DLL/so path. */
public final class IoVS {
  private IoVS() {}

  public static String getVersion() {
    try {
      String lib = System.getenv("IOVS_LIBRARY");
      if (lib == null || lib.isEmpty()) {
        throw new IllegalStateException("IOVS_LIBRARY is not set");
      }
      SymbolLookup lookup = SymbolLookup.libraryLookup(Path.of(lib), Arena.global());
      var mh =
          Linker.nativeLinker()
              .downcallHandle(
                  lookup.find("iovsGetVersion").orElseThrow(),
                  FunctionDescriptor.of(ValueLayout.ADDRESS));
      MemorySegment s = (MemorySegment) mh.invoke();
      return s.reinterpret(64).getUtf8String(0);
    } catch (Throwable t) {
      throw new RuntimeException(t);
    }
  }

  public static void main(String[] args) {
    System.out.println("java consumer ok version=" + getVersion());
  }
}
