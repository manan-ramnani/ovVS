package ai.ovvs;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;

/** JDK 21 FFM binding over libovvs. Set OVVS_LIBRARY to the DLL/so path. */
public final class OvVS {
  private OvVS() {}

  private static SymbolLookup lookup() {
    String lib = System.getenv("OVVS_LIBRARY");
    if (lib == null || lib.isEmpty()) {
      throw new IllegalStateException("OVVS_LIBRARY is not set");
    }
    Path p = Path.of(lib).toAbsolutePath();
    Path dir = p.getParent();
    if (dir != null) {
      Path ov = dir.resolve("openvino.dll");
      if (ov.toFile().exists()) System.load(ov.toString());
      System.load(p.toString());
    }
    return SymbolLookup.libraryLookup(p, Arena.global());
  }

  public static String getVersion() {
    try {
      var mh =
          Linker.nativeLinker()
              .downcallHandle(
                  lookup().find("ovvsGetVersion").orElseThrow(),
                  FunctionDescriptor.of(ValueLayout.ADDRESS));
      MemorySegment s = (MemorySegment) mh.invoke();
      return s.reinterpret(64).getUtf8String(0);
    } catch (Throwable t) {
      throw new RuntimeException(t);
    }
  }

  public static long[] bruteSearch(float[] data, int n, int dim, float[] query, int k) {
    Linker linker = Linker.nativeLinker();
    SymbolLookup lu = lookup();
    try (Arena arena = Arena.ofConfined()) {
      var create =
          linker.downcallHandle(
              lu.find("ovvsResourcesCreate").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
      var destroy =
          linker.downcallHandle(
              lu.find("ovvsResourcesDestroy").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
      var build =
          linker.downcallHandle(
              lu.find("ovvsBruteForceBuild").orElseThrow(),
              FunctionDescriptor.of(
                  ValueLayout.JAVA_INT,
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_INT,
                  ValueLayout.ADDRESS));
      var search =
          linker.downcallHandle(
              lu.find("ovvsBruteForceSearch").orElseThrow(),
              FunctionDescriptor.of(
                  ValueLayout.JAVA_INT,
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS));
      var bfd =
          linker.downcallHandle(
              lu.find("ovvsBruteForceDestroy").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
      MemorySegment resPtr = arena.allocate(ValueLayout.ADDRESS);
      int st = (int) create.invoke(resPtr);
      if (st != 0) throw new RuntimeException("create");
      MemorySegment res = resPtr.get(ValueLayout.ADDRESS, 0);
      MemorySegment ds = arena.allocateArray(ValueLayout.JAVA_FLOAT, data.length);
      for (int i = 0; i < data.length; i++) ds.setAtIndex(ValueLayout.JAVA_FLOAT, i, data[i]);
      MemorySegment ixPtr = arena.allocate(ValueLayout.ADDRESS);
      st = (int) build.invoke(res, ds, (long) n, (long) dim, 0, ixPtr);
      if (st != 0) throw new RuntimeException("build");
      MemorySegment ix = ixPtr.get(ValueLayout.ADDRESS, 0);
      MemorySegment q = arena.allocateArray(ValueLayout.JAVA_FLOAT, query.length);
      for (int i = 0; i < query.length; i++) q.setAtIndex(ValueLayout.JAVA_FLOAT, i, query[i]);
      MemorySegment nb = arena.allocateArray(ValueLayout.JAVA_LONG, k);
      MemorySegment dist = arena.allocateArray(ValueLayout.JAVA_FLOAT, k);
      st = (int) search.invoke(res, ix, q, 1L, (long) k, MemorySegment.NULL, nb, dist);
      if (st != 0) throw new RuntimeException("search");
      long[] out = new long[k];
      for (int i = 0; i < k; i++) out[i] = nb.getAtIndex(ValueLayout.JAVA_LONG, i);
      bfd.invoke(ix);
      destroy.invoke(res);
      return out;
    } catch (Throwable t) {
      throw new RuntimeException(t);
    }
  }

  public static long[] kmeansPredict(float[] data, int n, int dim, int nclusters) {
    Linker linker = Linker.nativeLinker();
    SymbolLookup lu = lookup();
    try (Arena arena = Arena.ofConfined()) {
      var create =
          linker.downcallHandle(
              lu.find("ovvsResourcesCreate").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
      var destroy =
          linker.downcallHandle(
              lu.find("ovvsResourcesDestroy").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
      var fit =
          linker.downcallHandle(
              lu.find("ovvsKMeansFit").orElseThrow(),
              FunctionDescriptor.of(
                  ValueLayout.JAVA_INT,
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_INT,
                  ValueLayout.JAVA_INT,
                  ValueLayout.ADDRESS));
      var pred =
          linker.downcallHandle(
              lu.find("ovvsKMeansPredict").orElseThrow(),
              FunctionDescriptor.of(
                  ValueLayout.JAVA_INT,
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS));
      var kmd =
          linker.downcallHandle(
              lu.find("ovvsKMeansDestroy").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
      MemorySegment resPtr = arena.allocate(ValueLayout.ADDRESS);
      int st = (int) create.invoke(resPtr);
      if (st != 0) throw new RuntimeException("create");
      MemorySegment res = resPtr.get(ValueLayout.ADDRESS, 0);
      MemorySegment ds = arena.allocateArray(ValueLayout.JAVA_FLOAT, data.length);
      for (int i = 0; i < data.length; i++) ds.setAtIndex(ValueLayout.JAVA_FLOAT, i, data[i]);
      MemorySegment mPtr = arena.allocate(ValueLayout.ADDRESS);
      st = (int) fit.invoke(res, ds, (long) n, (long) dim, nclusters, 8, mPtr);
      if (st != 0) throw new RuntimeException("kmeans fit");
      MemorySegment model = mPtr.get(ValueLayout.ADDRESS, 0);
      MemorySegment labs = arena.allocateArray(ValueLayout.JAVA_LONG, n);
      MemorySegment dist = arena.allocateArray(ValueLayout.JAVA_FLOAT, n);
      st = (int) pred.invoke(res, model, ds, (long) n, labs, dist);
      if (st != 0) throw new RuntimeException("kmeans predict");
      long[] out = new long[n];
      for (int i = 0; i < n; i++) out[i] = labs.getAtIndex(ValueLayout.JAVA_LONG, i);
      kmd.invoke(model);
      destroy.invoke(res);
      return out;
    } catch (Throwable t) {
      throw new RuntimeException(t);
    }
  }

  private static float l2sq(float[] a, float[] b, int off, int dim) {
    float s = 0f;
    for (int i = 0; i < dim; i++) {
      float t = a[i] - b[off + i];
      s += t * t;
    }
    return s;
  }

  public static void main(String[] args) {
    int n = 12, dim = 4, k = 3;
    float[] data = new float[n * dim];
    for (int i = 0; i < data.length; i++) data[i] = ((i * 17) % 100) / 50f - 1f;
    float[] q = new float[] {0.1f, -0.2f, 0.3f, 0.0f};
    long[] nb = bruteSearch(data, n, dim, q, k);
    int[] truth = new int[k];
    float[] td = new float[] {1e30f, 1e30f, 1e30f};
    java.util.Arrays.fill(truth, -1);
    for (int i = 0; i < n; i++) {
      float d = l2sq(q, data, i * dim, dim);
      for (int t = 0; t < k; t++) {
        if (d < td[t]) {
          for (int u = k - 1; u > t; u--) {
            td[u] = td[u - 1];
            truth[u] = truth[u - 1];
          }
          td[t] = d;
          truth[t] = i;
          break;
        }
      }
    }
    for (int t = 0; t < k; t++) {
      if (nb[t] != truth[t]) {
        throw new RuntimeException("mismatch t=" + t + " got=" + nb[t] + " expect=" + truth[t]);
      }
    }
    System.out.println(
        "java consumer ok neighbors=" + nb[0] + "," + nb[1] + "," + nb[2] + " version=" + getVersion());
    long[] labs = kmeansPredict(data, n, dim, 2);
    System.out.println("java kmeans ok label0=" + labs[0]);
  }
}
