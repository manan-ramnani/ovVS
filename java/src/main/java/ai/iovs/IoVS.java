package ai.iovs;

/** Panama/JNI placeholder wrapping the C ABI. */
public final class IoVS {
  static {
    System.loadLibrary("iovs");
  }

  private IoVS() {}

  public static native String getVersion();
}
