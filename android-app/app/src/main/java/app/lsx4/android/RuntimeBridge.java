package app.lsx4.android;

import android.view.Surface;

public final class RuntimeBridge {
    public static final int MANAGED_OPTIMIZATION_ARM_GPU_FAST_PATH = 1;
    public static final int MANAGED_OPTIMIZATION_COARSE_FRAGMENT_2X2 = 2;
    public static final int MANAGED_OPTIMIZATION_DISABLE_VK_ROBUSTNESS = 3;

    static {
        System.loadLibrary("lsx_runtime_bridge");
    }

    private RuntimeBridge() {
    }

    public static native String load(String path);
    public static native String loadPs5(String path);
    public static native String ps5Abi();
    public static native String ps5Status();

    public static native int extractRarArchive(
            String archivePath, String destinationPath, long maximumBytes, int maximumFiles);
    public static native int initializePs5(String rootDir, String userId);
    public static native int attachPs5Surface(Surface surface);
    public static native int detachPs5Surface();
    public static native int scanPs5Game(String path);
    public static native int launchPs5GameJit(String path);
    public static native int setPs5PadButton(int buttonMask, boolean pressed);
    public static native int setPs5PadAxis(int axis, int value);
    public static native String abi();
    public static native String version();
    public static native String systemInfo();
    public static native String firmwareStatus();
    public static native int initialize(String rootDir, String userId);
    public static native int installTranslatorHelper(String sourceDir);
    public static native int translatorRunGuestElf(String path);
    public static native int elfBox64ServiceRun(String elfPath, String translatorDir,
                                                String envBlob, String embedSoPath);
    public static native int startServicePresentPump(String translatorDir);
    public static native int stopServicePresentPump();
    public static native String probeDlopen(String path);
    public static native int loadEmbeddedBox64(String path);
    public static native String embeddedBox64Info();
    public static native int validateEmbeddedBox64MappedEntry();
    public static native int runEmbeddedBox64MappedEntry();
    public static native String embeddedBox64MappedEntryInfo();
    public static native int setPadButton(int buttonMask, boolean pressed);
    public static native int sendSystemUiText(String text);
    public static native int isSystemUiTextActive();
    public static native int submitSystemUiText();
    public static native int backspaceSystemUiText(int count);
    public static native int setPadAxis(int axis, int value);
    public static native int setTouchPad(boolean pressed, float x, float y);
    public static native int audioInit();
    public static native int setAudioEnabled(boolean enabled);
    public static native int setManagedOptimization(int option, boolean enabled);
    public static native int audioProbeTone();
    public static native String audioStatus();
    public static native int attachSurface(Surface surface);
    public static native int detachSurface();
    public static native String surfaceInfo();
    public static native int presentTestPattern(int width, int height);
    public static native int presentHomebrewLoaderFrame(int width, int height);
    public static native int presentFrameDumpFile(String path);
    public static native int vortekRingSelfTest();
    public static native int relocateImports();
    public static native int prepareBox64EntryTrampoline();
    public static native String box64EntryRequest();
    public static native String status();
    public static native String jitStatus();
    public static native long[] runtimeHudStats();
    public static native String jitSelfTest();
    public static native int scanGame(String path);
    public static native int launchGame(String path);
    public static native int launchGameJit(String path);
    public static native int prepareTranslatorLaunch(String path);
    public static native int reportTranslatorResult(int helperExitCode);
    public static native int reportTranslatorResultJson(String resultJson);
    public static native String translatorLaunchRequest();
}
