package app.lsx4.android;

import android.content.Context;
import android.content.pm.ApplicationInfo;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

final class TestModeManager {
    enum Activation {
        INSTALLED,
        MANAGED_MODULE_READY,
        USER_MODULE_READY
    }

    static final String MODULE_NAME = "libSceNgs2.sprx";
    private static final String ASSET_PATH = "test_mode/" + MODULE_NAME;
    private static final String MARKER_NAME = ".lsx4-test-mode-ngs2.sha256";
    private static final long MAX_MODULE_BYTES = 128L * 1024L * 1024L;

    private TestModeManager() {
    }

    static boolean isDebugBuild(Context context) {
        return (context.getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
    }

    static boolean hasBundledModule(Context context) {
        if (!isDebugBuild(context)) {
            return false;
        }
        try (InputStream ignored = context.getAssets().open(ASSET_PATH)) {
            return true;
        } catch (IOException ignored) {
            return false;
        }
    }

    static void reconcileBuildScope(Context context) {
        if (!isDebugBuild(context)) {
            SettingsActivity.prefs(context).edit()
                    .putBoolean(SettingsActivity.K_TEST_MODE, false).apply();
            removeManagedModule(context);
            return;
        }
        if (!SettingsActivity.prefs(context).getBoolean(SettingsActivity.K_TEST_MODE, false)) {
            return;
        }
        try {
            activate(context);
        } catch (IOException error) {
            SettingsActivity.prefs(context).edit()
                    .putBoolean(SettingsActivity.K_TEST_MODE, false).apply();
        }
    }

    static Activation activate(Context context) throws IOException {
        if (!isDebugBuild(context)) {
            throw new IOException("test mode is unavailable in this build");
        }

        File destination = moduleFile(context);
        File marker = markerFile(context);
        if (isSupportedModule(destination)) {
            return marker.isFile()
                    ? Activation.MANAGED_MODULE_READY
                    : Activation.USER_MODULE_READY;
        }

        File directory = destination.getParentFile();
        if (directory == null || (!directory.isDirectory() && !directory.mkdirs())) {
            throw new IOException("could not create the system module directory");
        }

        File temporary = new File(directory, "." + MODULE_NAME + ".test-mode-importing");
        long copied = 0;
        MessageDigest digest = sha256Digest();
        try (InputStream input = context.getAssets().open(ASSET_PATH);
             FileOutputStream output = new FileOutputStream(temporary)) {
            byte[] buffer = new byte[256 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                if (count == 0) {
                    continue;
                }
                if (copied > MAX_MODULE_BYTES - count) {
                    throw new IOException("the bundled NGS2 module is too large");
                }
                output.write(buffer, 0, count);
                digest.update(buffer, 0, count);
                copied += count;
            }
            output.getFD().sync();
        } catch (IOException error) {
            temporary.delete();
            throw error;
        }

        if (copied < 64 || !isSupportedModule(temporary)) {
            temporary.delete();
            throw new IOException("the bundled NGS2 module is not a decrypted PS4 module");
        }

        moveReplacing(temporary, destination);
        try {
            writeMarker(marker, toHex(digest.digest()));
        } catch (IOException error) {
            destination.delete();
            throw error;
        }
        return Activation.INSTALLED;
    }

    static boolean removeManagedModule(Context context) {
        File marker = markerFile(context);
        if (!marker.isFile()) {
            return false;
        }

        boolean removed = false;
        File destination = moduleFile(context);
        try {
            String expected = new String(Files.readAllBytes(marker.toPath()),
                    StandardCharsets.US_ASCII).trim();
            if (destination.isFile() && expected.equalsIgnoreCase(sha256(destination))) {
                removed = destination.delete();
            }
        } catch (IOException ignored) {
        }
        marker.delete();
        return removed;
    }

    static void forgetManagedModule(Context context) {
        markerFile(context).delete();
    }

    private static File moduleFile(Context context) {
        return new File(new File(new File(context.getFilesDir(), "lsx4-home"),
                "sys_modules"), MODULE_NAME);
    }

    private static File markerFile(Context context) {
        return new File(moduleFile(context).getParentFile(), MARKER_NAME);
    }

    private static boolean isSupportedModule(File module) throws IOException {
        if (!module.isFile() || module.length() < 64) {
            return false;
        }
        byte[] magic = new byte[4];
        try (FileInputStream input = new FileInputStream(module)) {
            if (input.read(magic) != magic.length) {
                return false;
            }
        }
        boolean elf = (magic[0] & 0xff) == 0x7f
                && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F';
        boolean ps4Self = (magic[0] & 0xff) == 0x4f
                && (magic[1] & 0xff) == 0x15
                && (magic[2] & 0xff) == 0x3d
                && (magic[3] & 0xff) == 0x1d;
        return elf || ps4Self;
    }

    private static void moveReplacing(File source, File destination) throws IOException {
        try {
            Files.move(source.toPath(), destination.toPath(),
                    StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(source.toPath(), destination.toPath(),
                    StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void writeMarker(File marker, String value) throws IOException {
        File temporary = new File(marker.getParentFile(), marker.getName() + ".writing");
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(value.getBytes(StandardCharsets.US_ASCII));
            output.getFD().sync();
        }
        moveReplacing(temporary, marker);
    }

    private static String sha256(File file) throws IOException {
        MessageDigest digest = sha256Digest();
        try (FileInputStream input = new FileInputStream(file)) {
            byte[] buffer = new byte[256 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                if (count > 0) {
                    digest.update(buffer, 0, count);
                }
            }
        }
        return toHex(digest.digest());
    }

    private static MessageDigest sha256Digest() throws IOException {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new IOException("SHA-256 is unavailable", error);
        }
    }

    private static String toHex(byte[] bytes) {
        char[] alphabet = "0123456789abcdef".toCharArray();
        char[] result = new char[bytes.length * 2];
        for (int index = 0; index < bytes.length; ++index) {
            int value = bytes[index] & 0xff;
            result[index * 2] = alphabet[value >>> 4];
            result[index * 2 + 1] = alphabet[value & 0x0f];
        }
        return new String(result);
    }
}
