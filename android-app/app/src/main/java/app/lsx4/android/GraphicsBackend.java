package app.lsx4.android;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Stable UI-to-renderer contract. Vulkan is the current implementation. OpenGL owns a distinct
 * configuration path now, so its renderer can be added without branching throughout activities.
 */
public interface GraphicsBackend {
    String id();

    boolean isImplemented();

    void writeLaunchConfiguration(Context context) throws Exception;

    static GraphicsBackend selected(Context context) {
        String id = SettingsActivity.prefs(context).getString(
                SettingsActivity.K_GPU_BACKEND, VulkanBackend.ID);
        if (OpenGlBackend.ID.equals(id)) {
            SettingsActivity.prefs(context).edit()
                    .putString(SettingsActivity.K_GPU_BACKEND, VulkanBackend.ID)
                    .apply();
        }
        return new VulkanBackend();
    }

    static void writeConfig(Context context, String id) throws Exception {
        File directory = new File(context.getFilesDir(), "runtime-config");
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IllegalStateException("Cannot create " + directory);
        }
        File target = new File(directory, "android-gpu-backend.ini");
        try (FileOutputStream output = new FileOutputStream(target, false)) {
            output.write(("backend=" + id + "\n").getBytes(StandardCharsets.UTF_8));
        }
    }

    final class VulkanBackend implements GraphicsBackend {
        static final String ID = "vulkan";

        @Override
        public String id() {
            return ID;
        }

        @Override
        public boolean isImplemented() {
            return true;
        }

        @Override
        public void writeLaunchConfiguration(Context context) throws Exception {
            writeConfig(context, ID);
        }
    }

    final class OpenGlBackend implements GraphicsBackend {
        static final String ID = "opengl";

        @Override
        public String id() {
            return ID;
        }

        @Override
        public boolean isImplemented() {
            return false;
        }

        @Override
        public void writeLaunchConfiguration(Context context) throws Exception {
            writeConfig(context, ID);
        }
    }
}
