package app.lsx4.android;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;

final class GameCacheManager {
    static final String ENABLE_MARKER = "run-jit-persistent-jit-cache";

    private static final String LEGACY_FORCE_MARKER =
            "run-jit-persistent-jit-cache-force";
    private static final String SETTING_FILE =
            "persistent-jit-cache-setting-v1";
    private static final Pattern TITLE_ID =
            Pattern.compile("[A-Z]{4}[0-9]{5}");
    // Native PS4 fallback keys are TITLE_<hex>. PS5 keys are the
    // upper-cased, filesystem-safe game-directory/configured-title text.
    // Keep the exact spelling for deletion, but reject separators, dot
    // entries and names which cannot be a single filesystem component.
    private static final Pattern CACHE_KEY =
            Pattern.compile("[A-Za-z0-9][A-Za-z0-9_-]{0,254}");
    private static final int MAX_METADATA_BYTES = 1024 * 1024;

    static final class Entry {
        final String titleId;
        final String title;
        final File icon;
        final long jitBytes;
        final long gpuBytes;
        final List<String> cacheKeys;

        Entry(String titleId, String title, File icon, long jitBytes,
              long gpuBytes, List<String> cacheKeys) {
            this.titleId = titleId;
            this.title = title;
            this.icon = icon;
            this.jitBytes = jitBytes;
            this.gpuBytes = gpuBytes;
            this.cacheKeys = cacheKeys;
        }

        long totalBytes() {
            return saturatingAdd(jitBytes, gpuBytes);
        }
    }

    private static final class Metadata {
        final String titleId;
        final String title;
        final File icon;
        final Set<String> aliases = new HashSet<>();

        Metadata(String titleId, String title, File icon) {
            this.titleId = titleId;
            this.title = title;
            this.icon = icon;
            aliases.add(titleId);
        }
    }

    private static final class MutableEntry {
        final String titleId;
        String title;
        File icon;
        long jitBytes;
        long gpuBytes;
        final Set<String> cacheKeys = new TreeSet<>();

        MutableEntry(String titleId) {
            this.titleId = titleId;
            this.title = titleId;
        }

        Entry freeze() {
            return new Entry(titleId, title, icon, jitBytes, gpuBytes,
                    new ArrayList<>(cacheKeys));
        }
    }

    private static final class CacheAmounts {
        long jitBytes;
        long gpuBytes;
    }

    private GameCacheManager() {
    }

    static File home(Context context) {
        return new File(context.getFilesDir(), "lsx4-home");
    }

    static File enableMarker(Context context) {
        return new File(home(context), ENABLE_MARKER);
    }

    static boolean applyStoredSetting(Context context,
                                      SharedPreferences preferences)
            throws IOException {
        File root = home(context);
        ensureDirectory(root);
        Boolean stored = readExplicitSetting(root);
        boolean enabled = stored != null
                ? stored
                : preferences.getBoolean(
                        SettingsActivity.K_PERSISTENT_JIT_CACHE, true);
        if (stored == null) {
            writeExplicitSetting(root, enabled);
        }
        if (!preferences.contains(SettingsActivity.K_PERSISTENT_JIT_CACHE)
                || preferences.getBoolean(
                        SettingsActivity.K_PERSISTENT_JIT_CACHE, true)
                        != enabled) {
            preferences.edit().putBoolean(
                    SettingsActivity.K_PERSISTENT_JIT_CACHE, enabled).commit();
        }
        applyMarker(root, enabled);
        return enabled;
    }

    static boolean configuredEnabled(Context context,
                                     SharedPreferences preferences) {
        try {
            Boolean stored = readExplicitSetting(home(context));
            if (stored != null) {
                return stored;
            }
        } catch (IOException ignored) {
        }
        return preferences.getBoolean(
                SettingsActivity.K_PERSISTENT_JIT_CACHE, true);
    }

    static void setUserEnabled(Context context, boolean enabled)
            throws IOException {
        File root = home(context);
        ensureDirectory(root);
        writeExplicitSetting(root, enabled);
        applyMarker(root, enabled);
    }

    static List<Entry> scan(Context context) throws IOException {
        File root = home(context);
        Map<String, Metadata> metadataById = loadInstalledMetadata(root);
        Map<String, Metadata> metadataByAlias = new HashMap<>();
        for (Metadata metadata : metadataById.values()) {
            for (String alias : metadata.aliases) {
                metadataByAlias.putIfAbsent(alias, metadata);
            }
        }

        Map<String, CacheAmounts> amountsByKey = new HashMap<>();
        File jitRoot = new File(new File(root, "cache"), "jit");
        collectDirectoryCache(jitRoot, true, amountsByKey);
        File gpuRoot = new File(new File(root, "native"), "cache");
        collectDirectoryCache(gpuRoot, false, amountsByKey);
        collectFingerprintCache(
                new File(new File(root, "cache"), "jit-fingerprints"),
                amountsByKey);

        Map<String, MutableEntry> grouped = new HashMap<>();
        for (Map.Entry<String, CacheAmounts> amount : amountsByKey.entrySet()) {
            CacheAmounts bytes = amount.getValue();
            if (bytes.jitBytes == 0 && bytes.gpuBytes == 0) {
                continue;
            }
            Metadata metadata = metadataByAlias.get(amount.getKey());
            if (metadata == null) {
                metadata = metadataByAlias.get(
                        amount.getKey().toUpperCase(Locale.ROOT));
            }
            String titleId = metadata == null
                    ? amount.getKey() : metadata.titleId;
            MutableEntry entry = grouped.get(titleId);
            if (entry == null) {
                entry = new MutableEntry(titleId);
                if (metadata == null) {
                    entry.title = context.getString(
                            R.string.cache_unknown_game, titleId);
                }
                grouped.put(titleId, entry);
            }
            entry.cacheKeys.add(amount.getKey());
            entry.jitBytes = saturatingAdd(entry.jitBytes, bytes.jitBytes);
            entry.gpuBytes = saturatingAdd(entry.gpuBytes, bytes.gpuBytes);
            if (metadata != null) {
                entry.title = metadata.title;
                entry.icon = metadata.icon;
            }
        }

        List<Entry> entries = new ArrayList<>(grouped.size());
        for (MutableEntry entry : grouped.values()) {
            entries.add(entry.freeze());
        }
        entries.sort(Comparator
                .comparing((Entry entry) ->
                        entry.title.toLowerCase(Locale.ROOT))
                .thenComparing(entry -> entry.titleId));
        return entries;
    }

    static void delete(Context context, Entry entry) throws IOException {
        if (entry == null || entry.cacheKeys.isEmpty()) {
            return;
        }
        File root = home(context);
        File jitRoot = new File(new File(root, "cache"), "jit");
        File fingerprintRoot =
                new File(new File(root, "cache"), "jit-fingerprints");
        File gpuRoot = new File(new File(root, "native"), "cache");
        for (String key : entry.cacheKeys) {
            if (!isSafeCacheKey(key)) {
                throw new IOException("Unsafe cache key: " + key);
            }
            deleteExactChild(jitRoot, key, true);
            deleteExactChild(gpuRoot, key, true);
            deleteExactChild(fingerprintRoot, key + ".txt", false);
        }
    }

    private static Boolean readExplicitSetting(File root) throws IOException {
        File setting = new File(root, SETTING_FILE);
        if (!isRegularFile(setting)) {
            return null;
        }
        byte[] data = readFileLimited(setting, 16);
        String value = new String(data, StandardCharsets.US_ASCII).trim();
        if ("1".equals(value)) {
            return true;
        }
        if ("0".equals(value)) {
            return false;
        }
        throw new IOException("Invalid persistent-cache setting");
    }

    private static void writeExplicitSetting(File root, boolean enabled)
            throws IOException {
        atomicWrite(new File(root, SETTING_FILE),
                enabled ? "1\n" : "0\n");
    }

    private static void applyMarker(File root, boolean enabled)
            throws IOException {
        File legacyForce = new File(root, LEGACY_FORCE_MARKER);
        if (legacyForce.exists() && !legacyForce.delete()) {
            throw new IOException(
                    "Cannot remove obsolete cache force marker");
        }
        File marker = new File(root, ENABLE_MARKER);
        if (enabled) {
            if (marker.exists() && !isRegularFile(marker)) {
                throw new IOException("Cache marker is not a regular file");
            }
            if (!marker.isFile()) {
                atomicWrite(marker, "1\n");
            }
        } else if (marker.exists() && !marker.delete()) {
            throw new IOException("Cannot remove cache marker");
        }
    }

    private static void atomicWrite(File target, String value)
            throws IOException {
        File parent = target.getParentFile();
        ensureDirectory(parent);
        File temporary = new File(parent,
                "." + target.getName() + ".tmp-" +
                        Long.toUnsignedString(System.nanoTime()));
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(value.getBytes(StandardCharsets.US_ASCII));
            output.getFD().sync();
        } catch (IOException error) {
            temporary.delete();
            throw error;
        }
        try {
            Files.move(temporary.toPath(), target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(temporary.toPath(), target.toPath(),
                    StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static Map<String, Metadata> loadInstalledMetadata(File root) {
        Map<String, Metadata> result = new HashMap<>();
        File runtimeApps = new File(
                new File(new File(root, "runtime-fs"), "user"), "app");
        loadMetadataRoot(runtimeApps, result);
        loadMetadataRoot(new File(root, "games"), result);
        return result;
    }

    private static void loadMetadataRoot(File root,
                                         Map<String, Metadata> result) {
        File[] children = root.listFiles();
        if (children == null) {
            return;
        }
        for (File directory : children) {
            if (!isDirectory(directory)) {
                continue;
            }
            Metadata metadata = readMetadata(directory);
            if (metadata == null) {
                continue;
            }
            Metadata previous = result.get(metadata.titleId);
            if (previous == null
                    || (previous.icon == null && metadata.icon != null)
                    || (previous.title.equals(previous.titleId)
                            && !metadata.title.equals(metadata.titleId))) {
                if (previous != null) {
                    metadata.aliases.addAll(previous.aliases);
                }
                result.put(metadata.titleId, metadata);
            } else {
                previous.aliases.addAll(metadata.aliases);
            }
        }
    }

    private static Metadata readMetadata(File directory) {
        try {
            String directoryId = normalizeTitleId(directory.getName());
            File sceSys = new File(directory, "sce_sys");
            File paramJson = new File(sceSys, "param.json");
            File paramSfo = new File(sceSys, "param.sfo");
            File manifest = new File(directory, "executor-installed.json");
            String titleId = null;
            String title = null;
            String contentId = null;

            if (isRegularFile(paramJson)) {
                JSONObject json = new JSONObject(new String(
                        readFileLimited(paramJson, MAX_METADATA_BYTES),
                        StandardCharsets.UTF_8));
                titleId = normalizeTitleId(json.optString("titleId", ""));
                JSONObject localized =
                        json.optJSONObject("localizedParameters");
                if (localized != null) {
                    String language =
                            localized.optString("defaultLanguage", "");
                    JSONObject selected =
                            localized.optJSONObject(language);
                    if (selected == null) {
                        java.util.Iterator<String> keys = localized.keys();
                        while (keys.hasNext() && selected == null) {
                            Object candidate = localized.opt(keys.next());
                            if (candidate instanceof JSONObject) {
                                selected = (JSONObject) candidate;
                            }
                        }
                    }
                    if (selected != null) {
                        title = clean(
                                selected.optString("titleName", ""));
                    }
                }
            }
            if (isRegularFile(paramSfo)) {
                byte[] data =
                        readFileLimited(paramSfo, MAX_METADATA_BYTES);
                if (titleId == null) {
                    titleId = normalizeTitleId(
                            PkgInstaller.sfoString(data, "TITLE_ID"));
                }
                if (title == null) {
                    title = clean(
                            PkgInstaller.sfoString(data, "TITLE"));
                }
                contentId = clean(
                        PkgInstaller.sfoString(data, "CONTENT_ID"));
            }
            if (isRegularFile(manifest)
                    && (titleId == null || title == null)) {
                JSONObject json = new JSONObject(new String(
                        readFileLimited(manifest, MAX_METADATA_BYTES),
                        StandardCharsets.UTF_8));
                if (titleId == null) {
                    titleId = normalizeTitleId(
                            json.optString("titleId", ""));
                }
                if (title == null) {
                    title = clean(json.optString("title", ""));
                }
            }
            if (titleId == null) {
                titleId = directoryId;
            }
            if (titleId == null) {
                return null;
            }
            if (title == null) {
                title = titleId;
            }
            File icon = new File(sceSys, "icon0.png");
            Metadata metadata = new Metadata(
                    titleId, title, isRegularFile(icon) ? icon : null);
            addArtifactAlias(metadata, directory.getName());
            addArtifactAlias(metadata, title);
            if (contentId != null && contentId.length() >= 16) {
                String serial = normalizeTitleId(
                        contentId.substring(7, 16));
                if (serial != null) {
                    metadata.aliases.add(serial);
                }
            }
            if (directoryId != null) {
                metadata.aliases.add(directoryId);
            }
            return metadata;
        } catch (Exception ignored) {
            return null;
        }
    }

    private static void collectDirectoryCache(
            File root, boolean jit, Map<String, CacheAmounts> output)
            throws IOException {
        File[] children = root.listFiles();
        if (children == null) {
            return;
        }
        for (File directory : children) {
            String key = directory.getName();
            if (!isSafeCacheKey(key) || !isDirectory(directory)) {
                continue;
            }
            long bytes = treeSize(directory);
            if (bytes == 0) {
                continue;
            }
            CacheAmounts amount =
                    output.computeIfAbsent(key, ignored -> new CacheAmounts());
            if (jit) {
                amount.jitBytes = saturatingAdd(amount.jitBytes, bytes);
            } else {
                amount.gpuBytes = saturatingAdd(amount.gpuBytes, bytes);
            }
        }
    }

    private static void collectFingerprintCache(
            File root, Map<String, CacheAmounts> output) {
        File[] children = root.listFiles();
        if (children == null) {
            return;
        }
        for (File file : children) {
            if (!isRegularFile(file)
                    || !file.getName().endsWith(".txt")) {
                continue;
            }
            String key = file.getName().substring(
                    0, file.getName().length() - 4);
            if (!isSafeCacheKey(key)) {
                continue;
            }
            CacheAmounts amount =
                    output.computeIfAbsent(key, ignored -> new CacheAmounts());
            amount.jitBytes =
                    saturatingAdd(amount.jitBytes, file.length());
        }
    }

    private static long treeSize(File root) throws IOException {
        Path path = root.toPath();
        if (Files.isSymbolicLink(path)) {
            return 0;
        }
        if (Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) {
            return Math.max(0, root.length());
        }
        if (!Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)) {
            return 0;
        }
        long bytes = 0;
        try (DirectoryStream<Path> children =
                     Files.newDirectoryStream(path)) {
            for (Path child : children) {
                if (Files.isSymbolicLink(child)) {
                    continue;
                }
                bytes = saturatingAdd(bytes, treeSize(child.toFile()));
            }
        }
        return bytes;
    }

    private static void deleteExactChild(
            File root, String name, boolean directory) throws IOException {
        if (!root.exists()) {
            return;
        }
        File canonicalRoot = root.getCanonicalFile();
        File target = new File(canonicalRoot, name);
        if (!target.exists()) {
            return;
        }
        if (!target.getParentFile().getCanonicalFile().equals(canonicalRoot)) {
            throw new IOException("Cache path escaped its root");
        }
        Path targetPath = target.toPath();
        if (Files.isSymbolicLink(targetPath)) {
            Files.delete(targetPath);
            return;
        }
        if (!directory) {
            if (!Files.isRegularFile(
                    targetPath, LinkOption.NOFOLLOW_LINKS)) {
                throw new IOException("Cache entry is not a regular file");
            }
            Files.delete(targetPath);
            return;
        }
        if (!Files.isDirectory(targetPath, LinkOption.NOFOLLOW_LINKS)) {
            throw new IOException("Cache entry is not a directory");
        }
        File quarantine = new File(canonicalRoot,
                ".deleting-" + name + "-" +
                        Long.toUnsignedString(System.nanoTime()));
        try {
            Files.move(targetPath, quarantine.toPath(),
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(targetPath, quarantine.toPath());
        }
        deleteTreeNoFollow(quarantine.toPath());
    }

    private static void deleteTreeNoFollow(Path path) throws IOException {
        if (Files.isSymbolicLink(path)
                || Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) {
            Files.deleteIfExists(path);
            return;
        }
        if (!Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)) {
            Files.deleteIfExists(path);
            return;
        }
        try (DirectoryStream<Path> children =
                     Files.newDirectoryStream(path)) {
            for (Path child : children) {
                deleteTreeNoFollow(child);
            }
        }
        Files.deleteIfExists(path);
    }

    private static void ensureDirectory(File directory) throws IOException {
        if (!directory.isDirectory() && !directory.mkdirs()) {
            throw new IOException("Cannot create " + directory);
        }
    }

    private static boolean isDirectory(File file) {
        return file != null && Files.isDirectory(
                file.toPath(), LinkOption.NOFOLLOW_LINKS);
    }

    private static boolean isRegularFile(File file) {
        return file != null && Files.isRegularFile(
                file.toPath(), LinkOption.NOFOLLOW_LINKS);
    }

    private static byte[] readFileLimited(File file, int maximum)
            throws IOException {
        long length = file.length();
        if (length < 0 || length > maximum) {
            throw new IOException("Metadata file is too large");
        }
        try (InputStream input = new FileInputStream(file);
             ByteArrayOutputStream output = new ByteArrayOutputStream(
                     (int) length)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) != -1) {
                if (output.size() > maximum - count) {
                    throw new IOException("Metadata file is too large");
                }
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static String normalizeTitleId(String value) {
        String clean = clean(value);
        if (clean == null) {
            return null;
        }
        String normalized = clean.toUpperCase(Locale.ROOT);
        return TITLE_ID.matcher(normalized).matches()
                ? normalized : null;
    }

    private static boolean isSafeCacheKey(String value) {
        return value != null && CACHE_KEY.matcher(value).matches();
    }

    private static void addArtifactAlias(Metadata metadata, String value) {
        if (metadata == null || value == null) {
            return;
        }
        String upper = value.toUpperCase(Locale.ROOT);
        StringBuilder sanitized = new StringBuilder(upper.length());
        for (int index = 0; index < upper.length(); index++) {
            char character = upper.charAt(index);
            if ((character >= 'A' && character <= 'Z')
                    || (character >= '0' && character <= '9')
                    || character == '_' || character == '-') {
                sanitized.append(character);
            }
        }
        String alias = sanitized.toString();
        if (isSafeCacheKey(alias)) {
            metadata.aliases.add(alias);
        }
    }

    private static String clean(String value) {
        if (value == null) {
            return null;
        }
        int nul = value.indexOf('\0');
        String clean = (nul >= 0 ? value.substring(0, nul) : value).trim();
        return clean.isEmpty() ? null : clean;
    }

    private static long saturatingAdd(long left, long right) {
        if (right > 0 && left > Long.MAX_VALUE - right) {
            return Long.MAX_VALUE;
        }
        return left + Math.max(0, right);
    }
}
