package app.lsx4.android;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.StatFs;
import android.provider.DocumentsContract;
import android.provider.OpenableColumns;
import android.util.Log;
import android.view.Gravity;
import android.view.HapticFeedbackConstants;
import android.view.Menu;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.apache.commons.compress.archivers.sevenz.SevenZArchiveEntry;
import org.apache.commons.compress.archivers.sevenz.SevenZFile;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class LauncherUi {
    private static final String TAG = "LSX4.Library";
    public static final int REQUEST_IMPORT_EXTRACTED_GAME = 4101;
    public static final int REQUEST_IMPORT_PKG = 4102;

    private static final int MENU_IMPORT_DIRECTORY = 5101;
    private static final int MENU_IMPORT_PKG = 5102;
    private static final String EXTRA_EMBEDDED_AARCH64_JIT_BACKEND =
            "embedded_aarch64_jit_backend";
    private static final String EXTRA_GAME_PLATFORM = "game_platform";
    private static final String PLATFORM_PS4 = "ps4";
    private static final String PLATFORM_PS5 = "ps5";
    private static final Pattern TITLE_ID = Pattern.compile("[A-Z0-9]{9}");
    private static final int MAX_SFO_BYTES = 1024 * 1024;
    private static final int COPY_BUFFER_BYTES = 1024 * 1024;
    private static final int MAX_TREE_DEPTH = 64;
    private static final int MAX_IMPORT_FILES = 200_000;
    private static final int MAX_IMPORT_NODES = 200_000;
    private static final int MAX_DIRECTORY_CHILDREN = 20_000;
    private static final int MAX_DOCUMENT_NAME_LENGTH = 255;
    private static final int MAX_DOCUMENT_ID_LENGTH = 4096;
    private static final int MAX_TREE_URI_LENGTH = 8192;
    private static final long MAX_DOCUMENT_METADATA_CHARS = 8L * 1024L * 1024L;
    private static final long FREE_SPACE_RESERVE_BYTES = 512L * 1024L * 1024L;
    private static final Object IMPORT_LOCK = new Object();
    private static final AtomicBoolean IMPORT_ACTIVE = new AtomicBoolean();

    private final Activity activity;
    private final AtomicBoolean workRunning = new AtomicBoolean();
    private final GameCompatibilityRepository compatibilityRepository;
    private final Handler holdHandler;
    private final int touchSlop;
    private LinearLayout gameList;
    private TextView status;
    private ProgressBar progress;
    private ImageView removeAction;
    private GameEntry selectedGame;
    private List<GameEntry> latestGames = Collections.emptyList();
    private View rootView;
    private volatile boolean destroyed;

    public LauncherUi(Activity activity) {
        if (activity == null) {
            throw new IllegalArgumentException("activity == null");
        }
        this.activity = activity;
        compatibilityRepository = new GameCompatibilityRepository(activity);
        holdHandler = new Handler(Looper.getMainLooper());
        touchSlop = ViewConfiguration.get(activity).getScaledTouchSlop();
    }

    public View buildView() {
        activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        LinearLayout root = new LinearLayout(activity);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xff111417);
        UiChrome.applyInsets(root, dp(16), dp(12), dp(16), dp(14));

        LinearLayout appBar = new LinearLayout(activity);
        appBar.setOrientation(LinearLayout.HORIZONTAL);
        appBar.setGravity(Gravity.CENTER_VERTICAL);
        appBar.setPadding(dp(14), dp(9), dp(6), dp(9));
        GradientDrawable appBarBackground = new GradientDrawable();
        appBarBackground.setColor(0xff1b2026);
        appBarBackground.setCornerRadius(dp(16));
        appBarBackground.setStroke(dp(1), 0xff2a313a);
        appBar.setBackground(appBarBackground);
        appBar.setElevation(dp(4));

        LinearLayout titleColumn = new LinearLayout(activity);
        titleColumn.setOrientation(LinearLayout.VERTICAL);
        titleColumn.addView(label("LSX4", 23, 0xffeef2f6), matchWrap());
        status = label(activity.getString(R.string.library_scanning), 12, 0xff9aa6b2);
        titleColumn.addView(status, matchWrap());
        appBar.addView(titleColumn, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        progress = new ProgressBar(activity);
        progress.setIndeterminate(true);
        progress.setVisibility(View.GONE);
        LinearLayout.LayoutParams progressLp = new LinearLayout.LayoutParams(dp(24), dp(24));
        progressLp.setMargins(dp(8), 0, dp(6), 0);
        appBar.addView(progress, progressLp);

        removeAction = new ImageView(activity);
        removeAction.setImageResource(R.drawable.ic_delete);
        removeAction.setPadding(dp(12), dp(12), dp(12), dp(12));
        removeAction.setContentDescription(activity.getString(R.string.menu_remove_game));
        removeAction.setClickable(true);
        removeAction.setFocusable(true);
        removeAction.setVisibility(View.GONE);
        removeAction.setOnClickListener(view -> removeSelectedGame());
        appBar.addView(removeAction, new LinearLayout.LayoutParams(dp(48), dp(48)));

        TextView add = toolbarAction("+", activity.getString(R.string.menu_add));
        add.setOnClickListener(this::showAddMenu);
        appBar.addView(add, new LinearLayout.LayoutParams(dp(48), dp(48)));

        ImageView settings = new ImageView(activity);
        settings.setImageResource(R.drawable.ic_tune);
        settings.setPadding(dp(12), dp(12), dp(12), dp(12));
        settings.setContentDescription(activity.getString(R.string.menu_settings));
        settings.setClickable(true);
        settings.setFocusable(true);
        settings.setOnClickListener(view ->
                activity.startActivity(new Intent(activity, SettingsActivity.class)));
        appBar.addView(settings, new LinearLayout.LayoutParams(dp(48), dp(48)));
        root.addView(appBar, matchWrap());

        gameList = new LinearLayout(activity);
        gameList.setOrientation(LinearLayout.VERTICAL);
        ScrollView scroll = new ScrollView(activity);
        scroll.setFillViewport(true);
        scroll.addView(gameList);
        LinearLayout.LayoutParams scrollLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f);
        scrollLp.setMargins(0, dp(10), 0, 0);
        root.addView(scroll, scrollLp);

        rootView = root;
        refresh();
        return root;
    }

    private TextView toolbarAction(String glyph, String description) {
        TextView action = label(glyph, 30, 0xffeef2f6);
        action.setGravity(Gravity.CENTER);
        action.setContentDescription(description);
        action.setClickable(true);
        action.setFocusable(true);
        return action;
    }

    private void showAddMenu(View anchor) {
        PopupMenu popup = new PopupMenu(activity, anchor, Gravity.END);
        Menu menu = popup.getMenu();
        menu.add(Menu.NONE, MENU_IMPORT_DIRECTORY, 0,
                activity.getString(R.string.menu_import_directory));
        menu.add(Menu.NONE, MENU_IMPORT_PKG, 1,
                activity.getString(R.string.menu_import_pkg));
        popup.setOnMenuItemClickListener(item -> {
            switch (item.getItemId()) {
            case MENU_IMPORT_DIRECTORY:
                pickExtractedGameTree();
                return true;
            case MENU_IMPORT_PKG:
                pickPkg();
                return true;
            default:
                return false;
            }
        });
        popup.show();
    }

    public View createView() {
        return buildView();
    }

    public View getView() {
        return rootView;
    }

    public void destroy() {
        destroyed = true;
        holdHandler.removeCallbacksAndMessages(null);
        rootView = null;
        gameList = null;
        status = null;
        progress = null;
        removeAction = null;
    }

    public boolean handleActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode != REQUEST_IMPORT_EXTRACTED_GAME && requestCode != REQUEST_IMPORT_PKG) {
            return false;
        }
        if (resultCode != Activity.RESULT_OK || data == null || data.getData() == null) {
            setStatus(activity.getString(R.string.import_cancelled));
            return true;
        }

        Uri uri = data.getData();
        takeReadPermission(uri, data.getFlags());
        if (requestCode == REQUEST_IMPORT_EXTRACTED_GAME) {
            runJob(activity.getString(R.string.import_checking_directory), () -> {
                try {
                    return importExtractedTree(uri);
                } finally {
                    releaseReadPermission(uri);
                }
            });
        } else {
            runJob(activity.getString(R.string.import_copying_pkg), () -> {
                try {
                    return importSelectedFile(uri);
                } finally {
                    releaseReadPermission(uri);
                }
            });
        }
        return true;
    }

    public void refresh() {
        if (!workRunning.compareAndSet(false, true)) {
            setStatus(activity.getString(R.string.operation_wait));
            return;
        }
        setBusy(true, activity.getString(R.string.library_scanning));
        Thread worker = new Thread(() -> {
            List<GameEntry> games;
            String message;
            try {
                synchronized (IMPORT_LOCK) {
                    games = scanInstalledGames();
                }
                message = games.isEmpty()
                        ? activity.getString(R.string.library_empty_status)
                        : activity.getString(R.string.library_found, games.size());
            } catch (Exception e) {
                games = Collections.emptyList();
                Log.e(TAG, activity.getString(R.string.scan_error)
                        + " [" + e.getClass().getSimpleName() + "]");
                message = activity.getString(R.string.scan_error);
            }
            workRunning.set(false);
            finishUiWork(games, message);
        }, "ps4-library-scan");
        worker.setDaemon(true);
        worker.start();
    }

    private void pickExtractedGameTree() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
                | Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
        try {
            activity.startActivityForResult(intent, REQUEST_IMPORT_EXTRACTED_GAME);
        } catch (ActivityNotFoundException e) {
            toast(activity.getString(R.string.folder_picker_unavailable));
        }
    }

    private void pickPkg() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        try {
            activity.startActivityForResult(intent, REQUEST_IMPORT_PKG);
        } catch (ActivityNotFoundException e) {
            toast(activity.getString(R.string.file_picker_unavailable));
        }
    }

    private void takeReadPermission(Uri uri, int returnedFlags) {
        int flags = returnedFlags & (Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        flags &= Intent.FLAG_GRANT_READ_URI_PERMISSION;
        if (flags == 0) {
            return;
        }
        try {
            activity.getContentResolver().takePersistableUriPermission(uri, flags);
        } catch (SecurityException | UnsupportedOperationException ignored) {
        }
    }

    private void releaseReadPermission(Uri uri) {
        try {
            activity.getContentResolver().releasePersistableUriPermission(
                    uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException | UnsupportedOperationException ignored) {
        }
    }

    private void runJob(String initialStatus, ImportJob job) {
        if (!workRunning.compareAndSet(false, true)) {
            setStatus(activity.getString(R.string.operation_wait));
            return;
        }
        if (!IMPORT_ACTIVE.compareAndSet(false, true)) {
            workRunning.set(false);
            setStatus(activity.getString(R.string.import_already_running));
            return;
        }
        setBusy(true, initialStatus);
        Thread worker = new Thread(() -> {
            String message;
            List<GameEntry> games;
            try {
                synchronized (IMPORT_LOCK) {
                    try {
                        checkCancelled();
                        message = job.run();
                    } catch (Exception e) {
                        Log.e(TAG, activity.getString(R.string.import_error)
                                + " [" + e.getClass().getSimpleName() + "]");
                        message = activity.getString(R.string.import_error);
                    }
                    try {
                        games = scanInstalledGames();
                    } catch (Exception ignored) {
                        games = Collections.emptyList();
                    }
                }
            } finally {
                IMPORT_ACTIVE.set(false);
                workRunning.set(false);
            }
            finishUiWork(games, message);
        }, "ps4-game-import");
        worker.setDaemon(true);
        worker.start();
    }

    private String importExtractedTree(Uri treeUri) throws Exception {
        ContentResolver resolver = activity.getContentResolver();
        if (treeUri.toString().length() > MAX_TREE_URI_LENGTH) {
            throw new IOException("слишком длинный URI выбранной папки");
        }
        String rootId = DocumentsContract.getTreeDocumentId(treeUri);
        if (rootId == null || rootId.length() > MAX_DOCUMENT_ID_LENGTH) {
            throw new IOException("некорректный document ID корневой папки");
        }
        Uri rootDocument = DocumentsContract.buildDocumentUriUsingTree(treeUri, rootId);

        SafNode eboot = findDirectChild(treeUri, rootDocument, "eboot.bin", false);
        SafNode sceSys = findDirectChild(treeUri, rootDocument, "sce_sys", true);
        if (eboot == null || sceSys == null) {
            throw new IOException("выбранная папка должна содержать eboot.bin и sce_sys");
        }
        SafNode paramSfo = findDirectChild(treeUri, sceSys.uri, "param.sfo", false);
        SafNode paramJson = findDirectChild(treeUri, sceSys.uri, "param.json", false);
        if (paramSfo == null && paramJson == null) {
            throw new IOException("в sce_sys отсутствует param.sfo или param.json");
        }
        if (eboot.size >= 0 && eboot.size < 4) {
            throw new IOException("eboot.bin пуст или обрезан");
        }

        String platform;
        String titleId;
        String title;
        if (paramJson != null) {
            Ps5Metadata metadata = readPs5Metadata(
                    readUriLimited(resolver, paramJson.uri, MAX_SFO_BYTES));
            platform = PLATFORM_PS5;
            titleId = metadata.titleId;
            title = metadata.title;
        } else {
            byte[] sourceSfo = readUriLimited(resolver, paramSfo.uri, MAX_SFO_BYTES);
            platform = PLATFORM_PS4;
            titleId = normalizeTitleId(PkgInstaller.sfoString(sourceSfo, "TITLE_ID"));
            title = cleanSfoString(PkgInstaller.sfoString(sourceSfo, "TITLE"));
        }
        if (!isValidTitleId(titleId)) {
            throw new IOException("некорректный title ID");
        }

        File home = homeDir();
        File appRoot = appInstallRoot();
        ensureDirectory(appRoot);
        File destination = checkedChild(appRoot, titleId);
        if (destination.exists() || Files.isSymbolicLink(destination.toPath())) {
            throw new IOException("игра " + titleId
                    + " уже установлена; автоматическое слияние отключено");
        }
        File stagingRoot = checkedChild(home, "import-staging");
        ensureDirectory(stagingRoot);
        File staging = checkedChild(stagingRoot,
                titleId + "-" + Long.toUnsignedString(System.nanoTime()));
        ensureDirectory(staging);

        CopyProgress copied = new CopyProgress(availableImportBudget(stagingRoot));
        boolean installed = false;
        try {
            copySafDirectory(treeUri, rootDocument, staging, staging, 0, copied,
                    new HashSet<>());

            File stagedEboot = checkedChild(staging, "eboot.bin");
            File stagedSceSys = checkedChild(staging, "sce_sys");
            File stagedParam = checkedChild(
                    stagedSceSys,
                    PLATFORM_PS5.equals(platform) ? "param.json" : "param.sfo");
            if (!isPlausibleEboot(stagedEboot, staging)
                    || !safeRegularFileWithin(stagedParam, staging)) {
                throw new IOException("копия игры не прошла проверку структуры");
            }
            String copiedId = PLATFORM_PS5.equals(platform)
                    ? readPs5Metadata(readFileLimited(stagedParam, MAX_SFO_BYTES)).titleId
                    : readSfoValue(stagedParam, "TITLE_ID");
            if (!titleId.equals(copiedId)) {
                throw new IOException("title ID изменился во время копирования");
            }

            writeInstallManifest(staging, titleId, title, treeUri.toString(), true,
                    "saf-extracted-tree", platform);
            installStagingTree(staging, destination, stagingRoot);
            installed = true;
        } finally {
            if (!installed && staging.exists()) {
                deleteTreeWithin(staging, stagingRoot);
            }
        }

        return activity.getString(R.string.import_directory_success,
                displayTitle(title, titleId), titleId, copied.files, formatBytes(copied.bytes));
    }

    private String importSelectedFile(Uri uri) throws Exception {
        String displayName = queryDisplayName(uri);
        String lowerName = displayName == null ? "" : displayName.toLowerCase(Locale.ROOT);
        if (lowerName.endsWith(".rar") || lowerName.endsWith(".7z")
                || lowerName.endsWith(".zip")) {
            return importArchive(uri, displayName, lowerName);
        }
        return importPkg(uri);
    }

    private String importArchive(Uri uri, String displayName, String lowerName) throws Exception {
        File home = homeDir();
        File inbox = checkedChild(home, "archive-inbox");
        File stagingRoot = checkedChild(home, "import-staging");
        ensureDirectory(inbox);
        ensureDirectory(stagingRoot);

        String fallback = lowerName.endsWith(".rar") ? "import.rar"
                : lowerName.endsWith(".7z") ? "import.7z" : "import.zip";
        String safeName = safeFileName(displayName, fallback);
        File archiveFile = uniqueChild(inbox, safeName);
        File extractionRoot = checkedChild(stagingRoot,
                "archive-" + Long.toUnsignedString(System.nanoTime()));
        ensureDirectory(extractionRoot);

        long copied = 0;
        boolean installed = false;
        try {
            long declaredSize = querySize(uri);
            long copyBudget = availableImportBudget(inbox);
            if (declaredSize > copyBudget) {
                throw new IOException("недостаточно свободного места для архива (нужно "
                        + formatBytes(declaredSize) + ")");
            }
            copied = copyUriToFile(uri, archiveFile, copyBudget);
            postStatus("Распаковка " + safeName + "…");

            CopyProgress extracted = new CopyProgress(availableImportBudget(stagingRoot));
            if (lowerName.endsWith(".rar")) {
                extractRar(archiveFile, extractionRoot, extracted);
            } else if (lowerName.endsWith(".7z")) {
                extractSevenZip(archiveFile, extractionRoot, extracted);
            } else {
                extractZip(archiveFile, extractionRoot, extracted);
            }

            File gameRoot = findExtractedGameRoot(extractionRoot, 0);
            if (gameRoot == null) {
                throw new IOException("архив не содержит eboot.bin и sce_sys/param.json");
            }
            File eboot = checkedChild(gameRoot, "eboot.bin");
            File sceSys = checkedChild(gameRoot, "sce_sys");
            File paramJson = checkedChild(sceSys, "param.json");
            File paramSfo = checkedChild(sceSys, "param.sfo");
            boolean ps5 = safeRegularFileWithin(paramJson, gameRoot);
            if (!isPlausibleEboot(eboot, gameRoot)
                    || (!ps5 && !safeRegularFileWithin(paramSfo, gameRoot))) {
                throw new IOException("распакованная игра не прошла проверку структуры");
            }

            String platform;
            String titleId;
            String title;
            if (ps5) {
                Ps5Metadata metadata = readPs5Metadata(
                        readFileLimited(paramJson, MAX_SFO_BYTES));
                platform = PLATFORM_PS5;
                titleId = metadata.titleId;
                title = metadata.title;
            } else {
                platform = PLATFORM_PS4;
                titleId = readSfoValue(paramSfo, "TITLE_ID");
                title = readSfoValue(paramSfo, "TITLE");
            }
            if (!isValidTitleId(titleId)) {
                throw new IOException("некорректный title ID в архиве");
            }

            File appRoot = appInstallRoot();
            ensureDirectory(appRoot);
            File destination = checkedChild(appRoot, titleId);
            if (destination.exists() || Files.isSymbolicLink(destination.toPath())) {
                throw new IOException("игра " + titleId + " уже установлена");
            }
            writeInstallManifest(gameRoot, titleId, title, uri.toString(), true,
                    "archive-import", platform);
            installStagingTree(gameRoot, destination, stagingRoot);
            installed = true;

            return "Импортирована " + displayTitle(title, titleId) + " (" + titleId + "), "
                    + extracted.files + " файлов, " + formatBytes(extracted.bytes)
                    + " распаковано из " + formatBytes(copied);
        } finally {
            if (archiveFile.exists() && !archiveFile.delete()) {
                Log.w(TAG, "Не удалось удалить временный архив " + archiveFile.getName());
            }
            if (extractionRoot.exists()) {
                try {
                    deleteTreeWithin(extractionRoot, stagingRoot);
                } catch (IOException cleanupError) {
                    if (!installed) {
                        throw cleanupError;
                    }
                    Log.w(TAG, "Не удалось очистить staging архива", cleanupError);
                }
            }
        }
    }

    private void extractRar(File archiveFile, File destination, CopyProgress progress)
            throws Exception {
        checkCancelled();
        int result = RuntimeBridge.extractRarArchive(
                archiveFile.getAbsolutePath(), destination.getAbsolutePath(),
                progress.maxBytes, MAX_IMPORT_FILES);
        if (result != 0) {
            throw new IOException("UnRAR завершился с кодом " + result);
        }
        accountExtractedTree(destination, destination, progress, 0);
    }

    private void accountExtractedTree(File path, File boundary, CopyProgress progress, int depth)
            throws Exception {
        checkCancelled();
        if (depth > MAX_TREE_DEPTH) {
            throw new IOException("слишком глубокий путь после распаковки");
        }
        ensureWithin(path, boundary);
        if (Files.isSymbolicLink(path.toPath())) {
            throw new IOException("символические ссылки в архиве запрещены");
        }
        if (path.isFile()) {
            progress.reserveNode(false);
            progress.reserveFile(path.length());
            progress.addExtracted(path.length());
            return;
        }
        if (!path.isDirectory()) {
            throw new IOException("некорректный объект после распаковки");
        }
        if (!path.equals(boundary)) {
            progress.reserveNode(true);
        }
        File[] children = path.listFiles();
        if (children == null) {
            throw new IOException("не удалось прочитать распакованный каталог");
        }
        for (File child : children) {
            accountExtractedTree(child, boundary, progress, depth + 1);
        }
    }

    private void extractSevenZip(File archiveFile, File destination, CopyProgress progress)
            throws Exception {
        try (SevenZFile archive = new SevenZFile(archiveFile)) {
            SevenZArchiveEntry entry;
            byte[] buffer = new byte[COPY_BUFFER_BYTES];
            while ((entry = archive.getNextEntry()) != null) {
                checkCancelled();
                File output = archiveOutput(destination, entry.getName(),
                        entry.isDirectory(), progress);
                if (entry.isDirectory()) {
                    ensureDirectory(output);
                    continue;
                }
                long declared = entry.getSize();
                progress.reserveFile(declared);
                ensureDirectory(output.getParentFile());
                try (OutputStream stream = new BudgetOutputStream(
                        new FileOutputStream(output), progress, declared)) {
                    int count;
                    while ((count = archive.read(buffer)) != -1) {
                        stream.write(buffer, 0, count);
                    }
                } catch (Exception error) {
                    output.delete();
                    throw error;
                }
            }
        }
    }

    private void extractZip(File archiveFile, File destination, CopyProgress progress)
            throws Exception {
        try (ZipInputStream archive = new ZipInputStream(new FileInputStream(archiveFile))) {
            byte[] buffer = new byte[COPY_BUFFER_BYTES];
            ZipEntry entry;
            while ((entry = archive.getNextEntry()) != null) {
                checkCancelled();
                File output = archiveOutput(destination, entry.getName(),
                        entry.isDirectory(), progress);
                if (entry.isDirectory()) {
                    ensureDirectory(output);
                    continue;
                }
                long declared = entry.getSize();
                progress.reserveFile(declared);
                ensureDirectory(output.getParentFile());
                try (OutputStream stream = new BudgetOutputStream(
                        new FileOutputStream(output), progress, declared)) {
                    int count;
                    while ((count = archive.read(buffer)) != -1) {
                        stream.write(buffer, 0, count);
                    }
                } catch (Exception error) {
                    output.delete();
                    throw error;
                }
                archive.closeEntry();
            }
        }
    }

    private File archiveOutput(File root, String rawName, boolean directory,
                               CopyProgress progress) throws Exception {
        if (rawName == null || rawName.isEmpty() || rawName.length() > MAX_DOCUMENT_ID_LENGTH) {
            throw new IOException("некорректное имя записи архива");
        }
        String normalized = rawName.replace('\\', '/');
        if (normalized.startsWith("/") || normalized.matches("^[A-Za-z]:.*")) {
            throw new IOException("абсолютный путь в архиве запрещён");
        }
        String[] segments = normalized.split("/");
        File current = root;
        int depth = 0;
        for (String segment : segments) {
            if (segment.isEmpty()) {
                continue;
            }
            if (++depth > MAX_TREE_DEPTH) {
                throw new IOException("слишком глубокий путь в архиве");
            }
            validateDocumentName(segment);
            if (segment.length() > MAX_DOCUMENT_NAME_LENGTH) {
                throw new IOException("слишком длинное имя в архиве");
            }
            current = checkedChild(current, segment);
        }
        if (current.equals(root)) {
            throw new IOException("пустой путь записи архива");
        }
        ensureWithin(current, root);
        progress.reserveNode(directory);
        return current;
    }

    private File findExtractedGameRoot(File directory, int depth) throws IOException {
        ensureWithin(directory, directory);
        if (depth > MAX_TREE_DEPTH || Files.isSymbolicLink(directory.toPath())) {
            return null;
        }
        File eboot = new File(directory, "eboot.bin");
        File sceSys = new File(directory, "sce_sys");
        if (isPlausibleEboot(eboot, directory) && sceSys.isDirectory()
                && !Files.isSymbolicLink(sceSys.toPath())) {
            File paramJson = new File(sceSys, "param.json");
            File paramSfo = new File(sceSys, "param.sfo");
            if (safeRegularFileWithin(paramJson, directory)
                    || safeRegularFileWithin(paramSfo, directory)) {
                return directory;
            }
        }
        File[] children = directory.listFiles();
        if (children == null) {
            return null;
        }
        File found = null;
        for (File child : children) {
            if (!child.isDirectory() || Files.isSymbolicLink(child.toPath())) {
                continue;
            }
            File candidate = findExtractedGameRoot(child, depth + 1);
            if (candidate != null) {
                if (found != null) {
                    throw new IOException("архив содержит несколько корней игр");
                }
                found = candidate;
            }
        }
        return found;
    }

    private String importPkg(Uri uri) throws Exception {
        File home = homeDir();
        File inbox = checkedChild(home, "pkg-inbox");
        ensureDirectory(inbox);

        String displayName = queryDisplayName(uri);
        String safeName = safeFileName(displayName, "import.pkg");
        if (!safeName.toLowerCase(Locale.ROOT).endsWith(".pkg")) {
            safeName += ".pkg";
        }
        File destination = uniqueChild(inbox, safeName);
        long importBudget = availableImportBudget(inbox);
        long declaredSize = querySize(uri);
        if (declaredSize > importBudget) {
            throw new IOException("недостаточно свободного места для PKG (нужно "
                    + formatBytes(declaredSize) + ")");
        }
        long copied = copyUriToFile(uri, destination, importBudget);
        postStatus(activity.getString(R.string.import_pkg_copied, formatBytes(copied)));
        boolean registered = false;
        try {
            PkgMetadata metadata = preflightPkg(destination);
            if (!isValidTitleId(metadata.titleId)) {
                throw new IOException("некорректный TITLE_ID в PKG");
            }

            File appRoot = appInstallRoot();
            ensureDirectory(appRoot);
            File finalDirectory = checkedChild(appRoot, metadata.titleId);
            if (finalDirectory.exists() || Files.isSymbolicLink(finalDirectory.toPath())) {
                throw new IOException("игра " + metadata.titleId
                        + " уже установлена; PKG не будет менять её файлы");
            }

            File metadataStagingRoot = checkedChild(home, "pkg-install-staging");
            ensureDirectory(metadataStagingRoot);
            File stagingHome = checkedChild(metadataStagingRoot,
                    metadata.titleId + "-" + Long.toUnsignedString(System.nanoTime()));
            ensureDirectory(stagingHome);
            try {
                PkgInstaller.Result result = PkgInstaller.install(destination, stagingHome);
                if (!result.ok) {
                    throw new IOException(result.message == null ? "PkgInstaller отклонил PKG"
                            : result.message);
                }
                if (!metadata.titleId.equals(result.titleId)) {
                    throw new IOException("PkgInstaller вернул другой TITLE_ID");
                }
                File stagedDirectory = checkedChild(checkedChild(checkedChild(
                        checkedChild(stagingHome, "runtime-fs"), "user"), "app"),
                        metadata.titleId);
                if (!stagedDirectory.isDirectory()
                        || !safeRegularFileWithin(new File(new File(stagedDirectory, "sce_sys"),
                                "param.sfo"), stagedDirectory)) {
                    throw new IOException("PkgInstaller не создал полный metadata staging");
                }
                installStagingTree(stagedDirectory, finalDirectory, metadataStagingRoot);
                registered = true;

                if (result.ebootExtracted) {
                    return activity.getString(R.string.import_pkg_registered,
                            displayTitle(result.title, result.titleId));
                }
                return activity.getString(R.string.import_pkg_metadata_only,
                        displayTitle(result.title, result.titleId));
            } finally {
                if (stagingHome.exists()) {
                    deleteTreeWithin(stagingHome, metadataStagingRoot);
                }
            }
        } finally {
            if (!registered && destination.exists()) {
                destination.delete();
            }
        }
    }

    private List<GameEntry> scanInstalledGames() throws IOException {
        File home = homeDir();
        File[] roots = new File[]{appInstallRoot(), checkedChild(home, "games")};
        List<GameEntry> merged = new ArrayList<>();
        for (int rootPriority = 0; rootPriority < roots.length; rootPriority++) {
            File[] children = roots[rootPriority].listFiles();
            if (children == null) {
                continue;
            }
            for (File directory : children) {
                if (!directory.isDirectory() || Files.isSymbolicLink(directory.toPath())) {
                    continue;
                }
                try {
                    ensureWithin(directory, roots[rootPriority]);
                } catch (IOException escapedRoot) {
                    continue;
                }
                GameEntry candidate = readGameEntry(directory, rootPriority, roots[rootPriority]);
                if (candidate == null) {
                    continue;
                }
                int duplicate = findDuplicate(merged, candidate);
                if (duplicate < 0) {
                    merged.add(candidate);
                } else if (prefer(candidate, merged.get(duplicate))) {
                    merged.set(duplicate, candidate);
                }
            }
        }

        merged.sort(Comparator
                .comparing((GameEntry entry) -> entry.title.toLowerCase(Locale.ROOT))
                .thenComparing(entry -> entry.titleId));
        return merged;
    }

    private GameEntry readGameEntry(File directory, int rootPriority, File scanRoot) {
        try {
            ensureWithin(directory, scanRoot);
            File eboot = new File(directory, "eboot.bin");
            File sceSys = new File(directory, "sce_sys");
            File sfo = new File(sceSys, "param.sfo");
            File paramJson = new File(sceSys, "param.json");
            File manifest = new File(directory, "executor-installed.json");
            boolean hasEboot = isPlausibleEboot(eboot, directory);
            boolean hasSfo = safeRegularFileWithin(sfo, directory);
            boolean hasParamJson = safeRegularFileWithin(paramJson, directory);
            boolean hasManifest = safeRegularFileWithin(manifest, directory);
            if (!hasEboot && !hasSfo && !hasParamJson && !hasManifest) {
                return null;
            }

            String platform = hasParamJson ? PLATFORM_PS5 : PLATFORM_PS4;
            Ps5Metadata ps5Metadata = hasParamJson
                    ? readPs5Metadata(readFileLimited(paramJson, MAX_SFO_BYTES))
                    : null;
            String titleId = ps5Metadata != null
                    ? ps5Metadata.titleId
                    : hasSfo ? readSfoValue(sfo, "TITLE_ID") : null;
            if (!isValidTitleId(titleId) && isValidTitleId(directory.getName())) {
                titleId = directory.getName();
            }
            if (!isValidTitleId(titleId)) {
                return null;
            }
            String title = ps5Metadata != null
                    ? ps5Metadata.title
                    : hasSfo ? readSfoValue(sfo, "TITLE") : null;
            if (title == null || title.isEmpty()) {
                title = titleId;
            }
            File icon = new File(sceSys, "icon0.png");
            return new GameEntry(directory, eboot,
                    safeRegularFileWithin(icon, directory) ? icon : null,
                    titleId, title, platform, hasEboot, rootPriority);
        } catch (Exception ignored) {
            Log.w(TAG, "Skipping unreadable game directory " + directory, ignored);
            return null;
        }
    }

    private static int findDuplicate(List<GameEntry> entries, GameEntry candidate) {
        for (int i = 0; i < entries.size(); i++) {
            GameEntry existing = entries.get(i);
            if (candidate.titleId.equals(existing.titleId)) {
                return i;
            }
        }
        return -1;
    }

    private static boolean prefer(GameEntry candidate, GameEntry existing) {
        if (candidate.launchable != existing.launchable) {
            return candidate.launchable;
        }
        if (candidate.rootPriority != existing.rootPriority) {
            return candidate.rootPriority < existing.rootPriority;
        }
        if ((candidate.icon != null) != (existing.icon != null)) {
            return candidate.icon != null;
        }
        return candidate.directory.lastModified() > existing.directory.lastModified();
    }

    private void renderGameList(List<GameEntry> games) {
        if (gameList == null) {
            return;
        }
        List<GameEntry> orderedGames = new ArrayList<>(games);
        orderedGames.sort(Comparator
                .comparingInt((GameEntry entry) -> compatibilityRepository
                        .statusOf(entry.titleId).librarySortPriority())
                .thenComparing(entry -> entry.title.toLowerCase(Locale.ROOT))
                .thenComparing(entry -> entry.titleId));
        latestGames = orderedGames;
        if (selectedGame != null && !orderedGames.contains(selectedGame)) {
            selectedGame = null;
            updateRemoveAction();
        }
        gameList.removeAllViews();
        if (orderedGames.isEmpty()) {
            TextView empty = label(activity.getString(R.string.library_empty),
                    14, 0xffc5cbd3);
            empty.setGravity(Gravity.CENTER);
            empty.setPadding(dp(22), dp(34), dp(22), dp(34));
            empty.setBackground(roundedBackground(0xff171b20, 0xff2a313a, 18));
            gameList.addView(empty, matchWrap());
            return;
        }
        for (GameEntry game : orderedGames) {
            gameList.addView(gameRow(game), matchWrapWithMargins(0, 0, 0, 10));
        }
    }

    private View gameRow(GameEntry game) {
        AccessibleGameRow row = new AccessibleGameRow(activity);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(12), dp(12), dp(12), dp(12));
        boolean selected = selectedGame == game;
        GameCompatibilityStatus compatibility = compatibilityRepository.statusOf(game.titleId);
        int fill = game.launchable ? 0xff1b2026 : 0xff171b20;
        int stroke = 0xff2a313a;
        if (compatibility == GameCompatibilityStatus.SUPPORTED) {
            fill = 0xff192820;
            stroke = 0xff397352;
        } else if (compatibility == GameCompatibilityStatus.UNSUPPORTED) {
            fill = 0xff2b201d;
            stroke = 0xff8c4f43;
        }
        row.setBackground(roundedBackground(
                selected ? 0xff203754 : fill,
                selected ? 0xff6fb4ff : stroke, 18));
        row.setElevation(dp(2));

        FrameLayout iconHost = new FrameLayout(activity);
        ImageView icon = new ImageView(activity);
        icon.setScaleType(ImageView.ScaleType.CENTER_CROP);
        LinearLayout.LayoutParams iconLp = new LinearLayout.LayoutParams(dp(82), dp(82));
        iconLp.setMargins(0, 0, dp(16), 0);
        iconHost.addView(icon, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));
        if (PLATFORM_PS5.equals(game.platform)) {
            TextView badge = label("PS5", 10, 0xffffffff);
            badge.setTypeface(android.graphics.Typeface.DEFAULT,
                    android.graphics.Typeface.BOLD);
            badge.setGravity(Gravity.CENTER);
            badge.setPadding(dp(5), dp(2), dp(5), dp(2));
            badge.setBackground(roundedBackground(0xff1769aa, 0xff89c8ff, 7));
            FrameLayout.LayoutParams badgeLp = new FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.END | Gravity.BOTTOM);
            badgeLp.setMargins(dp(3), dp(3), dp(3), dp(3));
            iconHost.addView(badge, badgeLp);
        }
        row.addView(iconHost, iconLp);
        Bitmap bitmap = decodeIcon(game.icon, 192);
        if (bitmap != null) {
            icon.setImageBitmap(bitmap);
        } else {
            icon.setBackgroundColor(0xff2a313a);
        }

        LinearLayout textColumn = new LinearLayout(activity);
        textColumn.setOrientation(LinearLayout.VERTICAL);
        TextView title = label(game.title, 16,
                game.launchable ? 0xfff2f5f8 : 0xff9aa6b2);
        title.setTypeface(android.graphics.Typeface.DEFAULT,
                android.graphics.Typeface.BOLD);
        textColumn.addView(title);
        textColumn.addView(label(game.titleId, 12, 0xff7f8a97));
        TextView holdHint = label(activity.getString(R.string.game_hold_waiting),
                11, 0xff7f8a97);
        holdHint.setPadding(0, dp(5), 0, 0);
        textColumn.addView(holdHint);
        ProgressBar holdProgress = new ProgressBar(
                activity, null, android.R.attr.progressBarStyleHorizontal);
        holdProgress.setMax((int) GameHoldAction.MARK_SUPPORTED_MS);
        holdProgress.setProgress(0);
        holdProgress.setProgressTintList(
                android.content.res.ColorStateList.valueOf(0xff6fb4ff));
        holdProgress.setProgressBackgroundTintList(
                android.content.res.ColorStateList.valueOf(0xff303741));
        holdProgress.setVisibility(View.GONE);
        LinearLayout.LayoutParams holdProgressLp =
                new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(4));
        holdProgressLp.setMargins(0, dp(5), 0, 0);
        textColumn.addView(holdProgress, holdProgressLp);
        if (!game.launchable) {
            TextView metadata = label(activity.getString(R.string.metadata_only),
                    11, 0xffffad66);
            metadata.setPadding(0, dp(3), 0, 0);
            textColumn.addView(metadata);
        }
        row.addView(textColumn, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        if (compatibility != GameCompatibilityStatus.UNCLASSIFIED) {
            TextView compatibilityMark = label(
                    compatibility == GameCompatibilityStatus.SUPPORTED ? "\u2713" : "\u00d7",
                    20,
                    compatibility == GameCompatibilityStatus.SUPPORTED
                            ? 0xff67d391 : 0xffff806f);
            compatibilityMark.setGravity(Gravity.CENTER);
            compatibilityMark.setContentDescription(activity.getString(
                    compatibility == GameCompatibilityStatus.SUPPORTED
                            ? R.string.game_status_supported
                            : R.string.game_status_unsupported));
            LinearLayout.LayoutParams markLp =
                    new LinearLayout.LayoutParams(dp(34), dp(34));
            markLp.setMargins(dp(8), 0, 0, 0);
            row.addView(compatibilityMark, markLp);
        }

        row.setClickable(true);
        row.setFocusable(true);
        row.setOnClickListener(view -> {
            if (selectedGame != null) {
                selectedGame = game;
                updateRemoveAction();
                renderGameList(latestGames);
            } else if (game.launchable) {
                launch(game);
            }
        });
        int statusText = compatibility == GameCompatibilityStatus.SUPPORTED
                ? R.string.game_status_supported
                : compatibility == GameCompatibilityStatus.UNSUPPORTED
                ? R.string.game_status_unsupported
                : R.string.game_status_unclassified;
        row.setContentDescription(activity.getString(R.string.game_item_accessibility,
                game.title, game.titleId, activity.getString(statusText)));
        row.setOnTouchListener(new GameHoldTouchListener(game, holdHint, holdProgress));
        return row;
    }

    private void handleGameHoldAction(GameEntry game, GameHoldAction action) {
        GameCompatibilityStatus status;
        switch (action) {
        case TAP:
            return;
        case SELECT_FOR_REMOVAL:
            boolean selected = selectedGame != game;
            selectedGame = selected ? game : null;
            updateRemoveAction();
            toast(activity.getString(selected
                    ? R.string.game_selected_for_removal
                    : R.string.game_selection_cleared, game.title));
            renderGameList(latestGames);
            return;
        case MARK_SUPPORTED:
            status = GameCompatibilityStatus.SUPPORTED;
            break;
        case MARK_UNSUPPORTED:
            status = GameCompatibilityStatus.UNSUPPORTED;
            break;
        default:
            return;
        }
        compatibilityRepository.setStatus(game.titleId, status);
        selectedGame = null;
        updateRemoveAction();
        toast(activity.getString(status == GameCompatibilityStatus.SUPPORTED
                ? R.string.game_marked_supported
                : R.string.game_marked_unsupported, game.title));
        renderGameList(latestGames);
    }

    private final class GameHoldTouchListener implements View.OnTouchListener {
        private final GameEntry game;
        private final TextView holdHint;
        private final ProgressBar holdProgress;
        private View pressedView;
        private android.graphics.drawable.Drawable restingBackground;
        private long downAtMs;
        private float downX;
        private float downY;
        private boolean active;
        private GameHoldAction reachedAction = GameHoldAction.TAP;
        private int currentStageText = R.string.game_hold_waiting;

        private final Runnable elapsedTicker = new Runnable() {
            @Override
            public void run() {
                if (!active || pressedView == null) {
                    return;
                }
                long elapsedMs = Math.max(0L,
                        android.os.SystemClock.uptimeMillis() - downAtMs);
                holdProgress.setProgress((int) Math.min(
                        elapsedMs, GameHoldAction.MARK_SUPPORTED_MS));
                double seconds =
                        Math.min(elapsedMs, GameHoldAction.MARK_SUPPORTED_MS) / 1000.0;
                holdHint.setText(activity.getString(R.string.game_hold_elapsed,
                        seconds, activity.getString(currentStageText)));
                holdHandler.postDelayed(this, 100L);
            }
        };

        private final Runnable selectionCue = () -> showHoldStage(
                GameHoldAction.SELECT_FOR_REMOVAL, R.string.game_hold_release_remove, 0xff6fb4ff,
                0xff203754, 0xff6fb4ff, HapticFeedbackConstants.CLOCK_TICK);
        private final Runnable unsupportedCue = () -> showHoldStage(
                GameHoldAction.MARK_UNSUPPORTED, R.string.game_hold_release_unsupported, 0xffff806f,
                0xff2b201d, 0xffb85d50, HapticFeedbackConstants.CLOCK_TICK);
        private final Runnable supportedCue = () -> showHoldStage(
                GameHoldAction.MARK_SUPPORTED, R.string.game_hold_release_supported, 0xff67d391,
                0xff192820, 0xff4f9a6c, HapticFeedbackConstants.LONG_PRESS);

        GameHoldTouchListener(GameEntry game, TextView holdHint,
                              ProgressBar holdProgress) {
            this.game = game;
            this.holdHint = holdHint;
            this.holdProgress = holdProgress;
        }

        @Override
        public boolean onTouch(View view, MotionEvent event) {
            switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                cancel();
                pressedView = view;
                restingBackground = view.getBackground();
                downAtMs = android.os.SystemClock.uptimeMillis();
                downX = event.getX();
                downY = event.getY();
                active = true;
                reachedAction = GameHoldAction.TAP;
                if (view.getParent() != null) {
                    view.getParent().requestDisallowInterceptTouchEvent(true);
                }
                currentStageText = R.string.game_hold_waiting;
                holdHint.setTypeface(android.graphics.Typeface.DEFAULT,
                        android.graphics.Typeface.BOLD);
                holdHint.setTextColor(0xff9aa6b2);
                holdHint.setPadding(dp(8), dp(6), dp(8), dp(6));
                holdHint.setBackground(roundedBackground(0xff151a20, 0xff39424d, 10));
                holdHint.setVisibility(View.VISIBLE);
                holdProgress.setProgress(0);
                holdProgress.setProgressTintList(
                        android.content.res.ColorStateList.valueOf(0xff6fb4ff));
                holdProgress.setVisibility(View.VISIBLE);
                elapsedTicker.run();
                holdHandler.postDelayed(selectionCue,
                        GameHoldAction.SELECT_FOR_REMOVAL_MS);
                holdHandler.postDelayed(unsupportedCue,
                        GameHoldAction.MARK_UNSUPPORTED_MS);
                holdHandler.postDelayed(supportedCue,
                        GameHoldAction.MARK_SUPPORTED_MS);
                return true;
            case MotionEvent.ACTION_MOVE:
                long moveElapsedMs = android.os.SystemClock.uptimeMillis() - downAtMs;
                if (active && moveElapsedMs < GameHoldAction.SELECT_FOR_REMOVAL_MS
                        && (Math.abs(event.getX() - downX) > touchSlop * 5
                        || Math.abs(event.getY() - downY) > touchSlop * 5)) {
                    cancel();
                }
                return true;
            case MotionEvent.ACTION_UP:
                if (!active) {
                    cancel();
                    return true;
                }
                long durationMs = Math.max(0L,
                        android.os.SystemClock.uptimeMillis() - downAtMs);
                GameHoldAction action = GameHoldAction.fromDuration(durationMs);
                cancel();
                if (action == GameHoldAction.TAP) {
                    view.performClick();
                } else {
                    handleGameHoldAction(game, action);
                }
                return true;
            case MotionEvent.ACTION_CANCEL:
            case MotionEvent.ACTION_POINTER_DOWN:
                cancel();
                return true;
            default:
                return true;
            }
        }

        private void cue(int feedbackConstant) {
            if (active && pressedView != null) {
                pressedView.performHapticFeedback(feedbackConstant);
            }
        }

        private void showHoldStage(GameHoldAction action, int textResource,
                                   int textColor, int fillColor,
                                   int strokeColor, int feedbackConstant) {
            if (!active || pressedView == null) {
                return;
            }
            reachedAction = action;
            holdHint.setText(textResource);
            currentStageText = textResource;
            holdHint.setTextColor(textColor);
            holdHint.setVisibility(View.VISIBLE);
            holdProgress.setProgressTintList(
                    android.content.res.ColorStateList.valueOf(textColor));
            pressedView.setBackground(roundedBackground(fillColor, strokeColor, 18));
            cue(feedbackConstant);
        }

        private void cancel() {
            active = false;
            reachedAction = GameHoldAction.TAP;
            holdHandler.removeCallbacks(selectionCue);
            holdHandler.removeCallbacks(unsupportedCue);
            holdHandler.removeCallbacks(supportedCue);
            holdHandler.removeCallbacks(elapsedTicker);
            holdHint.setText(R.string.game_hold_waiting);
            holdHint.setTypeface(android.graphics.Typeface.DEFAULT,
                    android.graphics.Typeface.NORMAL);
            holdHint.setTextColor(0xff7f8a97);
            holdHint.setPadding(0, dp(5), 0, 0);
            holdHint.setBackground(null);
            holdHint.setVisibility(View.VISIBLE);
            holdProgress.setProgress(0);
            holdProgress.setVisibility(View.GONE);
            if (pressedView != null && restingBackground != null) {
                pressedView.setBackground(restingBackground);
            }
            if (pressedView != null && pressedView.getParent() != null) {
                pressedView.getParent().requestDisallowInterceptTouchEvent(false);
            }
            pressedView = null;
            restingBackground = null;
        }
    }

    private static final class AccessibleGameRow extends LinearLayout {
        AccessibleGameRow(android.content.Context context) {
            super(context);
        }

        @Override
        public boolean performClick() {
            return super.performClick();
        }
    }

    private void updateRemoveAction() {
        if (removeAction != null) {
            removeAction.setVisibility(selectedGame == null ? View.GONE : View.VISIBLE);
        }
    }

    private void removeSelectedGame() {
        final GameEntry game = selectedGame;
        if (game == null) {
            return;
        }
        selectedGame = null;
        updateRemoveAction();
        if (!workRunning.compareAndSet(false, true)) {
            setStatus(activity.getString(R.string.operation_wait));
            return;
        }
        if (!IMPORT_ACTIVE.compareAndSet(false, true)) {
            workRunning.set(false);
            setStatus(activity.getString(R.string.operation_wait));
            return;
        }
        setBusy(true, activity.getString(R.string.game_removing, game.title));
        Thread worker = new Thread(() -> {
            String message;
            List<GameEntry> games;
            try {
                synchronized (IMPORT_LOCK) {
                    try {
                        File boundary = game.rootPriority == 0
                                ? appInstallRoot() : checkedChild(homeDir(), "games");
                        ensureWithin(game.directory, boundary);
                        deleteTreeWithin(game.directory, boundary);
                        compatibilityRepository.clear(game.titleId);
                        message = activity.getString(R.string.game_removed, game.title);
                    } catch (Exception error) {
                        Log.e(TAG, "Could not remove " + game.titleId, error);
                        message = activity.getString(R.string.game_remove_failed, game.title);
                    }
                    try {
                        games = scanInstalledGames();
                    } catch (Exception ignored) {
                        games = Collections.emptyList();
                    }
                }
            } finally {
                IMPORT_ACTIVE.set(false);
                workRunning.set(false);
            }
            finishUiWork(games, message);
        }, "lsx4-game-remove");
        worker.setDaemon(true);
        worker.start();
    }

    private void launch(GameEntry game) {
        if (workRunning.get() || IMPORT_ACTIVE.get()) {
            toast(activity.getString(R.string.import_wait_before_launch));
            return;
        }
        Intent intent = new Intent(activity, GameActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        intent.putExtra("autoload_existing_runtime", true);
        intent.putExtra("autoinit_runtime", true);
        String launchPath = PLATFORM_PS5.equals(game.platform)
                ? game.directory.getAbsolutePath()
                : game.eboot.getAbsolutePath();
        intent.putExtra("scan_game_path", launchPath);
        intent.putExtra("launch_game_path", launchPath);
        intent.putExtra(EXTRA_GAME_PLATFORM, game.platform);
        intent.putExtra("fullscreen_render", true);
        intent.putExtra(EXTRA_EMBEDDED_AARCH64_JIT_BACKEND, true);
        intent.putExtra(MainActivity.EXTRA_RENDER_RESOLUTION_MODE,
                SettingsActivity.prefs(activity).getInt(
                        SettingsActivity.K_RES_MODE, SettingsActivity.DEFAULT_RES_MODE));
        try {
            activity.startActivity(intent);
        } catch (ActivityNotFoundException e) {
            toast(activity.getString(R.string.game_activity_unavailable));
        }
    }

    private void copySafDirectory(Uri treeUri, Uri sourceDirectory, File destination,
                                  File stagingBoundary, int depth, CopyProgress copied,
                                  Set<String> visitedDocumentIds)
            throws Exception {
        checkCancelled();
        if (depth > MAX_TREE_DEPTH) {
            throw new IOException("слишком глубокое дерево папок");
        }
        String documentId = DocumentsContract.getDocumentId(sourceDirectory);
        if (!visitedDocumentIds.add(documentId)) {
            throw new IOException("провайдер вернул циклическое дерево папок");
        }
        ensureWithin(destination, stagingBoundary);
        ensureDirectory(destination);
        Set<String> childNames = new HashSet<>();
        int remainingNodes = MAX_IMPORT_NODES - copied.nodes;
        if (remainingNodes <= 0) {
            throw new IOException("слишком много объектов в импортируемой игре");
        }
        long remainingMetadataChars = MAX_DOCUMENT_METADATA_CHARS - copied.metadataChars;
        if (remainingMetadataChars <= 0) {
            throw new IOException("слишком много метаданных в дереве игры");
        }
        List<SafNode> children = listChildren(treeUri, sourceDirectory,
                Math.min(MAX_DIRECTORY_CHILDREN, remainingNodes), remainingMetadataChars);
        if (children.size() > MAX_IMPORT_NODES - copied.nodes) {
            throw new IOException("слишком много объектов в импортируемой игре");
        }
        long reservedMetadataChars = 0;
        for (SafNode child : children) {
            reservedMetadataChars += child.metadataChars;
        }
        if (reservedMetadataChars > MAX_DOCUMENT_METADATA_CHARS - copied.metadataChars) {
            throw new IOException("слишком много метаданных в дереве игры");
        }
        copied.nodes += children.size();
        copied.metadataChars += reservedMetadataChars;
        for (SafNode child : children) {
            validateDocumentName(child.name);
            if (!childNames.add(child.name)) {
                throw new IOException("повторяющееся имя в папке: " + child.name);
            }
            File output = checkedChild(destination, child.name);
            ensureWithin(output, stagingBoundary);
            if (child.directory) {
                copySafDirectory(treeUri, child.uri, output, stagingBoundary, depth + 1, copied,
                        visitedDocumentIds);
            } else {
                if (child.size >= 0 && child.size > copied.maxBytes - copied.bytes) {
                    throw new IOException("недостаточно места для файла " + child.name);
                }
                copyDocument(child.uri, output, copied);
            }
        }
    }

    private void copyDocument(Uri source, File destination, CopyProgress copied) throws Exception {
        checkCancelled();
        if (copied.files >= MAX_IMPORT_FILES) {
            throw new IOException("слишком много файлов в импортируемой игре");
        }
        ContentResolver resolver = activity.getContentResolver();
        try (InputStream input = resolver.openInputStream(source);
             FileOutputStream output = new FileOutputStream(destination)) {
            if (input == null) {
                throw new IOException("провайдер не открыл " + source);
            }
            byte[] buffer = new byte[COPY_BUFFER_BYTES];
            int count;
            while ((count = input.read(buffer)) != -1) {
                checkCancelled();
                if (copied.bytes > copied.maxBytes - count) {
                    throw new IOException("импорт остановлен: заканчивается свободное место");
                }
                output.write(buffer, 0, count);
                copied.bytes += count;
                if (copied.bytes - copied.lastReportedBytes >= 32L * 1024L * 1024L) {
                    copied.lastReportedBytes = copied.bytes;
                    postStatus(activity.getString(R.string.import_copy_progress,
                            copied.files, formatBytes(copied.bytes)));
                }
            }
            output.getFD().sync();
        }
        copied.files++;
    }

    private List<SafNode> listChildren(Uri treeUri, Uri parentDocument) throws IOException {
        return listChildren(treeUri, parentDocument, MAX_DIRECTORY_CHILDREN,
                MAX_DOCUMENT_METADATA_CHARS);
    }

    private List<SafNode> listChildren(Uri treeUri, Uri parentDocument, int maximumChildren,
                                       long maximumMetadataChars) throws IOException {
        if (maximumChildren <= 0) {
            throw new IOException("слишком много объектов в импортируемой игре");
        }
        if (maximumMetadataChars <= 0) {
            throw new IOException("слишком много метаданных в дереве игры");
        }
        String parentId = DocumentsContract.getDocumentId(parentDocument);
        Uri childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentId);
        String[] projection = new String[]{
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE,
                DocumentsContract.Document.COLUMN_SIZE
        };
        List<SafNode> nodes = new ArrayList<>();
        long metadataChars = 0;
        try (Cursor cursor = activity.getContentResolver().query(
                childrenUri, projection, null, null, null)) {
            if (cursor == null) {
                throw new IOException("провайдер не вернул содержимое папки");
            }
            while (cursor.moveToNext()) {
                if (nodes.size() >= maximumChildren) {
                    throw new IOException("слишком много объектов в одной папке");
                }
                String documentId = cursor.getString(0);
                String name = cursor.getString(1);
                String mime = cursor.getString(2);
                long size = cursor.isNull(3) ? -1 : cursor.getLong(3);
                if (documentId == null || documentId.length() > MAX_DOCUMENT_ID_LENGTH) {
                    throw new IOException("провайдер вернул некорректный document ID");
                }
                if (name == null || name.length() > MAX_DOCUMENT_NAME_LENGTH) {
                    throw new IOException("слишком длинное имя в выбранной папке");
                }
                Uri uri = DocumentsContract.buildDocumentUriUsingTree(treeUri, documentId);
                int nodeMetadataChars = documentId.length() + name.length()
                        + (mime == null ? 0 : mime.length()) + uri.toString().length() + 64;
                if (metadataChars > maximumMetadataChars - nodeMetadataChars) {
                    throw new IOException("слишком много метаданных в папке");
                }
                metadataChars += nodeMetadataChars;
                nodes.add(new SafNode(uri, name,
                        DocumentsContract.Document.MIME_TYPE_DIR.equals(mime), size,
                        nodeMetadataChars));
            }
        } catch (SecurityException e) {
            throw new IOException("нет доступа к выбранной папке", e);
        }
        return nodes;
    }

    private SafNode findDirectChild(Uri treeUri, Uri parent, String name, boolean directory)
            throws IOException {
        for (SafNode child : listChildren(treeUri, parent)) {
            if (name.equals(child.name) && child.directory == directory) {
                return child;
            }
        }
        return null;
    }

    private void installStagingTree(File staging, File destination, File stagingRoot)
            throws Exception {
        ensureWithin(staging, stagingRoot);
        ensureWithin(destination, appInstallRoot());
        if (destination.exists() || Files.isSymbolicLink(destination.toPath())) {
            throw new IOException("игра " + destination.getName()
                    + " уже установлена; слияние файлов запрещено");
        }
        try {
            Files.move(staging.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(staging.toPath(), destination.toPath());
        }
    }

    private static void moveReplace(File source, File target) throws IOException {
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void deleteTreeWithin(File path, File boundary) throws IOException {
        ensureWithin(path, boundary);
        File[] children = path.listFiles();
        if (children != null) {
            for (File child : children) {
                if (child.isDirectory()) {
                    deleteTreeWithin(child, boundary);
                } else if (!child.delete()) {
                    throw new IOException("не удалось удалить временный файл " + child.getName());
                }
            }
        }
        if (path.exists() && !path.delete()) {
            throw new IOException("не удалось очистить staging " + path.getName());
        }
    }

    private void writeInstallManifest(File directory, String titleId, String title, String source,
                                      boolean ebootExtracted, String installedBy,
                                      String platform) throws Exception {
        JSONObject json = new JSONObject();
        json.put("titleId", titleId);
        json.put("title", title == null ? "" : title);
        json.put("source", source);
        json.put("ebootExtracted", ebootExtracted);
        json.put("installedBy", installedBy);
        json.put("platform", platform);
        File target = checkedChild(directory, "executor-installed.json");
        File temporary = checkedChild(directory, ".executor-installed.json.tmp");
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(json.toString().getBytes(StandardCharsets.UTF_8));
            output.getFD().sync();
        }
        moveReplace(temporary, target);
    }

    private PkgMetadata preflightPkg(File pkg) throws Exception {
        try (RandomAccessFile file = new RandomAccessFile(pkg, "r")) {
            long length = file.length();
            if (length < 32 || u32be(file, 0) != 0x7f434e54L) {
                throw new IOException("файл не является PS4 PKG");
            }
            long entryCount = u32be(file, 0x10);
            long tableOffset = u32be(file, 0x18);
            if (entryCount <= 0 || entryCount > 8192 || tableOffset < 0
                    || tableOffset > length || entryCount > (length - tableOffset) / 32L) {
                throw new IOException("повреждена таблица PKG");
            }
            byte[] paramSfo = null;
            int paramSfoEntries = 0;
            for (long index = 0; index < entryCount; index++) {
                long entry = tableOffset + index * 32L;
                long id = u32be(file, entry);
                if (id != PkgInstaller.ENTRY_PARAM_SFO) {
                    continue;
                }
                paramSfoEntries++;
                if (paramSfoEntries > 1) {
                    throw new IOException("PKG содержит несколько param.sfo");
                }
                long offset = u32be(file, entry + 16);
                long size = u32be(file, entry + 20);
                if (size <= 0 || size > MAX_SFO_BYTES || offset > length
                        || size > length - offset) {
                    throw new IOException("повреждён param.sfo в PKG");
                }
                paramSfo = new byte[(int) size];
                file.seek(offset);
                file.readFully(paramSfo);
            }
            if (paramSfo == null) {
                throw new IOException("PKG не содержит param.sfo");
            }
            String id = normalizeTitleId(PkgInstaller.sfoString(paramSfo, "TITLE_ID"));
            String title = cleanSfoString(PkgInstaller.sfoString(paramSfo, "TITLE"));
            return new PkgMetadata(id, title);
        }
    }

    private long copyUriToFile(Uri uri, File destination, long maximumBytes) throws Exception {
        long copied = 0;
        try (InputStream input = activity.getContentResolver().openInputStream(uri);
             FileOutputStream output = new FileOutputStream(destination)) {
            if (input == null) {
                throw new IOException("провайдер не открыл выбранный PKG");
            }
            byte[] buffer = new byte[COPY_BUFFER_BYTES];
            int count;
            long lastReport = 0;
            while ((count = input.read(buffer)) != -1) {
                checkCancelled();
                if (copied > maximumBytes - count) {
                    throw new IOException("импорт остановлен: заканчивается свободное место");
                }
                output.write(buffer, 0, count);
                copied += count;
                if (copied - lastReport >= 64L * 1024L * 1024L) {
                    lastReport = copied;
                    postStatus(activity.getString(R.string.import_pkg_copy_progress,
                            formatBytes(copied)));
                }
            }
            output.getFD().sync();
        } catch (Exception e) {
            if (destination.exists()) {
                destination.delete();
            }
            throw e;
        }
        return copied;
    }

    private String queryDisplayName(Uri uri) {
        try (Cursor cursor = activity.getContentResolver().query(
                uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst() && !cursor.isNull(0)) {
                return cursor.getString(0);
            }
        } catch (Exception ignored) {
        }
        return uri.getLastPathSegment();
    }

    private long querySize(Uri uri) {
        try (Cursor cursor = activity.getContentResolver().query(
                uri, new String[]{OpenableColumns.SIZE}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst() && !cursor.isNull(0)) {
                return Math.max(0, cursor.getLong(0));
            }
        } catch (Exception ignored) {
        }
        return -1;
    }

    private static byte[] readUriLimited(ContentResolver resolver, Uri uri, int maximum)
            throws IOException {
        try (InputStream input = resolver.openInputStream(uri);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            if (input == null) {
                throw new IOException("провайдер не открыл param.sfo");
            }
            byte[] buffer = new byte[8192];
            int total = 0;
            int count;
            while ((count = input.read(buffer)) != -1) {
                total += count;
                if (total > maximum) {
                    throw new IOException("param.sfo слишком большой");
                }
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static byte[] readFileLimited(File file, int maximum) throws IOException {
        long length = file.length();
        if (length <= 0 || length > maximum) {
            throw new IOException("metadata file has invalid size");
        }
        byte[] data = new byte[(int) length];
        try (FileInputStream input = new FileInputStream(file)) {
            int offset = 0;
            while (offset < data.length) {
                int count = input.read(data, offset, data.length - offset);
                if (count < 0) {
                    throw new IOException("truncated metadata file");
                }
                offset += count;
            }
        }
        return data;
    }

    private static Ps5Metadata readPs5Metadata(byte[] data) throws IOException {
        try {
            JSONObject json = new JSONObject(new String(data, StandardCharsets.UTF_8));
            String titleId = normalizeTitleId(json.optString("titleId", ""));
            String title = "";
            JSONObject localized = json.optJSONObject("localizedParameters");
            if (localized != null) {
                String language = localized.optString("defaultLanguage", "");
                JSONObject selected = localized.optJSONObject(language);
                if (selected == null) {
                    java.util.Iterator<String> keys = localized.keys();
                    while (keys.hasNext() && selected == null) {
                        Object value = localized.opt(keys.next());
                        if (value instanceof JSONObject) {
                            selected = (JSONObject) value;
                        }
                    }
                }
                if (selected != null) {
                    title = selected.optString("titleName", "");
                }
            }
            return new Ps5Metadata(titleId, title.trim());
        } catch (Exception error) {
            throw new IOException("повреждён sce_sys/param.json", error);
        }
    }

    private String readSfoValue(File sfo, String key) throws IOException {
        long length = sfo.length();
        if (length <= 0 || length > MAX_SFO_BYTES) {
            return null;
        }
        byte[] data = new byte[(int) length];
        try (FileInputStream input = new FileInputStream(sfo)) {
            int offset = 0;
            while (offset < data.length) {
                int count = input.read(data, offset, data.length - offset);
                if (count < 0) {
                    throw new IOException("обрезанный param.sfo");
                }
                offset += count;
            }
        }
        String value = cleanSfoString(PkgInstaller.sfoString(data, key));
        return "TITLE_ID".equals(key) ? normalizeTitleId(value) : value;
    }

    private static long u32be(RandomAccessFile file, long offset) throws IOException {
        file.seek(offset);
        int b0 = file.read();
        int b1 = file.read();
        int b2 = file.read();
        int b3 = file.read();
        if ((b0 | b1 | b2 | b3) < 0) {
            throw new IOException("unexpected EOF at " + offset);
        }
        return ((long) b0 << 24) | ((long) b1 << 16) | ((long) b2 << 8) | b3;
    }

    private File homeDir() throws IOException {
        return checkedChild(activity.getFilesDir(), "lsx4-home");
    }

    private File appInstallRoot() throws IOException {
        File runtimeFs = checkedChild(homeDir(), "runtime-fs");
        File user = checkedChild(runtimeFs, "user");
        return checkedChild(user, "app");
    }

    private long availableImportBudget(File existingDirectory) throws IOException {
        ensureDirectory(existingDirectory);
        StatFs stat = new StatFs(existingDirectory.getAbsolutePath());
        long available = stat.getAvailableBytes();
        long budget = available - FREE_SPACE_RESERVE_BYTES;
        if (budget <= 0) {
            throw new IOException("недостаточно свободного места; требуется резерв не менее 512 MiB");
        }
        return budget;
    }

    private static File checkedChild(File parent, String name) throws IOException {
        validateDocumentName(name);
        File child = new File(parent, name);
        ensureWithin(child, parent);
        return child;
    }

    private static File uniqueChild(File parent, String requestedName) throws IOException {
        File candidate = checkedChild(parent, requestedName);
        if (candidate.createNewFile()) {
            return candidate;
        }
        int dot = requestedName.lastIndexOf('.');
        String stem = dot > 0 ? requestedName.substring(0, dot) : requestedName;
        String extension = dot > 0 ? requestedName.substring(dot) : "";
        for (int suffix = 1; suffix <= 10000; suffix++) {
            candidate = checkedChild(parent, stem + "-" + suffix + extension);
            if (candidate.createNewFile()) {
                return candidate;
            }
        }
        throw new IOException("слишком много файлов с одинаковым именем");
    }

    private static void validateDocumentName(String name) throws IOException {
        if (name == null || name.isEmpty() || ".".equals(name) || "..".equals(name)
                || name.indexOf('/') >= 0 || name.indexOf('\\') >= 0
                || name.indexOf('\0') >= 0) {
            throw new IOException("небезопасное имя файла");
        }
    }

    private static String safeFileName(String name, String fallback) {
        if (name == null || name.trim().isEmpty()) {
            return fallback;
        }
        String safe = name.replaceAll("[^A-Za-z0-9._ -]", "_").trim();
        while (safe.startsWith(".")) {
            safe = safe.substring(1);
        }
        return safe.isEmpty() || ".".equals(safe) || "..".equals(safe) ? fallback : safe;
    }

    private static void ensureWithin(File child, File parent) throws IOException {
        String childPath = child.getCanonicalPath();
        String parentPath = parent.getCanonicalPath();
        if (!childPath.equals(parentPath)
                && !childPath.startsWith(parentPath + File.separator)) {
            throw new IOException("путь выходит за границу импорта");
        }
    }

    private static void ensureDirectory(File directory) throws IOException {
        if (directory.isDirectory()) {
            return;
        }
        if (directory.exists() || !directory.mkdirs()) {
            throw new IOException("не удалось создать " + directory.getName());
        }
    }

    private static boolean safeRegularFileWithin(File file, File boundary) {
        try {
            if (!file.isFile() || Files.isSymbolicLink(file.toPath())) {
                return false;
            }
            ensureWithin(file, boundary);
            return true;
        } catch (IOException ignored) {
            return false;
        }
    }

    private static boolean isPlausibleEboot(File file, File boundary) {
        if (!safeRegularFileWithin(file, boundary) || file.length() < 4) {
            return false;
        }
        byte[] magic = new byte[4];
        try (FileInputStream input = new FileInputStream(file)) {
            if (input.read(magic) != magic.length) {
                return false;
            }
        } catch (IOException ignored) {
            return false;
        }
        boolean elf = (magic[0] & 0xff) == 0x7f
                && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F';
        boolean ps4Self = (magic[0] & 0xff) == 0x4f
                && (magic[1] & 0xff) == 0x15
                && (magic[2] & 0xff) == 0x3d
                && (magic[3] & 0xff) == 0x1d;
        return elf || ps4Self;
    }

    private void checkCancelled() throws IOException {
        if (destroyed || Thread.currentThread().isInterrupted()) {
            throw new IOException("импорт отменён: Activity закрыта");
        }
    }

    private static boolean isValidTitleId(String titleId) {
        return titleId != null && TITLE_ID.matcher(titleId).matches();
    }

    private static String normalizeTitleId(String value) {
        String clean = cleanSfoString(value);
        return clean == null ? null : clean.toUpperCase(Locale.ROOT);
    }

    private static String cleanSfoString(String value) {
        if (value == null) {
            return null;
        }
        int nul = value.indexOf('\0');
        String clean = (nul >= 0 ? value.substring(0, nul) : value).trim();
        return clean.isEmpty() ? null : clean;
    }

    private static String displayTitle(String title, String titleId) {
        return title == null || title.isEmpty() ? titleId : title;
    }

    private static String formatBytes(long bytes) {
        if (bytes >= 1024L * 1024L * 1024L) {
            return String.format(Locale.US, "%.2f GiB", bytes / (1024.0 * 1024.0 * 1024.0));
        }
        if (bytes >= 1024L * 1024L) {
            return String.format(Locale.US, "%.1f MiB", bytes / (1024.0 * 1024.0));
        }
        if (bytes >= 1024L) {
            return String.format(Locale.US, "%.1f KiB", bytes / 1024.0);
        }
        return bytes + " B";
    }

    private static String friendlyMessage(Throwable throwable) {
        String message = throwable.getMessage();
        return message == null || message.trim().isEmpty()
                ? throwable.getClass().getSimpleName() : message;
    }

    private static Bitmap decodeIcon(File icon, int requestedPixels) {
        if (icon == null || !icon.isFile()) {
            return null;
        }
        try {
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(icon.getAbsolutePath(), bounds);
            int sample = 1;
            while (bounds.outWidth / sample > requestedPixels * 2
                    || bounds.outHeight / sample > requestedPixels * 2) {
                sample *= 2;
            }
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inSampleSize = sample;
            return BitmapFactory.decodeFile(icon.getAbsolutePath(), options);
        } catch (Throwable ignored) {
            return null;
        }
    }

    private void finishUiWork(List<GameEntry> games, String message) {
        postUi(() -> {
            renderGameList(games);
            setBusy(false, message);
        });
    }

    private void setBusy(boolean busy, String message) {
        if (progress != null) {
            progress.setVisibility(busy ? View.VISIBLE : View.GONE);
        }
        if (status != null) {
            status.setText(message);
        }
    }

    private void setStatus(String message) {
        postUi(() -> {
            if (status != null) {
                status.setText(message);
            }
        });
    }

    private void postStatus(String message) {
        setStatus(message);
    }

    private void postUi(Runnable runnable) {
        activity.runOnUiThread(() -> {
            if (destroyed || activity.isFinishing()
                    || (Build.VERSION.SDK_INT >= 17 && activity.isDestroyed())) {
                return;
            }
            runnable.run();
        });
    }

    private void toast(String message) {
        postUi(() -> Toast.makeText(activity, message, Toast.LENGTH_LONG).show());
    }

    private TextView label(String text, int sp, int color) {
        TextView label = new TextView(activity);
        label.setText(text);
        label.setTextSize(sp);
        label.setTextColor(color);
        return label;
    }

    private GradientDrawable roundedBackground(int fill, int stroke, int radiusDp) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(fill);
        drawable.setCornerRadius(dp(radiusDp));
        drawable.setStroke(dp(1), stroke);
        return drawable;
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private LinearLayout.LayoutParams matchWrapWithMargins(int left, int top, int right,
                                                            int bottom) {
        LinearLayout.LayoutParams lp = matchWrap();
        lp.setMargins(dp(left), dp(top), dp(right), dp(bottom));
        return lp;
    }

    private int dp(int value) {
        return Math.round(value * activity.getResources().getDisplayMetrics().density);
    }

    private static int clamp(int value, int minimum, int maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private interface ImportJob {
        String run() throws Exception;
    }

    private static final class GameEntry {
        final File directory;
        final File eboot;
        final File icon;
        final String titleId;
        final String title;
        final String platform;
        final boolean launchable;
        final int rootPriority;

        GameEntry(File directory, File eboot, File icon, String titleId, String title,
                  String platform, boolean launchable, int rootPriority) {
            this.directory = directory;
            this.eboot = eboot;
            this.icon = icon;
            this.titleId = titleId;
            this.title = title;
            this.platform = platform;
            this.launchable = launchable;
            this.rootPriority = rootPriority;
        }
    }

    private static final class Ps5Metadata {
        final String titleId;
        final String title;

        Ps5Metadata(String titleId, String title) {
            this.titleId = titleId;
            this.title = title;
        }
    }

    private static final class SafNode {
        final Uri uri;
        final String name;
        final boolean directory;
        final long size;
        final int metadataChars;

        SafNode(Uri uri, String name, boolean directory, long size, int metadataChars) {
            this.uri = uri;
            this.name = name;
            this.directory = directory;
            this.size = size;
            this.metadataChars = metadataChars;
        }
    }

    private static final class CopyProgress {
        int nodes;
        int files;
        long bytes;
        long lastReportedBytes;
        long metadataChars;
        final long maxBytes;

        CopyProgress(long maxBytes) {
            this.maxBytes = maxBytes;
        }

        void reserveNode(boolean directory) throws IOException {
            nodes++;
            if (nodes > MAX_IMPORT_NODES) {
                throw new IOException("слишком много записей в архиве");
            }
            if (!directory) {
                files++;
                if (files > MAX_IMPORT_FILES) {
                    throw new IOException("слишком много файлов в архиве");
                }
            }
        }

        void reserveFile(long declaredBytes) throws IOException {
            if (declaredBytes > maxBytes || (declaredBytes >= 0
                    && bytes > maxBytes - declaredBytes)) {
                throw new IOException("распаковка остановлена: недостаточно свободного места");
            }
        }

        void addExtracted(long count) throws IOException {
            if (count < 0 || bytes > maxBytes - count) {
                throw new IOException("распаковка остановлена: недостаточно свободного места");
            }
            bytes += count;
        }
    }

    private static final class BudgetOutputStream extends OutputStream {
        private final OutputStream output;
        private final CopyProgress progress;
        private final long declaredBytes;
        private long written;

        BudgetOutputStream(OutputStream output, CopyProgress progress, long declaredBytes) {
            this.output = output;
            this.progress = progress;
            this.declaredBytes = declaredBytes;
        }

        @Override
        public void write(int value) throws IOException {
            byte[] one = new byte[]{(byte) value};
            write(one, 0, 1);
        }

        @Override
        public void write(byte[] buffer, int offset, int count) throws IOException {
            if (declaredBytes >= 0 && written > declaredBytes - count) {
                throw new IOException("запись архива превышает заявленный размер");
            }
            progress.addExtracted(count);
            output.write(buffer, offset, count);
            written += count;
        }

        @Override
        public void flush() throws IOException {
            output.flush();
        }

        @Override
        public void close() throws IOException {
            output.close();
        }
    }

    private static final class PkgMetadata {
        final String titleId;
        final String title;

        PkgMetadata(String titleId, String title) {
            this.titleId = titleId;
            this.title = title;
        }
    }
}
