package app.lsx4.android;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;

public final class PkgInstaller {

    public static final int ENTRY_PARAM_SFO = 0x1000;
    public static final int ENTRY_ICON0_PNG = 0x1200;
    public static final int ENTRY_PIC1_PNG = 0x1220;

    public static final class Result {
        public boolean ok;
        public String titleId;
        public String title;
        public boolean hasIcon;
        public boolean ebootExtracted;
        public File installDir;
        public String message;
    }

    private PkgInstaller() {
    }

    private static long u32be(RandomAccessFile f, long off) throws IOException {
        f.seek(off);
        int b0 = f.read(), b1 = f.read(), b2 = f.read(), b3 = f.read();
        if ((b0 | b1 | b2 | b3) < 0) throw new IOException("eof@" + off);
        return ((long) b0 << 24) | ((long) b1 << 16) | (b2 << 8) | b3;
    }

    public static Result install(File pkg, File homeDir) {
        Result r = new Result();
        try (RandomAccessFile f = new RandomAccessFile(pkg, "r")) {
            long magic = u32be(f, 0);
            if (magic != 0x7F434E54L) {
                r.message = "Not a PKG (bad magic 0x" + Long.toHexString(magic) + ")";
                return r;
            }
            long entryCount = u32be(f, 0x10);
            long tableOffset = u32be(f, 0x18);
            if (entryCount <= 0 || entryCount > 8192 || tableOffset <= 0 || tableOffset > f.length()) {
                r.message = "Bad PKG entry table (count=" + entryCount + " off=" + tableOffset + ")";
                return r;
            }

            byte[] paramSfo = null, icon0 = null, pic1 = null;
            for (long i = 0; i < entryCount; i++) {
                long e = tableOffset + i * 32L;
                if (e + 32 > f.length()) break;
                long id = u32be(f, e);
                long offset = u32be(f, e + 16);
                long size = u32be(f, e + 20);
                if (offset <= 0 || size <= 0 || offset + size > f.length()) continue;
                if (id == ENTRY_PARAM_SFO && size < 1 << 20) paramSfo = readAt(f, offset, (int) size);
                else if (id == ENTRY_ICON0_PNG && size < 8 << 20) icon0 = readAt(f, offset, (int) size);
                else if (id == ENTRY_PIC1_PNG && size < 16 << 20) pic1 = readAt(f, offset, (int) size);
            }

            if (paramSfo == null) {
                r.message = "PKG has no param.sfo entry";
                return r;
            }
            r.title = sfoString(paramSfo, "TITLE");
            r.titleId = sfoString(paramSfo, "TITLE_ID");
            if (r.titleId != null) {
                r.titleId = r.titleId.trim().toUpperCase(java.util.Locale.ROOT);
            }
            if (r.titleId == null || !r.titleId.matches("[A-Z0-9]{9}")) {
                r.message = "PKG has an invalid TITLE_ID; import refused";
                return r;
            }

            File appRoot = new File(new File(new File(homeDir, "runtime-fs"), "user"), "app");
            File dir = new File(appRoot, r.titleId);
            File sceSys = new File(dir, "sce_sys");
            if (!sceSys.exists() && !sceSys.mkdirs()) {
                r.message = "Cannot create " + sceSys;
                return r;
            }
            writeBytes(new File(sceSys, "param.sfo"), paramSfo);
            if (icon0 != null) { writeBytes(new File(sceSys, "icon0.png"), icon0); r.hasIcon = true; }
            if (pic1 != null) writeBytes(new File(sceSys, "pic1.png"), pic1);

            String safeTitle = r.title == null ? "" : r.title.replace("\"", "'");
            String json = "{\"titleId\":\"" + r.titleId + "\",\"title\":\"" + safeTitle
                    + "\",\"source\":\"" + pkg.getAbsolutePath().replace("\\", "/")
                    + "\",\"ebootExtracted\":false,\"installedBy\":\"on-device-pkg-import\"}";
            writeBytes(new File(dir, "executor-installed.json"), json.getBytes(StandardCharsets.UTF_8));

            r.installDir = dir;
            r.ebootExtracted = new File(dir, "eboot.bin").isFile();
            r.ok = true;
            r.message = "Imported " + (r.title != null ? r.title : r.titleId)
                    + (r.ebootExtracted ? " (launchable)" : " (metadata only; eboot extraction pending)");
            return r;
        } catch (Exception ex) {
            r.message = "Import failed: " + ex;
            return r;
        }
    }

    private static byte[] readAt(RandomAccessFile f, long off, int size) throws IOException {
        byte[] b = new byte[size];
        f.seek(off);
        f.readFully(b);
        return b;
    }

    private static void writeBytes(File dest, byte[] data) throws IOException {
        try (FileOutputStream o = new FileOutputStream(dest)) {
            o.write(data);
        }
    }

    public static String sfoString(byte[] b, String wantKey) {
        if (b == null || b.length < 20 || b[0] != 0x00 || b[1] != 0x50 || b[2] != 0x53 || b[3] != 0x46) {
            return null;
        }
        int keyTable = le32(b, 8);
        int dataTable = le32(b, 12);
        int count = le32(b, 16);
        for (int e = 0; e < count; e++) {
            int idx = 20 + e * 16;
            if (idx + 16 > b.length) break;
            int keyOff = (b[idx] & 0xff) | ((b[idx + 1] & 0xff) << 8);
            int dataLen = le32(b, idx + 4);
            int dataOff = le32(b, idx + 12);
            String key = cstr(b, keyTable + keyOff);
            if (wantKey.equals(key)) {
                int ds = dataTable + dataOff;
                int de = Math.min(ds + dataLen, b.length);
                if (ds < 0 || ds >= b.length) return null;
                return new String(b, ds, Math.max(0, de - ds), StandardCharsets.UTF_8).trim();
            }
        }
        return null;
    }

    private static int le32(byte[] b, int o) {
        if (o < 0 || o + 4 > b.length) return 0;
        return (b[o] & 0xff) | ((b[o + 1] & 0xff) << 8) | ((b[o + 2] & 0xff) << 16) | ((b[o + 3] & 0xff) << 24);
    }

    private static String cstr(byte[] b, int o) {
        if (o < 0 || o >= b.length) return "";
        int e = o;
        while (e < b.length && b[e] != 0) e++;
        return new String(b, o, e - o, StandardCharsets.UTF_8);
    }
}
