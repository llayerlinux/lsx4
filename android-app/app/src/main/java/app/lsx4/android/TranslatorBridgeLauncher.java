package app.lsx4.android;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.json.JSONArray;
import org.json.JSONObject;

final class TranslatorBridgeLauncher {
    private static final long HELPER_TIMEOUT_SECONDS = 6;

    static final class Result {
        final String helperLine;
        final String requestLine;
        final boolean reported;
        final int reportResult;

        Result(String requestLine, String helperLine, boolean reported, int reportResult) {
            this.requestLine = requestLine;
            this.helperLine = helperLine;
            this.reported = reported;
            this.reportResult = reportResult;
        }
    }

    private TranslatorBridgeLauncher() {
    }

    static Result launch(File runtimeRoot, File nativeLibraryDir) throws Exception {
        LaunchRequest request = LaunchRequest.fromRuntime(runtimeRoot, nativeLibraryDir);
        if (!request.ready) {
            return new Result(request.line, "Translator bridge helper result: request_not_ready",
                    false, 0);
        }
        if (request.argv.size() < 2) {
            return new Result(request.line, "Translator bridge helper result: invalid_argv",
                    false, 0);
        }

        if (!request.helperPath.isFile()) {
            return new Result(request.line, "Translator bridge helper result: missing " +
                    request.helperPath.getAbsolutePath(), false, 0);
        }
        if (!request.mappedEntrypoint.isFile()) {
            return new Result(request.line, "Translator bridge helper result: missing " +
                    request.mappedEntrypoint.getAbsolutePath(), false, 0);
        }
        if (request.requestFile != null && !request.requestFile.isFile()) {
            return new Result(request.line, "Translator bridge helper result: missing requestFile " +
                    request.requestFile.getAbsolutePath(), false, 0);
        }

        ProcessBuilder builder = new ProcessBuilder(request.argv);
        builder.directory(request.workingDirectory);
        builder.redirectErrorStream(true);
        builder.redirectOutput(request.stdoutLog);

        Map<String, String> env = builder.environment();
        env.clear();
        env.putAll(request.env);

        long started = System.nanoTime();
        Process process = builder.start();
        boolean finished = process.waitFor(request.timeoutMs, TimeUnit.MILLISECONDS);
        if (!finished) {
            process.destroyForcibly();
            return new Result(request.line, "Translator bridge helper result: timeout log=" +
                    readSmallFile(request.stdoutLog), false, 0);
        }

        int helperExit = process.exitValue();
        long durationMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started);
        String logTail = readSmallFile(request.stdoutLog);
        int reportResult = reportResult(request.requestId, request.resultFile, helperExit,
                durationMs, logTail);
        String line = "Translator bridge helper result: exit=" + helperExit +
                " durationMs=" + durationMs + " log=" + logTail;
        return new Result(request.line, line, true, reportResult);
    }

    private static int reportResult(
            String requestId,
            File resultFile,
            int helperExit,
            long durationMs,
            String logTail) throws Exception {
        JSONObject json = new JSONObject();
        json.put("schema", 1);
        json.put("requestId", requestId);
        json.put("exitCode", helperExit);
        json.put("durationMs", durationMs);
        json.put("logTail", logTail);
        if (resultFile != null) {
            json.put("resultFile", resultFile.getAbsolutePath());
            writeSmallFile(resultFile, json.toString());
        }

        int result = RuntimeBridge.reportTranslatorResultJson(json.toString());
        if (result == -2) {
            return RuntimeBridge.reportTranslatorResult(helperExit);
        }
        return result;
    }

    private static String readSmallFile(File file) {
        if (!file.isFile()) {
            return "";
        }

        try (FileInputStream in = new FileInputStream(file)) {
            byte[] buffer = new byte[(int) Math.min(file.length(), 4096)];
            int read = in.read(buffer);
            if (read <= 0) {
                return "";
            }
            return new String(buffer, 0, read, StandardCharsets.UTF_8)
                    .replace('\n', ' ')
                    .replace('\r', ' ')
                    .trim();
        } catch (Exception ignored) {
            return "";
        }
    }

    private static void writeSmallFile(File file, String value) throws Exception {
        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("Cannot create " + parent);
        }
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(value.getBytes(StandardCharsets.UTF_8));
        }
    }

    private static final class LaunchRequest {
        final boolean ready;
        final String line;
        final String requestId;
        final File requestFile;
        final File resultFile;
        final java.util.ArrayList<String> argv;
        final File helperPath;
        final File mappedEntrypoint;
        final File workingDirectory;
        final File stdoutLog;
        final long timeoutMs;
        final Map<String, String> env;

        LaunchRequest(
                boolean ready,
                String line,
                String requestId,
                File requestFile,
                File resultFile,
                java.util.ArrayList<String> argv,
                File helperPath,
                File mappedEntrypoint,
                File workingDirectory,
                File stdoutLog,
                long timeoutMs,
                Map<String, String> env) {
            this.ready = ready;
            this.line = line;
            this.requestId = requestId;
            this.requestFile = requestFile;
            this.resultFile = resultFile;
            this.argv = argv;
            this.helperPath = helperPath;
            this.mappedEntrypoint = mappedEntrypoint;
            this.workingDirectory = workingDirectory;
            this.stdoutLog = stdoutLog;
            this.timeoutMs = timeoutMs;
            this.env = env;
        }

        static LaunchRequest fromRuntime(File runtimeRoot, File nativeLibraryDir) throws Exception {
            String raw = RuntimeBridge.translatorLaunchRequest();
            if (raw == null || raw.isEmpty()) {
                File translatorDir = new File(runtimeRoot, "translator");
                return fallback(runtimeRoot, nativeLibraryDir, translatorDir);
            }

            JSONObject json = new JSONObject(raw);
            boolean ready = json.optBoolean("ready", false);
            String result = json.optString("result", "");
            String requestId = json.optString("requestId", "");
            File requestFile = optionalFile(json.optString("requestFile", ""));
            File resultFile = optionalFile(json.optString("resultFile", ""));
            String line = "Translator bridge request: ready=" + ready + " result=" + result;
            if (!requestId.isEmpty()) {
                line += " requestId=" + requestId;
            }
            if (requestFile != null) {
                line += " requestFile=" + requestFile.isFile();
            }
            if (resultFile != null) {
                line += " resultFile=" + resultFile.getName();
            }
            JSONObject guestJson = json.optJSONObject("guest");
            if (guestJson != null) {
                String entryVa = guestJson.optString("entryVa", "");
                JSONArray segments = guestJson.optJSONArray("executableSegments");
                int segmentCount = segments != null ? segments.length() : 0;
                if (!entryVa.isEmpty()) {
                    line += " guestEntryVa=" + entryVa + " guestExecSegments=" + segmentCount;
                }
                JSONObject memoryJson = guestJson.optJSONObject("memory");
                if (memoryJson != null) {
                    JSONArray memorySegments = memoryJson.optJSONArray("segments");
                    JSONArray writableSegments = memoryJson.optJSONArray("writableSegments");
                    String jitCacheDir = memoryJson.optString("jitCacheDir", "");
                    line += " guestSegments=" + (memorySegments != null ? memorySegments.length() : 0) +
                            " guestWritableSegments=" +
                            (writableSegments != null ? writableSegments.length() : 0) +
                            " jitCache=" + !jitCacheDir.isEmpty();
                }
                JSONObject hleJson = guestJson.optJSONObject("hle");
                if (hleJson != null) {
                    JSONArray callbacks = hleJson.optJSONArray("callbacks");
                    line += " hleCallbacks=" + (callbacks != null ? callbacks.length() : 0);
                }
            }

            File helperPath = new File(json.optString("helperPath", ""));
            File mappedEntrypoint = new File(json.optString("mappedEntrypoint", ""));
            java.util.ArrayList<String> argv = readArgv(json.optJSONArray("argv"));
            if (argv.isEmpty()) {
                argv.add(helperPath.getAbsolutePath());
                argv.add(mappedEntrypoint.getAbsolutePath());
            } else {
                helperPath = new File(argv.get(0));
                if (argv.size() == 2) {
                    mappedEntrypoint = new File(argv.get(1));
                }
            }

            File workingDirectory = new File(json.optString(
                    "workingDirectory",
                    new File(runtimeRoot, "translator").getAbsolutePath()));
            JSONObject logsJson = json.optJSONObject("logs");
            String stdoutPath = logsJson != null ? logsJson.optString("stdout", "") : "";
            if (stdoutPath.isEmpty()) {
                stdoutPath = json.optString(
                        "stdoutLog",
                        new File(workingDirectory, "box64.java-process.log").getAbsolutePath());
            }
            File stdoutLog = new File(stdoutPath);
            long timeoutMs = json.optLong("timeoutMs", 0);
            if (timeoutMs <= 0) {
                timeoutMs = json.optLong("timeoutSeconds", HELPER_TIMEOUT_SECONDS) * 1000;
            }

            JSONObject envJson = json.optJSONObject("env");
            java.util.HashMap<String, String> env = new java.util.HashMap<>();
            if (envJson != null) {
                java.util.Iterator<String> keys = envJson.keys();
                while (keys.hasNext()) {
                    String key = keys.next();
                    env.put(key, envJson.optString(key, ""));
                }
            }
            if (env.isEmpty()) {
                fillDefaultEnv(env, runtimeRoot, workingDirectory);
            }

            return new LaunchRequest(ready, line, requestId, requestFile, resultFile, argv,
                    helperPath, mappedEntrypoint, workingDirectory, stdoutLog, timeoutMs, env);
        }

        private static LaunchRequest fallback(File runtimeRoot, File nativeLibraryDir, File translatorDir) {
            java.util.HashMap<String, String> env = new java.util.HashMap<>();
            fillDefaultEnv(env, runtimeRoot, translatorDir);
            File helperPath = new File(nativeLibraryDir, "libbox64_executor.so");
            File mappedEntrypoint = new File(translatorDir, "mapped-entrypoint-from-eboot.elf");
            java.util.ArrayList<String> argv = new java.util.ArrayList<>();
            argv.add(helperPath.getAbsolutePath());
            argv.add(mappedEntrypoint.getAbsolutePath());
            return new LaunchRequest(
                    true,
                    "Translator bridge request: ready=true result=fallback",
                    "",
                    null,
                    null,
                    argv,
                    helperPath,
                    mappedEntrypoint,
                    translatorDir,
                    new File(translatorDir, "box64.java-process.log"),
                    HELPER_TIMEOUT_SECONDS * 1000,
                    env);
        }

        private static File optionalFile(String path) {
            if (path == null || path.isEmpty()) {
                return null;
            }
            return new File(path);
        }

        private static java.util.ArrayList<String> readArgv(JSONArray json) {
            java.util.ArrayList<String> argv = new java.util.ArrayList<>();
            if (json == null) {
                return argv;
            }

            for (int i = 0; i < json.length(); i++) {
                String arg = json.optString(i, "");
                if (!arg.isEmpty()) {
                    argv.add(arg);
                }
            }
            return argv;
        }

        private static void fillDefaultEnv(
                java.util.HashMap<String, String> env,
                File runtimeRoot,
                File translatorDir) {
            env.put("PATH", "/system/bin:/system/xbin:/apex/com.android.runtime/bin");
            env.put("HOME", runtimeRoot.getAbsolutePath());
            env.put("TMPDIR", translatorDir.getAbsolutePath());
            env.put("BOX64_NOBANNER", "1");
        }
    }
}
