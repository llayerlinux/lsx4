package app.lsx4.android;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Versioned, application-shipped patch database. UI code contains no title-specific branches. */
final class GamePatchCatalog {
    static final class Patch {
        final String id;
        final String preferenceKey;
        final int titleResource;
        final boolean inverted;

        Patch(String id, String preferenceKey, int titleResource, boolean inverted) {
            this.id = id;
            this.preferenceKey = preferenceKey;
            this.titleResource = titleResource;
            this.inverted = inverted;
        }
    }

    static final class Game {
        final String name;
        final List<String> titleIds;
        final List<Patch> patches;

        Game(String name, List<String> titleIds, List<Patch> patches) {
            this.name = name;
            this.titleIds = titleIds;
            this.patches = patches;
        }
    }

    private GamePatchCatalog() {
    }

    static List<Game> load(Context context) {
        try (InputStream input = context.getAssets().open("game_patches.json")) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                output.write(buffer, 0, count);
            }
            JSONObject root = new JSONObject(output.toString(StandardCharsets.UTF_8.name()));
            JSONArray games = root.getJSONArray("games");
            List<Game> result = new ArrayList<>();
            for (int i = 0; i < games.length(); ++i) {
                JSONObject game = games.getJSONObject(i);
                List<String> titleIds = new ArrayList<>();
                JSONArray ids = game.getJSONArray("titleIds");
                for (int j = 0; j < ids.length(); ++j) {
                    titleIds.add(ids.getString(j));
                }
                List<Patch> patches = new ArrayList<>();
                JSONArray entries = game.getJSONArray("patches");
                for (int j = 0; j < entries.length(); ++j) {
                    JSONObject patch = entries.getJSONObject(j);
                    int title = context.getResources().getIdentifier(
                            patch.getString("title"), "string", context.getPackageName());
                    patches.add(new Patch(patch.getString("id"),
                            patch.getString("preferenceKey"), title,
                            patch.optBoolean("inverted", true)));
                }
                result.add(new Game(game.getString("name"), titleIds, patches));
            }
            return result;
        } catch (Exception error) {
            android.util.Log.e("LSX4", "Patch catalog could not be loaded", error);
            return Collections.emptyList();
        }
    }
}
