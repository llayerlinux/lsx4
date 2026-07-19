package app.lsx4.android;

import android.app.LocaleManager;
import android.content.Context;
import android.content.res.Configuration;
import android.os.Build;
import android.os.LocaleList;

import java.util.Locale;

/** Application-owned locale with English as the deterministic first-install default. */
final class AppLocale {
    static final String[] TAGS = {"en", "es", "fr", "de", "ru", "zh-CN"};
    static final String[] NATIVE_LABELS = {
            "🇬🇧 English",
            "🇪🇸 Español",
            "🇫🇷 Français",
            "🇩🇪 Deutsch",
            "🇷🇺 Русский",
            "🇨🇳 中文"
    };
    private static final String PREFS = "lsx4_settings";
    private static final String KEY = "app_language";

    private AppLocale() {
    }

    static Context wrap(Context base) {
        String tag = base.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getString(KEY, "en");
        Locale locale = Locale.forLanguageTag(tag);
        Locale.setDefault(locale);
        Configuration configuration = new Configuration(base.getResources().getConfiguration());
        configuration.setLocale(locale);
        configuration.setLocales(new LocaleList(locale));
        return base.createConfigurationContext(configuration);
    }

    static String selectedTag(Context context) {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getString(KEY, "en");
    }

    static int selectedPosition(Context context) {
        String selected = selectedTag(context);
        for (int i = 0; i < TAGS.length; ++i) {
            if (TAGS[i].equals(selected)) {
                return i;
            }
        }
        return 0;
    }

    static void select(Context context, int position) {
        if (position < 0 || position >= TAGS.length) {
            position = 0;
        }
        String tag = TAGS[position];
        // Locale changes recreate the current Activity immediately. Persist first so
        // attachBaseContext() cannot race the asynchronous SharedPreferences writer.
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putString(KEY, tag)
                .commit();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            LocaleManager manager = context.getSystemService(LocaleManager.class);
            if (manager != null) {
                manager.setApplicationLocales(LocaleList.forLanguageTags(tag));
            }
        }
    }
}
