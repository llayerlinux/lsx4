package app.lsx4.android;

/**
 * Landscape/fullscreen render host.  The implementation intentionally stays in MainActivity's
 * proven game-host base while MainActivity itself presents the portrait library on a launcher
 * intent.  This also keeps old automation intents source-compatible through MainActivity's router.
 */
public final class GameActivity extends MainActivity {
}
