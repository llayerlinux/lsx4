package app.lsx4.android;

enum GameCompatibilityStatus {
    UNCLASSIFIED,
    UNSUPPORTED,
    SUPPORTED;

    int librarySortPriority() {
        switch (this) {
        case SUPPORTED:
            return 0;
        case UNSUPPORTED:
            return 1;
        case UNCLASSIFIED:
        default:
            return 2;
        }
    }
}
