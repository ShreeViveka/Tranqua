"""
categories.py — Classifies apps/windows into mental health relevant categories
"""

# Keywords matched against app executable name OR window title (lowercased)
CATEGORY_RULES = {
    "social_media": [
        "instagram", "facebook", "twitter", "tiktok", "snapchat",
        "reddit", "whatsapp", "telegram", "discord", "linkedin",
        "pinterest", "tumblr", "messenger", "wechat", "line",
        "x.com", "threads"
    ],
    "work": [
        "word", "excel", "powerpoint", "outlook", "teams", "zoom",
        "slack", "notion", "onenote", "vscode", "code", "pycharm",
        "intellij", "eclipse", "terminal", "cmd", "powershell",
        "notepad", "sublime", "atom", "vim", "emacs", "jupyter",
        "anaconda", "spyder", "matlab", "rstudio", "figma", "sketch",
        "postman", "github", "gitlab", "jira", "confluence", "trello",
        "asana", "monday", "clickup"
    ],
    "entertainment": [
        "netflix", "youtube", "spotify", "vlc", "mpv", "prime video",
        "disney", "hulu", "twitch", "steam", "epic games", "origin",
        "battle.net", "minecraft", "roblox", "valorant", "fortnite",
        "league of legends", "dota", "csgo", "gaming", "game",
        "media player", "itunes", "winamp", "kodi", "plex",
        "crunchyroll", "funimation", "hotstar"
    ],
    "browser": [
        "chrome", "firefox", "edge", "safari", "opera", "brave",
        "vivaldi", "internet explorer", "chromium"
    ],
    "health_fitness": [
        "fitness", "workout", "meditation", "calm", "headspace",
        "strava", "myfitnesspal", "nike", "adidas"
    ],
    "education": [
        "coursera", "udemy", "edx", "khan", "duolingo", "anki",
        "quizlet", "grammarly", "mendeley", "zotero", "adobe reader",
        "foxit", "pdf", "ebook"
    ],
    "communication": [
        "gmail", "mail", "thunderbird", "skype", "facetime",
        "google meet", "webex"
    ]
}

# Social media apps that are specifically linked to comparison/anxiety
HIGH_RISK_SOCIAL = {
    "instagram", "tiktok", "facebook", "twitter", "x.com",
    "snapchat", "pinterest", "threads"
}


def classify_app(app_name: str, window_title: str = "") -> str:
    """
    Returns one of:
    social_media / work / entertainment / browser /
    health_fitness / education / communication / other
    """
    text = (app_name + " " + (window_title or "")).lower()

    for category, keywords in CATEGORY_RULES.items():
        if any(kw in text for kw in keywords):
            return category

    return "other"


def is_high_risk_social(app_name: str, window_title: str = "") -> bool:
    """Returns True if this is a high-risk social media app."""
    text = (app_name + " " + (window_title or "")).lower()
    return any(kw in text for kw in HIGH_RISK_SOCIAL)


def is_late_night(hour: int) -> bool:
    """Returns True if hour is between 11pm and 4am."""
    return hour >= 23 or hour < 4


def get_category_display_name(category: str) -> str:
    names = {
        "social_media"   : "Social Media",
        "work"           : "Work / Study",
        "entertainment"  : "Entertainment",
        "browser"        : "Browsing",
        "health_fitness" : "Health & Fitness",
        "education"      : "Education",
        "communication"  : "Communication",
        "other"          : "Other"
    }
    return names.get(category, "Other")
