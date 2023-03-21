import jdatetime
# from persiantools import jdatetime

MORNING = "morning"
AFTERNOON = "afternoon"
EVENING = "evening"
NIGHT = "night"
MIDNIGHT = "midnight"
BASE = "none"

TIMES = {
    BASE: jdatetime.time(0,0),
    MORNING: jdatetime.time(5,0),
    AFTERNOON: jdatetime.time(12,0),
    EVENING: jdatetime.time(16,0),
    NIGHT: jdatetime.time(19,0),
    MIDNIGHT: jdatetime.time(23,59),
}

TIMES_NAMES = [
    MORNING,
    AFTERNOON,
    EVENING,
    NIGHT,
    MIDNIGHT
]

HOURS = [i for i in range(24)]

MONTH_NAMES_EN = [
    None,
    "Farvardin",
    "Ordibehesht",
    "Khordad",
    "Tir",
    "Mordad",
    "Shahrivar",
    "Mehr",
    "Aban",
    "Azar",
    "Dey",
    "Bahman",
    "Esfand",
]

MONTH_NAMES_FA = [
    None,
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
]

MONTH_NAMES_ABBR_EN = [
    None,
    "Far",
    "Ord",
    "Kho",
    "Tir",
    "Mor",
    "Sha",
    "Meh",
    "Aba",
    "Aza",
    "Dey",
    "Bah",
    "Esf",
]

MONTH_NAMES_ABBR_FA = [
    None,
    "فرو",
    "ارد",
    "خرد",
    "تیر",
    "مرد",
    "شهر",
    "مهر",
    "آبا",
    "آذر",
    "دی",
    "بهم",
    "اسف",
]

SHANBEH = "Shanbeh"
YEKSHANBEH = "Yekshanbeh"
DOSHANBEH = "Doshanbeh"
SESHANBEH = "Seshanbeh"
CHAHARSHANBEH = "Chaharshanbeh"
PANJSHANBEH = "Panjshanbeh"
JOMEH = "Jomeh"

WEEKDAY_NAMES_EN = [
    SHANBEH,
    YEKSHANBEH,
    DOSHANBEH,
    SESHANBEH,
    CHAHARSHANBEH,
    PANJSHANBEH,
    JOMEH,
]

WEEKDAY_NAMES_FA = ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه"]

WEEKDAY_NAMES_ABBR_EN = ["Sha", "Yek", "Dos", "Ses", "Cha", "Pan", "Jom"]
WEEKDAY_NAMES_ABBR_FA = ["ش", "ی", "د", "س", "چ", "پ", "ج"]

HOLIDAYS = [
    jdatetime.date(1398, 4, 8),
    jdatetime.date(1398, 5, 21),
    jdatetime.date(1398, 5, 29),
    jdatetime.date(1398, 6, 18),
    jdatetime.date(1398, 6, 19),
]





