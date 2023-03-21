import ast
import datetime
import random
import cv2
import face_recognition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hazm
import nltk
import jdatetime

from dateutil import parser
from matplotlib.cbook import boxplot_stats
from scipy.stats import shapiro, normaltest, anderson, ttest_1samp, ttest_ind, stats, rayleigh, boxcox
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.stats import weightstats

from src.python import Params, Tags, time_utils, Color, Alphabet
from src.python.File import get_files, create_directory
from src.python.TelegramHTMLParser import TelegramHTMLParser

# use for converting data
from src.python.plot_utils import set_diagram, set_plot, show_plot, mosaic_plot, hist
from src.python.time_utils import TIMES, WEEKDAY_NAMES_ABBR_EN, WEEKDAY_NAMES_EN, TIMES_NAMES, HOLIDAYS, HOURS, \
    PANJSHANBEH, JOMEH

name = "AmadNews"

filename = "{}.xlsx".format(name)
dirname = "{}".format(name)

raw_data_path = "/home/albert/Downloads/Telegram Desktop/{}/".format(dirname)

# use for merging
merge_path = "/home/albert/Projects/PycharmProjects/cognitive/src/python/{}/".format(dirname)

# use for reading data
complete_dir = "/home/albert/Documents/cognitive/complete_data2/"
complete_file = "/home/albert/Documents/cognitive/complete_data2/{}".format(filename)

# use for date integrating
data1_path = "/home/albert/Documents/cognitive/complete_data/{}".format(filename)
data2_path = "/home/albert/Documents/cognitive/zahra/noads/KhabarForiMohem-data-noads.xlsx"

# use for emotion
emotion_path = "/home/albert/Documents/cognitive/emotion/words-for-questionare.xlsx"
emotion_write_path = "/home/albert/Documents/cognitive/emotion/dataset/{}".format(filename)
emotion_points_path = "/home/albert/Documents/cognitive/emotion/aggregated_emotion_points.xlsx"

# use for views data
view_path = "/home/albert/Documents/cognitive/view_most_less/{}".format(filename)

########################
# TODO: emotion
# TODO: NLM

channel_names = {
    "خبرهای فوری / مهم🔖": "ForiMohem",
    "BBCPersian": "BBCPersian",
    "«صدای مردم» (آمدنیوز)": "AmadNews",
    "خبرفوری": "KhabarFori",
    "کانال آخرین خبر": "AkharinKhabar",
}

channel_ids = {
    "Khabar_Fouri": "ForiMohem",
    "BBCPersian": "BBCPersian",
    "Sedaiemardom": "AmadNews",
    "khabarfouri": "KhabarFori",
    "Akharinkhabar": "AkharinKhabar",
}


def get_data(file_path):
    raw_data = open(file_path, 'r').read()
    parser = TelegramHTMLParser()
    parser.feed(raw_data)
    data = parser.getDataFrame()
    return data


def print_to_file(data, path="./", filename="newFile"):
    data.to_excel(path + filename + ".xlsx")


def write_merge_file(data):
    filename = merge_path.split('/')[-2]
    print_to_file(data, path='./data/', filename=filename)


def write_file(data, path):
    data.to_excel(path)


def convert_data(path):
    #TODO: on click converting!
    dirname = path.split('/')[-2]
    create_directory(dirname)

    i = 1
    for file in get_files(path):
        print("reading file " + str(i))
        if i == 1:
            print("-----------------")
            data = get_data(path + file)
            print_to_file(data, "./" + dirname + "/", dirname + "_" + str(i))
        i += 1


def read_html_data(path):
    data = pd.DataFrame()
    for file in get_files(path):
        data = data.append(get_data(path + file))
    return data


def read_excel_data(path):
    data = pd.DataFrame()
    for file in get_files(path):
        d = pd.read_excel(path + file)
        data = data.append(d)
    return data


def merge_files(path):
    data = read_excel_data(path)
    data = data.reset_index().drop("Unnamed: 0", axis=1).drop("index", axis=1)
    return data


def get_day_time(date_time):
    # lib 1 - comparing by time

    base = TIMES[time_utils.BASE]
    midnight = TIMES[time_utils.MIDNIGHT]
    morning = TIMES[time_utils.MORNING]
    afternoon = TIMES[time_utils.AFTERNOON]
    evening = TIMES[time_utils.EVENING]
    night = TIMES[time_utils.NIGHT]
    time = date_time.time()

    if (time >= base) and (time < morning):
        return time_utils.MIDNIGHT
    elif (time >= morning) and (time < afternoon):
        return time_utils.MORNING
    elif (time >= afternoon) and (time < evening):
        return time_utils.AFTERNOON
    elif (time >= evening) and (time < night):
        return time_utils.EVENING
    elif (time >= night) and (time < midnight):
        return time_utils.NIGHT
    else:
        return time_utils.BASE


def _get_day_time1(date_time):
    # lib 1
    # using full datetime to compare
    # not working in JalaliDateTime having bugs in comparing
    # to view the bug you can use:
    #   jdatetime.JalaliDateTime(date_time.year, date_time.month, date_time.day, times[time_utils.BASE])

    base = jdatetime.datetime(date_time.year, date_time.month, date_time.day, TIMES[time_utils.BASE])
    midnight = jdatetime.datetime(date_time.year, date_time.month, date_time.day, TIMES[time_utils.MIDNIGHT])
    morning= jdatetime.datetime(date_time.year, date_time.month, date_time.day, TIMES[time_utils.MORNING])
    afternoon = jdatetime.datetime(date_time.year, date_time.month, date_time.day, TIMES[time_utils.AFTERNOON])
    evening = jdatetime.datetime(date_time.year, date_time.month, date_time.day, TIMES[time_utils.EVENING])
    night = jdatetime.datetime(date_time.year, date_time.month, date_time.day, TIMES[time_utils.NIGHT])

    if (date_time >= base) and (date_time < morning):
        return time_utils.MIDNIGHT
    elif (date_time >= morning) and (date_time < afternoon):
        return time_utils.MORNING
    elif (date_time >= afternoon) and (date_time < evening):
        return time_utils.AFTERNOON
    elif (date_time >= evening) and (date_time < night):
        return time_utils.EVENING
    elif (date_time >= night) and (date_time < midnight):
        return time_utils.NIGHT
    else:
        return time_utils.BASE


def _get_day_time2(date_time):
    # lib 2
    # comparing by hour values
    # Todo: effecting other parameters like month and day (not just hour!)

    base = TIMES[time_utils.BASE]
    midnight = TIMES[time_utils.MIDNIGHT]
    morning= TIMES[time_utils.MORNING]
    afternoon = TIMES[time_utils.AFTERNOON]
    evening = TIMES[time_utils.EVENING]
    night = TIMES[time_utils.NIGHT]

    if (date_time.hour >= base) and (date_time.hour < morning):
        return time_utils.MIDNIGHT
    elif (date_time.hour >= morning) and (date_time.hour < afternoon):
        return time_utils.MORNING
    elif (date_time.hour >= afternoon) and (date_time.hour < evening):
        return time_utils.AFTERNOON
    elif (date_time.hour >= evening) and (date_time.hour < night):
        return time_utils.EVENING
    elif (date_time.hour >= night) and (date_time.hour < midnight):
        return time_utils.NIGHT
    else:
        return time_utils.BASE


def assign_day_time(df):
    df = df.assign(**{Tags.DAYTIME:time_utils.BASE})
    df[Tags.DAYTIME] = df[Tags.TIMEDATE].apply(get_day_time)
    return df


def assign_hour(df):
    df = df.assign(hour=time_utils.BASE)
    df[Tags.HOUR] = df[Tags.TIMEDATE].apply(lambda time: time.hour)
    return df


def split_time_date(date_time_string):
    time, date = date_time_string.split()
    return time, date


def convert_digits(digits):
    standard_digits = ""
    for d_char in list(digits):
        d = 0
        if d_char == "۰":
            d = 0
        elif d_char == "۱":
            d = 1
        elif d_char == "۲":
            d = 2
        elif d_char == "۳":
            d = 3
        elif d_char == "۴":
            d = 4
        elif d_char == "۵":
            d = 5
        elif d_char == "۶":
            d = 6
        elif d_char == "۷":
            d = 7
        elif d_char == "۸":
            d = 8
        elif d_char == "۹":
            d = 9
        else:
            d = d_char
        standard_digits += str(d)
    return standard_digits


def standard_date_time(date_time_string):
    standard_string = convert_digits(date_time_string)
    time, date = split_time_date(standard_string)
    return time, date


def get_date_time(date_time_string):
    time, date = standard_date_time(date_time_string)
    h, m , s = list(map(int, time.split(":")))
    y, m, d = list(map(int, date.split("/")))

    # lib 1:
    date_time = jdatetime.datetime(y,m,d,h,m,s)

    # lib 2:
    #   date_time = jdatetime.JalaliDateTime(y, m, d, h, m, s)

    # in case of standard datetime:
    #   data[Tags.Date] = pd.to_datetime(data[Tags.Date], format="%H:%M")
    return date_time


def convert_date_time(data):
    data[Tags.TIMEDATE] = data[Tags.TIMEDATE].apply(get_date_time)
    return data


def get_date_range(date, period=1, year=0, month=0, day=1, hour=0, minute=0, second=0, week=0):
    # TODO: add to specific library
    # TODO: add options
    dif = datetime.timedelta(days=day, hours=hour, minutes=minute, seconds=second, weeks=week)
    dates = [date]
    for i in range(period):
        date += dif
        dates.append(date)
    return dates


def get_weekday(date_time):
    return (date_time.toordinal() + 4) % 7


def get_weekday_name(date_time):
    day_number = get_weekday(date_time)
    return WEEKDAY_NAMES_EN[day_number]


def assign_weekday_name(df):
    df = df.assign(weekday_name=time_utils.BASE)
    df[Tags.WEEKDAY_NAME] = df[Tags.TIMEDATE].apply(get_weekday_name)
    return df


def is_holiday(date_time):
    # TODO: use compare method in other functions
    date = date_time.date()
    if date in HOLIDAYS or\
                    get_weekday_name(date) == PANJSHANBEH or \
                    get_weekday_name(date) == JOMEH:
        return True
    else:
        return False


def assign_holidays(df):
    df = df.assign(is_holiday=time_utils.BASE)
    df[Tags.IS_HOLIDAY] = df[Tags.TIMEDATE].apply(is_holiday)
    return df


def get_face_locations(image_url):
    # Load the jpg file into a NumPy array
    image = face_recognition.load_image_file(image_url)

    # Find all the faces in the image
    face_locations = face_recognition.face_locations(image)
    return face_locations


def get_face_number(image_url):
    if str(image_url) == "nan":
        number_of_faces = 0
    else:
        image_url = raw_data_path + image_url
        face_locations = get_face_locations(image_url)
        number_of_faces = len(face_locations)
    return number_of_faces


def assign_face_number(df):
    df = df.assign(face_number=0)
    df[Tags.FACE_NUMBER] = df[Tags.MEDIA_URL].apply(get_face_number)
    return df


def get_spaces(locations):
    spaces = []
    for loc in locations:
        top, right, bottom, left = loc
        space = np.abs(top - bottom) * np.abs(right - left)
        spaces.append(space)
    return spaces


def get_face_space(image_url):
    if str(image_url) == "nan":
        face_spaces = []
    else:
        image_url = raw_data_path + image_url
        face_locations = get_face_locations(image_url)
        face_spaces = get_spaces(face_locations)
    return face_spaces


def assign_face_space(df):
    df = df.assign(**{Tags.FACE_SPACE: 0})
    df[Tags.FACE_SPACE] = df[Tags.MEDIA_URL].apply(get_face_space)
    return df


def get_face_center(image_url):
    raise NotImplementedError

    if str(image_url) == "nan":
        face_is_center = None
    else:
        image_url = raw_data_path + image_url
        face_locations = get_face_locations(image_url)
    return face_is_center


def assign_face_center(df):
    df = df.assign(**{Tags.FACE_IS_CENTER: False})
    df[Tags.FACE_IS_CENTER] = df[Tags.MEDIA_URL].apply(get_face_center)
    return df


def get_color_range(color):
    if color == Color.RED:
        return Color.RED_LOWER, Color.RED_UPPER
    elif color == Color.ORANGE:
        return Color.ORANGE_LOWER, Color.ORANGE_UPPER
    elif color == Color.GREEN:
        return Color.GREEN_LOWER, Color.GREEN_UPPER
    elif color == Color.BLUE:
        return Color.BLUE_LOWER, Color.BLUE_UPPER
    else:
        return None


def get_percent(image_url, color):
    image = cv2.imread(image_url)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower, upper = get_color_range(color)
    mask = cv2.inRange(image_hsv, lower, upper)
    n_color = cv2.countNonZero(mask)
    n = image.shape[0] * image.shape[1]
    percent = round(n_color / n, 2)
    return percent


def get_color_percents(image_url):
    color_percents = {}
    if str(image_url) == "nan":
        pass
    else:
        image_url = raw_data_path + image_url
        colors = [Color.RED, Color.ORANGE, Color.GREEN, Color.BLUE]
        color_percents = {}
        for color in colors:
            percent = get_percent(image_url, color)
            color_percents.update({color: percent})
    return color_percents


def assign_colors(df):
    df = df.assign(**{Tags.COLOR: 0})
    df[Tags.COLOR] = df[Tags.MEDIA_URL].apply(get_color_percents)
    return df


def get_date(date_time):
    return date_time.date()


def assign_date(df):
    df = df.assign(date=0)
    df[Tags.DATE] = df[Tags.TIMEDATE].apply(get_date)
    return df


def complete_date_time(data):
    data = convert_date_time(data)
    data = assign_day_time(data)
    data = assign_hour(data)
    data = assign_weekday_name(data)
    data = assign_holidays(data)
    data = assign_date(data)
    return data


def complete_image_features(data):
    data = assign_face_number(data)
    data = assign_face_space(data)
    # data = assign_face_center(data)
    data = assign_colors(data)
    return data


def analysis_day_time():
    counts = data[Tags.DAYTIME].value_counts().to_dict()
    categories = TIMES_NAMES
    values = [counts.get(c, 0) for c in categories]
    print(values)
    title = "Post Histogram - Day Time"
    xlabel = "timestamp"
    ylabel = "frequency"
    set_diagram(title,xlabel,ylabel)
    set_plot(categories, values, 'bar')
    show_plot()


def analysis_day_time2():
    averages = data.groupby(Tags.DAYTIME)[seen].mean()
    stds = data.groupby(Tags.DAYTIME)[seen].std()
    print(averages)
    print(stds)
    categories = TIMES_NAMES

    #TODO: in order to use counts to widen the bars
    # daytime_counts = data[Tags.DAYTIME].value_counts().to_dict()
    # counts = [daytime_counts.get(c, 0) for c in categories]

    values = [averages.get(c, 0) for c in categories]
    errors = [stds.get(c, 0) for c in categories]
    title = "Day Time Seen Average"
    xlabel = "timestamp"
    ylabel = "Average Seen"

    set_diagram(title,xlabel,ylabel)
    set_plot(categories, values, 'bar', errors=errors)
    show_plot()


def analysis_hour():
    counts = data[Tags.HOUR].value_counts().to_dict()
    categories = HOURS
    values = [counts.get(c, 0) for c in categories]
    print(values)
    title = "Post Histogram - Hour"
    xlabel = "timestamp"
    ylabel = "frequency"
    set_diagram(title,xlabel,ylabel)
    set_plot(categories, values, 'bar')
    show_plot()


def analysis_hour2():
    averages = data.groupby(Tags.HOUR)[seen].mean()
    stds = data.groupby(Tags.HOUR)[seen].std()
    print(averages)
    print(stds)
    categories = HOURS
    values = [averages.get(c, 0) for c in categories]
    errors = [stds.get(c, 0) for c in categories]
    title = "Hour Seen Average"
    xlabel = "timestamp"
    ylabel = "Average Seen"
    set_diagram(title,xlabel,ylabel)
    set_plot(categories, values, 'bar', errors=errors)
    show_plot()


def analysis_weekday():
    title = "POST Histogram - Weekday"
    xlabel = "weekday"
    ylabel = "frequency"
    counts = data[Tags.WEEKDAY_NAME].value_counts().to_dict()
    print(counts)
    categories = WEEKDAY_NAMES_EN
    values = [counts.get(c, 0) for c in categories]
    set_diagram(title, xlabel, ylabel)
    set_plot(categories, values, 'bar')
    show_plot()


def analysis_weekday2():
    title = "Weekday Seen Average"
    xlabel = "weekday"
    ylabel = "Average Seen"
    averages = data.groupby(Tags.WEEKDAY_NAME)[seen].mean()
    stds = data.groupby(Tags.WEEKDAY_NAME)[seen].std()
    categories = WEEKDAY_NAMES_EN
    values = [averages.get(c, 0) for c in categories]
    errors = [stds.get(c, 0) for c in categories]
    print(categories, "\n", values, '\n', errors)
    set_diagram(title, xlabel, ylabel)
    set_plot(categories, values, 'bar', errors=errors)
    show_plot()


def analysis_holiday():
    title = "POST Histogram - Holiday - Total"
    xlabel = "holiday or not"
    ylabel = "frequency"
    counts = data[Tags.IS_HOLIDAY].value_counts().to_dict()
    print(counts)
    categories = ["Holiday", "Not Holiday"]
    values = [counts[True], counts[False]]
    set_diagram(title, xlabel, ylabel)
    set_plot(categories, values, 'bar')
    show_plot()


def analysis_holiday_balanced():
    title = "POST Histogram - Holiday - Average"
    xlabel = "holiday or not"
    ylabel = "frequency"

    # counts = data.groupby([Tags.IS_HOLIDAY,Tags.DATE])["id"].count()
    count_table = data.pivot_table(values=Tags.ID, index=Tags.DATE, columns=Tags.IS_HOLIDAY, aggfunc='count')
    true_avg_counts = count_table[True].mean()
    true_std_counts = count_table[True].std()

    true_counts = len(count_table[count_table[True].notna()])
    false_random_data = pd.Series(random.choices(count_table[False], k=true_counts))

    false_avg_counts = false_random_data.mean()
    false_std_counts = false_random_data.std()

    categories = ["Holiday", "Not Holiday"]
    values = [true_avg_counts, false_avg_counts]
    print(categories, "\n", values)
    set_diagram(title, xlabel, ylabel)
    set_plot(categories, values, 'bar', errors=[true_std_counts, false_std_counts])
    show_plot()


def analysis_holiday_balanced2():
    title = "Holiday Average Seen"
    xlabel = "holiday or not"
    ylabel = "Average Seen"
    true_count = data[Tags.IS_HOLIDAY].value_counts().to_dict()[True]
    true_avg = data.groupby(Tags.IS_HOLIDAY)[seen].mean()[True]
    true_std = data.groupby(Tags.IS_HOLIDAY)[seen].std()[True]
    random_data = data[data[Tags.IS_HOLIDAY]==False].sample(n=true_count)
    false_avg = random_data.groupby(Tags.IS_HOLIDAY)[seen].mean()[False]
    false_std = random_data.groupby(Tags.IS_HOLIDAY)[seen].std()[False]
    categories = ["Holiday", "Not Holiday"]
    values = [true_avg, false_avg]
    print(categories, "\n", values)
    set_diagram(title, xlabel, ylabel)
    set_plot(categories, values, 'bar', errors=[true_std, false_std])
    show_plot()


def analysis_daytime_weekday():
    # average_df = data.groupby([Tags.DAYTIME,Tags.WEEKDAY_NAME])["id"].count()
    average_table = pd.pivot_table(data, index=Tags.WEEKDAY_NAME, columns=Tags.DAYTIME, values='id', aggfunc='count')
    average_table = average_table.reindex(WEEKDAY_NAMES_EN, axis=0)
    average_table = average_table.reindex(TIMES_NAMES, axis=1)
    print(average_table)
    title = "Weekday and Day Time heatmap - Count"
    ax = sns.heatmap(average_table, cmap=cmap)
    ax.set_title(title)
    ax.set_ylabel('Week Day')
    ax.set_xlabel('Day Time')
    show_plot()


def analysis_hour_weekday():
    # average_df = data.groupby([Tags.DAYTIME,Tags.WEEKDAY_NAME])["id"].count()
    average_table = pd.pivot_table(data, index=Tags.WEEKDAY_NAME, columns=Tags.HOUR, values=Tags.ID, aggfunc='count')
    average_table = average_table.reindex(WEEKDAY_NAMES_EN, axis=0)
    average_table = average_table.reindex(HOURS, axis=1)
    print(average_table)
    title = "Weekday and Hour heatmap - Count"
    ax = sns.heatmap(average_table, cmap=cmap)
    ax.set_title(title)
    ax.set_ylabel('Week Day')
    ax.set_xlabel('Hour')
    show_plot()


def analysis_daytime_weekday2():
    # average_df = data.groupby([Tags.DAYTIME,Tags.WEEKDAY_NAME])["id"].count()
    average_table = pd.pivot_table(data, index=Tags.WEEKDAY_NAME, columns=Tags.DAYTIME, values=seen, aggfunc='mean')
    average_table = average_table.reindex(WEEKDAY_NAMES_EN, axis=0)
    average_table = average_table.reindex(TIMES_NAMES, axis=1)
    print(average_table)
    title = "Weekday and Day Time heatmap - Seen Average"
    ax = sns.heatmap(average_table, cmap=cmap)
    ax.set_title(title)
    ax.set(xlabel='Day Time', ylabel='Week Day')
    show_plot()


def analysis_hour_weekday2():
    # average_df = data.groupby([Tags.DAYTIME,Tags.WEEKDAY_NAME])["id"].count()
    average_table = pd.pivot_table(data, index=Tags.WEEKDAY_NAME, columns=Tags.HOUR, values=seen, aggfunc='mean')
    average_table = average_table.reindex(WEEKDAY_NAMES_EN, axis=0)
    average_table = average_table.reindex(HOURS, axis=1)
    print(average_table)
    title = "Weekday and Hour heatmap - Seen Average"
    ax = sns.heatmap(average_table, cmap=cmap)
    ax.set_title(title)
    ax.set(xlabel='Hour', ylabel='Week Day')
    show_plot()


def get_difference(df1, df2):
    df = pd.concat([df1, df2])
    return df.drop_duplicates(keep=False)


def get_outliers(df1, column_name):
    #TODO: get rid of this messy code!
    Q1 = df1[column_name].quantile(0.25)
    Q3 = df1[column_name].quantile(0.75)
    IQR = Q3 - Q1

    filter = (df1[column_name] >= Q1 - 1.5 * IQR) & (df1[column_name] <= Q3 + 1.5 * IQR)
    df2 = df1.loc[filter]
    df = get_difference(df1,df2)
    return df


def analysis_seen():
    sns.set_style("whitegrid")
    sns.boxplot(data=data, x=Tags.BROADCASTER, y=Tags.VIEW_COUNT, showfliers=True)
    # sns.swarmplot(x=Tags.BROADCASTER, y=Tags.VIEW_COUNT, data=all_data)
    sns.set(font="Verdana") #!
    show_plot()


def analysis_outliers():
    indexes = data.groupby(Tags.BROADCASTER).indices
    for channel, index in indexes.items():
        outliers = get_outliers(pd.DataFrame(data, index=index), Tags.VIEW_COUNT_NORMALIZED)
        print(channel,'(', len(outliers), '/', len(index), "):")
        print(outliers)
        # write_file(outliers, './outliers/' + channels.get(channel, 'channel') + '-outliers.xlsx')


def analysis_face():
    pass


def analysis_color():
    pass


def analysis_video():
    pass


def analysis_density():
    df = data.sort_values(Tags.TIMEDATE)
    values = df[Tags.VIEW_COUNT]
    lables = pd.Series(df[Tags.TIMEDATE], dtype=str)
    title = "Seen Density Analysis"
    set_diagram(title=title, x_label="Times", y_label="Seen")
    set_plot(lables, values)
    show_plot()


def test_shapiro(column):
    # normality test
    stat, p = shapiro(column)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


def test_k2(column):
    # normality test
    stat, p = normaltest(column)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


def test_anderson(column, dist="norm"):
    # normality test
    result = anderson(column, dist=dist)
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


def test_normal(column):
    # plt.plot(list(counts.keys()), list(counts.values()), 'r')
    test_shapiro(column)

    plt.hist(column)
    show_plot()

    qqplot(column, line='s')
    show_plot()


def test_ttest1(col1, mu):
    tset, pval = ttest_1samp(col1, mu)
    print("p - values", pval)
    if pval < 0.05:  # alpha value is 0.05 or 5%
        print(" we are rejecting null hypothesis")
    else:
        print("we are accepting null hypothesis")


def test_ttest2(col1, col2):
    ttest, pval = ttest_ind(col1, col2)
    print("p-value", pval)
    if pval < 0.05:
        print("we reject null hypothesis")
    else:
        print("we accept null hypothesis")


def calc_stats(sample1, sample2):
    n = len(sample1)
    m1 = sample1[Tags.VIEW_COUNT].mean()
    m2 = sample2[Tags.VIEW_COUNT].mean()
    s1 = sample1[Tags.VIEW_COUNT].std()
    s2 = sample2[Tags.VIEW_COUNT].std()

    SE = np.sqrt((s1**2 + s2**2)/n)


def test_ttest_paired(col1, col2):
    ttest, pval = stats.ttest_rel(col1, col2)
    print(pval)
    if pval < 0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")


def test_ztest1(col1):
    ztest, pval = weightstats.ztest(col1, x2=None, value=156)
    print(float(pval))
    if pval < 0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")


def test_ztest2(col1, col2):
    ztest, pval = weightstats.ztest(col1, x2=col2, value=0, alternative='two-sided')
    print(float(pval))
    if pval < 0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")


def get_sample_size(population, column_name, n0=100):
    sample = population.sample(n=n0)
    s = population[column_name].std()
    se = sample[column_name].sem()
    ME = 1.96*se
    n = int(np.ceil(((1.96*s)/ME)**2))
    return n


def analysis_hour3():

    population1 = data[data[Tags.DAYTIME] == time_utils.MIDNIGHT]
    population2 = data[data[Tags.DAYTIME] != time_utils.MIDNIGHT]

    n0 = 40 # based on the midnight counts!
    column = Tags.VIEW_COUNT
    sample1, sample2 = get_samples(column, population1, population2, n0=n0)
    sample1 = sample1[column]
    sample2 = sample2[column]
    # sample2 = get_sample(population2, column, n0=n0)[column]

    # just for information!
    print(data[data[Tags.VIEW_COUNT] > 500000][[Tags.ORIGNIAL_LINK, Tags.BROADCASTER, Tags.TEXT,Tags.VIEW_COUNT]])

    # normal test - hist
    test_normal(sample1)
    test_normal(sample2)

    test_ztest2(sample1, sample2)


def filter_duplicates(df):
    df = remove_duplicate(df, Tags.TEXT)
    return df


def filter_text_length(df):
    df = df[df[Tags.TEXT].apply(lambda x: len(str(x)) > 10)]
    return df


def filter_ads(df):
    df = df[df[Tags.ADS]!=1]
    return df


def filter_data(df):
    df = filter_duplicates(df)

    df = filter_text_length(df)

    df = filter_ads(df)

    return df


def get_z_score(view_count):
    pass


def get_med_score(view_count):
    pass


def normalize_minmax(x):
    res = np.array(x / np.max(x))
    return res


def normalize_view(df):
    indexes = df.groupby(Tags.CHANNEL)[Tags.ID].indices
    for channel_name, index in indexes.items():
        group = pd.DataFrame(df, index=index)
        group[Tags.VIEW_COUNT_NORMALIZED] = stats.zscore(group[Tags.VIEW_COUNT])
        df.loc[index, Tags.VIEW_COUNT_NORMALIZED] = group[Tags.VIEW_COUNT_NORMALIZED]
    return df


def transform_view(df):
    indexes = df.groupby(Tags.CHANNEL)[Tags.ID].indices
    for channel_name, index in indexes.items():
        group = pd.DataFrame(df, index=index)
        group[Tags.VIEW_COUNT_TRANSFORMED], _ = boxcox(group[Tags.VIEW_COUNT])
        df.loc[index, Tags.VIEW_COUNT_TRANSFORMED] = group[Tags.VIEW_COUNT_TRANSFORMED]
    return df


def to_latin(channel_name):
    return channel_names.get(channel_name, "channel")


def get_sample(df, n):
    sample = df.sample(n=n)
    return sample


def integrate_data(data1, data2):
    data2 = data2.drop([Tags.MEDIA, Tags.TEXT, "Unnamed: 0.1", "idx"], axis=1)
    try:
        data2[Tags.ADS] = data2[Tags.ADS]
    except KeyError:
        data2 = data2.assign(**{Tags.ADS: 0})
    data1 = data1.drop([Tags.FROM], axis=1)
    data1[Tags.TIME] = data1[Tags.DATE]
    data1 = data1.drop(Tags.DATE, axis=1)
    channels = {v: k for k, v in channel_ids.items()}
    data1 = data1.assign(channel=channels.get(dirname))
    df = pd.merge(data1, data2, on=Tags.ID)
    return df


def read_data(path):
    df = pd.read_excel(path)
    df = df.drop("Unnamed: 0", axis=1)
    return df


def preprocess_data(df):
    df = complete_date_time(df)
    df = normalize_view(df)
    df = transform_view(df)
    df = complete_view_quality(df)
    df = filter_data(df)
    return df


def get_lemmatized_word(word):
    lemmatizer = hazm.Lemmatizer()
    lem_word = lemmatizer.lemmatize(word)
    return lem_word


def get_lemmatized_words(words):
    lem_words = [get_lemmatized_word(w) for w in words]
    return lem_words


def get_lemmatized_words_from_text(text):
    words = hazm.word_tokenize(text)
    lem_words = get_lemmatized_words(words)
    return lem_words


def normalize_words(text):
    normalizer = hazm.Normalizer()
    nomalized_text = normalizer.normalize(text)
    return nomalized_text


def get_conjugated_word(word):
    lemmatizer = hazm.Lemmatizer()
    try:
        cw = lemmatizer.lemmatize(word)
    except ValueError:
        cw = word
    return cw


def get_conjugated_words(words):
    conjugated_words = []
    for word in words:
        cw = get_conjugated_word(word)
        conjugated_words.append(cw)
    return conjugated_words


def get_spaced_word(word):
    word = str(word).strip()
    ww = word.split("\u200c")
    if len(ww) > 0:
        new_word = " ".join(ww)
    else:
        new_word = word
    return new_word


def extend_words(words):
    extended_words = list(words)
    for word in words:
        space_words = get_spaced_word(word)
        extended_words.append(space_words)
    return extended_words


def add_spaced_words_in_dataframe(df, column):
    temp_df = df.copy()
    temp_df[column] = df[column].apply(get_spaced_word)
    df = df.append(temp_df)
    df = df.drop_duplicates()
    return df


def remove_duplicate(df, column):
    index = df[column].drop_duplicates().index
    df = pd.DataFrame(df, index=index)
    return df


def get_emotion_words():
    # words
    emotion_words = pd.read_excel(emotion_path)["words"]
    emotion_words = set(emotion_words)
    # lemmatization
    lemmatized_emotion_words = get_lemmatized_words(emotion_words)
    emotion_words = emotion_words.union(lemmatized_emotion_words)
    # extension
    extended_emotion_words = extend_words(emotion_words)
    emotion_words = emotion_words.union(extended_emotion_words)

    return emotion_words


def get_emotion_words_data_frame(extend=True):
    # words
    emotion_words_df = pd.read_excel(emotion_points_path)
    emotion_words_df = remove_duplicate(emotion_words_df, Tags.EMO_WORDS)

    if extend:
        # add spaced words
        emotion_words_df = add_spaced_words_in_dataframe(emotion_words_df, Tags.EMO_WORDS)

        # lemmatization
        emotion_words_df[Tags.EMO_WORDS_LEMMATIZED] = emotion_words_df[Tags.EMO_WORDS].apply(get_lemmatized_words_in_phrase)

        # add infitives
        emotion_words_df = add_infinitives_in_dataframe(emotion_words_df, Tags.EMO_WORDS_LEMMATIZED)

    # polarity
    emotion_words_df[Tags.EMO_POLARITY_POINT] = emotion_words_df[Tags.EMO_POLARITY].apply(get_unique_point)

    # arousing
    emotion_words_df[Tags.EMO_AROUSING_POINT] = emotion_words_df[Tags.EMO_AROUSING].apply(get_unique_point)

    return emotion_words_df


def add_infinitives_in_dataframe(df, column):
    temp_df = df.copy()
    temp_df[column] = temp_df[column].apply(get_infinitive_word_in_phrase)
    df = df.append(temp_df)
    df = df.drop_duplicates()
    return df


def intersection(lst1, lst2):
    # Use of hybrid method
    if isinstance(lst2, set):
        temp = lst2
    else:
        temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def select_emotion_words(lemmatized_words):
    selected_words = intersection(lemmatized_words, emotion_df[Tags.EMO_WORDS_LEMMATIZED])
    return selected_words


def get_ngrams(tokens):
    ngrams = ast.literal_eval(tokens)
    n23grams = nltk.everygrams(ngrams, 2, 3)
    n23grams = [" ".join(grams) for grams in n23grams]
    ngrams.extend(n23grams)
    return ngrams


def get_lemmatized_words_in_phrase(phrase):
    lemmatized_phrase = " ".join(get_lemmatized_words(phrase.split()))
    return lemmatized_phrase


def get_infinitive_word_in_phrase(phrase):
    word = phrase
    words = phrase.split()
    if str(words[-1]).__contains__("#"):
        past = words[-1].split("#")[0]
        infinitive = past + Alphabet.N
        word = " ".join(words[:-1] + [infinitive])
    return word


def get_lemmatized_ngrams(ngrams):
    ngrams = ast.literal_eval(ngrams)
    #TODO: check split VS tokenizer
    lemmatized_ngrams = [get_lemmatized_words_in_phrase(ngram) for ngram in ngrams]
    return lemmatized_ngrams


def assign_emotion_words(df):
    #TODO: n-grams (ongoing) OR new lemmatization on text OR conjugate the emotion words (Failed) OR sim func
    df = df.assign(**{Tags.TEXT_WORD_LEMMA_GRAMS: None})
    #method1
    # df[Tags.TEXT_WORD_LEMMA_GRAMS] = df[Tags.GRAMS].apply(get_lemmatized_ngrams)

    #method2
    df[Tags.TEXT_WORD_LEMMA_GRAMS] = df[Tags.TEXT_WORD_LEMMA_NOPUNC_NOEMOJI].apply(get_ngrams)

    df = df.assign(**{Tags.EMOTION_WORDS: None})
    df[Tags.EMOTION_WORDS] = df[Tags.TEXT_WORD_LEMMA_GRAMS].apply(select_emotion_words)
    return df


def get_unique_point(points_string):
    #TODO: correct?
    points_list = ast.literal_eval(points_string)
    points_list = [i for i in points_list if i != 'None'] #TODO: remove to evaluate!
    points = np.array(points_list, dtype=np.int)
    return np.mean(points)


def get_polarity_point(words):
    emotion_points = emotion_df[emotion_df[Tags.EMO_WORDS_LEMMATIZED].isin(words)][Tags.EMO_POLARITY_POINT]
    point = emotion_points.sum()
    return point


def get_arousing_point(words):
    emotion_points = emotion_df[emotion_df[Tags.EMO_WORDS_LEMMATIZED].isin(words)][Tags.EMO_AROUSING_POINT]
    point = emotion_points.sum()
    return point


def get_emotion_point(emotion_words_list_string):
    emotion_words_list = ast.literal_eval(emotion_words_list_string)
    pass


def assign_emotion_points(df):
    df[Tags.EMOTION_POLARITY_POINT] = df[Tags.EMOTION_WORDS].apply(get_polarity_point) / df[Tags.WORD_COUNT]
    df[Tags.EMOTION_AROUSING_POINT] = df[Tags.EMOTION_WORDS].apply(get_arousing_point) / df[Tags.WORD_COUNT]
    return df


def complete_emotion_features(df):
    df = assign_emotion_words(df)
    df = assign_emotion_points(df)
    return df


def count_none_in_list(list_string):
    dlist = ast.literal_eval(list_string)
    n = sum(x=='None' for x in dlist)
    return n


def emotion_none():
    df = pd.read_excel(emotion_points_path)
    df["arousing_none"] = df["arousing"].apply(count_none_in_list)
    df["polarity_none"] = df["polarity"].apply(count_none_in_list)
    df_none = pd.DataFrame(df, index=df[(df["arousing_none"] >0)|(df[ "polarity_none"] > 0)].index)
    df_none.to_excel("/home/albert/Documents/cognitive/emotion/questionare/emotion_nones.xlsx")


def get_main_populations(df, column):
    m = df[column].mean()
    s = df[column].std()
    df_most = df[df[column] > m + 1*s]
    df_less = df[df[column] < m - 1*s]
    return df_most, df_less


def write_views(df_most, df_less):
    write_file(df_most, view_path + "-most.xlsx")
    write_file(df_less, view_path + "-less.xlsx")


def assign_view_quality(df):
    df = df.assign(**{Tags.VIEW_QUALITY: 0})
    df_most, df_less = get_main_populations(df, Tags.VIEW_COUNT_TRANSFORMED)
    df.loc[df_most.index, Tags.VIEW_QUALITY] = 1
    df.loc[df_less.index, Tags.VIEW_QUALITY] = -1
    return df


def complete_view_quality(df):
    df = assign_view_quality(df)
    return df


def get_populations():
    df_most = data[data[Tags.VIEW_QUALITY] == 1]
    df_less = data[data[Tags.VIEW_QUALITY] == -1]
    return df_most, df_less


def get_samples(column, pop1=None, pop2=None, n0=80):
    if pop1 is None and pop2 is None:
        df_most, df_less = get_populations()
    else:
        df_most, df_less = pop1, pop2
    n1 = get_sample_size(df_most, column, n0)
    n2 = get_sample_size(df_less, column, n0)
    n = max(n1, n2)
    sample1 = get_sample(df_most, n)
    sample2 = get_sample(df_less, n)
    print("sample size estimation: ", n)
    print("sample size proportion: %", n/len(df_most))
    print("sample size proportion: %", n/len(df_less))
    return sample1, sample2


def analysis_emotion():
    emotion_df = pd.read_excel(emotion_points_path)
    min_word = emotion_df.sort_values(Tags.EMO_POLARITY_POINT).iloc[0]
    print(min_word)
    max_word = emotion_df.sort_values(Tags.EMO_POLARITY_POINT, ascending=False).iloc[0]
    print(max_word)

    set_plot(emotion_df[Tags.EMO_POLARITY_POINT], emotion_df[Tags.EMO_AROUSING_POINT])
    show_plot()


def fit_powerlaw(d):
    import powerlaw
    fit = powerlaw.Fit(np.array(d) + 1, xmin=1, discrete=True)
    fit.power_law.plot_pdf(color='b', linestyle='--', label='fit ccdf')
    fit.plot_pdf(color='b')
    print('alpha= ', fit.power_law.alpha, '  sigma= ', fit.power_law.sigma)


def analysis_dist2():
    counts = data[Tags.VIEW_COUNT]
    fit_powerlaw(counts)
    show_plot()


def analysis_dist1():
    title = "Histogram - {}".format(dirname)
    xlabel = "view counts"
    ylabel = "frequency"
    counts = data[Tags.VIEW_COUNT]

    stat = counts.describe()
    print(stat)

    test_normal(counts)

    set_diagram(title, xlabel, ylabel)
    hist(counts)
    show_plot()


def test_correlation(arr1, arr2):
    coef, pvalue = stats.pearsonr(arr1, arr2)
    alpha = 0.05
    if pvalue < alpha:
        print("H0 rejected, there is some relation by coefficient: {}".format(coef))
    else:
        print("Failed to reject H0, no relation seems to exist. coefficient {}".format(coef))


def get_correlation(arr1, arr2):
    coef, pvalue = stats.pearsonr(arr1, arr2)
    return coef


def get_reformed_sample(col1, col2, n0):
    n1 = get_sample_size(data, col1, n0)
    n2 = get_sample_size(data, col2, n0)
    n = max(n1, n2)
    sample = get_sample(data, n)
    print("sample size estimation: ", n)
    print("sample size proportion: %", n/len(data))
    return sample


def test_analysis_correlation(col1, col2):
    n0 = 100
    sample = get_reformed_sample(col1, col2, n0)

    # normal test & sample size
    test_normal(sample[col1])
    test_normal(sample[col2])

    var1 = sample[col1]
    var2 = sample[col2]
    print(col1, col2)
    plt.scatter(var1, var2)
    test_correlation(var1, var2)
    show_plot()


def check_correlation(col1, col2):
    var1 = data[col1]
    var2 = data[col2]
    print(col1, col2)
    print(get_correlation(var1, var2))
    set_plot(var1, var2, 'bo')
    set_diagram(" - ".join([col1, col2]), col1, col2)
    show_plot()


def analysis_correlation():
    col1 = Tags.EMOTION_POLARITY_POINT
    col2 = Tags.EMOTION_AROUSING_POINT
    check_correlation(col1, col2)

    col1 = Tags.TEXT_EMOJI_LENGTH
    col2 = Tags.EMOTION_AROUSING_POINT
    check_correlation(col1, col2)

    col1 = Tags.RED
    col2 = Tags.EMOTION_AROUSING_POINT
    check_correlation(col1, col2)

    col1 = Tags.BLUE
    col2 = Tags.EMOTION_AROUSING_POINT
    check_correlation(col1, col2)

    col1 = Tags.FACE_NUMBER
    col2 = Tags.EMOTION_AROUSING_POINT
    check_correlation(col1, col2)

    col1 = Tags.FACE_BIG
    col2 = Tags.EMOTION_AROUSING_POINT
    check_correlation(col1, col2)

    print("------------------")

    col1 = Tags.EMOTION_POLARITY_POINT
    col2 = Tags.VIEW_COUNT
    check_correlation(col1, col2)

    col1 = Tags.EMOTION_AROUSING_POINT
    col2 = Tags.VIEW_COUNT
    check_correlation(col1, col2)

    col1 = Tags.TEXT_EMOJI_LENGTH
    col2 = Tags.VIEW_COUNT
    check_correlation(col1, col2)

    col1 = Tags.RED
    col2 = Tags.VIEW_COUNT
    check_correlation(col1, col2)

    col1 = Tags.ORANGE
    col2 = Tags.VIEW_COUNT
    check_correlation(col1, col2)

    col1 = Tags.BLUE
    col2 =Tags.VIEW_COUNT
    check_correlation(col1, col2)

    col1 = Tags.GREEN
    col2 = Tags.VIEW_COUNT
    check_correlation(col1, col2)

    col1 = Tags.FACE_NUMBER
    col2 = Tags.VIEW_COUNT
    check_correlation(col1, col2)

    col1 = Tags.FACE_BIG
    col2 = Tags.VIEW_COUNT
    check_correlation(col1, col2)


def get_length(list_string):
    lst = ast.literal_eval(list_string)
    return len(lst)


def assign_emoji_length(df):
    df[Tags.TEXT_EMOJI_LENGTH] = df[Tags.TEXT_EMOJIS].apply(get_length)
    return df


def complete_text_features(df):
    df = assign_emoji_length(df)
    return df


def normal_column(df, col, new_col):
    df[new_col], lamb = boxcox(df[col])
    return df[new_col]


def assign_normal_polarity(df):
    df[Tags.EMOTION_POLARITY_POINT_TRANS] = df[Tags.EMOTION_POLARITY_POINT] + 2

    df[Tags.EMOTION_POLARITY_POINT_TRANS] = normal_column(df, Tags.EMOTION_POLARITY_POINT_TRANS, Tags.EMOTION_POLARITY_POINT_TRANS)

    return df


def assign_normal_arousing(df):
    df[Tags.EMOTION_AROUSING_POINT_TRANS] = df[Tags.EMOTION_AROUSING_POINT] + 1
    df[Tags.EMOTION_AROUSING_POINT_TRANS] = normal_column(df, Tags.EMOTION_AROUSING_POINT_TRANS,
                                                          Tags.EMOTION_AROUSING_POINT_TRANS)
    return df


def assign_normal_emoji_length(df):
    df[Tags.TEXT_EMOJI_LENGTH_TRANS] = df[Tags.TEXT_EMOJI_LENGTH] + 1
    df[Tags.TEXT_EMOJI_LENGTH_TRANS] = normal_column(df, Tags.TEXT_EMOJI_LENGTH_TRANS, Tags.TEXT_EMOJI_LENGTH_TRANS)
    return df


def normal_data(df):
    df = assign_normal_polarity(df)
    df = assign_normal_arousing(df)
    df = assign_normal_emoji_length(df)
    return df


def get_red(color_str):
    col = ast.literal_eval(color_str)
    try:
        red = col[Tags.RED]
    except KeyError:
        red = 0
    return red


def get_green(color_str):
    col = ast.literal_eval(color_str)
    try:
        red = col[Tags.GREEN]
    except KeyError:
        red = 0
    return red


def get_orange(color_str):
    col = ast.literal_eval(color_str)
    try:
        red = col[Tags.ORANGE]
    except KeyError:
        red = 0
    return red


def get_blue(color_str):
    col = ast.literal_eval(color_str)
    try:
        red = col[Tags.BLUE]
    except KeyError:
        red = 0
    return red


def add_colors(df):
    df[Tags.RED] = df[Tags.COLOR].apply(get_red)
    df[Tags.GREEN] = df[Tags.COLOR].apply(get_green)
    df[Tags.BLUE] = df[Tags.COLOR].apply(get_blue)
    df[Tags.ORANGE] = df[Tags.COLOR].apply(get_orange)
    return df


def get_max(arr_str):
    arr = ast.literal_eval(arr_str)
    if len(arr) == 0:
        return 0
    else:
        return max(arr)


def add_big_face(df):
    df[Tags.FACE_BIG] = df[Tags.FACE_SPACE].apply(get_max)
    return df


def add_redundant_features(df):
    df = add_colors(df)
    df = add_big_face(df)
    return df


def analysis():
    analysis_dist1()


if __name__ == '__main__':

    # phase1 - converting - done
    convert_data(raw_data_path)

    # phase 2-1 - integrating - done
    data = merge_files(merge_path)
    data = complete_image_features(data)
    write_merge_file(data)

    # phase 2-2 - integrating (me & zahra) - done
    data1 = read_data(data1_path)
    data2 = read_data(data2_path)
    data = integrate_data(data1, data2)
    write_file(data, complete_file)

    # phase3 - reading & preprocessing
    # single_data = read_data(complete_file)
    all_data = merge_files(complete_dir)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    # in last version it is preprocessed
    # data = preprocess_data(single_data)
    data = all_data

    # phase4 - time analysis
    seen = Tags.VIEW_COUNT
    cmap = sns.cm.rocket_r
    
    # by frequency
    analysis_day_time()
    analysis_hour()
    analysis_weekday()
    analysis_holiday_balanced()
    
    # by seen number
    analysis_day_time2()
    analysis_hour2()
    analysis_weekday2()
    analysis_holiday_balanced2()
    
    # together
    analysis_daytime_weekday()
    analysis_hour_weekday()
    analysis_daytime_weekday2()
    analysis_hour_weekday2()
    
    # significance
    analysis_hour3()

    # phase 5 - seen analysis
    data = all_data
    analysis_seen()
    analysis_outliers()

    # phase 6 - image features
    analysis_face()
    analysis_color()
    analysis_video()

    # # phase 7 - process
    analysis_density()

    # phase 8 - emotion
    emotion_df = get_emotion_words_data_frame()
    data = complete_emotion_features(data)
    write_file(data, emotion_write_path)
    analysis_emotion()

    # phase 9 - correlation
    data = complete_text_features(data)
    data = add_redundant_features(data)
    analysis_correlation()

    print(get_date_range(jdatetime.datetime(1398,2,2),10))

    i = pd.date_range('2020-04-09', periods=4, freq='1D20min')
    data = pd.DataFrame(data, index=i)
    print(data[Tags.TIMEDATE])
    print(data[Tags.TIMEDATE].between_time("01:20","02:00"))

