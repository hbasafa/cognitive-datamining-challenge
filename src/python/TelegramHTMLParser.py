import pandas as pd
import re
from html.parser import HTMLParser

from src.python import Params, Tags


class TelegramHTMLParser(HTMLParser):
    _isMessage = False
    _isMessageBody = False
    _isMessageDate = False
    _isMessageFromName = False
    _isMessageMedia = False
    _isMessageText = False
    _isMessageImage = False
    _isMessageVideo = False
    _isMessageReplied = False
    content = []
    current_id = -1

    def __init__(self):
        HTMLParser.__init__(self)

    def getDataFrame(self):
        return pd.DataFrame(self.content)

    def createMessage(self):
        return {Tags.ID: 0, Tags.DATE: None, Tags.FROM: None,
                Tags.MEDIA: None, Tags.MEDIA_URL: None, Tags.VIDEO: None,
                Tags.REPLIED: None, Tags.TEXT: None}

    def handle_starttag(self, tag, attrs):
        # print("Encountered a start tag :", tag)
        attrs = dict(attrs)

        if self.isMessageTag(tag, attrs):
            self.current_id += 1
            self.content.append({Tags.ID: self.getID(attrs)})
            self._isMessage = True

        if self.isMessageBodyTag(tag, attrs):
            self._isMessageBody = True

        if self.isMessageDateTag(tag, attrs):
            self._isMessageDate = True

        if self.isMessageFromName(tag, attrs):
            self._isMessageFromName = True

        if self.isMessageMedia(tag, attrs):
            self._isMessageMedia = True

        if self.isMessageImage(tag, attrs):
            self._isMessageImage = True
            self.set_image_url(attrs)

        if self.isMessageVideo(tag, attrs):
            self._isMessageVideo = True

        if self.isMessageReplied(tag, attrs):
            self._isMessageReplied = True

        if self.isMessageRepliedLink(tag, attrs):
            self.set_replied_id(attrs)

        if self.isMessageText(tag, attrs):
            self._isMessageText = True

    def handle_endtag(self, tag):
        # print("Encountered an end tag :", tag)

        # the last div tag is used to close the text tag
        if self.isDivTag(tag) and self._isMessageText:
            self._isMessageText = False

        if self.isDivTag(tag) and self._isMessageMedia:
            #TODO: real end tag!
            self._isMessageMedia = False

        if self.isDivTag(tag) and self._isMessageVideo:
            #TODO: real end tag!
            self._isMessageVideo = False

        if self.isAddress(tag) and self._isMessageReplied:
            self._isMessageReplied = False

    def handle_data(self, data):
        # print("Encountered some data  :", data)

        if self._isMessageDate:
            self.content[self.current_id].update({Tags.DATE: data.strip()})
            self._isMessageDate = False

        if self._isMessageFromName:
            self.content[self.current_id].update({Tags.FROM: data.strip()})
            self._isMessageFromName = False

        if self._isMessageMedia:
            self.content[self.current_id].update({Tags.MEDIA: True})

        if self._isMessageVideo:
            self.content[self.current_id].update({Tags.VIDEO: True})

        if self._isMessageText and self._isMessage:
            # this tag is multi sections, so it is not closed here.
            self.content[self.current_id].update(
                {Tags.TEXT: self._getText() + data.strip()})

    def set_image_url(self, attrs):
        if self._isMessageMedia and self._isMessageImage:
            try:
                self.content[self.current_id].update({Tags.MEDIA_URL: attrs[Params.SRC]})
            except KeyError:
                self.content[self.current_id].update({Tags.MEDIA_URL: None})

    def set_replied_id(self, attrs):
        if self._isMessageReplied:
            try:
                href = attrs[Params.HREF]
                index = re.search("\d+", href)
                message_id = href[index.start():index.end()]
                self.content[self.current_id].update({Tags.REPLIED: message_id})
            except KeyError:
                self.content[self.current_id].update({Tags.REPLIED: None})

    def _getText(self):
        try:
            return self.content[self.current_id][Tags.TEXT] + "\n"
        except KeyError:
            return ""

    def isMessageBodyTag(self, tag, attrs):
        try:
            if self.isDivTag(tag) and attrs[Params.CLASS] == "body":
                return True
        except KeyError:
            return False
        return False

    def isDivTag(self, tag):
        if tag == Params.DIV:
            return True
        return False

    def isMessageTag(self, tag, attrs):
        if self.isDivTag(tag) and self.isMessageAttrs(attrs):
            return True
        return False

    def isImageTag(self, tag):
        if tag == Params.IMG:
            return True
        return False

    def isAddress(self, tag):
        if tag == Params.ADDRESS:
            return True
        return False

    def isMessageAttrs(self, attrs):
        if self.hasClass(attrs, "message"):
            return True
        return False

    def getID(self, attrs, pattern="\d+"):
        try:
            index = re.search(pattern, attrs[Params.ID])
            message_id = attrs[Params.ID][index.start():index.end()]
            return message_id
        except KeyError:
            return None

    def hasClass(self, attrs, value):
        try:
            if attrs[Params.CLASS].__contains__(value):
                return True
        except KeyError:
            return False
        return False

    def isMessageDateTag(self, tag, attrs):
        if self.isDivTag(tag) and self.hasClass(attrs, "date"):
            return True
        return False

    def isMessageFromName(self, tag, attrs):
        if self.isDivTag(tag) and self.hasClass(attrs, "from_name"):
            return True
        return False

    def isMessageMedia(self, tag, attrs):
        if self.isDivTag(tag) and self.hasClass(attrs, "media_wrap"):
            return True
        return False

    def isMessageVideo(self, tag, attrs):
        if self.isDivTag(tag) and self.hasClass(attrs, "media_video"):
            return True
        return False

    def isMessageImage(self, tag, attrs):
        if self.isImageTag(tag):
            return True
        return False

    def isMessageText(self, tag, attrs):
        if self.isDivTag(tag) and self.hasClass(attrs, "text"):
            return True
        return False

    def isMessageReplied(self, tag, attrs):
        if self.isDivTag(tag) and self.hasClass(attrs, "reply_to"):
            return True
        return False

    def isMessageRepliedLink(self, tag, attrs):
        if self.isAddress(tag) and self._isMessageReplied:
            return True
        return False
